import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__:
    from .data_preprocessing import (
        DEFAULT_DATA_PATH,
        HISTORY,
        INTERVAL,
        SENSORS,
        load_raw,
    )
else:
    from data_preprocessing import (
        DEFAULT_DATA_PATH,
        HISTORY,
        INTERVAL,
        SENSORS,
        load_raw,
    )


ENERGY_LAGS = (
    0, 1, 2, 3, 5, 6, 11,
    17, 18, 23, 35, 71, 143,
)

ROLLING_WINDOWS = (6, 18, 24, 72, 144)


def check_horizon(horizon: int) -> None:
    """Validate a horizon expressed in ten-minute steps."""
    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, np.integer))
        or not 1 <= horizon <= 144
    ):
        raise ValueError(
            "Horizon must be an integer from 1 to 144 "
            "ten-minute steps."
        )


def calendar_features(
    dates: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    """Encode known calendar information using cyclic features."""
    hour = dates.dt.hour + dates.dt.minute / 60.0
    weekday = dates.dt.dayofweek

    return pd.DataFrame(
        {
            f"{prefix}_hour_sin": np.sin(
                2 * np.pi * hour / 24
            ),
            f"{prefix}_hour_cos": np.cos(
                2 * np.pi * hour / 24
            ),
            f"{prefix}_weekday_sin": np.sin(
                2 * np.pi * weekday / 7
            ),
            f"{prefix}_weekday_cos": np.cos(
                2 * np.pi * weekday / 7
            ),
            f"{prefix}_is_weekend": (
                weekday >= 5
            ).astype(float),
        },
        index=dates.index,
    )


def origin_features(
    raw: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Create tabular features available at each forecast origin.

    At origin t:
        - Measurements through t are already observed.
        - The prediction target is Appliances[t + horizon].
        - No future energy or sensor values are included.

    Input must already be validated by load_raw() or
    validate_frame().
    """
    check_horizon(horizon)

    features = raw[SENSORS].astype(float).copy()
    energy = raw["Appliances"]

    # Lag offsets are measured relative to the forecast origin.
    # Lag 0 is the consumption already observed at origin t.
    for lag in ENERGY_LAGS:
        features[f"energy_lag_{lag}"] = energy.shift(lag)

    # These trailing windows include the observed origin value.
    # This is valid because the target is strictly later.
    for window in ROLLING_WINDOWS:
        rolling = energy.rolling(
            window=window,
            min_periods=window,
        )

        features[f"energy_mean_{window}"] = rolling.mean()
        features[f"energy_std_{window}"] = rolling.std(
            ddof=0
        )
        features[f"energy_max_{window}"] = rolling.max()

    features["energy_change_1"] = energy.diff()

    # Yesterday's consumption at the future target's clock time.
    #
    # Example:
    #   origin = today 10:00
    #   horizon = 1
    #   target = today 10:10
    #   reference = yesterday 10:10
    #
    # This reference is already observed at the origin.
    features["previous_day_target"] = energy.shift(
        144 - horizon
    )

    features["indoor_temp_humidity"] = (
        raw["T1"] * raw["RH_1"]
    )

    origin_calendar = calendar_features(
        raw["date"],
        prefix="origin",
    )

    # Future calendar information is known in advance.
    # Future sensor measurements are not used.
    target_dates = raw["date"] + horizon * INTERVAL

    target_calendar = calendar_features(
        target_dates,
        prefix="target",
    )

    return pd.concat(
        [
            features,
            origin_calendar,
            target_calendar,
        ],
        axis=1,
    )


def sequence_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create observed per-step inputs for LSTM/GRU models.

    Scaling will be fitted on training data later.
    """
    measurements = raw[
        ["Appliances", *SENSORS]
    ].astype(float)

    calendar = calendar_features(
        raw["date"],
        prefix="observed",
    )

    return pd.concat(
        [measurements, calendar],
        axis=1,
    )


@dataclass
class Samples:
    """Aligned tabular inputs, labels and forecast timestamps."""

    X: pd.DataFrame
    y: np.ndarray
    origins: np.ndarray
    origin_time: pd.Series
    target_time: pd.Series


def make_samples(
    raw: pd.DataFrame,
    horizon: int = 1,
) -> Samples:
    """
    Construct supervised forecasting samples.

    Each sample uses features at origin t and the label at t+h.

    The first HISTORY-1 origins provide warm-up history.
    The last h origins have no observed future label.
    Raw observations are not deleted or modified.
    """
    check_horizon(horizon)

    minimum_rows = HISTORY + horizon

    if len(raw) < minimum_rows:
        raise ValueError(
            f"At least {minimum_rows} observations are "
            f"needed to construct labeled samples for "
            f"horizon={horizon}."
        )

    features = origin_features(raw, horizon)

    origins = np.arange(
        HISTORY - 1,
        len(raw) - horizon,
        dtype=int,
    )

    target_positions = origins + horizon

    X = features.iloc[origins].reset_index(drop=True)

    y = raw["Appliances"].to_numpy(
        dtype=float
    )[target_positions]

    origin_time = raw["date"].iloc[
        origins
    ].reset_index(drop=True)

    target_time = raw["date"].iloc[
        target_positions
    ].reset_index(drop=True)

    if not np.isfinite(
        X.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Engineered features contain missing "
            "or non-finite values."
        )

    expected_difference = horizon * INTERVAL

    if not (
        target_time - origin_time
    ).eq(expected_difference).all():
        raise ValueError(
            "Forecast origins and targets are "
            "not correctly aligned."
        )

    return Samples(
        X=X,
        y=y,
        origins=origins,
        origin_time=origin_time,
        target_time=target_time,
    )


def partition_masks(
    raw: pd.DataFrame,
    samples: Samples,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Assign samples using their target timestamps.

    Boundaries match the EDA:
        - First 70% of the raw timeline: training
        - Next 15%: validation
        - Final 15%: testing

    Earlier observed history remains available as context
    at validation and testing boundaries.
    """
    train_end = int(len(raw) * 0.70)
    validation_end = int(len(raw) * 0.85)

    first_validation = raw["date"].iloc[train_end]
    first_test = raw["date"].iloc[validation_end]

    targets = samples.target_time

    masks = {
        "train": (
            targets < first_validation
        ).to_numpy(),
        "validation": (
            (targets >= first_validation)
            & (targets < first_test)
        ).to_numpy(),
        "test": (
            targets >= first_test
        ).to_numpy(),
    }

    for name, mask in masks.items():
        if int(mask.sum()) < 20:
            raise ValueError(
                f"The {name} partition contains fewer "
                "than 20 labeled samples."
            )

    membership = sum(
        mask.astype(int)
        for mask in masks.values()
    )

    if not np.all(membership == 1):
        raise ValueError(
            "Every sample must belong to exactly "
            "one chronological partition."
        )

    boundaries = {
        "validation_starts": str(first_validation),
        "test_starts": str(first_test),
    }

    return masks, boundaries


def sequence_windows(
    values: np.ndarray,
    origins: np.ndarray,
    lookback: int = 24,
) -> np.ndarray:
    """
    Build histories ending at each forecast origin.

    For lookback=24, each sequence contains four hours
    of observed ten-minute intervals.

    The future target row is never included.
    """
    values = np.asarray(values)
    origins = np.asarray(origins)

    if values.ndim != 2:
        raise ValueError(
            "Sequence inputs must be a two-dimensional array."
        )

    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, (int, np.integer))
        or not 1 <= lookback <= HISTORY
    ):
        raise ValueError(
            f"Lookback must be an integer from 1 to {HISTORY}."
        )

    if (
        origins.ndim != 1
        or origins.size == 0
        or not np.issubdtype(origins.dtype, np.integer)
    ):
        raise ValueError(
            "Origins must be a nonempty, one-dimensional "
            "array of integer positions."
        )

    if (
        np.any(origins < lookback - 1)
        or np.any(origins >= len(values))
    ):
        raise ValueError(
            "Sequence origins are outside the available history."
        )

    offsets = np.arange(
        -lookback + 1,
        1,
        dtype=int,
    )

    positions = (
        origins[:, None] + offsets[None, :]
    )

    return values[positions].astype(np.float32)


def main() -> None:
    """Check feature construction without fitting or saving models."""
    parser = argparse.ArgumentParser(
        description="Check forecasting feature construction."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the original energy CSV.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in ten-minute steps.",
    )
    args = parser.parse_args()

    raw = load_raw(args.data)
    samples = make_samples(raw, args.horizon)

    masks, boundaries = partition_masks(
        raw,
        samples,
    )

    print("Feature engineering checks passed.")
    print(f"Forecast horizon: {args.horizon * 10} minutes")
    print(f"Raw observations retained: {len(raw):,}")
    print(f"Tabular feature columns: {samples.X.shape[1]}")
    print(f"Eligible labeled samples: {len(samples.y):,}")

    print("\nLabeled chronological partition sizes:")
    for name, mask in masks.items():
        print(f"{name.title()}: {int(mask.sum()):,}")

    print("\nTarget-time boundaries:")
    print(
        "Validation starts:",
        boundaries["validation_starts"],
    )
    print(
        "Testing starts:",
        boundaries["test_starts"],
    )

    observed_inputs = sequence_features(raw)

    # Demonstrate the first three windows only.
    example_windows = sequence_windows(
        observed_inputs.to_numpy(dtype=float),
        samples.origins[:3],
        lookback=24,
    )

    print(
        "\nExample sequence shape:",
        example_windows.shape,
    )
    print(
        "Shape meaning: "
        "(samples, historical steps, input features)"
    )

    preview = pd.DataFrame(
        {
            "origin_time": samples.origin_time.head(5),
            "target_time": samples.target_time.head(5),
            "observed_energy_Wh": (
                samples.X["energy_lag_0"].head(5)
            ),
            "future_target_Wh": samples.y[:5],
        }
    )

    print("\nAlignment preview:")
    print(preview.to_string(index=False))

    print(
        "\nNo scaling, model training or file saving "
        "was performed."
    )


if __name__ == "__main__":
    main()