import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "energy_data_set.csv"
)

INTERVAL = pd.Timedelta(minutes=10)

# One day of 10-minute observations.
HISTORY = 144

# Directly recorded building measurements.
SENSORS = ["lights"] + [
    column
    for number in range(1, 10)
    for column in (f"T{number}", f"RH_{number}")
]

REQUIRED = ["date", "Appliances", *SENSORS]

HUMIDITY_COLUMNS = [
    column
    for column in SENSORS
    if column.startswith("RH_")
]


def validate_frame(
    frame: pd.DataFrame,
    min_rows: int = HISTORY,
) -> pd.DataFrame:
    """
    Return a validated, chronologically sorted copy of the observations.

    Validation rules:
        - Required columns must exist.
        - Timestamps must be complete, unique and timezone-naive.
        - Consecutive timestamps must be exactly ten minutes apart.
        - Measurements must be numeric, complete and finite.
        - Appliance and lighting energy must be nonnegative.
        - Relative humidity must be between zero and 100 percent.

    High-consumption observations are retained.
    Missing values are not interpolated or filled.
    Extra columns are excluded from the returned forecasting inputs.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if min_rows < 1:
        raise ValueError("min_rows must be positive.")

    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")

    missing_columns = sorted(
        set(REQUIRED) - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = frame.loc[:, REQUIRED].copy()

    # Parse timestamps without silently discarding invalid values.
    try:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="raise",
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "The date column contains invalid timestamps."
        ) from exc

    if df["date"].isna().any():
        raise ValueError("Timestamps cannot be missing.")

    try:
        timezone = df["date"].dt.tz
    except AttributeError as exc:
        raise ValueError(
            "Use consistent timezone-naive timestamps."
        ) from exc

    if timezone is not None:
        raise ValueError(
            "Use timezone-naive timestamps consistently, "
            "as in the supplied dataset."
        )

    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < min_rows:
        raise ValueError(
            f"At least {min_rows} observations are required; "
            f"received {len(df)}."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found. "
            "Resolve them before forecasting."
        )

    time_differences = df["date"].diff().dropna()

    if not time_differences.eq(INTERVAL).all():
        raise ValueError(
            "Observations must have uninterrupted "
            "10-minute timestamps."
        )

    measurement_columns = REQUIRED[1:]

    for column in measurement_columns:
        try:
            df[column] = pd.to_numeric(
                df[column],
                errors="raise",
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{column}' must contain numeric values."
            ) from exc

    if df[measurement_columns].isna().any().any():
        raise ValueError(
            "Measurements contain missing values. "
            "Provide complete observations."
        )

    measurements = df[measurement_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(measurements).all():
        raise ValueError(
            "Measurements cannot contain infinity "
            "or other non-finite values."
        )

    if (df[["Appliances", "lights"]] < 0).any().any():
        raise ValueError(
            "Appliance and lighting energy cannot be negative."
        )

    invalid_humidity = (
        (df[HUMIDITY_COLUMNS] < 0)
        | (df[HUMIDITY_COLUMNS] > 100)
    )

    if invalid_humidity.any().any():
        raise ValueError(
            "Relative humidity must be between "
            "zero and 100 percent."
        )

    return df


def load_raw(
    path: str | Path = DEFAULT_DATA_PATH,
    min_rows: int = HISTORY,
) -> pd.DataFrame:
    """Read a CSV and validate its forecasting inputs."""
    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {csv_path}"
        )

    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            "The CSV file could not be parsed."
        ) from exc

    return validate_frame(frame, min_rows=min_rows)


def main() -> None:
    """Run an input-validation check without training or saving files."""
    parser = argparse.ArgumentParser(
        description="Validate the appliance-energy dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the original energy dataset CSV.",
    )
    args = parser.parse_args()

    df = load_raw(args.data)

    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    print("Data validation passed.")
    print(f"Observations retained: {len(df):,}")
    print(f"Forecasting input columns: {len(df.columns)}")
    print(f"First timestamp: {df['date'].iloc[0]}")
    print(f"Last timestamp: {df['date'].iloc[-1]}")
    print("Timestamp interval: 10 minutes")
    print("Rows removed: 0")

    print("\nRaw chronological partition sizes:")
    print(f"Training: {train_end:,}")
    print(f"Validation: {validation_end - train_end:,}")
    print(f"Testing: {len(df) - validation_end:,}")

    print("\nNo scaling, model training or file saving was performed.")


if __name__ == "__main__":
    main()