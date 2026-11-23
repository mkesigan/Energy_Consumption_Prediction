"""Evaluation metrics for appliance-energy forecasting."""

import json

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def _as_vector(values, name: str) -> np.ndarray:
    """Convert single-target observations into a finite 1D array."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain numeric values."
        ) from exc

    if array.ndim == 0:
        array = array.reshape(1)

    elif array.ndim == 2 and array.shape[1] == 1:
        array = array.reshape(-1)

    elif array.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array "
            "or a single-column array."
        )

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains missing or non-finite values."
        )

    return array


def peak_threshold_from_training(
    y_train,
    quantile: float = 0.90,
) -> float:
    """
    Calculate a high-demand threshold from training targets only.

    Do not calculate this threshold from validation or test targets.
    Values must be in original Wh units.
    """
    if not 0 < quantile < 1:
        raise ValueError(
            "quantile must be strictly between zero and one."
        )

    training_values = _as_vector(
        y_train,
        name="y_train",
    )

    return float(
        np.quantile(training_values, quantile)
    )


def calculate_metrics(
    y_true,
    y_pred,
    peak_threshold: float | None = None,
) -> dict[str, float | int | None]:
    """
    Evaluate single-target predictions in original Wh units.

    Metrics:
        MAE_Wh:
            Mean absolute prediction error.

        RMSE_Wh:
            Root mean squared error; larger errors receive
            more weight than in MAE.

        R2:
            Regression coefficient of determination.
            This is not an accuracy percentage.
            Returns None for fewer than two observations
            or a constant target.

        bias_Wh:
            Mean predicted minus actual consumption.
            Positive means average overprediction.
            Negative means average underprediction.

        peak_MAE_Wh:
            MAE for actual consumption at or above the
            supplied training-derived threshold.

        peak_count:
            Number of evaluated high-demand observations.
            None means no threshold was supplied.

        n:
            Number of evaluated observations.

    This function does not scale or clip predictions.
    Neural target scaling must be reversed before evaluation.
    Apply the chosen nonnegative-output policy consistently
    before calling this function.
    """
    actual = _as_vector(
        y_true,
        name="y_true",
    )
    predicted = _as_vector(
        y_pred,
        name="y_pred",
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted arrays must have "
            f"matching lengths: received {len(actual)} "
            f"and {len(predicted)}."
        )

    mae = float(
        mean_absolute_error(actual, predicted)
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(actual, predicted)
        )
    )

    if len(actual) < 2 or np.all(actual == actual[0]):
        r2 = None
    else:
        r2 = float(
            r2_score(actual, predicted)
        )

    bias = float(
        np.mean(predicted - actual)
    )

    peak_mae = None
    peak_count = None

    if peak_threshold is not None:
        threshold = float(peak_threshold)

        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError(
                "peak_threshold must be a finite, "
                "nonnegative value in Wh."
            )

        peak_mask = actual >= threshold
        peak_count = int(peak_mask.sum())

        if peak_count > 0:
            peak_mae = float(
                mean_absolute_error(
                    actual[peak_mask],
                    predicted[peak_mask],
                )
            )

    return {
        "MAE_Wh": mae,
        "RMSE_Wh": rmse,
        "R2": r2,
        "bias_Wh": bias,
        "peak_MAE_Wh": peak_mae,
        "peak_count": peak_count,
        "n": int(len(actual)),
    }


def main() -> None:
    """
    Demonstrate metric calculations using synthetic examples.

    These values are not project results.
    No dataset is loaded and no model is trained.
    """
    example_training_targets = np.array(
        [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        dtype=float,
    )

    example_actual = np.array(
        [50, 100, 150, 200, 250],
        dtype=float,
    )

    example_predictions = np.array(
        [60, 90, 160, 190, 240],
        dtype=float,
    )

    threshold = peak_threshold_from_training(
        example_training_targets,
        quantile=0.90,
    )

    metrics = calculate_metrics(
        example_actual,
        example_predictions,
        peak_threshold=threshold,
    )

    print("Synthetic metric demonstration.")
    print("These are not dataset or model results.")
    print(
        f"Example training-derived peak threshold: "
        f"{threshold:.2f} Wh"
    )

    print(
        json.dumps(
            metrics,
            indent=2,
            allow_nan=False,
        )
    )

    print(
        "\nNo dataset loading, model training "
        "or file saving was performed."
    )


if __name__ == "__main__":
    main()