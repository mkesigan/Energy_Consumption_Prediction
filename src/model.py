import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class ColumnForecast(RegressorMixin, BaseEstimator):
    """
    Predict using an already observed feature column.

    Examples:
        energy_lag_0:
            Persistence forecast using consumption at the origin.

        previous_day_target:
            Yesterday's consumption at the target's clock time.
    """

    def __init__(self, column: str = "energy_lag_0"):
        self.column = column

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "ColumnForecast requires a pandas DataFrame."
            )

        if self.column not in X.columns:
            raise ValueError(
                f"Required feature is missing: {self.column}"
            )

        values = X[self.column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Feature '{self.column}' contains "
                "missing or non-finite values."
            )

    def fit(self, X: pd.DataFrame, y=None):
        """Validate the input contract; no parameters are learned."""
        self._validate_input(X)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.is_fitted_ = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the observed reference values."""
        check_is_fitted(self, attributes=["is_fitted_"])
        self._validate_input(X)

        return X[self.column].to_numpy(
            dtype=float,
            copy=True,
        )


class BaselineModels:
    """Construct fresh, unfitted tabular model candidates."""

    def __init__(
        self,
        random_state: int = 42,
        quick: bool = False,
    ):
        self.random_state = random_state
        self.quick = quick

    def get_models(self) -> dict:
        """
        Return unfitted candidate estimators.

        StandardScaler is inside each linear-model pipeline.
        It will learn its parameters only when that pipeline
        is fitted on training data.
        """
        models = {
            "Persistence": ColumnForecast(
                column="energy_lag_0"
            ),
            "DailySeasonal": ColumnForecast(
                column="previous_day_target"
            ),
            "LinearRegression": make_pipeline(
                StandardScaler(),
                LinearRegression(),
            ),
        }

        ridge_alphas = (
            [10.0]
            if self.quick
            else [1.0, 10.0, 100.0]
        )

        for alpha in ridge_alphas:
            models[f"Ridge_a{alpha:g}"] = make_pipeline(
                StandardScaler(),
                Ridge(alpha=alpha),
            )

        forest_leaf_sizes = (
            [4]
            if self.quick
            else [2, 8]
        )

        for leaf_size in forest_leaf_sizes:
            name = f"RandomForest_leaf{leaf_size}"

            models[name] = RandomForestRegressor(
                n_estimators=40 if self.quick else 160,
                min_samples_leaf=leaf_size,
                max_features=0.8,
                random_state=self.random_state,
                n_jobs=2,
            )

        boosting_leaf_counts = (
            [15]
            if self.quick
            else [15, 31]
        )

        for leaf_count in boosting_leaf_counts:
            name = (
                f"HistGradientBoosting_leaves{leaf_count}"
            )

            models[name] = HistGradientBoostingRegressor(
                max_iter=80 if self.quick else 250,
                max_leaf_nodes=leaf_count,
                learning_rate=0.05,
                l2_regularization=1.0,
                # Avoid an internally shuffled validation split.
                # Our training workflow uses chronological validation.
                early_stopping=False,
                random_state=self.random_state,
            )

        return models


class EnergyModel(nn.Module):
    """
    LSTM or GRU sequence-to-one regression model.

    Input:
        (batch_size, lookback_steps, input_dim)

    Output:
        (batch_size,)

    Input and target scaling are handled by the training
    workflow, not inside this architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout_prob: float = 0.2,
        model_type: str = "LSTM",
    ):
        super().__init__()

        for name, value in {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if not 0 <= dropout_prob < 1:
            raise ValueError(
                "dropout_prob must be at least zero "
                "and less than one."
            )

        model_type = model_type.upper()

        if model_type not in {"LSTM", "GRU"}:
            raise ValueError(
                "model_type must be 'LSTM' or 'GRU'."
            )

        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout_prob": float(dropout_prob),
            "model_type": model_type,
        }

        recurrent_class = (
            nn.LSTM
            if model_type == "LSTM"
            else nn.GRU
        )

        self.rnn = recurrent_class(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout_prob
                if num_layers > 1
                else 0.0
            ),
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict one future value from each historical sequence."""
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch_size, lookback_steps, input_dim)."
            )

        if x.shape[1] < 1:
            raise ValueError(
                "Each input sequence must contain "
                "at least one historical step."
            )

        if x.shape[2] != self.config["input_dim"]:
            raise ValueError(
                f"Expected {self.config['input_dim']} "
                f"input features, received {x.shape[2]}."
            )

        recurrent_output, _ = self.rnn(x)

        # The last step is the latest observed interval.
        last_observed_state = recurrent_output[:, -1, :]

        prediction = self.regression_head(
            last_observed_state
        )

        # Preserve the batch dimension, including batch_size=1.
        return prediction.squeeze(-1)

    def get_config(self) -> dict:
        """Return the configuration needed to rebuild this model."""
        return self.config.copy()


def nonnegative(predictions) -> np.ndarray:
    """
    Apply the physical output constraint in original Wh units.

    For neural models, call this AFTER reversing target scaling.
    Do not clip standardized target predictions before inversion.
    """
    values = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(values).all():
        raise ValueError(
            "Predictions contain non-finite values."
        )

    return np.maximum(values, 0.0)


def main() -> None:
    """Check architecture shapes without fitting or saving models."""
    parser = argparse.ArgumentParser(
        description="Check forecasting model definitions."
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=25,
        help="Number of per-step neural input features.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=24,
        help="Number of observed steps per sequence.",
    )
    args = parser.parse_args()

    if args.input_dim < 1 or args.lookback < 1:
        parser.error(
            "--input-dim and --lookback must be positive."
        )

    candidates = BaselineModels().get_models()

    print("Tabular model definitions loaded.")
    print(f"Candidate count: {len(candidates)}")

    for name in candidates:
        print(f"  - {name}")

    # Dummy inputs are used only to verify tensor shapes.
    # They are not observations or meaningful forecasts.
    dummy_inputs = torch.zeros(
        2,
        args.lookback,
        args.input_dim,
        dtype=torch.float32,
    )

    print(
        "\nNeural input shape:",
        tuple(dummy_inputs.shape),
    )

    for model_type in ("LSTM", "GRU"):
        model = EnergyModel(
            input_dim=args.input_dim,
            hidden_dim=64,
            num_layers=1,
            dropout_prob=0.2,
            model_type=model_type,
        )

        model.eval()

        with torch.inference_mode():
            output = model(dummy_inputs)

        if output.shape != (2,):
            raise RuntimeError(
                f"{model_type} returned an unexpected "
                f"output shape: {tuple(output.shape)}"
            )

        print(
            f"{model_type} output shape:",
            tuple(output.shape),
        )

    print("\nArchitecture checks passed.")
    print(
        "No model fitting, optimizer updates "
        "or file saving was performed."
    )


if __name__ == "__main__":
    main()