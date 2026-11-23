import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import torch

from src.data_preprocessing import (
    DEFAULT_DATA_PATH,
    HISTORY,
    INTERVAL,
    load_raw,
    validate_frame,
)
from src.feature_engineering import (
    check_horizon,
    make_samples,
    origin_features,
    partition_masks,
    sequence_features,
    sequence_windows,
)
from src.metrics import calculate_metrics
from src.model import EnergyModel, nonnegative


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


class EnergyForecaster:
    """Load a completed experiment and reuse its fitted artifacts."""

    def __init__(
        self,
        run_name="energy_v2",
        horizon=1,
        threads=2,
    ):
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
            run_name,
        ):
            raise ValueError("Invalid experiment name.")

        check_horizon(horizon)

        if threads < 1:
            raise ValueError("threads must be positive.")

        torch.set_num_threads(threads)

        self.run_name = run_name
        self.horizon = horizon

        self.run_root = (
            PROJECT_ROOT / "models" / run_name
        )
        self.model_root = (
            self.run_root / f"horizon_{horizon}"
        )
        self.report_root = (
            PROJECT_ROOT
            / "reports"
            / run_name
            / f"horizon_{horizon}"
        )

        manifest_path = (
            self.run_root / "manifest.json"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Experiment not found: {self.run_root}"
            )

        self.manifest = read_json(manifest_path)

        if self.manifest.get("status") != "complete":
            raise ValueError(
                "The selected experiment is not complete."
            )

        if horizon not in self.manifest["horizons_steps"]:
            raise ValueError(
                "This forecast horizon was not trained "
                "in the selected experiment."
            )

        self.champion = read_json(
            self.model_root / "champion.json"
        )

        self.model_id = self.champion["model_id"]

        self.peak_threshold = float(
            self.champion["peak_threshold_Wh"]
        )

        self._models = {}
        self._load_model(self.model_id, set())

    @staticmethod
    def _check_model_id(model_id):
        if (
            not isinstance(model_id, str)
            or model_id in {".", ".."}
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]*",
                model_id,
            )
        ):
            raise ValueError("Invalid saved model identifier.")

    def _load_model(self, model_id, ancestors):
        """Load each required artifact once."""
        self._check_model_id(model_id)

        if model_id in ancestors:
            raise ValueError(
                "Circular ensemble references were found."
            )

        if model_id in self._models:
            return

        directory = self.model_root / model_id
        metadata = read_json(
            directory / "model.json"
        )

        kind = metadata["kind"]

        node = {
            "kind": kind,
            "metadata": metadata,
        }

        if kind == "ensemble":
            components = metadata.get("components", [])

            if not components:
                raise ValueError(
                    "The ensemble has no components."
                )

            weights = np.asarray(
                [
                    float(component["weight"])
                    for component in components
                ],
                dtype=float,
            )

            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.isclose(
                    weights.sum(),
                    1.0,
                    rtol=0,
                    atol=1e-9,
                )
            ):
                raise ValueError(
                    "Ensemble weights must be finite, "
                    "nonnegative and sum to one."
                )

            next_ancestors = ancestors | {model_id}

            for component in components:
                self._load_model(
                    component["model_id"],
                    next_ancestors,
                )

        elif kind == "tabular":
            node["model"] = joblib.load(
                directory / "estimator.joblib"
            )

        elif kind == "neural":
            node["scalers"] = joblib.load(
                directory / "scalers.joblib"
            )

            model = EnergyModel(
                **metadata["architecture"]
            )

            weights = torch.load(
                directory / "weights.pt",
                map_location="cpu",
                weights_only=True,
            )

            model.load_state_dict(weights)
            model.eval()

            node["model"] = model

        else:
            raise ValueError(
                f"Unsupported model kind: {kind}"
            )

        self._models[model_id] = node

    def _predict_model(
        self,
        model_id,
        features,
        source,
        origins,
        cache,
    ):
        """Predict a batch using already loaded artifacts."""
        if model_id in cache:
            return cache[model_id]

        node = self._models[model_id]
        metadata = node["metadata"]

        if node["kind"] == "ensemble":
            predictions = np.zeros(
                len(features),
                dtype=float,
            )

            for component in metadata["components"]:
                component_predictions = (
                    self._predict_model(
                        component["model_id"],
                        features,
                        source,
                        origins,
                        cache,
                    )
                )

                predictions += (
                    float(component["weight"])
                    * component_predictions
                )

        elif node["kind"] == "tabular":
            predictions = node["model"].predict(
                features.loc[
                    :, metadata["feature_columns"]
                ]
            )

        else:
            scalers = node["scalers"]

            observed = source.loc[
                :, metadata["sequence_columns"]
            ]

            scaled = scalers["x_scaler"].transform(
                observed
            ).astype(np.float32)

            windows = sequence_windows(
                scaled,
                origins,
                lookback=metadata["lookback"],
            )

            batches = []

            with torch.inference_mode():
                for start in range(0, len(windows), 512):
                    inputs = torch.from_numpy(
                        windows[start:start + 512]
                    )

                    batches.append(
                        node["model"](inputs).numpy()
                    )

            scaled_predictions = np.concatenate(
                batches
            )

            predictions = (
                scalers["y_scaler"].inverse_transform(
                    scaled_predictions.reshape(-1, 1)
                ).ravel()
            )

        predictions = nonnegative(predictions)
        cache[model_id] = predictions

        return predictions

    def predict_at_origins(self, history, origins):
        """
        Predict at specified observed row positions.

        Features and neural windows use data through each origin.
        No future target row is included in its inputs.
        """
        raw = validate_frame(history)
        origins = np.asarray(origins)

        if (
            origins.ndim != 1
            or origins.size == 0
            or not np.issubdtype(
                origins.dtype,
                np.integer,
            )
        ):
            raise ValueError(
                "Origins must be a nonempty integer vector."
            )

        if (
            np.any(origins < HISTORY - 1)
            or np.any(origins >= len(raw))
        ):
            raise ValueError(
                "Forecast origins require at least "
                "144 observed rows and must be within history."
            )

        # Discard rows later than the latest requested origin.
        raw = raw.iloc[
            :int(origins.max()) + 1
        ].copy()

        features = origin_features(
            raw,
            self.horizon,
        ).iloc[origins]

        if not np.isfinite(
            features.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Prediction features contain non-finite values."
            )

        source = sequence_features(raw)

        # This cache belongs only to the current prediction batch.
        return self._predict_model(
            self.model_id,
            features,
            source,
            origins,
            cache={},
        )

    def forecast(self, history):
        """
        Predict one future interval from the latest observed history.

        No future input row or future target value is required.
        """
        raw = validate_frame(history)

        # Current features require at most one day of history.
        recent = raw.tail(
            HISTORY
        ).reset_index(drop=True)

        prediction = float(
            self.predict_at_origins(
                recent,
                np.array(
                    [len(recent) - 1],
                    dtype=int,
                ),
            )[0]
        )

        observed_through = recent["date"].iloc[-1]
        forecast_time = (
            observed_through
            + self.horizon * INTERVAL
        )

        metadata = self._models[
            self.model_id
        ]["metadata"]

        components = (
            metadata["components"]
            if metadata["kind"] == "ensemble"
            else [
                {
                    "model_id": self.model_id,
                    "weight": 1.0,
                }
            ]
        )

        return {
            "run_name": self.run_name,
            "model_id": self.model_id,
            "horizon_minutes": self.horizon * 10,
            "observed_through": str(observed_through),
            "forecast_time": str(forecast_time),
            "predicted_Wh": prediction,
            "training_peak_threshold_Wh": (
                self.peak_threshold
            ),
            "high_demand_flag": bool(
                prediction >= self.peak_threshold
            ),
            "components": components,
            "interpretation": (
                "Energy for one future 10-minute interval. "
                "The demand flag is not a probability."
            ),
        }

    def verify_saved_predictions(self, data_path):
        """
        Compare reloaded predictions with the saved test report.

        This performs inference only; no models or scalers are fitted.
        """
        data_path = Path(data_path)

        digest = hashlib.sha256(
            data_path.read_bytes()
        ).hexdigest()

        if digest != self.manifest["dataset_sha256"]:
            raise ValueError(
                "Verification requires the same original CSV "
                "used for this experiment."
            )

        raw = load_raw(data_path)
        samples = make_samples(raw, self.horizon)
        masks, _ = partition_masks(raw, samples)

        report_path = (
            self.report_root / "test_predictions.csv"
        )

        if not report_path.is_file():
            raise FileNotFoundError(
                f"Test prediction report not found: {report_path}"
            )

        report = pd.read_csv(
            report_path,
            parse_dates=["origin_time", "target_time"],
        )

        required = {
            "origin_time",
            "target_time",
            "actual_Wh",
            self.model_id,
        }

        if not required.issubset(report.columns):
            raise ValueError(
                "The saved test report is missing required columns."
            )

        expected_origins = samples.origin_time[
            masks["test"]
        ].to_numpy()

        expected_targets = samples.target_time[
            masks["test"]
        ].to_numpy()

        actual = samples.y[masks["test"]]

        if len(report) != len(actual):
            raise ValueError(
                "The report and reconstructed test set "
                "have different lengths."
            )

        if not np.array_equal(
            report["origin_time"].to_numpy(),
            expected_origins,
        ):
            raise ValueError(
                "Report forecast origins do not match."
            )

        if not np.array_equal(
            report["target_time"].to_numpy(),
            expected_targets,
        ):
            raise ValueError(
                "Report target timestamps do not match."
            )

        if not np.allclose(
            report["actual_Wh"].to_numpy(dtype=float),
            actual,
            rtol=0,
            atol=1e-9,
        ):
            raise ValueError(
                "Reported actual values do not match the dataset."
            )

        predictions = self.predict_at_origins(
            raw,
            samples.origins[masks["test"]],
        )

        saved_predictions = report[
            self.model_id
        ].to_numpy(dtype=float)

        differences = np.abs(
            predictions - saved_predictions
        )

        passed = np.allclose(
            predictions,
            saved_predictions,
            rtol=1e-6,
            atol=1e-3,
        )

        if not passed:
            raise RuntimeError(
                "Reloaded predictions differ from the report. "
                f"Maximum difference: {differences.max():.6f} Wh."
            )

        return {
            "verification_passed": True,
            "model_id": self.model_id,
            "test_observations": len(actual),
            "maximum_absolute_difference_Wh": float(
                differences.max()
            ),
            "recomputed_metrics": calculate_metrics(
                actual,
                predictions,
                peak_threshold=self.peak_threshold,
            ),
        }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--run-name",
        default="energy_v2",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare reloaded predictions with the test report.",
    )
    args = parser.parse_args()

    forecaster = EnergyForecaster(
        run_name=args.run_name,
        horizon=args.horizon,
        threads=args.threads,
    )

    if args.verify:
        result = forecaster.verify_saved_predictions(
            args.history
        )
    else:
        result = forecaster.forecast(
            load_raw(args.history)
        )

    print(
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()