import argparse
import hashlib
import platform
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch

from threadpoolctl import threadpool_limits

from src.data_preprocessing import (
    DEFAULT_DATA_PATH,
    HISTORY,
    load_raw,
)
from src.feature_engineering import (
    check_horizon,
    make_samples,
    partition_masks,
    sequence_features,
)
from src.metrics import (
    calculate_metrics,
    peak_threshold_from_training,
)
from src.model import BaselineModels
from src.train import (
    fit_neural,
    predict_saved,
    read_json,
    save_plots,
    write_json,
)


def train_horizon(
    raw,
    horizon,
    args,
    run_root,
    reports_root,
):
    """Train initial candidates, freeze selection, then evaluate testing."""
    samples = make_samples(raw, horizon)
    masks, boundaries = partition_masks(raw, samples)
    source = sequence_features(raw)

    model_root = run_root / f"horizon_{horizon}"
    report_root = reports_root / f"horizon_{horizon}"

    model_root.mkdir()
    report_root.mkdir()

    X_train = samples.X.loc[masks["train"]]
    y_train = samples.y[masks["train"]]

    X_validation = samples.X.loc[masks["validation"]]
    validation_origins = samples.origins[
        masks["validation"]
    ]
    validation_actual = samples.y[
        masks["validation"]
    ]

    peak_threshold = peak_threshold_from_training(
        y_train,
        quantile=0.90,
    )

    validation_rows = []

    print(
        f"\n=== Initial training: "
        f"{horizon * 10} minutes ahead ===",
        flush=True,
    )
    print(
        "Labeled samples:",
        {
            name: int(mask.sum())
            for name, mask in masks.items()
        },
        flush=True,
    )

    def evaluate_validation(model_id):
        predictions = predict_saved(
            model_root / model_id,
            X_validation,
            source,
            validation_origins,
        )

        metrics = calculate_metrics(
            validation_actual,
            predictions,
            peak_threshold=peak_threshold,
        )

        validation_rows.append(
            {
                "model": model_id,
                "phase": "initial",
                **metrics,
            }
        )

        print(
            f"Validation RMSE: "
            f"{metrics['RMSE_Wh']:.4f} Wh",
            flush=True,
        )

    # All initial tabular candidates use all engineered features.
    candidates = BaselineModels(
        random_state=args.seed,
        quick=False,
    ).get_models()

    for model_id, estimator in candidates.items():
        print(f"\nTraining {model_id}", flush=True)

        estimator.fit(
            X_train,
            y_train,
        )

        directory = model_root / model_id
        directory.mkdir()

        joblib.dump(
            estimator,
            directory / "estimator.joblib",
            compress=3,
        )

        write_json(
            directory / "model.json",
            {
                "kind": "tabular",
                "phase": "initial",
                "feature_columns": (
                    samples.X.columns.tolist()
                ),
                "estimator_description": repr(estimator),
                "output_policy": "clip below zero",
            },
        )

        # Reloading also checks the saved artifact's inference.
        evaluate_validation(model_id)

    # Initial recurrent configurations match the earlier baseline:
    # one layer, dropout 0.2, hidden units 32 or 64.
    for family in ("LSTM", "GRU"):
        for hidden_dim in (32, 64):
            model_id = f"{family}_hidden{hidden_dim}"
            directory = model_root / model_id

            configuration = {
                "hidden_dim": hidden_dim,
                "num_layers": 1,
                "dropout_prob": 0.2,
                "lookback": args.lookback,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "weight_decay": 1e-5,
                "epochs": args.epochs,
            }

            print(f"\nTraining {model_id}", flush=True)

            fit_neural(
                family=family,
                configuration=configuration,
                raw=raw,
                source=source,
                samples=samples,
                masks=masks,
                directory=directory,
                args=args,
            )

            # The shared helper is also used by optimization.
            # Label these particular configurations as initial.
            metadata = read_json(
                directory / "model.json"
            )
            metadata["phase"] = "initial"

            write_json(
                directory / "model.json",
                metadata,
            )

            evaluate_validation(model_id)

    validation_frame = pd.DataFrame(
        validation_rows
    ).sort_values(
        ["RMSE_Wh", "model"]
    )

    validation_frame.to_csv(
        report_root / "validation_metrics.csv",
        index=False,
    )

    selected = min(
        validation_rows,
        key=lambda row: (
            row["RMSE_Wh"],
            row["model"],
        ),
    )

    champion = {
        "model_id": selected["model"],
        "horizon_steps": horizon,
        "horizon_minutes": horizon * 10,
        "selection_metric": "validation RMSE in Wh",
        "validation_metrics": selected,
        "peak_threshold_Wh": peak_threshold,
        "boundaries": boundaries,
        "partition_sizes": {
            name: int(mask.sum())
            for name, mask in masks.items()
        },
        "selection_order": (
            "frozen before test predictions and metrics"
        ),
    }

    write_json(
        model_root / "champion.json",
        champion,
    )

    print(
        f"\nFrozen initial winner: {selected['model']}",
        flush=True,
    )
    print("Evaluating testing...", flush=True)

    test_predictions = pd.DataFrame(
        {
            "origin_time": (
                samples.origin_time[
                    masks["test"]
                ].to_numpy()
            ),
            "target_time": (
                samples.target_time[
                    masks["test"]
                ].to_numpy()
            ),
            "actual_Wh": samples.y[masks["test"]],
        }
    )

    test_rows = []

    for row in validation_rows:
        model_id = row["model"]

        predictions = predict_saved(
            model_root / model_id,
            samples.X.loc[masks["test"]],
            source,
            samples.origins[masks["test"]],
        )

        metrics = calculate_metrics(
            samples.y[masks["test"]],
            predictions,
            peak_threshold=peak_threshold,
        )

        test_rows.append(
            {
                "model": model_id,
                "phase": "initial",
                "selected_by_validation": (
                    model_id == selected["model"]
                ),
                **metrics,
            }
        )

        test_predictions[model_id] = predictions

    test_frame = pd.DataFrame(test_rows)

    test_frame.to_csv(
        report_root / "test_metrics.csv",
        index=False,
    )
    test_predictions.to_csv(
        report_root / "test_predictions.csv",
        index=False,
    )

    save_plots(
        test_predictions,
        test_frame,
        selected["model"],
        report_root,
    )

    selected_test = next(
        row
        for row in test_rows
        if row["model"] == selected["model"]
    )

    print("\nInitial test results:")
    print(
        test_frame[
            [
                "model",
                "selected_by_validation",
                "MAE_Wh",
                "RMSE_Wh",
                "R2",
            ]
        ].to_string(index=False)
    )

    return {
        **champion,
        "test_metrics": selected_test,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--run-name",
        default="energy_v1",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
        args.run_name,
    ):
        parser.error("Invalid experiment name.")

    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.threads < 1
    ):
        parser.error(
            "Epochs, patience, batch size and threads "
            "must be positive."
        )

    if (
        not np.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        parser.error(
            "Learning rate must be finite and positive."
        )

    if not 1 <= args.lookback <= HISTORY:
        parser.error(
            f"Lookback must be between 1 and {HISTORY}."
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA is unavailable. Use --device cpu."
        )

    horizons = list(dict.fromkeys(args.horizons))

    for horizon in horizons:
        check_horizon(horizon)

    run_root = (
        PROJECT_ROOT / "models" / args.run_name
    )
    reports_root = (
        PROJECT_ROOT / "reports" / args.run_name
    )

    if run_root.exists() or reports_root.exists():
        raise FileExistsError(
            "This experiment already exists. "
            "Choose another --run-name."
        )

    raw = load_raw(args.data)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    run_root.mkdir(parents=True)
    reports_root.mkdir(parents=True)

    manifest = {
        "schema_version": 1,
        "project": "Appliance Energy Forecasting",
        "run_name": args.run_name,
        "status": "training",
        "horizons_steps": horizons,
        "dataset_rows": len(raw),
        "dataset_sha256": hashlib.sha256(
            args.data.read_bytes()
        ).hexdigest(),
        "seed": args.seed,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lookback": args.lookback,
        "threads": args.threads,
        "device": args.device,
        "rows_removed_as_outliers": 0,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "pytorch": torch.__version__,
        },
        "forecast_contract": (
            "Predict y[t+h] using measurements "
            "observed through t."
        ),
        "evaluation": (
            "Chronological historical rolling-origin evaluation."
        ),
    }

    write_json(
        run_root / "manifest.json",
        manifest,
    )

    try:
        with threadpool_limits(limits=args.threads):
            summaries = [
                train_horizon(
                    raw,
                    horizon,
                    args,
                    run_root,
                    reports_root,
                )
                for horizon in horizons
            ]

        write_json(
            run_root / "summary.json",
            summaries,
        )
        write_json(
            reports_root / "summary.json",
            summaries,
        )

        manifest["status"] = "complete"

        write_json(
            run_root / "manifest.json",
            manifest,
        )
        write_json(
            PROJECT_ROOT / "models" / "active_run.json",
            {"run_name": args.run_name},
        )

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)

        write_json(
            run_root / "manifest.json",
            manifest,
        )
        raise

    print("\nInitial training completed.")
    print(f"Models: {run_root}")
    print(f"Reports: {reports_root}")


if __name__ == "__main__":
    main()