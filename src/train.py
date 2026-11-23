import argparse
import hashlib
import itertools
import json
import platform
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_preprocessing import DEFAULT_DATA_PATH, load_raw
from src.feature_engineering import (
    check_horizon,
    make_samples,
    partition_masks,
    sequence_features,
    sequence_windows,
)
from src.metrics import (
    calculate_metrics,
    peak_threshold_from_training,
)
from src.model import EnergyModel, nonnegative


def plain(value):
    """Convert NumPy scalars and nested configurations to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: plain(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]

    return value


def write_json(path, content):
    """Write JSON without partially replacing an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            plain(content),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def sample_configurations(space, count, seed, anchors):
    """
    Include predefined comparison configurations, then sample more.

    All configurations are determined before test evaluation.
    """
    total = len(ParameterGrid(space))

    if count > total:
        raise ValueError(
            f"Requested {count} trials, but only "
            f"{total} configurations exist."
        )

    draw_count = min(
        total,
        count + len(anchors) + 20,
    )

    sampled = ParameterSampler(
        space,
        n_iter=draw_count,
        random_state=seed,
    )

    configurations = []
    seen = set()

    for configuration in [*anchors, *sampled]:
        configuration = plain(configuration)
        signature = json.dumps(
            configuration,
            sort_keys=True,
        )

        if signature in seen:
            continue

        configurations.append(configuration)
        seen.add(signature)

        if len(configurations) == count:
            break

    if len(configurations) != count:
        raise RuntimeError(
            "Could not generate enough unique configurations."
        )

    return configurations


def family_from_name(name):
    for prefix, family in (
        ("HistGradientBoosting", "Boosting"),
        ("RandomForest", "RandomForest"),
        ("LinearRegression", "LinearRegression"),
        ("Ridge", "Ridge"),
        ("LSTM", "LSTM"),
        ("GRU", "GRU"),
    ):
        if name.startswith(prefix):
            return family

    return name


def neural_output(model, inputs, device):
    """Return standardized predictions without updating weights."""
    model.eval()
    predictions = []

    with torch.inference_mode():
        for start in range(0, len(inputs), 512):
            batch = torch.from_numpy(
                inputs[start:start + 512]
            ).to(device)

            predictions.append(
                model(batch).cpu().numpy()
            )

    return np.concatenate(predictions)


def predict_saved(directory, X, source, origins):
    """Reload a saved individual model or weighted ensemble."""
    directory = Path(directory)
    metadata = read_json(directory / "model.json")

    if metadata["kind"] == "ensemble":
        predictions = np.zeros(len(X), dtype=float)

        for component in metadata["components"]:
            component_predictions = predict_saved(
                directory.parent / component["model_id"],
                X,
                source,
                origins,
            )

            predictions += (
                component["weight"]
                * component_predictions
            )

        return nonnegative(predictions)

    if metadata["kind"] == "tabular":
        estimator = joblib.load(
            directory / "estimator.joblib"
        )

        predictions = estimator.predict(
            X.loc[:, metadata["feature_columns"]]
        )

        return nonnegative(predictions)

    if metadata["kind"] != "neural":
        raise ValueError(
            f"Unknown saved model kind: {metadata['kind']}"
        )

    scalers = joblib.load(
        directory / "scalers.joblib"
    )

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

    model = EnergyModel(
        **metadata["architecture"]
    )

    weights = torch.load(
        directory / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)

    scaled_predictions = neural_output(
        model,
        windows,
        torch.device("cpu"),
    )

    predictions = scalers["y_scaler"].inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()

    return nonnegative(predictions)


def fit_tabular(
    family,
    configuration,
    X_train,
    y_train,
    directory,
    seed,
):
    """Fit feature selection and the estimator on training data only."""
    parameters = configuration.copy()
    keep = parameters.pop("keep")

    selection = (
        "passthrough"
        if keep is None
        else SelectKBest(
            score_func=f_regression,
            k=keep,
        )
    )

    if family == "Ridge":
        estimator = Ridge(**parameters)

        pipeline = Pipeline(
            [
                ("selection", selection),
                ("scaling", StandardScaler()),
                ("estimator", estimator),
            ]
        )

    elif family == "RandomForest":
        estimator = RandomForestRegressor(
            **parameters,
            random_state=seed,
            n_jobs=2,
        )

        pipeline = Pipeline(
            [
                ("selection", selection),
                ("estimator", estimator),
            ]
        )

    elif family == "Boosting":
        estimator = HistGradientBoostingRegressor(
            **parameters,
            early_stopping=False,
            random_state=seed,
        )

        pipeline = Pipeline(
            [
                ("selection", selection),
                ("estimator", estimator),
            ]
        )

    else:
        raise ValueError(f"Unsupported family: {family}")

    pipeline.fit(X_train, y_train)
    directory.mkdir()

    joblib.dump(
        pipeline,
        directory / "estimator.joblib",
        compress=3,
    )

    if keep is None:
        selected_features = X_train.columns.tolist()
    else:
        selector = pipeline.named_steps["selection"]

        selected_features = X_train.columns[
            selector.get_support()
        ].tolist()

        pd.DataFrame(
            {
                "feature": X_train.columns,
                "training_F_score": selector.scores_,
                "selected": selector.get_support(),
            }
        ).to_csv(
            directory / "feature_selection.csv",
            index=False,
        )

    write_json(
        directory / "model.json",
        {
            "kind": "tabular",
            "family": family,
            "phase": "optimized",
            "configuration": configuration,
            "feature_columns": X_train.columns.tolist(),
            "selected_features": selected_features,
            "output_policy": "clip below zero",
        },
    )


def fit_neural(
    family,
    configuration,
    raw,
    source,
    samples,
    masks,
    directory,
    args,
):
    """Tune one neural configuration using chronological validation."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    training_cutoff = int(len(raw) * 0.70)

    x_scaler = StandardScaler().fit(
        source.iloc[:training_cutoff]
    )

    y_scaler = StandardScaler().fit(
        samples.y[masks["train"]].reshape(-1, 1)
    )

    scaled_source = x_scaler.transform(
        source
    ).astype(np.float32)

    X_train = sequence_windows(
        scaled_source,
        samples.origins[masks["train"]],
        configuration["lookback"],
    )

    X_validation = sequence_windows(
        scaled_source,
        samples.origins[masks["validation"]],
        configuration["lookback"],
    )

    y_train = y_scaler.transform(
        samples.y[masks["train"]].reshape(-1, 1)
    ).ravel().astype(np.float32)

    y_validation_scaled = y_scaler.transform(
        samples.y[masks["validation"]].reshape(-1, 1)
    ).ravel()

    validation_actual = samples.y[
        masks["validation"]
    ]

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        ),
        batch_size=configuration["batch_size"],
        shuffle=False,
    )

    model = EnergyModel(
        input_dim=int(source.shape[1]),
        hidden_dim=configuration["hidden_dim"],
        num_layers=configuration["num_layers"],
        dropout_prob=configuration["dropout_prob"],
        model_type=family,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration["learning_rate"],
        weight_decay=configuration["weight_decay"],
    )
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_weights = None
    best_epoch = None
    stalled = 0
    history = []

    for epoch in range(1, configuration["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_count = 0

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite neural training loss."
                )

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

            total_loss += loss.item() * len(inputs)
            total_count += len(inputs)

        scaled_predictions = neural_output(
            model,
            X_validation,
            device,
        )

        validation_predictions = nonnegative(
            y_scaler.inverse_transform(
                scaled_predictions.reshape(-1, 1)
            ).ravel()
        )

        validation_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        validation_predictions
                        - validation_actual
                    ) ** 2
                )
            )
        )

        if not np.isfinite(validation_rmse):
            raise RuntimeError(
                "Non-finite neural validation RMSE."
            )

        history.append(
            {
                "epoch": epoch,
                "train_MSE_scaled": (
                    total_loss / total_count
                ),
                "validation_MSE_scaled": float(
                    np.mean(
                        (
                            scaled_predictions
                            - y_validation_scaled
                        ) ** 2
                    )
                ),
                "validation_RMSE_Wh": validation_rmse,
            }
        )

        print(
            f"  Epoch {epoch:02d} | "
            f"validation RMSE: {validation_rmse:.4f} Wh",
            flush=True,
        )

        if validation_rmse < best_rmse - 0.0001:
            best_rmse = validation_rmse
            best_epoch = epoch
            stalled = 0

            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stalled += 1

        if stalled >= args.patience:
            print("  Early stopping.", flush=True)
            break

    if best_weights is None:
        raise RuntimeError(
            "No valid neural checkpoint was produced."
        )

    directory.mkdir()

    torch.save(
        best_weights,
        directory / "weights.pt",
    )

    joblib.dump(
        {
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
        },
        directory / "scalers.joblib",
    )

    write_json(
        directory / "model.json",
        {
            "kind": "neural",
            "family": family,
            "phase": "optimized",
            "architecture": model.get_config(),
            "lookback": configuration["lookback"],
            "sequence_columns": source.columns.tolist(),
            "configuration": configuration,
            "best_epoch": best_epoch,
            "output_policy": (
                "inverse target scaling, then clip below zero"
            ),
        },
    )

    pd.DataFrame(history).to_csv(
        directory / "training_history.csv",
        index=False,
    )


def ml_search_spaces():
    """Predefined spaces; no test results influence these choices."""
    return {
        "Ridge": (
            {
                "keep": [None, 20, 30, 40],
                "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            },
            {"alpha": 10.0},
        ),
        "RandomForest": (
            {
                "keep": [None, 20, 30, 40],
                "n_estimators": [150, 200, 250],
                "min_samples_leaf": [2, 5, 8, 15],
                "max_features": [0.5, 0.8, 1.0],
                "max_depth": [None, 8, 16],
            },
            {
                "n_estimators": 200,
                "min_samples_leaf": 8,
                "max_features": 0.8,
                "max_depth": None,
            },
        ),
        "Boosting": (
            {
                "keep": [None, 20, 30, 40],
                "max_iter": [100, 200, 350],
                "max_leaf_nodes": [7, 15, 31],
                "learning_rate": [0.02, 0.05, 0.1],
                "l2_regularization": [1.0, 10.0],
                "min_samples_leaf": [20, 50],
            },
            {
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "learning_rate": 0.05,
                "l2_regularization": 1.0,
                "min_samples_leaf": 20,
            },
        ),
    }


def neural_search_space(maximum_epochs):
    shorter_limit = max(1, maximum_epochs // 2)

    space = {
        "hidden_dim": [32, 64],
        "num_layers": [1, 2],
        "dropout_prob": [0.1, 0.2, 0.3],
        "lookback": [12, 24, 48],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [128, 256],
        "weight_decay": [1e-5, 1e-4],
        "epochs": list(
            dict.fromkeys(
                [shorter_limit, maximum_epochs]
            )
        ),
    }

    anchors = [
        {
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout_prob": 0.3,
            "lookback": 24,
            "learning_rate": 0.0005,
            "batch_size": 128,
            "weight_decay": 1e-5,
            "epochs": maximum_epochs,
        },
        {
            "hidden_dim": 32,
            "num_layers": 1,
            "dropout_prob": 0.1,
            "lookback": 48,
            "learning_rate": 0.001,
            "batch_size": 256,
            "weight_decay": 1e-4,
            "epochs": shorter_limit,
        },
    ]

    return space, anchors


def find_ensemble(records, validation_predictions, actual):
    """
    Search convex weights for the best three distinct model families.

    Components and weights are chosen using validation data only.
    """
    best_per_family = {}

    for row in sorted(
        records,
        key=lambda item: (
            item["RMSE_Wh"],
            item["model"],
        ),
    ):
        best_per_family.setdefault(
            row["family"],
            row,
        )

    members = list(best_per_family.values())[:3]

    if len(members) < 2:
        raise RuntimeError(
            "At least two families are needed for an ensemble."
        )

    best = None
    weight_options = [0.0, 0.25, 0.5, 0.75, 1.0]

    for weights in itertools.product(
        weight_options,
        repeat=len(members),
    ):
        if abs(sum(weights) - 1.0) > 1e-9:
            continue

        if sum(weight > 0 for weight in weights) < 2:
            continue

        predictions = np.zeros(
            len(actual),
            dtype=float,
        )

        components = []

        for member, weight in zip(members, weights):
            if weight == 0:
                continue

            predictions += (
                weight
                * validation_predictions[member["model"]]
            )

            components.append(
                {
                    "model_id": member["model"],
                    "weight": weight,
                }
            )

        rmse = float(
            np.sqrt(
                np.mean(
                    (predictions - actual) ** 2
                )
            )
        )

        if best is None or rmse < best["rmse"]:
            best = {
                "rmse": rmse,
                "components": components,
            }

    if best is None:
        raise RuntimeError(
            "No valid ensemble configuration was found."
        )

    return best


def save_plots(predictions, metrics, champion, report_root):
    shown = predictions.iloc[:288]

    fig, ax = plt.subplots(figsize=(12, 4))
    dates = pd.to_datetime(shown["target_time"])

    ax.plot(
        dates,
        shown["actual_Wh"],
        label="Observed",
        linewidth=1,
    )
    ax.plot(
        dates,
        shown[champion],
        label=champion,
        linewidth=1,
    )
    ax.set(
        xlabel="Target timestamp",
        ylabel="Energy per interval (Wh)",
        title="Validation-selected forecast: test period",
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(report_root / "forecast.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(
        predictions[champion],
        predictions["actual_Wh"] - predictions[champion],
        s=8,
        alpha=0.3,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        xlabel="Predicted energy (Wh)",
        ylabel="Observed minus predicted (Wh)",
        title="Test residuals",
    )
    fig.tight_layout()
    fig.savefig(report_root / "residuals.png", dpi=150)
    plt.close(fig)

    height = max(5, 0.3 * len(metrics))
    fig, ax = plt.subplots(figsize=(12, height))

    colors = [
        "#0f766e" if selected else "#94a3b8"
        for selected in metrics["selected_by_validation"]
    ]

    ax.barh(
        metrics["model"],
        metrics["RMSE_Wh"],
        color=colors,
    )
    ax.set(
        xlabel="Test RMSE (Wh)",
        title="Frozen validation winner highlighted",
    )
    fig.tight_layout()
    fig.savefig(
        report_root / "model_comparison.png",
        dpi=150,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )
    parser.add_argument(
        "--base-run",
        default="energy_v1",
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
        "--ml-trials",
        type=int,
        default=4,
        help="Optimization trials per tabular family.",
    )
    parser.add_argument(
        "--nn-trials",
        type=int,
        default=4,
        help="Optimization trials per neural architecture.",
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

    for name in (args.base_run, args.run_name):
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}",
            name,
        ):
            parser.error("Invalid experiment name.")

    if args.base_run == args.run_name:
        parser.error(
            "The new experiment must have a different name."
        )

    if args.ml_trials < 2 or args.nn_trials < 2:
        parser.error(
            "Use at least two trials per family/architecture."
        )

    if (
        args.epochs < 1
        or args.patience < 1
        or args.threads < 1
    ):
        parser.error(
            "Epochs, patience and threads must be positive."
        )

    check_horizon(args.horizon)

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        parser.error(
            "CUDA is unavailable. Use --device cpu."
        )

    base_root = (
        PROJECT_ROOT / "models" / args.base_run
    )
    base_manifest_path = base_root / "manifest.json"

    if not base_manifest_path.is_file():
        raise FileNotFoundError(
            f"Initial experiment not found: {base_root}"
        )

    base_manifest = read_json(base_manifest_path)

    if base_manifest.get("status") != "complete":
        raise ValueError(
            "The initial experiment must be complete."
        )

    if args.horizon not in base_manifest["horizons_steps"]:
        raise ValueError(
            "The requested horizon is not available "
            "in the initial experiment."
        )

    digest = hashlib.sha256(
        args.data.read_bytes()
    ).hexdigest()

    if digest != base_manifest["dataset_sha256"]:
        raise ValueError(
            "The dataset differs from the initial experiment."
        )

    run_root = (
        PROJECT_ROOT / "models" / args.run_name
    )
    reports_root = (
        PROJECT_ROOT / "reports" / args.run_name
    )

    if run_root.exists() or reports_root.exists():
        raise FileExistsError(
            "This experiment already exists. "
            "Choose a new --run-name."
        )

    raw = load_raw(args.data)
    samples = make_samples(raw, args.horizon)
    masks, boundaries = partition_masks(raw, samples)
    source = sequence_features(raw)

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

    model_root = (
        run_root / f"horizon_{args.horizon}"
    )
    report_root = (
        reports_root / f"horizon_{args.horizon}"
    )

    model_root.mkdir(parents=True)
    report_root.mkdir(parents=True)

    manifest = {
        "schema_version": 2,
        "project": "Appliance Energy Forecasting",
        "run_name": args.run_name,
        "base_run": args.base_run,
        "status": "optimizing",
        "horizons_steps": [args.horizon],
        "dataset_sha256": digest,
        "dataset_rows": len(raw),
        "seed": args.seed,
        "ml_trials_per_family": args.ml_trials,
        "neural_trials_per_architecture": args.nn_trials,
        "maximum_epochs": args.epochs,
        "patience": args.patience,
        "threads": args.threads,
        "device": args.device,
        "feature_selection": (
            "SelectKBest fitted inside training-only pipelines"
        ),
        "ensemble_selection": (
            "Validation-selected convex weights"
        ),
        "test_note": (
            "Historical benchmark already inspected in "
            "earlier experiments; not new external data."
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "pytorch": torch.__version__,
        },
    }

    write_json(
        run_root / "manifest.json",
        manifest,
    )

    records = []
    validation_predictions = {}

    def register(model_id, phase, family, configuration):
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

        records.append(
            {
                "model": model_id,
                "phase": phase,
                "family": family,
                "configuration": json.dumps(
                    plain(configuration),
                    sort_keys=True,
                ),
                **metrics,
            }
        )
        validation_predictions[model_id] = predictions

        print(
            f"Validation RMSE: "
            f"{metrics['RMSE_Wh']:.4f} Wh",
            flush=True,
        )

    try:
        with threadpool_limits(limits=args.threads):
            # Reuse the actual completed initial models.
            # Initial test scores are not read for selection.
            initial_directory = (
                base_root
                / f"horizon_{args.horizon}"
            )

            initial_files = sorted(
                initial_directory.glob("*/model.json")
            )

            if not initial_files:
                raise ValueError(
                    "No initial candidate artifacts were found."
                )

            print("\n=== Initial model comparison ===")

            for metadata_file in initial_files:
                original_directory = metadata_file.parent
                original_name = original_directory.name

                model_id = f"initial__{original_name}"
                destination = model_root / model_id

                shutil.copytree(
                    original_directory,
                    destination,
                )

                metadata = read_json(
                    destination / "model.json"
                )

                if metadata["kind"] not in {
                    "tabular",
                    "neural",
                }:
                    raise ValueError(
                        "The base run must contain individual "
                        "initial models."
                    )

                family = family_from_name(original_name)

                metadata["phase"] = "initial"
                metadata["family"] = family
                metadata["source_run"] = args.base_run

                write_json(
                    destination / "model.json",
                    metadata,
                )

                print(f"\nChecking {model_id}", flush=True)

                register(
                    model_id,
                    "initial",
                    family,
                    {"source_model": original_name},
                )

            initial_winner = min(
                records,
                key=lambda row: (
                    row["RMSE_Wh"],
                    row["model"],
                ),
            )

            print("\n=== Tabular optimization ===")

            for index, (family, specification) in enumerate(
                ml_search_spaces().items()
            ):
                space, anchor = specification

                anchors = [
                    {**anchor, "keep": None},
                    {**anchor, "keep": 30},
                ]

                configurations = sample_configurations(
                    space,
                    args.ml_trials,
                    args.seed + index,
                    anchors,
                )

                for trial, configuration in enumerate(
                    configurations,
                    start=1,
                ):
                    model_id = (
                        f"optimized__{family}_{trial:02d}"
                    )

                    print(
                        f"\nTraining {model_id}\n"
                        f"{configuration}",
                        flush=True,
                    )

                    fit_tabular(
                        family,
                        configuration,
                        X_train,
                        y_train,
                        model_root / model_id,
                        args.seed,
                    )

                    register(
                        model_id,
                        "optimized",
                        family,
                        configuration,
                    )

            print("\n=== Neural optimization ===")

            space, anchors = neural_search_space(
                args.epochs
            )

            for index, family in enumerate(("LSTM", "GRU")):
                configurations = sample_configurations(
                    space,
                    args.nn_trials,
                    args.seed + 100 + index,
                    anchors,
                )

                for trial, configuration in enumerate(
                    configurations,
                    start=1,
                ):
                    model_id = (
                        f"optimized__{family}_{trial:02d}"
                    )

                    print(
                        f"\nTraining {model_id}\n"
                        f"{configuration}",
                        flush=True,
                    )

                    fit_neural(
                        family,
                        configuration,
                        raw,
                        source,
                        samples,
                        masks,
                        model_root / model_id,
                        args,
                    )

                    register(
                        model_id,
                        "optimized",
                        family,
                        configuration,
                    )

            print("\n=== Weighted ensemble search ===")

            ensemble = find_ensemble(
                records,
                validation_predictions,
                validation_actual,
            )

            ensemble_id = "ensemble__weighted"
            ensemble_directory = (
                model_root / ensemble_id
            )
            ensemble_directory.mkdir()

            write_json(
                ensemble_directory / "model.json",
                {
                    "kind": "ensemble",
                    "phase": "ensemble",
                    "components": ensemble["components"],
                    "output_policy": "clip below zero",
                },
            )

            register(
                ensemble_id,
                "ensemble",
                "Ensemble",
                {"components": ensemble["components"]},
            )

            selected = min(
                records,
                key=lambda row: (
                    row["RMSE_Wh"],
                    row["model"],
                ),
            )

            validation_frame = pd.DataFrame(
                records
            ).sort_values(
                ["RMSE_Wh", "model"]
            )

            validation_frame.to_csv(
                report_root / "validation_metrics.csv",
                index=False,
            )

            champion = {
                "model_id": selected["model"],
                "initial_model_id": initial_winner["model"],
                "horizon_steps": args.horizon,
                "horizon_minutes": args.horizon * 10,
                "selection_metric": "validation RMSE in Wh",
                "peak_threshold_Wh": peak_threshold,
                "boundaries": boundaries,
                "partition_sizes": {
                    key: int(value.sum())
                    for key, value in masks.items()
                },
                "validation_metrics": selected,
                "selection_order": (
                    "frozen before test evaluation"
                ),
            }

            # No test outcomes have been evaluated in this run.
            write_json(
                model_root / "champion.json",
                champion,
            )

            print(
                "\nFrozen initial comparator:",
                initial_winner["model"],
            )
            print(
                "Frozen final winner:",
                selected["model"],
            )
            print(
                "Ensemble components:",
                ensemble["components"],
            )

            print("\n=== Final test evaluation ===")

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
                    "actual_Wh": samples.y[
                        masks["test"]
                    ],
                }
            )

            test_rows = []

            for record in records:
                model_id = record["model"]

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
                        "phase": record["phase"],
                        "family": record["family"],
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

            test_lookup = {
                row["model"]: row
                for row in test_rows
            }

            comparison_rows = []

            for stage, record in (
                ("Initial", initial_winner),
                ("Final validation choice", selected),
            ):
                comparison_rows.append(
                    {
                        "stage": stage,
                        "model": record["model"],
                        "validation_RMSE_Wh": record["RMSE_Wh"],
                        **{
                            key: value
                            for key, value in test_lookup[
                                record["model"]
                            ].items()
                            if key not in {
                                "model",
                                "phase",
                                "family",
                                "selected_by_validation",
                            }
                        },
                    }
                )

            comparison = pd.DataFrame(comparison_rows)

            comparison.to_csv(
                report_root / "before_after.csv",
                index=False,
            )

            before_rmse = test_lookup[
                initial_winner["model"]
            ]["RMSE_Wh"]

            after_rmse = test_lookup[
                selected["model"]
            ]["RMSE_Wh"]

            reduction = (
                100 * (before_rmse - after_rmse) / before_rmse
                if before_rmse > 0
                else None
            )

            summary = {
                **champion,
                "initial_test_metrics": test_lookup[
                    initial_winner["model"]
                ],
                "final_test_metrics": test_lookup[
                    selected["model"]
                ],
                "test_RMSE_reduction_percent": reduction,
                "ensemble_components": ensemble["components"],
            }

            write_json(
                run_root / "summary.json",
                [summary],
            )
            write_json(
                reports_root / "summary.json",
                [summary],
            )

            save_plots(
                test_predictions,
                test_frame,
                selected["model"],
                report_root,
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

        print("\nBefore/after comparison:")
        print(comparison.to_string(index=False))

        print("\nTest results:")
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

        print(
            "\nTest RMSE reduction versus the initial "
            f"validation winner: {reduction}"
        )
        print(
            "A negative reduction means test RMSE increased."
        )
        print(f"\nModels: {run_root}")
        print(f"Reports: {reports_root}")

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)

        write_json(
            run_root / "manifest.json",
            manifest,
        )
        raise


if __name__ == "__main__":
    main()