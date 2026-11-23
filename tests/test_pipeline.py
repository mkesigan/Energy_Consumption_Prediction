import numpy as np
import pandas as pd
import pytest
import torch

from src.data_preprocessing import (
    DEFAULT_DATA_PATH,
    HISTORY,
    INTERVAL,
    SENSORS,
    load_raw,
    validate_frame,
)
from src.feature_engineering import (
    make_samples,
    origin_features,
    partition_masks,
    sequence_features,
    sequence_windows,
)
from src.metrics import (
    calculate_metrics,
    peak_threshold_from_training,
)
from src.model import EnergyModel
from src.predict import EnergyForecaster, PROJECT_ROOT


@pytest.fixture
def history():
    """Create a small, valid synthetic recording."""
    positions = np.arange(600)

    frame = pd.DataFrame(
        {
            "date": pd.date_range(
                "2026-01-01",
                periods=len(positions),
                freq="10min",
            ),
            "Appliances": (
                60.0 + 20.0 * np.sin(positions / 8)
            ),
            "lights": np.where(
                positions % 12 < 3,
                10,
                0,
            ),
        }
    )

    for number in range(1, 10):
        frame[f"T{number}"] = (
            20.0
            + number * 0.1
            + 0.5 * np.sin(positions / 30)
        )
        frame[f"RH_{number}"] = (
            40.0
            + number
            + 0.5 * np.cos(positions / 20)
        )

    # A genuine high reading must remain in the data.
    frame.loc[200, "Appliances"] = 1080.0

    return validate_frame(frame)


def test_validation_retains_rows_and_consumption_peaks(history):
    result = validate_frame(history)

    assert len(result) == len(history)
    assert result["Appliances"].max() == 1080.0


def test_validation_sorts_observations_without_changing_them(history):
    shuffled = history.sample(
        frac=1,
        random_state=42,
    )

    result = validate_frame(shuffled)

    pd.testing.assert_frame_equal(result, history)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_column",
        "duplicate_timestamp",
        "timestamp_gap",
        "missing_measurement",
        "infinite_measurement",
        "negative_energy",
        "invalid_humidity",
        "insufficient_history",
        "timezone_aware",
    ],
)
def test_invalid_history_is_rejected(history, problem):
    invalid = history.copy()

    if problem == "missing_column":
        invalid = invalid.drop(columns="T1")

    elif problem == "duplicate_timestamp":
        invalid.loc[100, "date"] = invalid.loc[99, "date"]

    elif problem == "timestamp_gap":
        invalid = invalid.drop(index=100)

    elif problem == "missing_measurement":
        invalid.loc[100, "Appliances"] = np.nan

    elif problem == "infinite_measurement":
        invalid.loc[100, "T1"] = np.inf

    elif problem == "negative_energy":
        invalid.loc[100, "Appliances"] = -1.0

    elif problem == "invalid_humidity":
        invalid.loc[100, "RH_1"] = 101.0

    elif problem == "insufficient_history":
        invalid = invalid.iloc[:HISTORY - 1]

    elif problem == "timezone_aware":
        invalid["date"] = invalid["date"].dt.tz_localize(
            "UTC"
        )

    with pytest.raises(ValueError):
        validate_frame(invalid)


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_labels_match_the_correct_future_interval(history, horizon):
    samples = make_samples(history, horizon)

    expected_targets = history["Appliances"].to_numpy()[
        samples.origins + horizon
    ]

    np.testing.assert_array_equal(
        samples.y,
        expected_targets,
    )

    assert (
        samples.target_time - samples.origin_time
    ).eq(horizon * INTERVAL).all()


def test_feature_construction_does_not_modify_input(history):
    original = history.copy(deep=True)

    make_samples(history, horizon=1)

    pd.testing.assert_frame_equal(
        history,
        original,
    )


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_future_values_cannot_change_earlier_features(history, horizon):
    origin = 300

    before = origin_features(
        history,
        horizon,
    )

    changed = history.copy()

    changed.loc[
        origin + 1:, "Appliances"
    ] += 500.0

    changed.loc[
        origin + 1:, "lights"
    ] += 10

    for sensor in SENSORS:
        if sensor != "lights":
            changed.loc[
                origin + 1:, sensor
            ] += 0.5

    # The modified recording is still structurally valid.
    changed = validate_frame(changed)

    after = origin_features(
        changed,
        horizon,
    )

    pd.testing.assert_frame_equal(
        before.iloc[:origin + 1],
        after.iloc[:origin + 1],
    )


def test_partitions_use_target_timestamps(history):
    samples = make_samples(history, horizon=6)
    masks, boundaries = partition_masks(
        history,
        samples,
    )

    validation_start = pd.Timestamp(
        boundaries["validation_starts"]
    )
    test_start = pd.Timestamp(
        boundaries["test_starts"]
    )

    assert samples.target_time[
        masks["train"]
    ].max() < validation_start

    assert samples.target_time[
        masks["validation"]
    ].min() == validation_start

    assert samples.target_time[
        masks["validation"]
    ].max() < test_start

    assert samples.target_time[
        masks["test"]
    ].min() == test_start

    membership = sum(
        mask.astype(int)
        for mask in masks.values()
    )

    assert np.all(membership == 1)


@pytest.mark.parametrize("horizon", [1, 6, 144])
def test_daily_reference_is_already_observed(history, horizon):
    samples = make_samples(history, horizon)

    reference_positions = (
        samples.origins + horizon - 144
    )

    assert np.all(
        reference_positions <= samples.origins
    )

    expected = history["Appliances"].to_numpy()[
        reference_positions
    ]

    np.testing.assert_allclose(
        samples.X["previous_day_target"],
        expected,
    )


def test_sequences_end_at_the_forecast_origin(history):
    source = sequence_features(history).to_numpy(
        dtype=float
    )
    origins = np.array([200, 300])

    windows = sequence_windows(
        source,
        origins,
        lookback=24,
    )

    assert windows.shape == (
        2,
        24,
        source.shape[1],
    )

    for index, origin in enumerate(origins):
        expected = source[
            origin - 23:origin + 1
        ]

        np.testing.assert_allclose(
            windows[index],
            expected,
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.parametrize("model_type", ["LSTM", "GRU"])
def test_neural_prediction_preserves_single_sample_dimension(
    model_type,
):
    model = EnergyModel(
        input_dim=3,
        hidden_dim=4,
        num_layers=2,
        dropout_prob=0.2,
        model_type=model_type,
    )
    model.eval()

    with torch.inference_mode():
        output = model(
            torch.zeros(
                1,
                6,
                3,
                dtype=torch.float32,
            )
        )

    assert output.shape == (1,)


def test_metric_calculations():
    threshold = peak_threshold_from_training(
        [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    )

    metrics = calculate_metrics(
        [50, 100, 150, 200, 250],
        [60, 90, 160, 190, 240],
        peak_threshold=threshold,
    )

    assert threshold == pytest.approx(182.0)
    assert metrics["MAE_Wh"] == pytest.approx(10.0)
    assert metrics["RMSE_Wh"] == pytest.approx(10.0)
    assert metrics["R2"] == pytest.approx(0.98)
    assert metrics["bias_Wh"] == pytest.approx(-2.0)
    assert metrics["peak_MAE_Wh"] == pytest.approx(10.0)
    assert metrics["peak_count"] == 2
    assert metrics["n"] == 5


def test_single_observation_has_no_r_squared():
    metrics = calculate_metrics(
        [100],
        [90],
    )

    assert metrics["R2"] is None
    assert metrics["MAE_Wh"] == pytest.approx(10.0)


@pytest.fixture(scope="module")
def trained_forecaster():
    """Use existing artifacts; skip when they are not distributed."""
    manifest = (
        PROJECT_ROOT
        / "models"
        / "energy_v2"
        / "manifest.json"
    )

    if not manifest.is_file():
        pytest.skip(
            "Saved energy_v2 artifacts are not available."
        )

    return EnergyForecaster(
        run_name="energy_v2",
        horizon=1,
        threads=2,
    )


@pytest.fixture(scope="module")
def original_recording():
    if not DEFAULT_DATA_PATH.is_file():
        pytest.skip(
            "Original dataset is not available."
        )

    return load_raw(DEFAULT_DATA_PATH)


def test_saved_model_reproduces_the_report(
    trained_forecaster,
    original_recording,
):
    result = trained_forecaster.verify_saved_predictions(
        DEFAULT_DATA_PATH
    )

    assert result["verification_passed"] is True
    assert result["test_observations"] > 0


def test_single_history_forecasts_match_batch_predictions(
    trained_forecaster,
    original_recording,
):
    raw = original_recording

    first_test_position = int(len(raw) * 0.85)

    origins = np.array(
        [
            first_test_position - 1,
            first_test_position + 25,
            len(raw) - 2,
        ],
        dtype=int,
    )

    batch_predictions = (
        trained_forecaster.predict_at_origins(
            raw,
            origins,
        )
    )

    for index, origin in enumerate(origins):
        observed_history = raw.iloc[
            origin - HISTORY + 1:origin + 1
        ]

        result = trained_forecaster.forecast(
            observed_history
        )

        assert result["predicted_Wh"] == pytest.approx(
            batch_predictions[index],
            rel=1e-6,
            abs=1e-3,
        )

        expected_time = (
            raw["date"].iloc[origin]
            + trained_forecaster.horizon * INTERVAL
        )

        assert pd.Timestamp(
            result["forecast_time"]
        ) == expected_time


def test_trained_forecast_ignores_future_measurements(
    trained_forecaster,
    original_recording,
):
    raw = original_recording
    origin = int(len(raw) * 0.85) + 20

    before = trained_forecaster.predict_at_origins(
        raw,
        np.array([origin]),
    )

    changed = raw.copy()
    changed.loc[
        origin + 1:, "Appliances"
    ] += 200.0
    changed.loc[
        origin + 1:, "T1"
    ] += 1.0

    after = trained_forecaster.predict_at_origins(
        changed,
        np.array([origin]),
    )

    np.testing.assert_allclose(
        before,
        after,
        rtol=1e-10,
        atol=1e-8,
    )