import hashlib
import io
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.data_preprocessing import (
    DEFAULT_DATA_PATH,
    HISTORY,
    INTERVAL,
    SENSORS,
    load_raw,
    validate_frame,
)
from src.metrics import calculate_metrics
from src.predict import EnergyForecaster


st.set_page_config(
    page_title="EnergyLens",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def get_forecaster(run_name, horizon):
    """Reuse loaded model artifacts without retraining."""
    return EnergyForecaster(
        run_name=run_name,
        horizon=horizon,
        threads=2,
    )


@st.cache_data
def get_original_data():
    return load_raw(DEFAULT_DATA_PATH)


@st.cache_data
def get_original_checksum():
    return hashlib.sha256(
        DEFAULT_DATA_PATH.read_bytes()
    ).hexdigest()


@st.cache_data
def get_report(path):
    return pd.read_csv(path)


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def completed_experiments():
    experiments = {}

    for path in sorted(
        (PROJECT_ROOT / "models").glob("*/manifest.json")
    ):
        manifest = read_json(path)

        if manifest.get("status") == "complete":
            experiments[path.parent.name] = manifest

    return experiments


def set_input_history(frame):
    """Replace the session history and reset its forecast records."""
    validated = validate_frame(frame)

    st.session_state["input_history"] = (
        validated.tail(HISTORY)
        .reset_index(drop=True)
    )
    st.session_state["forecast_logs"] = {}


def resolved_metrics(records, threshold):
    """Evaluate only predictions whose actual values have arrived."""
    resolved = [
        record
        for record in records
        if record["actual_Wh"] is not None
    ]

    if not resolved:
        return None

    return calculate_metrics(
        [record["actual_Wh"] for record in resolved],
        [record["forecast_Wh"] for record in resolved],
        peak_threshold=threshold,
    )


def show_session_metrics(records, threshold):
    metrics = resolved_metrics(records, threshold)

    count, mae, rmse = st.columns(3)

    if metrics is None:
        count.metric("Evaluated forecasts", 0)
        mae.metric("Session MAE", "Waiting")
        rmse.metric("Session RMSE", "Waiting")
    else:
        count.metric("Evaluated forecasts", metrics["n"])
        mae.metric(
            "Session MAE",
            f"{metrics['MAE_Wh']:.2f} Wh",
        )
        rmse.metric(
            "Session RMSE",
            f"{metrics['RMSE_Wh']:.2f} Wh",
        )


def update_forecast_log(history, result, identity):
    """Resolve earlier predictions and record the current forecast."""
    logs = st.session_state["forecast_logs"]
    records = logs.setdefault(identity, [])

    observed = history.set_index("date")["Appliances"]

    for record in records:
        target_time = pd.Timestamp(record["target_time"])

        if (
            record["actual_Wh"] is None
            and target_time in observed.index
        ):
            record["actual_Wh"] = float(
                observed.loc[target_time]
            )

    already_recorded = any(
        record["origin_time"] == result["observed_through"]
        and record["target_time"] == result["forecast_time"]
        for record in records
    )

    if not already_recorded:
        records.append(
            {
                "origin_time": result["observed_through"],
                "target_time": result["forecast_time"],
                "forecast_Wh": result["predicted_Wh"],
                "actual_Wh": None,
            }
        )

    return records


def render_new_values(forecaster, raw, identity):
    st.subheader("Predict from new observations")

    st.write(
        "The model needs one day of historical readings. "
        "Upload that history, or add a new reading to the "
        "included example."
    )

    if "input_history" not in st.session_state:
        set_input_history(raw.tail(HISTORY))

    uploaded = st.file_uploader(
        "History CSV",
        type=["csv"],
        key="history_upload",
    )

    load_column, reset_column, example_column = st.columns(3)

    if load_column.button(
        "Load uploaded history",
        disabled=uploaded is None,
    ):
        try:
            frame = pd.read_csv(
                io.BytesIO(uploaded.getvalue())
            )
            set_input_history(frame)
            st.success("History loaded.")
        except (
            ValueError,
            TypeError,
            UnicodeDecodeError,
        ) as exc:
            st.error(str(exc))

    if reset_column.button("Reset example history"):
        set_input_history(raw.tail(HISTORY))

    example_column.download_button(
        "Download example CSV",
        raw.tail(HISTORY).to_csv(index=False),
        file_name="history_example.csv",
        mime="text/csv",
    )

    history = st.session_state["input_history"]
    latest = history.iloc[-1]
    next_timestamp = latest["date"] + INTERVAL

    st.caption(
        f"History currently ends at {latest['date']}. "
        "The included example is from the 2016 dataset."
    )

    with st.expander(
        "Enter a new reading or test a scenario",
        expanded=False,
    ):
        st.write(
            "Enter values for a newly observed interval. "
            "The model then predicts a later interval. "
            "Unchanged fields reuse the previous reading; "
            "replace them with measurements for actual-use testing."
        )
        st.write(
            f"New reading timestamp: **{next_timestamp}**"
        )

        with st.form("new_reading_form"):
            values = {"date": next_timestamp}

            values["Appliances"] = st.number_input(
                "Observed appliance consumption (Wh)",
                min_value=0.0,
                value=float(latest["Appliances"]),
                step=10.0,
                key=f"appliances_{next_timestamp}",
            )

            columns = st.columns(3)

            for index, sensor in enumerate(SENSORS):
                if sensor == "lights":
                    unit = "Wh"
                    limits = {"min_value": 0.0}
                elif sensor.startswith("RH_"):
                    unit = "%"
                    limits = {
                        "min_value": 0.0,
                        "max_value": 100.0,
                    }
                else:
                    unit = "°C"
                    limits = {}

                with columns[index % 3]:
                    values[sensor] = st.number_input(
                        f"{sensor} ({unit})",
                        value=float(latest[sensor]),
                        format="%.2f",
                        key=(
                            f"measurement_"
                            f"{next_timestamp}_{sensor}"
                        ),
                        **limits,
                    )

            submitted = st.form_submit_button(
                "Add reading and predict"
            )

        if submitted:
            try:
                updated = pd.concat(
                    [history, pd.DataFrame([values])],
                    ignore_index=True,
                )

                st.session_state["input_history"] = (
                    validate_frame(updated)
                    .tail(HISTORY)
                    .reset_index(drop=True)
                )
            except (ValueError, TypeError) as exc:
                st.error(str(exc))
            else:
                st.rerun()

    try:
        result = forecaster.forecast(history)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    prediction_column, flag_column = st.columns(2)

    prediction_column.metric(
        "Predicted interval consumption",
        f"{result['predicted_Wh']:.2f} Wh",
    )
    flag_column.metric(
        "Demand flag",
        (
            "High"
            if result["high_demand_flag"]
            else "Below threshold"
        ),
    )

    st.write(
        f"Forecast timestamp: **{result['forecast_time']}**"
    )
    st.caption(
        "The prediction is energy for one future "
        f"10-minute interval, {result['horizon_minutes']} "
        "minutes after the latest observation."
    )
    st.caption(
        "The high-demand flag compares the point forecast "
        f"with the training-derived threshold of "
        f"{result['training_peak_threshold_Wh']:.0f} Wh. "
        "It is not a probability."
    )

    records = update_forecast_log(
        history,
        result,
        identity,
    )

    st.download_button(
        "Download forecast JSON",
        json.dumps(result, indent=2),
        file_name="forecast.json",
        mime="application/json",
    )

    with st.expander("Ensemble components"):
        components = pd.DataFrame(result["components"])
        components["weight_percent"] = (
            components["weight"] * 100
        )
        st.dataframe(
            components,
            width="stretch",
        )

    st.divider()
    st.subheader("Forecasts from this session")

    st.write(
        "Earlier predictions are evaluated when you add "
        "their actual target readings. Hypothetical inputs "
        "produce scenario results rather than measured "
        "real-world performance."
    )

    show_session_metrics(
        records,
        forecaster.peak_threshold,
    )

    session_frame = pd.DataFrame(records)

    st.dataframe(
        session_frame.tail(20),
        width="stretch",
    )

    st.download_button(
        "Download session predictions",
        session_frame.to_csv(index=False),
        file_name="session_predictions.csv",
        mime="text/csv",
    )

    with st.expander("Current input history"):
        st.dataframe(
            history,
            width="stretch",
        )


def issue_replay_prediction(state, forecaster, raw):
    target_position = (
        state["cursor"] + forecaster.horizon
    )

    if target_position >= len(raw):
        return

    history = raw.iloc[
        state["cursor"] - HISTORY + 1:
        state["cursor"] + 1
    ].reset_index(drop=True)

    # Only already observed history reaches the predictor.
    result = forecaster.forecast(history)

    state["records"].append(
        {
            "origin_time": result["observed_through"],
            "target_time": result["forecast_time"],
            "target_position": target_position,
            "forecast_Wh": result["predicted_Wh"],
            "actual_Wh": None,
        }
    )


def new_replay_state(forecaster, raw, identity):
    first_test_position = int(len(raw) * 0.85)

    cursor = (
        first_test_position - forecaster.horizon
    )

    if cursor < HISTORY - 1:
        raise ValueError(
            "Not enough historical context for replay."
        )

    state = {
        "identity": identity,
        "cursor": cursor,
        "records": [],
        "playing": False,
        "last_tick": time.monotonic(),
    }

    issue_replay_prediction(
        state,
        forecaster,
        raw,
    )

    return state


def advance_replay(state, forecaster, raw):
    if state["cursor"] >= len(raw) - 1:
        state["playing"] = False
        return

    state["cursor"] += 1

    # Reveal actual values only after their timestamps arrive.
    for record in state["records"]:
        if (
            record["actual_Wh"] is None
            and record["target_position"]
            <= state["cursor"]
        ):
            record["actual_Wh"] = float(
                raw["Appliances"].iloc[
                    record["target_position"]
                ]
            )

    issue_replay_prediction(
        state,
        forecaster,
        raw,
    )

    if state["cursor"] == len(raw) - 1:
        state["playing"] = False


@st.fragment(run_every=1.0)
def render_replay(forecaster, raw, identity):
    state = st.session_state.get("replay_state")

    if (
        state is None
        or state["identity"] != identity
    ):
        state = new_replay_state(
            forecaster,
            raw,
            identity,
        )
        st.session_state["replay_state"] = state

    start, pause, step, reset = st.columns(4)

    if start.button("Start", key="replay_start"):
        state["playing"] = True
        state["last_tick"] = time.monotonic()

    if pause.button("Pause", key="replay_pause"):
        state["playing"] = False

    manual_step = step.button(
        "Next reading",
        key="replay_next",
    )

    if manual_step:
        state["playing"] = False

    if reset.button("Reset", key="replay_reset"):
        state = new_replay_state(
            forecaster,
            raw,
            identity,
        )
        st.session_state["replay_state"] = state
        manual_step = False

    delay = st.select_slider(
        "Replay speed",
        options=[1, 2, 5],
        value=2,
        format_func=lambda seconds: (
            f"One recorded interval every {seconds} seconds"
        ),
        key="replay_delay",
    )

    automatic_step = (
        state["playing"]
        and time.monotonic() - state["last_tick"] >= delay
    )

    if manual_step or automatic_step:
        advance_replay(
            state,
            forecaster,
            raw,
        )
        state["last_tick"] = time.monotonic()

    observed_time = raw["date"].iloc[
        state["cursor"]
    ]

    st.write(
        f"Observed through: **{observed_time}**"
    )

    pending = [
        record
        for record in state["records"]
        if record["actual_Wh"] is None
    ]

    if pending:
        latest = pending[-1]

        st.write(
            f"Latest forecast: "
            f"**{latest['forecast_Wh']:.2f} Wh** "
            f"for **{latest['target_time']}**."
        )
        st.caption(
            "Its actual value has not been observed "
            "by the replay yet."
        )

    show_session_metrics(
        state["records"],
        forecaster.peak_threshold,
    )

    observed = raw.iloc[
        max(0, state["cursor"] - 143):
        state["cursor"] + 1
    ]

    chart = observed.set_index("date")[
        ["Appliances"]
    ].rename(
        columns={"Appliances": "Observed Wh"}
    )

    issued = pd.DataFrame(state["records"])
    issued["target_time"] = pd.to_datetime(
        issued["target_time"]
    )

    forecast_series = issued.set_index(
        "target_time"
    )[["forecast_Wh"]].rename(
        columns={"forecast_Wh": "Forecast Wh"}
    )

    forecast_series = forecast_series.loc[
        forecast_series.index >= observed["date"].iloc[0]
    ]

    chart = chart.join(
        forecast_series,
        how="outer",
    ).sort_index()

    st.line_chart(
        chart,
        height=320,
    )

    display_records = issued.drop(
        columns=["target_position"]
    )

    st.dataframe(
        display_records.tail(20),
        width="stretch",
    )

    st.download_button(
        "Download replay predictions",
        display_records.to_csv(index=False),
        file_name="replay_predictions.csv",
        mime="text/csv",
    )

    if state["cursor"] == len(raw) - 1:
        st.success(
            "Replay completed. All available target "
            "readings have arrived."
        )


def render_evaluation(forecaster):
    report_root = forecaster.report_root

    st.subheader("Recorded experiment results")

    comparison_path = (
        report_root / "before_after.csv"
    )

    if comparison_path.is_file():
        comparison = get_report(
            str(comparison_path)
        )

        st.write("Initial versus final validation choice")
        st.dataframe(
            comparison.round(4),
            width="stretch",
        )

        if (
            "peak_MAE_Wh" in comparison.columns
            and len(comparison) == 2
        ):
            before_peak = comparison.iloc[0][
                "peak_MAE_Wh"
            ]
            after_peak = comparison.iloc[1][
                "peak_MAE_Wh"
            ]

            st.caption(
                f"High-demand MAE: {before_peak:.2f} Wh "
                f"initially and {after_peak:.2f} Wh "
                "for the final choice."
            )

    test_metrics = get_report(
        str(report_root / "test_metrics.csv")
    )

    st.write("All candidate test results")
    st.dataframe(
        test_metrics.round(4),
        width="stretch",
    )

    st.bar_chart(
        test_metrics.set_index("model")[["RMSE_Wh"]]
    )

    with st.expander("Validation results used for selection"):
        st.dataframe(
            get_report(
                str(
                    report_root
                    / "validation_metrics.csv"
                )
            ),
            width="stretch",
        )

    predictions = get_report(
        str(report_root / "test_predictions.csv")
    )

    st.subheader("Historical observed versus predicted energy")

    displayed_intervals = st.slider(
        "Intervals to display",
        min_value=24,
        max_value=min(1440, len(predictions)),
        value=min(288, len(predictions)),
        step=1,
    )

    chart = predictions.iloc[
        :displayed_intervals
    ].copy()

    chart["target_time"] = pd.to_datetime(
        chart["target_time"]
    )

    chart = chart.set_index("target_time")[
        ["actual_Wh", forecaster.model_id]
    ].rename(
        columns={
            "actual_Wh": "Observed Wh",
            forecaster.model_id: "Forecast Wh",
        }
    )

    st.line_chart(chart, height=320)

    residual_path = report_root / "residuals.png"

    if residual_path.is_file():
        st.image(str(residual_path))

    st.download_button(
        "Download complete test predictions",
        predictions.to_csv(index=False),
        file_name="test_predictions.csv",
        mime="text/csv",
    )

    histories = sorted(
        forecaster.model_root.glob(
            "*/training_history.csv"
        )
    )

    if histories:
        st.subheader("Neural learning curves")

        selected_history = st.selectbox(
            "Neural candidate",
            histories,
            format_func=lambda path: path.parent.name,
        )

        history = get_report(
            str(selected_history)
        ).set_index("epoch")

        loss_columns = [
            column
            for column in (
                "train_MSE_scaled",
                "validation_MSE_scaled",
            )
            if column in history.columns
        ]

        if loss_columns:
            st.caption("Losses use standardized target units.")
            st.line_chart(history[loss_columns])

        if "validation_RMSE_Wh" in history.columns:
            st.caption("Validation RMSE in original Wh units.")
            st.line_chart(
                history[["validation_RMSE_Wh"]]
            )


def main():
    st.caption("ENERGYLENS")
    st.title("⚡ Appliance Energy Forecasting")
    st.write(
        "Test new readings, inspect forecasts, "
        "and replay historical energy consumption."
    )

    experiments = completed_experiments()

    if not experiments:
        st.info(
            "No completed experiments were found. "
            "Complete training before opening this dashboard."
        )
        st.stop()

    run_names = list(experiments)

    active_path = (
        PROJECT_ROOT / "models" / "active_run.json"
    )

    active_name = (
        read_json(active_path)["run_name"]
        if active_path.is_file()
        else run_names[-1]
    )

    default_index = (
        run_names.index(active_name)
        if active_name in run_names
        else 0
    )

    with st.sidebar:
        st.title("⚡ EnergyLens")

        run_name = st.selectbox(
            "Experiment",
            run_names,
            index=default_index,
        )

        manifest = experiments[run_name]

        horizon = st.selectbox(
            "Forecast ahead",
            manifest["horizons_steps"],
            format_func=lambda value: (
                f"{value * 10} minutes"
            ),
        )

        st.caption(
            "Each prediction describes one future "
            "10-minute interval."
        )

    try:
        forecaster = get_forecaster(
            run_name,
            horizon,
        )
        raw = get_original_data()
    except (
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
    ) as exc:
        st.error(str(exc))
        st.stop()

    identity = f"{run_name}:{horizon}"

    metrics = get_report(
        str(
            forecaster.report_root
            / "test_metrics.csv"
        )
    )

    selected = metrics.loc[
        metrics["model"] == forecaster.model_id
    ].iloc[0]

    mae, rmse, r2 = st.columns(3)

    mae.metric(
        "Recorded test MAE",
        f"{selected['MAE_Wh']:.2f} Wh",
    )
    rmse.metric(
        "Recorded test RMSE",
        f"{selected['RMSE_Wh']:.2f} Wh",
    )
    r2.metric(
        "Recorded test R²",
        f"{selected['R2']:.4f}",
    )

    st.caption(
        "These are frozen historical test results. "
        "Entering new values does not change them."
    )

    new_tab, replay_tab, evaluation_tab, details_tab = st.tabs(
        [
            "Test new values",
            "Streaming replay",
            "Model evaluation",
            "Project details",
        ]
    )

    with new_tab:
        render_new_values(
            forecaster,
            raw,
            identity,
        )

    with replay_tab:
        st.subheader("Forecast first, reveal the actual later")
        st.write(
            "This accelerated replay simulates incoming "
            "readings from the historical test period."
        )

        matching_source = (
            get_original_checksum()
            == forecaster.manifest["dataset_sha256"]
        )

        if matching_source:
            render_replay(
                forecaster,
                raw,
                identity,
            )
        else:
            st.error(
                "The source CSV differs from this experiment. "
                "Restore the original recording for historical "
                "replay. New history can be tested in the first tab."
            )

        st.caption(
            "Replay is a simulation, not a live sensor connection."
        )

    with evaluation_tab:
        render_evaluation(forecaster)

    with details_tab:
        st.subheader("Forecasting method")

        st.write(
            "After measurements through time t are observed, "
            "the model predicts appliance energy at t+h."
        )
        st.write(
            "The pipeline preserves consumption peaks, "
            "uses chronological partitions and fits learned "
            "preprocessing using training data only."
        )
        st.write(
            "The final model and ensemble weights were selected "
            "using validation RMSE before test evaluation."
        )
        st.write(
            "A real-time deployment requires a compatible "
            "external feed of energy and building-sensor readings."
        )

        st.json(
            {
                "experiment": forecaster.run_name,
                "selected_model": forecaster.model_id,
                "horizon_minutes": horizon * 10,
                "partition_sizes": (
                    forecaster.champion["partition_sizes"]
                ),
                "boundaries": (
                    forecaster.champion["boundaries"]
                ),
            }
        )

        st.caption(
            "The supplied dataset describes one building in "
            "Belgium in 2016. Performance on other buildings "
            "or present-day live feeds has not been established."
        )

        st.markdown(
            "Dataset: [UCI Appliances Energy Prediction]"
            "(https://archive.ics.uci.edu/dataset/374/"
            "appliances+energy+prediction), "
            "Candanedo (2017), CC BY 4.0."
        )


if __name__ == "__main__":
    main()