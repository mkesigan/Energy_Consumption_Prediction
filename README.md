# EnergyLens — Appliance Energy Forecasting

A multivariate time-series forecasting project that predicts appliance energy consumption using historical energy readings and building sensors.

EnergyLens compares conventional machine learning, LSTM and GRU models, performs bounded hyperparameter optimization and training-only feature selection, and evaluates a validation-selected weighted ensemble.

A Streamlit application supports CSV history uploads, manually entered readings, historical streaming replay and model evaluation.

## Project objective

After receiving measurements through timestamp `t`, predict appliance energy at a future timestamp `t + h`.

The default forecasting horizon is **10 minutes**.

Predictions represent energy in **Wh for one future 10-minute interval**. They are not power in watts or cumulative consumption over the entire forecast horizon.

For example, observations available through 10:00 are used to forecast consumption at timestamp 10:10.

## Main features

- Exploratory data analysis with chronological training-only exploration.
- Validation of missing values, duplicate timestamps and sampling continuity.
- Retention of genuine high-consumption observations.
- Historical energy features, trailing statistics and cyclic calendar encodings.
- Persistence and daily seasonal forecasting baselines.
- Linear Regression, Ridge, Random Forest and histogram gradient boosting.
- LSTM and GRU architectures.
- Training-only feature selection.
- Bounded hyperparameter search.
- Early stopping with best-checkpoint restoration.
- Validation-selected ensemble components and weights.
- Saved preprocessing and models for independent inference.
- Interactive testing and historical replay.
- Automated checks for data quality, temporal alignment and inference consistency.

## Dataset

The project uses the **UCI Appliances Energy Prediction** dataset.

| Property | Value |
|---|---|
| Raw observations | 19,735 |
| Source columns | 29 |
| Sampling interval | 10 minutes |
| Recording period | January–May 2016 |
| Target | `Appliances` |
| Target unit | Wh per recorded interval |
| Setting | One building in Belgium |

The forecasting input contract retains:

- `date`
- `Appliances`
- `lights`
- `T1`–`T9`
- `RH_1`–`RH_9`

These represent 21 retained columns, including the timestamp and target.

The random control variables `rv1` and `rv2` are excluded. Source-interpolated airport weather is also excluded from the operational input contract.

The original CSV is retained at:

```text
data/raw/energy_data_set.csv
```

No target-based outlier filtering is applied.

## Forecasting methodology

### Historical features

Tabular models receive 60 engineered features before optional feature selection, including:

- Current observed building measurements.
- Historical energy values.
- Trailing means, standard deviations and maxima.
- Recent consumption changes.
- Previous-day consumption at the target's clock time.
- Temperature–humidity interaction.
- Cyclic origin-time and target-time calendar features.

The initial neural models use 24 historical steps, representing four hours, with 25 inputs per step. Optimization evaluates additional sequence lengths and architecture settings.

Tabular and neural models use different representations but share the same forecast cutoff and evaluation target timestamps.

### Chronological partitions

Raw timeline boundaries are:

| Partition | Raw observations |
|---|---:|
| Training | 13,814 |
| Validation | 2,960 |
| Testing | 2,961 |

For the 10-minute forecast, after historical warm-up and future-label alignment:

| Partition | Labeled samples |
|---|---:|
| Training | 13,670 |
| Validation | 2,960 |
| Testing | 2,961 |

Samples are assigned using their **target timestamps**.

Earlier observed history remains available as context at validation and test boundaries.

### Information timing

At forecast origin `t`, consumption at `t` is already observed and may be used to predict `t + h`.

Trailing statistics may therefore include the origin's measured energy. Future energy and future sensor measurements are excluded from that forecast.

Input and target scalers are fitted using training data only.

Evaluation uses historical rolling origins: as time advances, newly observed readings become available for subsequent forecasts. This is not an open-loop prediction of the entire test period issued at its beginning.

## Experiments

### Initial experiment: `energy_v1`

The initial comparison contains 14 candidates:

- Persistence.
- Daily seasonal baseline.
- Linear Regression.
- Three Ridge configurations.
- Two Random Forest configurations.
- Two histogram gradient boosting configurations.
- Two LSTM configurations.
- Two GRU configurations.

The initial validation-selected winner was Random Forest with minimum leaf size 8.

### Optimized experiment: `energy_v2`

The completed optimization run evaluated:

- 14 reused initial candidates.
- 12 additional tabular trials.
- Eight additional neural trials.
- One validation-selected weighted ensemble.

Tabular optimization compares all features against training-only selection within fitted pipelines.

Neural optimization varies hidden units, layers, dropout, sequence length, learning rate, batch size, weight decay and epoch limits.

The search is bounded and includes predefined comparison configurations. It is not exhaustive.

### Final ensemble

The validation-selected ensemble combines:

| Component | Weight | Configuration |
|---|---:|---|
| Random Forest | 50% | 200 trees, minimum leaf size 8, 30 selected features |
| Histogram gradient boosting | 25% | 200 iterations, 15 leaves, 30 selected features |
| GRU | 25% | 64 hidden units, two layers, dropout 0.3, 24-step history |

Components and weights were chosen using validation performance.

The final choice was saved before test evaluation. An ensemble is not forced to win if an individual candidate has better validation RMSE.

## Results

Both comparisons use the same 2,961 historical test intervals, with consumption peaks retained.

| Metric | Initial validation winner | Final ensemble |
|---|---:|---:|
| Validation RMSE | 57.7050 Wh | 55.8196 Wh |
| Test MAE | 32.8992 Wh | 24.8115 Wh |
| Test RMSE | 62.0844 Wh | 57.1953 Wh |
| Test R² | 0.5332 | 0.6039 |
| Mean prediction bias | +13.1336 Wh | +0.7594 Wh |
| High-demand MAE | 113.2028 Wh | 121.8633 Wh |

The final ensemble achieved:

- **7.87% lower test RMSE** than the initial validation-selected model.
- **24.58% lower test MAE** than the initial validation-selected model.
- Approximately **14.4% lower test RMSE** than persistence.

High-demand MAE increased relative to the initial Random Forest. The improvement therefore does not hold across every metric or consumption regime.

High demand is defined using the training-target 90th percentile, which was 210 Wh in this experiment. The test set contains 248 such intervals.

**R² is a regression statistic, not an accuracy percentage.**

Detailed results are available under:

```text
reports/energy_v1/
reports/energy_v2/
```

Earlier assessment reports, if retained, describe a previous implementation. The current benchmark results are the experiment CSVs above.

## Project files

| File or directory | Responsibility |
|---|---|
| `notebooks/EDA.ipynb` | Data audit and exploratory analysis |
| `src/data_preprocessing.py` | Input validation and CSV loading |
| `src/feature_engineering.py` | Historical features, future labels and chronological masks |
| `src/model.py` | Baseline estimators and recurrent architecture |
| `src/metrics.py` | Original-unit regression metrics |
| `src/train_initial.py` | Initial training from the source CSV |
| `src/train.py` | Optimization, ensemble selection and final evaluation |
| `src/predict.py` | Saved-model inference and report verification |
| `app.py` | Streamlit frontend |
| `tests/test_pipeline.py` | Automated pipeline checks |
| `requirements.txt` | Runtime dependencies |
| `requirements-dev.txt` | Runtime, testing and notebook dependencies |

## Environment

The verified model-training, inference and test environment used:

| Dependency | Version |
|---|---|
| Python | 3.13.3 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.9.1+cpu |
| Streamlit | 1.51.0 |

Exact dependency pins are recorded in the requirements files.

The setup commands below use Windows PowerShell.

## Installation

Clone the repository and open a terminal at its root.

Create and activate an environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

For notebook development and automated tests:

```powershell
python -m pip install -r requirements-dev.txt
```

The runtime requirements include the PyTorch CPU-wheel index.

For EDA, open `notebooks/EDA.ipynb` and select the project environment as the notebook kernel.

## Training from a fresh clone

Trained model directories are excluded from ordinary Git commits. Generate them locally before running inference or the frontend.

### 1. Train the initial candidates

```powershell
python src/train_initial.py --run-name energy_v1
```

### 2. Run optimization and ensemble selection

```powershell
python src/train.py --base-run energy_v1 --run-name energy_v2
```

The default optimization budget is four trials per tabular family and four trials per neural architecture.

For a larger explicitly configured search:

```powershell
python src/train.py --base-run energy_v1 --run-name energy_v3 --ml-trials 8 --nn-trials 8
```

Training time depends on the hardware and search budget. Better validation or test performance is not guaranteed by increasing the budget.

Experiment directories are not overwritten. Choose a new run name for a new experiment.

The initial-training command also supports additional horizons:

```powershell
python src/train_initial.py --run-name initial_multi --horizons 1 3 6
```

Optimize each available horizon into a separate named experiment:

```powershell
python src/train.py --base-run initial_multi --run-name optimized_30min --horizon 3
```

The published result table above covers the **10-minute experiment only**.

## Using existing local artifacts

If completed `energy_v1` and `energy_v2` directories are already present, training is unnecessary.

Verify the selected saved model:

```powershell
python src/predict.py --run-name energy_v2 --verify
```

Predict from the latest included history:

```powershell
python src/predict.py --run-name energy_v2
```

Predict from another compatible CSV:

```powershell
python src/predict.py --run-name energy_v2 --history data/raw/my_history.csv
```

New history must contain at least 144 consecutive ten-minute observations with all required measurement columns.

The included dataset ends in 2016. Its subsequent-interval prediction therefore also has a 2016 timestamp; it is not a live household feed.

## Frontend

Launch:

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The application provides:

- **Test new values:** upload history or enter a newly observed reading.
- **Session evaluation:** compare earlier forecasts with subsequently entered actual readings.
- **Streaming replay:** simulate incoming historical test readings with Start, Pause, Next reading and Reset.
- **Model evaluation:** inspect initial/final comparisons, candidate metrics, residuals and learning curves.

The manual form reuses unchanged values from the previous reading. Replace them with measured values for actual-input testing; hypothetical values represent scenarios.

Recorded test metrics remain fixed when new values are entered. Session metrics describe only the supplied session observations.

## Verification and tests

Run:

```powershell
python -m pytest tests/test_pipeline.py -q
```

The working project completed **30 passing tests**.

Checks cover:

- Input validation and preserved consumption peaks.
- Future-target alignment.
- Chronological partition boundaries.
- Future-data mutation invariance.
- Historical sequence boundaries.
- Single-sample neural output shape.
- Metric calculations.
- Saved-model/report agreement.
- Consistency between individual-history and batch inference.

Tests requiring trained artifacts are skipped when those artifacts are absent.

Saved ensemble predictions reproduced the test report with a maximum absolute difference of approximately **1.14 × 10⁻¹³ Wh**.

The added initial-training entry point passed its import/CLI check. A complete fresh-environment rebuild using that new entry point has not been independently repeated.

## Saved artifacts

Each completed model experiment contains:

- A manifest with versions, settings, dataset checksum and completion status.
- A frozen validation-selected model identifier.
- Fitted tabular pipelines.
- Neural weights, architecture and fitted input/target scalers.
- Ensemble component identifiers and weights where applicable.

Reports include validation metrics, test metrics, timestamped predictions and evaluation plots.

Preserve a complete experiment directory when transferring models so that ensemble components and preprocessing remain available.

## Limitations

- The dataset describes one building and approximately 4.5 months in 2016.
- Performance on other buildings, countries or modern live feeds has not been established.
- The historical test period has already been inspected during project development; these are benchmark results, not new external validation.
- Selection uses one chronological validation block. Rolling development validation and repeated seeds remain possible extensions.
- High-demand prediction remains difficult.
- The project provides point forecasts without calibrated uncertainty intervals.
- The high-demand flag is a threshold comparison, not a probability.
- Live deployment requires a compatible sensor feed and verified measurement-arrival timing.
- No measured electricity-bill savings are claimed.

## Dataset attribution

Candanedo, L. (2017). *Appliances Energy Prediction* [Dataset]. UCI Machine Learning Repository.

- Dataset: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- DOI: https://doi.org/10.24432/C5VC8G
- Dataset license: Creative Commons Attribution 4.0 International.

## Author

Kesigan Mukuntharathan