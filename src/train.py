import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import BaselineModels, EnergyModel

import copy
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# Metrics
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n{name} Performance")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# Sequence Builder
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values)
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)


# Initialize preprocessing and feature engineering classes
pre = DataPreprocessor()
fe = FeatureEngineer()

# Load raw dataset and convert date column to datetime
df = pre.load_data("data/raw/energy_data_set.csv")
# Remove duplicate records
df = pre.handle_duplicates(df)
# Handle missing values using time-based interpolation
df = pre.handle_missing(df)
# Remove extreme outliers using IQR method
df = pre.remove_outliers(df)


# Create time-based features (hour, day, month, weekend)
df = fe.create_time_features(df)
# Create lag features to capture past energy consumption
df = fe.create_lag_features(df)
# Create rolling statistics (mean, std) for trend smoothing
df = fe.create_rolling_features(df)
# Create interaction features (temperature × humidity)
df = fe.create_interaction_features(df)
# Create domain-specific features (peak hours, night indicator)
df = fe.create_domain_features(df)
# Remove rows with NaN values
df = fe.clean_after_feature_engineering(df)

# Save Processed Data
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

processed_path = os.path.join(PROCESSED_DIR, "processed_energy_data.csv")
df.to_csv(processed_path, index=False)

print(f"Processed dataset saved to: {processed_path}")



# Split dataset chronologically (80% train, 20% test)
train_df, test_df = pre.split_data(df)

# Separate features and target for training data
X_train_raw, y_train_raw = fe.split_features_target(train_df)

# Select most important features using Random Forest importance
selected_features = fe.select_features(X_train_raw, y_train_raw)

# Scale features using MinMaxScaler
X_train_scaled, X_test_scaled, y_train, y_test = pre.scale_data(train_df, test_df)

# Keep only selected features
X_train = X_train_scaled[selected_features]
X_test = X_test_scaled[selected_features]



# BASELINE MODELS
# Train classical machine learning models for benchmarking
results = {}
baseline_models = BaselineModels().get_models()

for name, model in baseline_models.items():
    model.fit(X_train, y_train)                      # Train model
    preds = model.predict(X_test)                    # Predict on test data
    results[name] = evaluate(y_test, preds, name)    # Evaluate performance




# Convert tabular data into sequences (required for LSTM/GRU)
# Each sequence uses past 24 time steps
X_train_seq, y_train_seq = create_sequences(X_train, y_train)
X_test_seq, y_test_seq = create_sequences(X_test, y_test)

# Convert numpy arrays to PyTorch tensors
X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)
y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
y_test_t = torch.tensor(y_test_seq, dtype=torch.float32)

# Create DataLoader for batch training
# shuffle=False to preserve time order in time-series data
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)



# Deep Learning Models(LSTM + GRU)
deep_models = {
    "LSTM": "LSTM",
    "GRU": "GRU"
}

EPOCHS = 100         # Number of training iterations

for name, model_type in deep_models.items():

    print(f"\n===== Training {name} =====")

    # Initialize model
    model = EnergyModel(
        input_dim=X_train_t.shape[2],           # Number of input features
        hidden_dim=64,                          # Hidden units
        num_layers=2,                           # Number of RNN layers
        dropout_prob=0.2,                       # Regularization
        model_type=model_type                   # LSTM or GRU
    )

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()         # Regression loss

    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        t_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()               # Reset gradients
            out = model(xb).squeeze()           # Forward pass
            loss = criterion(out, yb)           # Compute loss
            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update weights
            t_loss += loss.item()

        train_losses.append(t_loss / len(train_loader))

        model.eval()
        v_loss = 0
        with torch.no_grad():                    # Disable gradient computation
            for xb, yb in test_loader:
                out = model(xb).squeeze()
                v_loss += criterion(out, yb).item()

        val_losses.append(v_loss / len(test_loader))

        print(f"{name} Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

    

    # EVALUATION
    # Set model to evaluation mode
    model.eval()
    preds, actuals = [], []

    # Disable gradient computation for inference
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb).squeeze()       # Model prediction
            preds.extend(out.numpy())       # Store predictions
            actuals.extend(yb.numpy())      # Store actual values

    # Convert to numpy arrays
    preds = np.array(preds)
    actuals = np.array(actuals)

    # Evaluate model performance using regression metrics
    results[name] = evaluate(actuals, preds, name)


    # Visualization

    # 1. Actual vs Predicted Plot
    plt.figure()
    plt.plot(actuals[:200], label="Actual")
    plt.plot(preds[:200], label="Predicted")
    plt.legend()
    plt.title(f"{name}: Actual vs Predicted")
    plt.savefig(f"{REPORT_DIR}/{name}_actual_vs_pred.png")
    plt.close()

    # 2. Residual Plot
    plt.figure()
    plt.scatter(preds, actuals - preds, alpha=0.3)
    plt.axhline(0)
    plt.title(f"{name}: Residual Plot")
    plt.savefig(f"{REPORT_DIR}/{name}_residual.png")
    plt.close()

    # 3. Training vs Validation Loss Curve
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.title(f"{name}: Loss Curve")
    plt.savefig(f"{REPORT_DIR}/{name}_loss.png")
    plt.close()



# MODEL COMPARISON
# Compare models using MAE (lower is better)
plt.figure()
plt.bar(results.keys(), [v["MAE"] for v in results.values()])
plt.title("Model Comparison (MAE)")
plt.savefig(f"{REPORT_DIR}/model_comparison.png")
plt.close()

print("\nTraining Completed. All reports saved in /reports")




# MODEL OPTIMIZATION
print("\n========== MODEL OPTIMIZATION ==========")

# Track best results for BOTH models
best_models = {
    "LSTM": {"loss": float("inf"), "weights": None, "config": None},
    "GRU": {"loss": float("inf"), "weights": None, "config": None}
}

# Ensure model directory exists
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# Hyperparameter Grid
# Define search space for Grid Search
param_grid = {
    "model_type": ["LSTM", "GRU"],
    "lr": [0.001, 0.0005],              # Learning rates
    "hidden_dim": [32, 64],             # Number of hidden units
    "dropout": [0.2, 0.3]               # Dropout rates for regularization
}

best_val_loss = float("inf")
best_weights = None
best_config = None
best_model_type = None

# Training settings for optimization
EPOCHS_OPT = 50
PATIENCE = 7                # Early stopping patience


# Grid Search + Early Stopping

# Iterate through all hyperparameter combinations
for model_type in param_grid["model_type"]:
    for lr in param_grid["lr"]:
        for hidden in param_grid["hidden_dim"]:
            for drop in param_grid["dropout"]:

                print(f"\nTesting {model_type} | LR={lr}, Hidden={hidden}, Dropout={drop}")

                # Initialize model with current configuration
                model_opt = EnergyModel(
                    input_dim=X_train_t.shape[2],
                    hidden_dim=hidden,
                    dropout_prob=drop,
                    model_type=model_type
                )

                optimizer = torch.optim.Adam(model_opt.parameters(), lr=lr)
                criterion = nn.MSELoss()

                # Track best validation loss for current config
                patience_counter = 0
                local_best = float("inf")
                local_weights = None

                for epoch in range(EPOCHS_OPT):

                    # ---- Training ----
                    model_opt.train()
                    for xb, yb in train_loader:
                        optimizer.zero_grad()
                        loss = criterion(model_opt(xb).squeeze(), yb)
                        loss.backward()
                        optimizer.step()

                    # ---- Validation ----
                    model_opt.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for xb, yb in test_loader:
                            val_loss += criterion(model_opt(xb).squeeze(), yb).item()
                    val_loss /= len(test_loader)

                    print(f"{model_type} Epoch {epoch+1}: Val Loss={val_loss:.4f}")

                    # ---- Early Stopping ----
                    if val_loss < local_best:
                        local_best = val_loss
                        local_weights = copy.deepcopy(model_opt.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Stop training if no improvement
                    if patience_counter >= PATIENCE:
                        break

                # ---- Save Best Per Model ----
                if local_best < best_models[model_type]["loss"]:
                    best_models[model_type]["loss"] = local_best
                    best_models[model_type]["weights"] = local_weights
                    best_models[model_type]["config"] = (lr, hidden, drop)


# Evaluate BOTH Optimized Models (LSTM & GRU)
optimized_results = {}
optimized_preds = {}

for model_type in ["LSTM", "GRU"]:

    print(f"\n=== Evaluating Optimized {model_type} ===")

    config = best_models[model_type]["config"]
    weights = best_models[model_type]["weights"]

    # Rebuild model with best hyperparameters
    opt_model = EnergyModel(
        input_dim=X_train_t.shape[2],
        hidden_dim=config[1],
        dropout_prob=config[2],
        model_type=model_type
    )

    opt_model.load_state_dict(weights)
    opt_model.eval()

    preds = []

    # Generate predictions
    with torch.no_grad():
        for xb, _ in test_loader:
            preds.extend(opt_model(xb).squeeze().numpy())

    preds = np.array(preds)
    optimized_preds[model_type] = preds

    # Evaluate
    optimized_results[model_type] = evaluate(
        actuals, preds, f"Optimized {model_type}"
    )


# Select Best Final Model
best_model_type = min(
    optimized_results,
    key=lambda x: optimized_results[x]["RMSE"]
)

print(f"\nBest Final Model Selected: {best_model_type}")


# Save Final Best Model
final_config = best_models[best_model_type]["config"]
final_weights = best_models[best_model_type]["weights"]

# Rebuild final model
final_model = EnergyModel(
    input_dim=X_train_t.shape[2],
    hidden_dim=final_config[1],
    dropout_prob=final_config[2],
    model_type=best_model_type
)

final_model.load_state_dict(final_weights)

# Save model weights
torch.save(final_model.state_dict(), f"{MODEL_DIR}/trained_model.pth")

# Save hyperparameters
joblib.dump(
    {"model_type": best_model_type, "config": final_config},
    f"{MODEL_DIR}/best_hyperparams.pkl"
)

print("Final best model saved successfully.")



# Visuals for Best Model
best_preds = optimized_preds[best_model_type]

# Actual vs Predicted
plt.figure(figsize=(12,5))
plt.plot(actuals[:200], label="Actual")
plt.plot(best_preds[:200], label="Optimized Prediction")
plt.legend()
plt.title(f"{best_model_type}: Actual vs Predicted")
plt.savefig(f"{REPORT_DIR}/optimized_actual_vs_pred.png")
plt.close()

# Residual plot
plt.figure(figsize=(8,5))
plt.scatter(best_preds, actuals - best_preds, alpha=0.3)
plt.axhline(0)
plt.title(f"{best_model_type}: Residual Plot")
plt.savefig(f"{REPORT_DIR}/optimized_residual.png")
plt.close()

# Model comparison
plt.figure(figsize=(8,5))
plt.bar(optimized_results.keys(), [v["MAE"] for v in optimized_results.values()])
plt.title("Optimized Model Comparison (MAE)")
plt.savefig(f"{REPORT_DIR}/optimized_comparison.png")
plt.close()