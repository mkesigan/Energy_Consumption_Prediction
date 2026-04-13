import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model import BaselineModels, EnergyModel


# PATH SETUP
MODEL_DIR = "models"
REPORT_DIR = "reports"
PROCESSED_DIR = "data/processed"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# METRICS
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print(f"\n{name} Performance")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# SPLIT (TRAIN / VAL / TEST)
def split_data_3way(df, train_size=0.7, val_size=0.15):
    train_end = int(len(df) * train_size)
    val_end = int(len(df) * (train_size + val_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    return train_df, val_df, test_df


# SCALING 
def scale_data_3way(pre, train_df, val_df, test_df, target_column='Appliances'):

    # Split
    X_train = train_df.drop(columns=[target_column, 'date'])
    y_train = train_df[target_column]

    X_val = val_df.drop(columns=[target_column, 'date'])
    y_val = val_df[target_column]

    X_test = test_df.drop(columns=[target_column, 'date'])
    y_test = test_df[target_column]

    # Fit only on train
    X_train_scaled = pre.scaler.fit_transform(X_train)
    X_val_scaled = pre.scaler.transform(X_val)
    X_test_scaled = pre.scaler.transform(X_test)

    # Convert back
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print("Scaling completed (Train/Val/Test)")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


# SEQUENCE BUILDER
def create_sequences(X, y, seq_len=24):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values)
        ys.append(y.iloc[i+seq_len])
    return np.array(Xs), np.array(ys)


# DATALOADER BUILDER
def build_loader(X, y, batch_size=32):
    X_seq, y_seq = create_sequences(X, y)
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    y_t = torch.tensor(y_seq, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=False)


# TRAIN FUNCTION
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # TRAIN
        model.train()
        t_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)  # FIXED
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        train_losses.append(t_loss / len(train_loader))

        # VALIDATION
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb).squeeze(-1), yb).item()  # FIXED

        val_losses.append(v_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

    return train_losses, val_losses


# VISUALIZATION
def plot_results(actuals, preds, train_losses, val_losses, name):

    plt.figure()
    plt.plot(actuals[:200], label="Actual")
    plt.plot(preds[:200], label="Predicted")
    plt.legend()
    plt.title(f"{name}: Actual vs Predicted")
    plt.savefig(f"{REPORT_DIR}/{name}_actual_vs_pred.png")
    plt.close()

    plt.figure()
    plt.scatter(preds, actuals - preds, alpha=0.3)
    plt.axhline(0)
    plt.title(f"{name}: Residual")
    plt.savefig(f"{REPORT_DIR}/{name}_residual.png")
    plt.close()

    if len(train_losses) > 0:
        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Validation")
        plt.legend()
        plt.title(f"{name}: Loss Curve")
        plt.savefig(f"{REPORT_DIR}/{name}_loss.png")
        plt.close()


# PIPELINE
pre = DataPreprocessor()
fe = FeatureEngineer()

df = pre.load_data("data/raw/energy_data_set.csv")
df = pre.handle_duplicates(df)
df = pre.handle_missing(df)
df = pre.remove_outliers(df)

df = fe.create_time_features(df)
df = fe.create_lag_features(df)
df = fe.create_rolling_features(df)
df = fe.create_interaction_features(df)
df = fe.create_domain_features(df)
df = fe.clean_after_feature_engineering(df)

df.to_csv(f"{PROCESSED_DIR}/processed_energy_data.csv", index=False)


# SPLIT
train_df, val_df, test_df = split_data_3way(df)

# Feature selection (ONLY TRAIN)
X_train_raw, y_train_raw = fe.split_features_target(train_df)
selected_features = fe.select_features(X_train_raw, y_train_raw)

X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = \
    scale_data_3way(pre, train_df, val_df, test_df)

X_train = X_train_scaled[selected_features]
X_val = X_val_scaled[selected_features]
X_test = X_test_scaled[selected_features]


# BASELINE MODELS
results = {}
baseline_models = BaselineModels().get_models()

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = evaluate(y_test, preds, name)


# LOADERS
train_loader = build_loader(X_train, y_train)
val_loader = build_loader(X_val, y_val)
test_loader = build_loader(X_test, y_test)


# DEEP LEARNING
deep_models = {"LSTM": "LSTM", "GRU": "GRU"}
EPOCHS = 100

for name, model_type in deep_models.items():

    print(f"\n===== Training {name} =====")

    model = EnergyModel(
        input_dim=X_train.shape[1],
        hidden_dim=64,
        num_layers=2,
        dropout_prob=0.2,
        model_type=model_type
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, EPOCHS
    )

    # TEST
    model.eval()
    preds, actuals = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb).squeeze(-1)  
            preds.extend(out.numpy())
            actuals.extend(yb.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    results[name] = evaluate(actuals, preds, name)

    plot_results(actuals, preds, train_losses, val_losses, name)


# FINAL COMPARISON
plt.figure()
plt.bar(results.keys(), [v["MAE"] for v in results.values()])
plt.title("Model Comparison (MAE)")
plt.savefig(f"{REPORT_DIR}/model_comparison.png")
plt.close()

print("\nTraining Completed Successfully.")


# MODEL OPTIMIZATION
print("\n=== MODEL OPTIMIZATION ===")

best_models = {
    "LSTM": {"loss": float("inf"), "weights": None, "config": None},
    "GRU": {"loss": float("inf"), "weights": None, "config": None}
}

# Hyperparameter grid
param_grid = {
    "model_type": ["LSTM", "GRU"],
    "lr": [0.001, 0.0005],
    "hidden_dim": [32, 64],
    "dropout": [0.2, 0.3]
}

EPOCHS_OPT = 50
PATIENCE = 7


# TRAIN WITH EARLY STOPPING
def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion):
    best_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(EPOCHS_OPT):

        # TRAIN
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(-1), yb)
            loss.backward()
            optimizer.step()

        # VALIDATION 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb).squeeze(-1), yb).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}")

        # EARLY STOPPING
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_loss, best_weights


# GRID SEARCH
for model_type in param_grid["model_type"]:
    for lr in param_grid["lr"]:
        for hidden in param_grid["hidden_dim"]:
            for drop in param_grid["dropout"]:

                print(f"\nTesting {model_type} | LR={lr}, Hidden={hidden}, Dropout={drop}")

                model = EnergyModel(
                    input_dim=X_train.shape[1],
                    hidden_dim=hidden,
                    dropout_prob=drop,
                    model_type=model_type
                )

                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.MSELoss()

                val_loss, weights = train_with_early_stopping(
                    model, train_loader, val_loader, optimizer, criterion
                )
                # Save best config per model
                if val_loss < best_models[model_type]["loss"]:
                    best_models[model_type]["loss"] = val_loss
                    best_models[model_type]["weights"] = weights
                    best_models[model_type]["config"] = (lr, hidden, drop)


# EVALUATE OPTIMIZED MODELS
optimized_results = {}
optimized_preds = {}

for model_type in ["LSTM", "GRU"]:

    print(f"\n=== Evaluating Optimized {model_type} ===")

    config = best_models[model_type]["config"]

    if config is None:
        continue

    weights = best_models[model_type]["weights"]

    model = EnergyModel(
        input_dim=X_train.shape[1],
        hidden_dim=config[1],
        dropout_prob=config[2],
        model_type=model_type
    )

    model.load_state_dict(weights)
    model.eval()

    preds, actuals = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb).squeeze(-1)
            preds.extend(out.numpy())
            actuals.extend(yb.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)

    optimized_preds[model_type] = preds
    optimized_results[model_type] = evaluate(actuals, preds, f"Optimized {model_type}")


# SELECT FINAL MODEL
best_model_type = min(
    optimized_results,
    key=lambda x: optimized_results[x]["RMSE"]
)

print(f"\nBest Final Model Selected: {best_model_type}")


# SAVE FINAL MODEL
final_config = best_models[best_model_type]["config"]
final_weights = best_models[best_model_type]["weights"]

final_model = EnergyModel(
    input_dim=X_train.shape[1],
    hidden_dim=final_config[1],
    dropout_prob=final_config[2],
    model_type=best_model_type
)

final_model.load_state_dict(final_weights)

torch.save(final_model.state_dict(), f"{MODEL_DIR}/trained_model.pth")

joblib.dump(
    {"model_type": best_model_type, "config": final_config},
    f"{MODEL_DIR}/best_hyperparams.pkl"
)

print("Final best model saved successfully.")

# VISUALIZATION 
best_preds = optimized_preds[best_model_type]

actuals_final = []

with torch.no_grad():
    for xb, yb in test_loader:
        actuals_final.extend(yb.numpy())

actuals_final = np.array(actuals_final)

plot_results(actuals_final, best_preds, [], [], "optimized")