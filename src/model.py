import torch
import torch.nn as nn


class BaselineModels:
    def __init__(self):

        # Import baseline ML models
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor


        # Initialize Linear Regression model
        self.lr_model = LinearRegression()

        # Initialize Random Forest model with fixed parameters
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    def get_models(self):

        # Return dictionary of baseline models for training
        return {
            "LinearRegression": self.lr_model,
            "RandomForest": self.rf_model
        }


class EnergyModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=64,
        num_layers=2,
        dropout_prob=0.2,
        model_type="LSTM",
        bidirectional=False
    ):
        super(EnergyModel, self).__init__()

        # Choose between GRU and LSTM based on model_type
        if model_type == "GRU":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

        # Adjust output size if bidirectional RNN is used
        direction_factor = 2 if bidirectional else 1

        # Fully connected layers for final prediction
        self.fc1 = nn.Linear(hidden_dim * direction_factor, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(32, 1) # Output single value (energy consumption)

    def forward(self, x):
        # Pass input sequence through RNN (LSTM/GRU)
        out, _ = self.rnn(x)

        # Take output from the last time step
        out = out[:, -1, :]

        # Pass through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Final output layer
        out = self.fc2(out)
        return out