import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class FeatureEngineer:

    # 1. Time-Based Features
    def create_time_features(self, df):
        df = df.copy()

        # Extract temporal features from timestamp
        df['hour'] = df['date'].dt.hour                   # Hour of day (0–23)
        df['day_of_week'] = df['date'].dt.dayofweek       # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month                 # Month (1–12)

        # Weekend indicator (Saturday=5, Sunday=6)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        print("Time features created")
        return df


    # 2. Lag Features
    def create_lag_features(self, df, lags=[1, 2, 3, 6]):
        df = df.copy()

        # Ensure chronological order before creating lags
        df = df.sort_values('date')

        # Create lag features (previous time steps of target variable)
        for lag in lags:
            df[f'lag_{lag}'] = df['Appliances'].shift(lag)

        print("Lag features created:", lags)
        return df

    # 3. Rolling Features
    def create_rolling_features(self, df):
        df = df.copy()

        # 10-minute interval data 
        # Rolling mean captures trend
        df['rolling_mean_1h'] = df['Appliances'].rolling(window=6, min_periods=1).mean()
        df['rolling_mean_3h'] = df['Appliances'].rolling(window=18, min_periods=1).mean()

        # Rolling std captures variability (fluctuation)
        df['rolling_std_1h'] = df['Appliances'].rolling(window=6, min_periods=1).std()

        print("Rolling features created")
        return df

    # 4. Interaction Features
    def create_interaction_features(self, df):
        df = df.copy()

        # Combine outdoor temperature & humidity
        if 'T_out' in df.columns and 'RH_out' in df.columns:
            df['temp_humidity'] = df['T_out'] * df['RH_out']

        # Combine indoor temperature & humidity
        if 'T1' in df.columns and 'RH_1' in df.columns:
            df['indoor_temp_humidity'] = df['T1'] * df['RH_1']

        # These features help model combined environmental effects
        print("Interaction features created")
        return df

    # 5. Domain-Specific Features
    def create_domain_features(self, df):
        df = df.copy()

        # Evening peak hours
        df['is_peak_hour'] = df['hour'].isin([17, 18, 19, 20]).astype(int)

        # Night time indicator
        df['is_night'] = df['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

        # These features encode human behavior patterns
        print("Domain features created")
        return df

    # 6. Clean After Feature Engineering
    def clean_after_feature_engineering(self, df):
        df = df.copy()

        # Remove rows with NaN
        df = df.dropna()

        print("Dropped NaN after feature engineering")
        return df

    # 7. Split Features and Target
    def split_features_target(self, df, target='Appliances'):
        df = df.copy()

        # Separate input features and target variable
        X = df.drop(columns=[target, 'date'])
        y = df[target]

        return X, y

    # 8. Feature Selection 
    def select_features(self, X_train, y_train, threshold=0.006):
        
        # Use Random Forest to estimate feature importance
        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # Create importance dataframe
        importances = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        print("\n--- Feature Importances (Train Set) ---")
        print(importances.head(15))

        # Select features above threshold
        selected_features = importances[
            importances['Importance'] >= threshold
        ]['Feature'].tolist()

        print(f"\nSelected {len(selected_features)} features (threshold={threshold})")
        print("Final Features:", selected_features)

        return selected_features