import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore


class DataPreprocessor:
    def __init__(self):
        # Initialize Min-Max scaler for feature normalization
        self.scaler = MinMaxScaler()

    # 1. Load Data
    def load_data(self, path):
        # Load dataset from CSV file
        df = pd.read_csv(path)

        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Sort data chronologically (for time-series)
        df = df.sort_values('date')

        print("Data Loaded:", df.shape)
        return df

    # 2. Handle Duplicates
    def handle_duplicates(self, df):
        # Count duplicate rows
        before = df.duplicated().sum()
        # Remove duplicates
        df = df.drop_duplicates()
        after = df.duplicated().sum()

        print(f"Duplicates removed: {before - after}")
        return df

    # 3. Handle Missing Values
    def handle_missing(self, df):
        # Check missing values before processing
        print("Missing values before:", df.isnull().sum().sum())

        # Set 'date' as index for time-based operations
        df = df.set_index('date')

        # Time-based interpolation
        df = df.interpolate(method='time')

        # Fill remaining values
        df = df.bfill().ffill()

        df = df.reset_index()

        print("Missing values after:", df.isnull().sum().sum())
        return df

    # 4. Detect Outliers (Z-score)
    def detect_outliers(self, df):
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Compute Z-scores (ignore NaNs)
        z_scores = np.abs(zscore(numeric_df, nan_policy='omit'))

        # Count values beyond threshold
        outliers = (z_scores > 3).sum().sum()

        print("Outliers detected (Z-score):", outliers)
        return outliers

    # 5. Remove Outliers (IQR)
    def remove_outliers(self, df):
        # Calculate Q1 and Q3 for target variable
        Q1 = df['Appliances'].quantile(0.25)
        Q3 = df['Appliances'].quantile(0.75)

        # Compute Interquartile Range (IQR)
        IQR = Q3 - Q1

        before = len(df)

        # Filter values within acceptable IQR range
        df = df[
            (df['Appliances'] >= Q1 - 1.5 * IQR) &
            (df['Appliances'] <= Q3 + 1.5 * IQR)
        ]

        after = len(df)

        print(f"Outliers removed (IQR): {before - after}")
        return df

    # 6. Split Data (Time-Based)
    def split_data(self, df, test_size=0.2):
        # Determine split index based on time order
        split_index = int(len(df) * (1 - test_size))

        # Split dataset chronologically (no shuffling)
        train_df = df.iloc[:split_index].copy()
        test_df = df.iloc[split_index:].copy()

        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)

        return train_df, test_df

    # 7. Scaling (Min-Max)
    def scale_data(self, train_df, test_df, target_column='Appliances'):

        # Separate features and target
        X_train = train_df.drop(columns=[target_column, 'date'])
        y_train = train_df[target_column]

        X_test = test_df.drop(columns=[target_column, 'date'])
        y_test = test_df[target_column]

        # Fit scaler on training data only (avoid data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Apply same transformation to test data
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for consistency
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        print("Scaling completed using MinMaxScaler")
        print("Train features:", X_train_scaled.shape)
        print("Test features:", X_test_scaled.shape)

        return X_train_scaled, X_test_scaled, y_train, y_test