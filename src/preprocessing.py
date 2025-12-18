"""
Preprocessing Module

Handles data cleaning, encoding, and transformation for student retention models.
Ensures the same preprocessing pipeline can be applied to training and inference data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional
import pickle


class StudentDataPreprocessor:
    """
    Preprocessor for student retention forecasting data.

    Handles:
    - Target encoding (Dropout/Graduate/Enrolled -> numeric)
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    """

    def __init__(self, scale_features: bool = False):
        """
        Initialize preprocessor.

        Args:
            scale_features: Whether to apply StandardScaler to numeric features
        """
        self.scale_features = scale_features
        self.target_encoder = None
        self.categorical_encoders = {}
        self.scaler = None
        self.numeric_features = []
        self.categorical_features = []
        self.feature_names = []

    def fit(self, df: pd.DataFrame, features: list, target_col: str = 'Target'):
        """
        Fit preprocessing transformations on training data.

        Args:
            df: Training dataframe
            features: List of feature column names to use
            target_col: Name of target column

        Returns:
            self
        """
        self.feature_names = features

        # Identify feature types
        df_subset = df[features]
        self.numeric_features = df_subset.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df_subset.select_dtypes(include=['object', 'category']).columns.tolist()

        # Fit target encoder
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df[target_col])

        # Fit categorical encoders
        for col in self.categorical_features:
            encoder = LabelEncoder()
            # Handle potential missing values by including them in fitting
            values = df[col].fillna('missing').astype(str)
            encoder.fit(values)
            self.categorical_encoders[col] = encoder

        # Fit scaler if requested
        if self.scale_features and len(self.numeric_features) > 0:
            self.scaler = StandardScaler()
            self.scaler.fit(df[self.numeric_features])

        return self

    def transform(
        self,
        df: pd.DataFrame,
        features: list,
        target_col: Optional[str] = 'Target',
        include_target: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Transform data using fitted preprocessing.

        Args:
            df: Dataframe to transform
            features: List of feature columns
            target_col: Name of target column
            include_target: Whether to encode and return target

        Returns:
            Tuple of (X_transformed, y_encoded) or (X_transformed, None)
        """
        df_transformed = df[features].copy()

        # Handle missing values
        for col in df_transformed.columns:
            if df_transformed[col].isnull().any():
                if col in self.numeric_features:
                    # Fill numeric with median
                    df_transformed[col].fillna(df_transformed[col].median(), inplace=True)
                else:
                    # Fill categorical with 'missing'
                    df_transformed[col].fillna('missing', inplace=True)

        # Encode categorical features
        for col in self.categorical_features:
            if col in df_transformed.columns:
                encoder = self.categorical_encoders[col]
                # Handle unseen categories
                values = df_transformed[col].fillna('missing').astype(str)
                # Map unseen values to a default class (first class)
                transformed = []
                for val in values:
                    if val in encoder.classes_:
                        transformed.append(encoder.transform([val])[0])
                    else:
                        transformed.append(0)  # Default to first class for unseen values
                df_transformed[col] = transformed

        # Scale numeric features if enabled
        if self.scale_features and self.scaler is not None:
            numeric_cols = [col for col in self.numeric_features if col in df_transformed.columns]
            if numeric_cols:
                df_transformed[numeric_cols] = self.scaler.transform(df_transformed[numeric_cols])

        # Encode target if requested
        y = None
        if include_target and target_col in df.columns:
            y = pd.Series(
                self.target_encoder.transform(df[target_col]),
                index=df.index,
                name=target_col
            )

        return df_transformed, y

    def fit_transform(
        self,
        df: pd.DataFrame,
        features: list,
        target_col: str = 'Target'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform in one step.

        Args:
            df: Training dataframe
            features: List of feature columns
            target_col: Name of target column

        Returns:
            Tuple of (X_transformed, y_encoded)
        """
        self.fit(df, features, target_col)
        return self.transform(df, features, target_col, include_target=True)

    def inverse_transform_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """
        Convert encoded target back to original labels.

        Args:
            y_encoded: Encoded target values

        Returns:
            Original target labels
        """
        return self.target_encoder.inverse_transform(y_encoded)

    def get_target_classes(self) -> list:
        """
        Get original target class names.

        Returns:
            List of class names
        """
        return self.target_encoder.classes_.tolist()

    def save(self, filepath: str):
        """
        Save preprocessor to disk.

        Args:
            filepath: Path to save preprocessor
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str) -> 'StudentDataPreprocessor':
        """
        Load preprocessor from disk.

        Args:
            filepath: Path to saved preprocessor

        Returns:
            Loaded preprocessor instance
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor


def prepare_data_for_phase(
    df: pd.DataFrame,
    phase: str,
    feature_list: list,
    target_col: str = 'Target',
    scale_features: bool = False
) -> Tuple[pd.DataFrame, pd.Series, StudentDataPreprocessor]:
    """
    Prepare data for a specific modeling phase.

    Args:
        df: Raw dataframe
        phase: Modeling phase name (for logging)
        feature_list: List of features for this phase
        target_col: Target column name
        scale_features: Whether to scale features

    Returns:
        Tuple of (X_processed, y_encoded, preprocessor)
    """
    preprocessor = StudentDataPreprocessor(scale_features=scale_features)
    X, y = preprocessor.fit_transform(df, feature_list, target_col)

    print(f"\n{'='*60}")
    print(f"Preprocessing for {phase.upper()} phase")
    print(f"{'='*60}")
    print(f"Features: {len(feature_list)}")
    print(f"Samples: {len(X)}")
    print(f"Target classes: {preprocessor.get_target_classes()}")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")
    print(f"{'='*60}\n")

    return X, y, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_student_data
    from feature_groups import get_features_for_phase

    print("Testing preprocessing pipeline...")

    # Load data
    df = load_student_data()

    # Test with enrollment phase features
    features = get_features_for_phase('enrollment')
    X, y, preprocessor = prepare_data_for_phase(df, 'enrollment', features, scale_features=True)

    print("Preprocessing test successful!")
    print(f"\nProcessed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFirst 5 processed samples:")
    print(X.head())
