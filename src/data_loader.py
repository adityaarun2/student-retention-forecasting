"""
Data Loader Module

This module is responsible for loading the UCI Student Dropout and Academic Success dataset.
It provides a clean interface to fetch the data without any preprocessing or modeling logic.

Dataset: UCI ML Repository ID 697
Source: fetch_ucirepo(id=697)
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
from typing import Tuple


def load_student_data() -> pd.DataFrame:
    """
    Load the UCI Student Dropout and Academic Success dataset.

    Returns:
        pd.DataFrame: Raw dataset with original column names preserved

    Raises:
        Exception: If dataset cannot be fetched from UCI repository
    """
    try:
        dataset = fetch_ucirepo(id=697)
        df = dataset.data.original

        # Basic validation
        if df is None or df.empty:
            raise ValueError("Dataset is empty or could not be loaded")

        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        raise Exception(f"Failed to load dataset from UCI repository: {str(e)}")


def get_dataset_info() -> dict:
    """
    Get metadata about the dataset.

    Returns:
        dict: Dataset metadata including shape, columns, and target distribution
    """
    df = load_student_data()

    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'target_column': 'Target',
        'target_classes': df['Target'].value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'dtypes': df.dtypes.to_dict()
    }

    return info


if __name__ == "__main__":
    # Test the data loader
    df = load_student_data()
    print("\nDataset Preview:")
    print(df.head())
    print("\nTarget Distribution:")
    print(df['Target'].value_counts())
    print("\nDataset Info:")
    info = get_dataset_info()
    print(f"Shape: {info['shape']}")
    print(f"Missing values: {info['missing_values']}")
