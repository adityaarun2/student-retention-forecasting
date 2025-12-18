"""
Training Module

Handles model training, validation, and persistence for student retention forecasting.
Supports phased modeling with proper train/test splitting and metric logging.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from typing import Tuple, Dict, Any, Optional
import pickle
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Handles training and validation of retention forecasting models.
    """

    def __init__(
        self,
        model,
        model_name: str,
        phase: str,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize trainer.

        Args:
            model: Sklearn-compatible model instance
            model_name: Name identifier for the model
            phase: Modeling phase (enrollment, semester_1, semester_2)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.model_name = model_name
        self.phase = phase
        self.test_size = test_size
        self.random_state = random_state
        self.is_trained = False
        self.training_metrics = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], Any]:
        """
        Train the model on provided data.

        Args:
            X: Feature matrix
            y: Target vector
            verbose: Whether to print training progress

        Returns:
            Tuple of (training_metrics, trained_model)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training: {self.model_name} | Phase: {self.phase.upper()}")
            print(f"{'='*70}")
            print(f"Total samples: {len(X)}")
            print(f"Features: {X.shape[1]}")
            print(f"Class distribution:\n{pd.Series(y).value_counts()}")

        # Train the model
        self.model.fit(X, y)
        self.is_trained = True

        if verbose:
            print(f"✓ Model trained successfully")
            print(f"{'='*70}\n")

        return self.model

    def train_with_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation for performance estimation.

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds
            verbose: Whether to print results

        Returns:
            Dictionary containing training results and CV scores
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training with {cv_folds}-Fold Cross-Validation")
            print(f"Model: {self.model_name} | Phase: {self.phase.upper()}")
            print(f"{'='*70}")

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        scoring = {
            'accuracy': 'accuracy',
            'roc_auc_ovr': 'roc_auc_ovr',
            'f1_weighted': 'f1_weighted'
        }

        cv_results = cross_validate(
            self.model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        # Train on full data
        self.model.fit(X, y)
        self.is_trained = True

        # Aggregate results
        results = {
            'model_name': self.model_name,
            'phase': self.phase,
            'cv_folds': cv_folds,
            'train_accuracy': cv_results['train_accuracy'].mean(),
            'val_accuracy': cv_results['test_accuracy'].mean(),
            'val_accuracy_std': cv_results['test_accuracy'].std(),
            'val_roc_auc': cv_results['test_roc_auc_ovr'].mean(),
            'val_roc_auc_std': cv_results['test_roc_auc_ovr'].std(),
            'val_f1': cv_results['test_f1_weighted'].mean(),
            'val_f1_std': cv_results['test_f1_weighted'].std()
        }

        if verbose:
            print(f"\nCross-Validation Results:")
            print(f"  Accuracy: {results['val_accuracy']:.4f} (±{results['val_accuracy_std']:.4f})")
            print(f"  ROC-AUC:  {results['val_roc_auc']:.4f} (±{results['val_roc_auc_std']:.4f})")
            print(f"  F1:       {results['val_f1']:.4f} (±{results['val_f1_std']:.4f})")
            print(f"✓ Model trained on full dataset")
            print(f"{'='*70}\n")

        self.training_metrics = results
        return results

    def save_model(self, output_dir: str, include_metadata: bool = True):
        """
        Save trained model to disk.

        Args:
            output_dir: Directory to save model
            include_metadata: Whether to save training metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_filename = f"{self.model_name}_{self.phase}.pkl"
        model_path = output_path / model_filename

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"✓ Model saved: {model_path}")

        # Save metadata
        if include_metadata and self.training_metrics:
            metadata_filename = f"{self.model_name}_{self.phase}_metadata.json"
            metadata_path = output_path / metadata_filename

            with open(metadata_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)

            print(f"✓ Metadata saved: {metadata_path}")

    @staticmethod
    def load_model(filepath: str):
        """
        Load a trained model from disk.

        Args:
            filepath: Path to saved model file

        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"✓ Model loaded from {filepath}")
        return model


def train_model_for_phase(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    model_name: str,
    phase: str,
    use_cv: bool = True,
    cv_folds: int = 5,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience function to train a model for a specific phase.

    Args:
        X: Feature matrix
        y: Target vector
        model: Model instance to train
        model_name: Model identifier
        phase: Modeling phase
        use_cv: Whether to use cross-validation
        cv_folds: Number of CV folds
        save_dir: Optional directory to save model
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, metrics)
    """
    trainer = ModelTrainer(model, model_name, phase)

    if use_cv:
        metrics = trainer.train_with_validation(X, y, cv_folds=cv_folds, verbose=verbose)
    else:
        trainer.train(X, y, verbose=verbose)
        metrics = {}

    if save_dir:
        trainer.save_model(save_dir)

    return trainer.model, metrics


def train_all_phases(
    df: pd.DataFrame,
    model_factory_func,
    model_name: str,
    feature_groups: Dict[str, list],
    target_col: str = 'Target',
    preprocessor_class=None,
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train a model across all temporal phases.

    Args:
        df: Raw dataframe
        model_factory_func: Function that returns a new model instance
        model_name: Model identifier
        feature_groups: Dict mapping phase names to feature lists
        target_col: Target column name
        preprocessor_class: Preprocessor class (if None, uses default)
        save_dir: Optional directory to save models
        verbose: Whether to print progress

    Returns:
        Dictionary mapping phase names to results
    """
    results = {}

    for phase, features in feature_groups.items():
        if verbose:
            print(f"\n{'#'*70}")
            print(f"PHASE: {phase.upper()}")
            print(f"{'#'*70}")

        # Create new model instance for this phase
        model = model_factory_func()

        # Preprocess data if preprocessor provided
        if preprocessor_class:
            preprocessor = preprocessor_class()
            X, y = preprocessor.fit_transform(df, features, target_col)

            # Save preprocessor if save_dir provided
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                preprocessor.save(f"{save_dir}/{model_name}_{phase}_preprocessor.pkl")
        else:
            X = df[features]
            y = df[target_col]

        # Train model
        trained_model, metrics = train_model_for_phase(
            X, y, model, model_name, phase,
            use_cv=True,
            save_dir=save_dir,
            verbose=verbose
        )

        results[phase] = {
            'model': trained_model,
            'metrics': metrics,
            'n_features': len(features),
            'n_samples': len(X)
        }

    return results


if __name__ == "__main__":
    # Test training pipeline
    from data_loader import load_student_data
    from feature_groups import get_features_for_phase
    from preprocessing import StudentDataPreprocessor
    from models import create_model

    print("Testing training pipeline...")

    # Load data
    df = load_student_data()

    # Get features for enrollment phase
    features = get_features_for_phase('enrollment')

    # Preprocess
    preprocessor = StudentDataPreprocessor()
    X, y = preprocessor.fit_transform(df, features, 'Target')

    # Create model
    model = create_model('logistic_regression')

    # Train
    trainer = ModelTrainer(model, 'logistic_regression', 'enrollment')
    results = trainer.train_with_validation(X, y, cv_folds=5)

    print("\n✓ Training pipeline test successful!")
    print(f"\nValidation ROC-AUC: {results['val_roc_auc']:.4f}")
