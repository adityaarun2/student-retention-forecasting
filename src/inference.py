"""
Inference Module

Production inference pipeline for student retention forecasting.
Loads trained models and preprocessors to make predictions on new student data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class RetentionPredictor:
    """
    Production-ready predictor for student retention forecasting.
    """

    def __init__(self, model_path: str, preprocessor_path: str, phase: str):
        """
        Initialize predictor with trained model and preprocessor.

        Args:
            model_path: Path to saved model file
            preprocessor_path: Path to saved preprocessor file
            phase: Modeling phase (enrollment, semester_1, semester_2)
        """
        self.phase = phase
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self.class_names = self.preprocessor.get_target_classes()

    @staticmethod
    def _load_model(path: str):
        """Load trained model from disk."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded from {path}")
        return model

    @staticmethod
    def _load_preprocessor(path: str):
        """Load preprocessor from disk."""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✓ Preprocessor loaded from {path}")
        return preprocessor

    def predict_single_student(
        self,
        student_data: Dict[str, Union[int, float, str]],
        return_probabilities: bool = True,
        return_risk_tier: bool = True
    ) -> Dict[str, any]:
        """
        Make prediction for a single student.

        Args:
            student_data: Dictionary of student features
            return_probabilities: Whether to return class probabilities
            return_risk_tier: Whether to assign risk tier

        Returns:
            Dictionary containing predictions and metadata
        """
        # Convert to DataFrame
        df = pd.DataFrame([student_data])

        # Preprocess
        X, _ = self.preprocessor.transform(
            df,
            self.preprocessor.feature_names,
            include_target=False
        )

        # Predict
        prediction = self.model.predict(X)[0]
        predicted_class = self.class_names[prediction]

        result = {
            'predicted_class': predicted_class,
            'predicted_class_index': int(prediction),
            'phase': self.phase
        }

        # Add probabilities
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[0]
            result['probabilities'] = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }

            # Dropout probability (assuming class index 1 or class name 'Dropout')
            if 'Dropout' in self.class_names:
                dropout_idx = self.class_names.index('Dropout')
            elif len(self.class_names) > 1:
                dropout_idx = 1  # Default assumption
            else:
                dropout_idx = 0

            result['dropout_probability'] = float(probabilities[dropout_idx])

            # Risk tier
            if return_risk_tier:
                risk_tier = self._assign_risk_tier(probabilities[dropout_idx])
                result['risk_tier'] = risk_tier

        return result

    def predict_batch(
        self,
        df: pd.DataFrame,
        return_probabilities: bool = True,
        return_risk_tier: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions for multiple students.

        Args:
            df: DataFrame with student features
            return_probabilities: Whether to return class probabilities
            return_risk_tier: Whether to assign risk tiers

        Returns:
            DataFrame with predictions
        """
        # Preprocess
        X, _ = self.preprocessor.transform(
            df,
            self.preprocessor.feature_names,
            include_target=False
        )

        # Predict
        predictions = self.model.predict(X)
        predicted_classes = [self.class_names[pred] for pred in predictions]

        results_df = pd.DataFrame({
            'predicted_class': predicted_classes,
            'predicted_class_index': predictions
        })

        # Add probabilities
        if return_probabilities and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)

            # Add probability for each class
            for i, class_name in enumerate(self.class_names):
                results_df[f'prob_{class_name}'] = probabilities[:, i]

            # Dropout probability
            if 'Dropout' in self.class_names:
                dropout_idx = self.class_names.index('Dropout')
            elif len(self.class_names) > 1:
                dropout_idx = 1
            else:
                dropout_idx = 0

            results_df['dropout_probability'] = probabilities[:, dropout_idx]

            # Risk tiers
            if return_risk_tier:
                results_df['risk_tier'] = results_df['dropout_probability'].apply(
                    self._assign_risk_tier
                )

        return results_df

    @staticmethod
    def _assign_risk_tier(dropout_probability: float) -> str:
        """
        Assign risk tier based on dropout probability.

        Args:
            dropout_probability: Predicted probability of dropout

        Returns:
            Risk tier label
        """
        if dropout_probability < 0.3:
            return 'Low'
        elif dropout_probability < 0.6:
            return 'Medium'
        else:
            return 'High'

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance if model supports it.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.preprocessor.feature_names

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)

            return importance_df

        elif hasattr(self.model, 'coef_'):
            # For linear models
            coef = np.abs(self.model.coef_[0])
            feature_names = self.preprocessor.feature_names

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coef
            }).sort_values('importance', ascending=False).head(top_n)

            return importance_df

        else:
            print("Model does not support feature importance extraction")
            return pd.DataFrame()


class MultiPhasePredictor:
    """
    Manages predictions across multiple temporal phases.
    """

    def __init__(self, models_dir: str):
        """
        Initialize with directory containing saved models.

        Args:
            models_dir: Directory path containing model and preprocessor files
        """
        self.models_dir = Path(models_dir)
        self.predictors = {}

    def load_phase(self, model_name: str, phase: str):
        """
        Load model and preprocessor for a specific phase.

        Args:
            model_name: Name of the model (e.g., 'xgboost')
            phase: Phase name (e.g., 'enrollment')
        """
        model_path = self.models_dir / f"{model_name}_{phase}.pkl"
        preprocessor_path = self.models_dir / f"{model_name}_{phase}_preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")

        predictor = RetentionPredictor(str(model_path), str(preprocessor_path), phase)
        self.predictors[phase] = predictor

        print(f"✓ Loaded {model_name} for {phase} phase")

    def predict_all_phases(self, student_data: Dict) -> Dict[str, Dict]:
        """
        Make predictions using all loaded phases.

        Args:
            student_data: Dictionary of student features

        Returns:
            Dictionary mapping phase names to prediction results
        """
        results = {}

        for phase, predictor in self.predictors.items():
            try:
                prediction = predictor.predict_single_student(student_data)
                results[phase] = prediction
            except Exception as e:
                results[phase] = {'error': str(e)}

        return results

    def compare_phases(self, student_data: Dict) -> pd.DataFrame:
        """
        Compare predictions across phases for a single student.

        Args:
            student_data: Dictionary of student features

        Returns:
            DataFrame comparing predictions across phases
        """
        results = self.predict_all_phases(student_data)

        comparison = []
        for phase, result in results.items():
            if 'error' not in result:
                comparison.append({
                    'phase': phase,
                    'predicted_class': result.get('predicted_class', 'N/A'),
                    'dropout_probability': result.get('dropout_probability', np.nan),
                    'risk_tier': result.get('risk_tier', 'N/A')
                })

        return pd.DataFrame(comparison)


def create_example_student(phase: str = 'enrollment') -> Dict:
    """
    Create example student data for testing.

    Args:
        phase: Which phase to create example for

    Returns:
        Dictionary of student features
    """
    example = {
        # Demographics
        'Marital Status': 1,
        'Nacionality': 1,
        'Age at enrollment': 20,
        'Gender': 1,
        'International': 0,

        # Application
        'Application mode': 17,
        'Application order': 1,
        'Course': 9254,
        'Daytime/evening attendance': 1,

        # Prior Education
        'Previous qualification': 1,
        'Previous qualification (grade)': 150.0,
        'Admission grade': 140.0,

        # Parental Education
        "Mother's qualification": 19,
        "Father's qualification": 12,
        "Mother's occupation": 5,
        "Father's occupation": 10,

        # Financial
        'Displaced': 0,
        'Educational special needs': 0,
        'Debtor': 0,
        'Tuition fees up to date': 1,
        'Scholarship holder': 0,

        # Economic Context
        'Unemployment rate': 10.8,
        'Inflation rate': 1.4,
        'GDP': 1.74
    }

    if phase in ['semester_1', 'semester_2']:
        # Add semester 1 features
        example.update({
            'Curricular units 1st sem (credited)': 0,
            'Curricular units 1st sem (enrolled)': 6,
            'Curricular units 1st sem (evaluations)': 6,
            'Curricular units 1st sem (approved)': 5,
            'Curricular units 1st sem (grade)': 13.5,
            'Curricular units 1st sem (without evaluations)': 0
        })

    if phase == 'semester_2':
        # Add semester 2 features
        example.update({
            'Curricular units 2nd sem (credited)': 0,
            'Curricular units 2nd sem (enrolled)': 6,
            'Curricular units 2nd sem (evaluations)': 6,
            'Curricular units 2nd sem (approved)': 5,
            'Curricular units 2nd sem (grade)': 13.0,
            'Curricular units 2nd sem (without evaluations)': 0
        })

    return example


if __name__ == "__main__":
    print("Inference module loaded successfully.")
    print("\nKey classes:")
    print("  - RetentionPredictor: Single-phase prediction")
    print("  - MultiPhasePredictor: Multi-phase comparison")
    print("\nExample usage:")
    print("  predictor = RetentionPredictor(model_path, preprocessor_path, 'enrollment')")
    print("  result = predictor.predict_single_student(student_data)")
