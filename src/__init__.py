"""
Student Retention Forecasting Package

A methodologically sound, academically rigorous package for student retention forecasting
using the UCI Student Dropout and Academic Success dataset.

Key Features:
- Phased modeling to prevent temporal leakage
- Proper evaluation metrics (PR-AUC, Recall@K)
- Probability calibration
- Production-ready inference

Modules:
- data_loader: Load dataset from UCI repository
- feature_groups: Define temporal feature phases
- preprocessing: Data cleaning and transformation
- models: Model factories and configurations
- training: Model training with cross-validation
- evaluation: Comprehensive evaluation metrics
- calibration: Probability calibration
- inference: Production inference pipeline
"""

from .data_loader import load_student_data, get_dataset_info
from .feature_groups import (
    get_features_for_phase,
    validate_features,
    get_phase_description,
    ENROLLMENT_FEATURES,
    SEMESTER_1_FEATURES,
    SEMESTER_2_FEATURES,
    TARGET
)
from .preprocessing import StudentDataPreprocessor, prepare_data_for_phase
from .models import ModelFactory, create_model, get_model_configs
from .training import ModelTrainer, train_model_for_phase, train_all_phases
from .evaluation import RetentionEvaluator, compare_models
from .calibration import ProbabilityCalibrator, plot_calibration_curve
from .inference import RetentionPredictor, MultiPhasePredictor, create_example_student

__version__ = '1.0.0'
__author__ = 'Student Retention Research Team'

__all__ = [
    # Data
    'load_student_data',
    'get_dataset_info',

    # Features
    'get_features_for_phase',
    'validate_features',
    'get_phase_description',
    'ENROLLMENT_FEATURES',
    'SEMESTER_1_FEATURES',
    'SEMESTER_2_FEATURES',
    'TARGET',

    # Preprocessing
    'StudentDataPreprocessor',
    'prepare_data_for_phase',

    # Models
    'ModelFactory',
    'create_model',
    'get_model_configs',

    # Training
    'ModelTrainer',
    'train_model_for_phase',
    'train_all_phases',

    # Evaluation
    'RetentionEvaluator',
    'compare_models',

    # Calibration
    'ProbabilityCalibrator',
    'plot_calibration_curve',

    # Inference
    'RetentionPredictor',
    'MultiPhasePredictor',
    'create_example_student',
]
