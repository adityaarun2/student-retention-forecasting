"""
Models Module

Defines model factories and configurations for student retention forecasting.
This module only creates model instances - training happens in training.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any


class ModelFactory:
    """
    Factory class for creating configured model instances.
    """

    @staticmethod
    def create_logistic_regression(
        class_weight: str = 'balanced',
        max_iter: int = 1000,
        random_state: int = 42,
        **kwargs
    ) -> LogisticRegression:
        """
        Create a Logistic Regression model with class weighting.

        Args:
            class_weight: Strategy for handling class imbalance
            max_iter: Maximum iterations for solver
            random_state: Random seed
            **kwargs: Additional parameters for LogisticRegression

        Returns:
            Configured LogisticRegression instance
        """
        return LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs',
            **kwargs
        )

    @staticmethod
    def create_random_forest(
        n_estimators: int = 100,
        max_depth: int = 10,
        class_weight: str = 'balanced',
        random_state: int = 42,
        **kwargs
    ) -> RandomForestClassifier:
        """
        Create a Random Forest classifier with class weighting.

        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            class_weight: Strategy for handling class imbalance
            random_state: Random seed
            **kwargs: Additional parameters for RandomForestClassifier

        Returns:
            Configured RandomForestClassifier instance
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

    @staticmethod
    def create_xgboost(
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = None,
        random_state: int = 42,
        **kwargs
    ) -> XGBClassifier:
        """
        Create an XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            scale_pos_weight: Weight for positive class (for imbalance)
            random_state: Random seed
            **kwargs: Additional parameters for XGBClassifier

        Returns:
            Configured XGBClassifier instance
        """
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'eval_metric': 'mlogloss',
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        if scale_pos_weight is not None:
            params['scale_pos_weight'] = scale_pos_weight

        params.update(kwargs)

        return XGBClassifier(**params)


def get_model_configs() -> Dict[str, Dict[str, Any]]:
    """
    Get predefined model configurations for experiments.

    Returns:
        Dictionary mapping model names to their configurations
    """
    configs = {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'params': {
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            },
            'description': 'Linear baseline with class balancing'
        },
        'random_forest': {
            'name': 'Random Forest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'description': 'Ensemble decision trees with balanced classes'
        },
        'random_forest_deep': {
            'name': 'Random Forest (Deep)',
            'params': {
                'n_estimators': 200,
                'max_depth': 20,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'description': 'Deeper random forest for complex patterns'
        },
        'xgboost': {
            'name': 'XGBoost',
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            },
            'description': 'Gradient boosting for best performance'
        },
        'xgboost_tuned': {
            'name': 'XGBoost (Tuned)',
            'params': {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'description': 'Fine-tuned XGBoost with regularization'
        }
    }

    return configs


def create_model(model_name: str, **override_params) -> Any:
    """
    Create a model instance by name with optional parameter overrides.

    Args:
        model_name: Name of model configuration
        **override_params: Parameters to override defaults

    Returns:
        Configured model instance

    Raises:
        ValueError: If model_name is not recognized
    """
    configs = get_model_configs()

    if model_name not in configs:
        available = ', '.join(configs.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    config = configs[model_name]
    params = config['params'].copy()
    params.update(override_params)

    # Determine which factory method to use
    if 'logistic' in model_name.lower():
        return ModelFactory.create_logistic_regression(**params)
    elif 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
        return ModelFactory.create_random_forest(**params)
    elif 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
        return ModelFactory.create_xgboost(**params)
    else:
        raise ValueError(f"Cannot determine model type for '{model_name}'")


if __name__ == "__main__":
    # Test model creation
    print("Available Model Configurations:")
    print("=" * 70)

    configs = get_model_configs()
    for name, config in configs.items():
        print(f"\n{config['name']}")
        print(f"  Key: {name}")
        print(f"  Description: {config['description']}")
        print(f"  Parameters: {config['params']}")

    print("\n" + "=" * 70)
    print("\nTesting model instantiation...")

    # Test creating each model type
    for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
        model = create_model(model_name)
        print(f"✓ Created: {model.__class__.__name__}")

    print("\nAll models created successfully!")
