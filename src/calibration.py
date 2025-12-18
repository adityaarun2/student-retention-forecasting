"""
Calibration Module

Probability calibration for retention forecasting models.
Well-calibrated probabilities are critical for:
- Setting intervention thresholds
- Resource allocation decisions
- Trustworthy risk communication
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import pickle


class ProbabilityCalibrator:
    """
    Handles probability calibration for retention forecasting models.
    """

    def __init__(self, base_model, method: str = 'sigmoid', cv: int = 5):
        """
        Initialize calibrator.

        Args:
            base_model: Trained sklearn-compatible model
            method: Calibration method ('sigmoid' for Platt, 'isotonic' for isotonic regression)
            cv: Number of cross-validation folds for calibration
        """
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.calibrated_model = None
        self.is_calibrated = False

    def calibrate(self, X, y, verbose: bool = True):
        """
        Calibrate the model's probability predictions.

        Args:
            X: Features for calibration
            y: True labels
            verbose: Whether to print progress

        Returns:
            Calibrated model
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Calibrating model using {self.method} method")
            print(f"Cross-validation folds: {self.cv}")
            print(f"{'='*60}")

        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=self.method,
            cv=self.cv
        )

        self.calibrated_model.fit(X, y)
        self.is_calibrated = True

        if verbose:
            print("✓ Calibration complete")
            print(f"{'='*60}\n")

        return self.calibrated_model

    def predict_proba(self, X):
        """
        Get calibrated probability predictions.

        Args:
            X: Features

        Returns:
            Calibrated probability predictions
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before prediction")

        return self.calibrated_model.predict_proba(X)

    def predict(self, X):
        """
        Get class predictions using calibrated probabilities.

        Args:
            X: Features

        Returns:
            Predicted classes
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before prediction")

        return self.calibrated_model.predict(X)

    def save(self, filepath: str):
        """
        Save calibrated model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before saving")

        with open(filepath, 'wb') as f:
            pickle.dump(self.calibrated_model, f)

        print(f"✓ Calibrated model saved to {filepath}")

    @staticmethod
    def load(filepath: str):
        """
        Load calibrated model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Calibrated model
        """
        with open(filepath, 'rb') as f:
            calibrated_model = pickle.load(f)

        print(f"✓ Calibrated model loaded from {filepath}")
        return calibrated_model


def plot_calibration_curve(
    y_true,
    y_proba_uncalibrated,
    y_proba_calibrated,
    n_bins: int = 10,
    class_label: str = 'Dropout',
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram comparing uncalibrated vs calibrated probabilities.

    Args:
        y_true: True binary labels
        y_proba_uncalibrated: Uncalibrated predicted probabilities
        y_proba_calibrated: Calibrated predicted probabilities
        n_bins: Number of bins for calibration curve
        class_label: Label for the positive class
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Compute calibration curves
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_proba_uncalibrated, n_bins=n_bins, strategy='uniform'
    )

    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_proba_calibrated, n_bins=n_bins, strategy='uniform'
    )

    # Plot 1: Calibration curves
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.plot(mean_pred_uncal, fraction_pos_uncal, 's-', label='Uncalibrated', linewidth=2)
    ax1.plot(mean_pred_cal, fraction_pos_cal, 'o-', label='Calibrated', linewidth=2)

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Calibration Curve ({class_label})')
    ax1.legend(loc='best')
    ax1.grid(alpha=0.3)

    # Plot 2: Probability histograms
    ax2.hist(y_proba_uncalibrated, bins=30, alpha=0.5, label='Uncalibrated', density=True)
    ax2.hist(y_proba_calibrated, bins=30, alpha=0.5, label='Calibrated', density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Probability Distribution')
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def compare_calibration(
    models_dict: dict,
    X_test,
    y_test,
    dropout_class: int = 1,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Compare calibration across multiple models.

    Args:
        models_dict: Dictionary mapping model names to trained models
        X_test: Test features
        y_test: Test labels (will be binarized for dropout class)
        dropout_class: Index of dropout class
        n_bins: Number of bins for calibration curve
        figsize: Figure size
        save_path: Optional path to save figure
    """
    plt.figure(figsize=figsize)

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    # Binary target for dropout class
    y_binary = (y_test == dropout_class).astype(int)

    # Plot calibration curve for each model
    for model_name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, dropout_class]

        fraction_pos, mean_pred = calibration_curve(
            y_binary, y_proba, n_bins=n_bins, strategy='uniform'
        )

        plt.plot(mean_pred, fraction_pos, 's-', label=model_name, linewidth=2, markersize=6)

    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Comparison Across Models')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def calculate_calibration_error(
    y_true,
    y_proba,
    n_bins: int = 10
) -> Tuple[float, float]:
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        n_bins: Number of bins

    Returns:
        Tuple of (ECE, MCE)
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )

    # Calculate bin sizes
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sizes = np.bincount(bin_indices, minlength=n_bins)
    total = len(y_proba)

    # Expected Calibration Error (ECE)
    ece = 0.0
    for i in range(len(fraction_pos)):
        bin_size = bin_sizes[i]
        if bin_size > 0:
            ece += (bin_size / total) * abs(fraction_pos[i] - mean_pred[i])

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(fraction_pos - mean_pred))

    return ece, mce


def find_optimal_threshold(
    y_true,
    y_proba,
    metric: str = 'f1',
    thresholds: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for a given metric.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
        thresholds: Optional array of thresholds to evaluate

    Returns:
        Tuple of (optimal_threshold, optimal_metric_value)
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    best_threshold = 0.5
    best_score = 0.0

    from sklearn.metrics import f1_score, precision_score, recall_score

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tp = ((y_true == 1) & (y_pred == 1)).sum()

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def create_risk_tiers(
    y_proba,
    low_threshold: float = 0.3,
    high_threshold: float = 0.7
) -> np.ndarray:
    """
    Create risk tier labels based on probability thresholds.

    Args:
        y_proba: Predicted probabilities
        low_threshold: Threshold below which students are low risk
        high_threshold: Threshold above which students are high risk

    Returns:
        Array of risk tier labels
    """
    risk_tiers = np.full(len(y_proba), 'Medium', dtype=object)
    risk_tiers[y_proba < low_threshold] = 'Low'
    risk_tiers[y_proba >= high_threshold] = 'High'

    return risk_tiers


if __name__ == "__main__":
    print("Calibration module loaded successfully.")
    print("\nKey functions:")
    print("  - ProbabilityCalibrator: Calibrate model probabilities")
    print("  - plot_calibration_curve: Visualize calibration")
    print("  - calculate_calibration_error: Compute ECE and MCE")
    print("  - find_optimal_threshold: Optimize decision threshold")
    print("  - create_risk_tiers: Assign risk levels to students")
