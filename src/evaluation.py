"""
Evaluation Module

Comprehensive evaluation metrics for student retention forecasting.
Emphasizes metrics that matter for retention interventions:
- PR-AUC (handles class imbalance better than ROC-AUC)
- Recall@K (operational metric for limited advisor capacity)
- Calibration assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RetentionEvaluator:
    """
    Comprehensive evaluator for retention forecasting models.
    """

    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series, class_names: List[str] = None):
        """
        Initialize evaluator.

        Args:
            model: Trained sklearn-compatible model
            X_test: Test features
            y_test: Test labels
            class_names: Names of target classes
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names or [str(i) for i in sorted(y_test.unique())]
        self.n_classes = len(self.class_names)

        # Generate predictions
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    def compute_all_metrics(self, dropout_class: int = 1) -> Dict[str, float]:
        """
        Compute comprehensive metrics for retention forecasting.

        Args:
            dropout_class: Index of the dropout class for binary metrics

        Returns:
            Dictionary of metric names and values
        """
        metrics = {}

        # Basic metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics['accuracy'] = accuracy_score(self.y_test, self.y_pred)
        metrics['f1_weighted'] = f1_score(self.y_test, self.y_pred, average='weighted')
        metrics['precision_weighted'] = precision_score(self.y_test, self.y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(self.y_test, self.y_pred, average='weighted')

        # ROC-AUC (multi-class)
        if self.y_proba is not None:
            try:
                if self.n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(self.y_test, self.y_proba[:, 1])
                else:
                    metrics['roc_auc_ovr'] = roc_auc_score(self.y_test, self.y_proba, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(self.y_test, self.y_proba, multi_class='ovo')
            except Exception as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")

            # PR-AUC (critical for imbalanced data)
            try:
                if self.n_classes == 2:
                    metrics['pr_auc'] = average_precision_score(self.y_test, self.y_proba[:, 1])
                else:
                    # Compute PR-AUC for each class
                    pr_aucs = []
                    for i in range(self.n_classes):
                        y_true_binary = (self.y_test == i).astype(int)
                        pr_aucs.append(average_precision_score(y_true_binary, self.y_proba[:, i]))
                    metrics['pr_auc_macro'] = np.mean(pr_aucs)

                    # Dropout class specific
                    if dropout_class < self.n_classes:
                        y_dropout = (self.y_test == dropout_class).astype(int)
                        metrics['pr_auc_dropout'] = average_precision_score(y_dropout, self.y_proba[:, dropout_class])
            except Exception as e:
                print(f"Warning: Could not compute PR-AUC: {e}")

        return metrics

    def recall_at_k(self, k_percent: float, dropout_class: int = 1) -> Tuple[float, int]:
        """
        Compute Recall@K: recall when intervening on top K% of students.

        This is the KEY operational metric: given limited advisor capacity,
        how many actual dropout cases do we catch if we intervene on the
        top K% highest risk students?

        Args:
            k_percent: Percentage of students to flag (e.g., 10 = top 10%)
            dropout_class: Index of dropout class

        Returns:
            Tuple of (recall_value, number_of_students_flagged)
        """
        if self.y_proba is None:
            raise ValueError("Model must support predict_proba for Recall@K")

        # Get dropout probabilities
        dropout_proba = self.y_proba[:, dropout_class]

        # Calculate number of students in top K%
        n_total = len(dropout_proba)
        k_students = int(np.ceil(n_total * k_percent / 100))

        # Get indices of top K students by dropout probability
        top_k_indices = np.argsort(dropout_proba)[-k_students:]

        # Calculate recall: what fraction of actual dropouts are in top K?
        y_dropout = (self.y_test == dropout_class).astype(int)
        n_actual_dropouts = y_dropout.sum()

        if n_actual_dropouts == 0:
            return 0.0, k_students

        n_caught = y_dropout.iloc[top_k_indices].sum()
        recall = n_caught / n_actual_dropouts

        return recall, k_students

    def compute_recall_at_multiple_k(
        self,
        k_values: List[float] = [5, 10, 15, 20, 25],
        dropout_class: int = 1
    ) -> pd.DataFrame:
        """
        Compute Recall@K for multiple K values.

        Args:
            k_values: List of K percentages to evaluate
            dropout_class: Index of dropout class

        Returns:
            DataFrame with K values and corresponding recalls
        """
        results = []
        for k in k_values:
            recall, n_students = self.recall_at_k(k, dropout_class)
            results.append({
                'k_percent': k,
                'recall': recall,
                'n_students_flagged': n_students,
                'intervention_rate': f"{k}%"
            })

        return pd.DataFrame(results)

    def plot_confusion_matrix(self, figsize: Tuple[int, int] = (8, 6), save_path: Optional[str] = None):
        """
        Plot confusion matrix.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        cm = confusion_matrix(self.y_test, self.y_pred)

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_precision_recall_curve(
        self,
        dropout_class: int = 1,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot Precision-Recall curve for dropout class.

        Args:
            dropout_class: Index of dropout class
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.y_proba is None:
            print("Cannot plot PR curve: model does not support predict_proba")
            return

        y_dropout = (self.y_test == dropout_class).astype(int)
        dropout_proba = self.y_proba[:, dropout_class]

        precision, recall, thresholds = precision_recall_curve(y_dropout, dropout_proba)
        pr_auc = average_precision_score(y_dropout, dropout_proba)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Dropout Class)')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(
        self,
        dropout_class: int = 1,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve for dropout class.

        Args:
            dropout_class: Index of dropout class
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.y_proba is None:
            print("Cannot plot ROC curve: model does not support predict_proba")
            return

        y_dropout = (self.y_test == dropout_class).astype(int)
        dropout_proba = self.y_proba[:, dropout_class]

        fpr, tpr, _ = roc_curve(y_dropout, dropout_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Dropout Class)')
        plt.legend(loc='best')
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_recall_at_k(
        self,
        k_values: List[float] = [5, 10, 15, 20, 25, 30],
        dropout_class: int = 1,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot Recall@K for different intervention rates.

        Args:
            k_values: List of K percentages
            dropout_class: Index of dropout class
            figsize: Figure size
            save_path: Optional path to save figure
        """
        df_recall = self.compute_recall_at_multiple_k(k_values, dropout_class)

        plt.figure(figsize=figsize)
        plt.plot(df_recall['k_percent'], df_recall['recall'], marker='o', linewidth=2, markersize=8)
        plt.xlabel('Intervention Rate (Top K%)')
        plt.ylabel('Recall (Dropout Detection)')
        plt.title('Recall@K: Dropout Detection vs Intervention Rate')
        plt.grid(alpha=0.3)
        plt.xticks(k_values)

        # Add value labels
        for _, row in df_recall.iterrows():
            plt.text(row['k_percent'], row['recall'] + 0.02, f"{row['recall']:.2f}",
                    ha='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, dropout_class: int = 1, show_plots: bool = True) -> pd.DataFrame:
        """
        Generate comprehensive evaluation report.

        Args:
            dropout_class: Index of dropout class
            show_plots: Whether to display plots

        Returns:
            DataFrame with all metrics
        """
        print("=" * 70)
        print("RETENTION FORECASTING EVALUATION REPORT")
        print("=" * 70)

        # Compute metrics
        metrics = self.compute_all_metrics(dropout_class)

        print("\n--- CLASSIFICATION METRICS ---")
        for metric, value in metrics.items():
            print(f"{metric:25s}: {value:.4f}")

        # Recall@K
        print("\n--- RECALL@K (OPERATIONAL METRICS) ---")
        df_recall = self.compute_recall_at_multiple_k([10, 20, 30], dropout_class)
        print(df_recall.to_string(index=False))

        # Classification report
        print("\n--- DETAILED CLASSIFICATION REPORT ---")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        # Plots
        if show_plots and self.y_proba is not None:
            print("\nGenerating evaluation plots...")
            self.plot_confusion_matrix()
            self.plot_precision_recall_curve(dropout_class)
            self.plot_roc_curve(dropout_class)
            self.plot_recall_at_k(dropout_class=dropout_class)

        print("=" * 70)

        # Return metrics as DataFrame
        metrics_df = pd.DataFrame([metrics])
        return metrics_df


def compare_models(
    evaluators: Dict[str, RetentionEvaluator],
    metrics_to_compare: List[str] = ['accuracy', 'roc_auc_ovr', 'pr_auc_dropout', 'f1_weighted'],
    dropout_class: int = 1
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        evaluators: Dictionary mapping model names to evaluators
        metrics_to_compare: List of metric names to include
        dropout_class: Index of dropout class

    Returns:
        DataFrame with comparison results
    """
    results = []

    for model_name, evaluator in evaluators.items():
        metrics = evaluator.compute_all_metrics(dropout_class)
        recall_10, _ = evaluator.recall_at_k(10, dropout_class)
        recall_20, _ = evaluator.recall_at_k(20, dropout_class)

        row = {'model': model_name}
        for metric in metrics_to_compare:
            row[metric] = metrics.get(metric, np.nan)

        row['recall@10%'] = recall_10
        row['recall@20%'] = recall_20

        results.append(row)

    df_comparison = pd.DataFrame(results)
    return df_comparison


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")
    print("This requires a trained model. See training.py for examples.")
