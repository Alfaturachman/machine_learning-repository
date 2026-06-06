import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    plt.close()
    return cm


def plot_roc_curve(y_true, y_proba, title='ROC Curve', return_thresholds=False):
    """Plot ROC curve and compute AUC. Optionally return thresholds for optimal cut-off."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.close()
    if return_thresholds:
        optimal_idx = (tpr - fpr).argmax()
        return roc_auc, thresholds[optimal_idx], fpr[optimal_idx], tpr[optimal_idx]
    return roc_auc


def print_classification_report(y_true, y_pred, target_names=None):
    """Print a formatted scikit-learn classification report."""
    print(classification_report(y_true, y_pred, target_names=target_names))
