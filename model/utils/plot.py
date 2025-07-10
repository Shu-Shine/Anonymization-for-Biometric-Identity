import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



def calculate_AUROC_AUPR(y_true_bin, y_pred_probs, n_classes):
    auroc_list, aupr_list = [], []
    for i in range(n_classes):
        if y_true_bin.shape[1] > i:
            # AUROC
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
                auroc = auc(fpr, tpr) if not (np.isnan(fpr).any() or np.isnan(tpr).any()) else 0.0
            except:
                auroc = 0.0
            auroc_list.append(auroc)

            # AUPR
            try:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
                aupr = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i]) if not (
                    np.isnan(precision).any() or np.isnan(recall).any()) else 0.0
            except:
                aupr = 0.0
            aupr_list.append(aupr)
        else:
            auroc_list.append(0.0)
            aupr_list.append(0.0)
    return auroc_list, aupr_list


def plot_ROC_PR_curves(y_true_bin, y_pred_probs, class_labels, save_path, curve_type="roc"):
    plt.figure(figsize=(10, 8))
    for i in range(len(class_labels)):
        if y_true_bin.shape[1] > i:
            try:
                if curve_type == "roc":
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
                    auc_score = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{class_labels[i]} (AUC = {auc_score:.3f})')
                elif curve_type == "pr":
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
                    ap_score = average_precision_score(y_true_bin[:, i], y_pred_probs[:, i])
                    plt.plot(recall, precision, label=f'{class_labels[i]} (AP = {ap_score:.3f})')
            except:
                continue
    if curve_type == "roc":
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
    elif curve_type == "pr":
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves (One-vs-Rest)')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {curve_type.upper()} curves: {save_path}")


def plot_confusion_matrix(cm_tensor, save_path, class_labels, title="Confusion Matrix", fmt='d'):
    if cm_tensor is None:
        print(f"--- WARNING: Confusion matrix tensor is None for {title}.")
        return

    try:
        cm_numpy = cm_tensor.cpu().numpy()
        fig, ax = plt.subplots(
            figsize=(max(8, len(class_labels) * 0.9), max(6, len(class_labels) * 0.8)))
        sns.heatmap(cm_numpy, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved {title.lower()}: {save_path}")
    except Exception as e:
        print(f"Could not plot/save {title.lower()}: {e}")
