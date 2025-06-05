import matplotlib.pyplot as plt
import seaborn as sns

# Utility function to plot confusion matrix
def plot_confusion_matrix(cm_tensor, output_path, class_labels, title="Confusion Matrix", fmt='d'):
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
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved {title.lower()}: {output_path}")
    except Exception as e:
        print(f"Could not plot/save {title.lower()}: {e}")
