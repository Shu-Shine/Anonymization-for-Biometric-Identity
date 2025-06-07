import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import re
from pathlib import Path
from utils import increment_path

plt.switch_backend('Agg')


def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = name.replace(" ", "_")
    return name


def find_metric_files(base_log_dir, specific_versions=None):
    metric_files = []
    if specific_versions:
        for version in specific_versions:
            path = os.path.join(base_log_dir, version, "metrics.csv")
            if os.path.exists(path):
                metric_files.append({"path": path, "name": f"{os.path.basename(base_log_dir)}_{version}"})
            else:
                print(f"Warning: metrics.csv not found for specific version: {path}")
    else:
        for version_dir in glob.glob(os.path.join(base_log_dir, "version_*")):
            path = os.path.join(version_dir, "metrics.csv")
            if os.path.exists(path):
                parent_dir_name = os.path.basename(os.path.dirname(version_dir))
                version_name = os.path.basename(version_dir)
                run_name = f"{parent_dir_name}_{version_name}"
                metric_files.append({"path": path, "name": run_name})
    return metric_files


def plot_metrics(metrics_data_list, metric_pairs_to_plot, output_dir, model_name, combined_plot=False):
    """
    Plots specified pairs of train/validation metrics.
    metrics_data_list: List of dicts, each with 'name' and 'df'.
    metric_pairs_to_plot: List of tuples, e.g., [('train_loss', 'val_loss'), ('train_acc', 'val_acc')].
    output_dir: Directory to save plots.
    combined_plot: If True, plots all runs on the same graph for each metric pair.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Define how to get display names for metrics
    def get_display_name(metric_key):
        name = metric_key.replace("_epoch", "").replace("_step", "").replace("train_", "")
        if "loss" in name: return "Loss"
        if "acc" in name: return "Accuracy"
        # Add more specific display names as needed
        return name.replace("_", " ").title()

    def get_base_metric_name(metric_key):  # e.g. from val_loss_epoch -> loss
        name = metric_key.replace("_epoch", "").replace("_step", "")
        if name.startswith("val_") or name.startswith("train_"):
            name = name[4:]  # remove val_ or train_
        return name

    if combined_plot:
        for train_metric_key, val_metric_key in metric_pairs_to_plot:
            plt.figure(figsize=(12, 7))

            # Use a common base name for the plot title, e.g., "Loss", "Accuracy"
            base_display_name = get_display_name(train_metric_key)  # Assuming train_metric gives a good base

            for item in metrics_data_list:
                df = item['df']
                run_name = item['name']

                if 'epoch' not in df.columns:
                    print(
                        f"Warning: 'epoch' column not found in {run_name}. Skipping this run for {base_display_name} plot.")
                    continue

                # Plot training metric
                if train_metric_key in df.columns:
                    plot_df_train = df[['epoch', train_metric_key]].dropna()
                    if not plot_df_train.empty:
                        plt.plot(plot_df_train['epoch'], plot_df_train[train_metric_key],
                                 label=f"{run_name} Train", linestyle='--')  # Dashed for train
                else:
                    print(f"Warning: Training metric '{train_metric_key}' not found in {run_name}.")

                # Plot validation metric
                if val_metric_key in df.columns:
                    plot_df_val = df[['epoch', val_metric_key]].dropna()
                    if not plot_df_val.empty:
                        plt.plot(plot_df_val['epoch'], plot_df_val[val_metric_key],
                                 label=f"{run_name} Val", linestyle='-')  # Solid for val
                else:
                    print(f"Warning: Validation metric '{val_metric_key}' not found in {run_name}.")

            plt.title(f"Combined {base_display_name} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(base_display_name)
            plt.legend(loc='best', fontsize='small')
            plt.grid(True)
            plt.tight_layout()
            sanitized_metric_name = sanitize_filename(base_display_name.lower())
            save_path = os.path.join(output_dir, f"combined_{sanitized_metric_name}_train_val_vs_epoch.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved combined plot: {save_path}")

    else:  # Individual plots per run, but still train/val on the same axes for that run
        for item in metrics_data_list:
            df = item['df']
            # run_name = item['name']
            run_name = model_name  # Use model_name as plot title, instead of folder name
            sanitized_run_name = sanitize_filename(run_name)

            if 'epoch' not in df.columns:
                print(f"Warning: 'epoch' column not found in {run_name}. Skipping plots for this run.")
                continue

            for train_metric_key, val_metric_key in metric_pairs_to_plot:
                base_display_name = get_display_name(train_metric_key)
                plt.figure(figsize=(10, 6))

                has_data_to_plot = False
                # Plot training metric
                if train_metric_key in df.columns:
                    plot_df_train = df[['epoch', train_metric_key]].dropna()
                    if not plot_df_train.empty:
                        plt.plot(plot_df_train['epoch'], plot_df_train[train_metric_key], label="Train", linestyle='--')
                        has_data_to_plot = True
                else:
                    print(f"Warning: Training metric '{train_metric_key}' not found in {run_name}.")

                # Plot validation metric
                if val_metric_key in df.columns:
                    plot_df_val = df[['epoch', val_metric_key]].dropna()
                    if not plot_df_val.empty:
                        plt.plot(plot_df_val['epoch'], plot_df_val[val_metric_key], label="Validation", linestyle='-')
                        has_data_to_plot = True
                else:
                    print(f"Warning: Validation metric '{val_metric_key}' not found in {run_name}.")

                if not has_data_to_plot:
                    print(f"No data to plot for {base_display_name} for run {run_name}. Skipping this plot.")
                    plt.close()  # Close empty figure
                    continue

                plt.title(f"{base_display_name} vs. Epoch ({run_name})")
                plt.xlabel("Epoch")
                plt.ylabel(base_display_name)
                plt.legend(loc='best')
                plt.grid(True)
                plt.tight_layout()
                sanitized_metric_name = sanitize_filename(base_display_name.lower())
                save_path = os.path.join(output_dir,
                                         f"{sanitized_run_name}_{sanitized_metric_name}_train_val_vs_epoch.png")
                plt.savefig(save_path)
                plt.close()
                print(f"Saved plot: {save_path}")


def main():
    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]  # model/utils → model → root
    output_dir = increment_path(os.path.join(root_dir, "output", "plots"), "training_plots")

    parser = argparse.ArgumentParser(description="Plot training and validation curves from PyTorch Lightning CSV logs.")
    parser.add_argument("--model_name", type=str, default="DINOv2",help="Name of the model.")
    parser.add_argument(
        "--log_dirs", type=str, required=False, nargs='+',
        default=[os.path.join(root_dir, "output", "run_logs")],
        help="Base directory(ies) containing experiment versions."
    )
    parser.add_argument(
        "--versions", type=str, default="version_24",
        help="Optional: Comma-separated list of specific version subdirectories."
    )

    parser.add_argument(
        "--output_dir", type=str, default=output_dir,
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="If set, plot all runs on a single combined graph for each metric set."
    )

    parser.add_argument(
        "--metric_sets", type=str,
        default="loss,acc,f1_macro",
        help="Comma-separated list of metric sets to plot (e.g., loss,acc,f1_macro). "
             "Script will look for appropriate train/val versions based on CSV headers."
    )

    args = parser.parse_args()

    metric_pairs_to_plot = []
    requested_sets = [s.strip().lower() for s in args.metric_sets.split(',')]

    metric_name_mappings = {
        "loss": ("train_loss_epoch", "val_loss"),  # train_loss_epoch from step log, val_loss from epoch log_dict
        "acc": ("train_acc", "val_acc"),  # Both from epoch log_dict
        # macros averaged metrics
        "f1_macro": ("train_f1_macro", "val_f1_macro"),
        "precision_macro": ("train_precision_macro", "val_precision_macro"),
        "recall_macro": ("train_recall_macro", "val_recall_macro"),
        "auroc_macro": ("train_auroc_macro", "val_auroc_macro"),
        "aupr_macro": ("train_aupr_macro", "val_aupr_macro"),
        # weighted averaged metrics
        "f1_weighted": ("train_f1_weighted", "val_f1_weighted"),
        "precision_weighted": ("train_precision_weighted", "val_precision_weighted"),
        "recall_weighted": ("train_recall_weighted", "val_recall_weighted"),
        "auroc_weighted": ("train_auroc_weighted", "val_auroc_weighted"),
        "aupr_weighted": ("train_aupr_weighted", "val_aupr_weighted"),
    }

    for metric_set_name in requested_sets:
        if metric_set_name in metric_name_mappings:
            metric_pairs_to_plot.append(metric_name_mappings[metric_set_name])
        else:
            print(f"Warning: Metric set '{metric_set_name}' is not pre-configured in metric_name_mappings. "
                  f"Skipping this set. Please add it if desired.")
            # Optionally, you could try a generic fallback, but explicit mapping is safer:
            # print(f"Attempting generic train_{metric_set_name} and val_{metric_set_name}.")
            # metric_pairs_to_plot.append((f"train_{metric_set_name}", f"val_{metric_set_name}"))

    if not metric_pairs_to_plot:
        print("No valid metric sets specified to plot or found in mappings. Exiting.")
        print(f"Requested sets: {requested_sets}")
        print(f"Available mappings: {list(metric_name_mappings.keys())}")
        return

    all_metrics_data = []
    specific_versions_list = [v.strip() for v in args.versions.split(',')] if args.versions else None

    log_dirs_to_process = args.log_dirs
    # Ensure log_dirs_to_process is a list
    if not isinstance(log_dirs_to_process, list):
        log_dirs_to_process = [log_dirs_to_process]

    for i, log_dir_base in enumerate(log_dirs_to_process):
        current_versions_to_check = specific_versions_list if i == 0 and specific_versions_list else None
        found_files = find_metric_files(log_dir_base, specific_versions=current_versions_to_check)
        for file_info in found_files:
            try:
                df = pd.read_csv(file_info['path'])
                all_metrics_data.append({'name': file_info['name'], 'df': df})
            except Exception as e:
                print(f"Error reading or processing {file_info['path']}: {e}")

    if not all_metrics_data:
        print("No metric files found or processed. Exiting.")
        return

    plot_metrics(all_metrics_data, metric_pairs_to_plot, args.output_dir, args.model_name, combined_plot=args.combined)
    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()