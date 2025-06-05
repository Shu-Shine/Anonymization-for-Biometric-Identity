# run_kfold.py
import subprocess
import os
import pandas as pd
import sys
from model.utils.utils import increment_path  # todo
import yaml

# --- Configuration for this K-Fold CV Experiment ---
BASE_CONFIG_FILE = "config/base_cv_config.yaml"

# Top-level directory for all outputs of this CV experiment
Results_dir = "kfold_cv_outputs"
CV_EXPERIMENT_NAME = "cv_experiment_run"
CV_EXPERIMENT_OUTPUT_DIR = increment_path(Results_dir, name=CV_EXPERIMENT_NAME)

os.makedirs(CV_EXPERIMENT_OUTPUT_DIR, exist_ok=True)
print(f"K-Fold CV Experiment outputting to: {CV_EXPERIMENT_OUTPUT_DIR}")


with open(BASE_CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

# --- Load configuration parameters ---
NUM_FOLDS = config["data"]["k_fold_num_splits"]
logger_name = config["trainer"]["logger"]["name"] if "logger" in config["trainer"] else "run_logs"
primary_metric_to_aggregate = config["model_checkpoint"]["monitor"]  # make sure set in the YAML config, and not covered by CLI args


# --- Loop through folds for Training and Validation ---
for i in range(NUM_FOLDS):
    current_fold_idx = i
    # Create a unique output directory for this fold
    fold_run_dir = os.path.join(CV_EXPERIMENT_OUTPUT_DIR, f"fold_{current_fold_idx + 1}")

    print(f"\n===== RUNNING CV FOLD {current_fold_idx + 1}/{NUM_FOLDS} =====")
    print(f"Output directory for this fold: {fold_run_dir}")

    command = [
        # "python",
        sys.executable,  # Use the current Python interpreter
        "main.py", "fit",
        "--config", BASE_CONFIG_FILE,  # Load static parameters from YAML
        # --- Dynamic Overrides for the current fold ---
        f"--data.k_fold_current_fold={current_fold_idx}",
        f"--trainer.logger.save_dir={fold_run_dir}",  # Save logs and outputs for current fold
        f"--trainer.logger.name={logger_name}",  # Reset logger name as in the config, instead of CLI args
        f"--model_checkpoint.dirpath={os.path.join(fold_run_dir, 'checkpoints')}",
    ]

    print("\nExecuting FIT command for CV fold:")
    print(" ".join(command))
    print("\n")

    try:
        subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FIT for CV fold {current_fold_idx + 1}: {e}")
        # break # Stop if a fold fails

print(f"\n===== K-FOLD CV FOR EXPERIMENT '{CV_EXPERIMENT_NAME}' COMPLETE =====")

# --- Aggregation of Validation Metrics (after all folds are done) ---
all_fold_val_metrics = []

print(f"\n--- Aggregating '{primary_metric_to_aggregate}' from CV Folds ---")
for i in range(NUM_FOLDS):
    current_fold_idx = i
    fold_run_dir = os.path.join(CV_EXPERIMENT_OUTPUT_DIR, f"fold_{current_fold_idx + 1}")
    metrics_file_path = os.path.join(fold_run_dir, logger_name, "version_0", "metrics.csv")

    if not os.path.exists(metrics_file_path):  # Try to find latest version
        log_dir_base = os.path.join(fold_run_dir, logger_name)
        if os.path.exists(log_dir_base):
            versions = sorted([d for d in os.listdir(log_dir_base) if
                               d.startswith("version_") and os.path.isdir(os.path.join(log_dir_base, d))],
                              key=lambda x: int(x.split('_')[1]))
            if versions: metrics_file_path = os.path.join(log_dir_base, versions[-1], "metrics.csv")

    if os.path.exists(metrics_file_path):
        try:
            df = pd.read_csv(metrics_file_path)
            metric_rows = df[df[primary_metric_to_aggregate].notna()]
            if not metric_rows.empty:
                # todo
                # Depending on model_checkpoint.save_top_k and how often val is run,
                # you might want the value from the epoch that ModelCheckpoint selected,
                # or simply the max value achieved for that metric.
                best_metric_for_fold = metric_rows[primary_metric_to_aggregate].max()
                all_fold_val_metrics.append(best_metric_for_fold)
                print(f"Fold {current_fold_idx + 1}: Best {primary_metric_to_aggregate} = {best_metric_for_fold:.4f}")
            else:
                print(
                    f"Fold {current_fold_idx + 1}: '{primary_metric_to_aggregate}' not found or all NaN in {metrics_file_path}")
                all_fold_val_metrics.append(float('nan'))
        except Exception as e:
            print(f"Error reading/processing metrics for fold {current_fold_idx + 1} from {metrics_file_path}: {e}")
            all_fold_val_metrics.append(float('nan'))
    else:
        print(f"Metrics file not found for fold {current_fold_idx + 1}: {metrics_file_path}")
        all_fold_val_metrics.append(float('nan'))

if all_fold_val_metrics:
    valid_metrics = [m for m in all_fold_val_metrics if not pd.isna(m)]
    if valid_metrics:
        average_metric = sum(valid_metrics) / len(valid_metrics)
        std_dev_metric = pd.Series(valid_metrics).std()
        print(
            f"\nAverage {primary_metric_to_aggregate} across {len(valid_metrics)} successful folds: {average_metric:.4f}")
        print(f"Std Dev of {primary_metric_to_aggregate}: {std_dev_metric:.4f}")
        summary_path = os.path.join(CV_EXPERIMENT_OUTPUT_DIR, "cv_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"CV Experiment: {CV_EXPERIMENT_NAME}\n")
            f.write(f"Base Config: {BASE_CONFIG_FILE}\n")
            f.write(f"Individual Fold {primary_metric_to_aggregate}: {all_fold_val_metrics}\n")
            f.write(f"Average {primary_metric_to_aggregate}: {average_metric:.4f}\n")
            f.write(f"Std Dev {primary_metric_to_aggregate}: {std_dev_metric:.4f}\n")
        print(f"CV summary saved to: {summary_path}")
    else:
        print("No valid validation metrics found across folds.")
else:
    print("No validation metrics collected.")