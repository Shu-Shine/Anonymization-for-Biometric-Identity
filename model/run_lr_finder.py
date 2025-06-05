# run_lr_finder.py
import os
import yaml
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib, good for scripts
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from jsonargparse import ArgumentParser
import pathlib
import torch

# Explicitly import Tuner - this is the fallback approach
from pytorch_lightning.tuner.tuning import Tuner

from src.data import DataModule
from src.model import ClassificationModel


def check_path_exists(path_str):
    """Custom type for argparse to check if a path exists and is a file."""
    path = pathlib.Path(path_str)
    if not path.exists():
        raise ArgumentParser.ArgumentTypeError(f"Path does not exist: {path_str}")
    if not path.is_file():
        raise ArgumentParser.ArgumentTypeError(f"Path is not a file: {path_str}")
    return path_str


def main():
    parser = ArgumentParser(description="PyTorch Lightning LR Finder Script (Fallback Tuner Usage)")
    parser.add_argument(
        "-c", "--config",
        type=check_path_exists,
        required=True,
        help="Path to the YAML configuration file for LR finding."
    )
    # lr_finder can't be defined in the config file, so we define it here
    parser.add_argument("--lr_finder.min_lr", type=float, default=1e-7, help="Minimum learning rate for scan.")
    parser.add_argument("--lr_finder.max_lr", type=float, default=0.1, help="Maximum learning rate for scan.")
    parser.add_argument("--lr_finder.num_training", type=int, default=500, help="Number of training steps for LR find.")
    parser.add_argument("--lr_finder.mode", type=str, default="exponential", choices=["linear", "exponential"],
                        help="LR finder mode.")
    parser.add_argument("--lr_finder.early_stop_threshold", type=float, default=4.0,
                        help="Early stop threshold for loss divergence.")
    # parser.add_argument("--lr_finder.plot_save_path", type=str, default="lr_finder_plots/lr_plot.png",
    #                     help="Path to save the LR finder plot.")  # Defined later in the script

    args = parser.parse_args()

    # Load base configuration from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Instantiate DataModule ---
    data_config = config.get('data', {})
    if not data_config:
        raise ValueError("Data configuration ('data:') is missing in the config file.")
    datamodule = DataModule(**data_config)
    # It's good practice to call setup if your dataloaders depend on it
    # datamodule.prepare_data() # Call if you have downloads or one-time setup
    datamodule.setup(stage='fit')  # Ensure train_dataloader is available

    # --- Instantiate Model ---
    model_config = config.get('model', {})
    if not model_config:
        raise ValueError("Model configuration ('model:') is missing in the config file.")
    # Link num_classes and image_size from data_config if not directly in model_config
    if 'num_classes' in data_config and 'n_classes' not in model_config:
        model_config['n_classes'] = data_config['num_classes']
    if 'size' in data_config and 'image_size' not in model_config:
        model_config['image_size'] = data_config['size']
    model = ClassificationModel(**model_config)

    # --- Instantiate Trainer (minimal for LR finding, Tuner will use this) ---
    trainer_config = config.get('trainer', {})
    # Ensure trainer_config is a dict, even if empty in YAML
    if trainer_config is None: trainer_config = {}

    trainer_instance_for_tuner = pl.Trainer(
        accelerator=trainer_config.get('accelerator', 'auto'),
        devices=trainer_config.get('devices', 'auto'),
        precision=trainer_config.get('precision', '32-true'),  # Default to 32-true if not specified
        max_epochs=trainer_config.get('max_epochs', 3),  # LR find usually needs few epochs/steps
        max_steps=trainer_config.get('max_steps', -1),  # Can also control by steps
        logger=False,  # No extensive logging needed for LR find
        callbacks=None,  # No callbacks needed for LR find
        enable_checkpointing=False  # No checkpointing needed for LR find
    )

    # define a flexible path for saving the LR finder plot
    PATH =f"lr_finder_plots/{model_config['model_name']}_lr_finder_plot.png"

    print("Starting Learning Rate Finder...")
    print(f"  Config file: {args.config}")
    print(f"  Min LR: {args.lr_finder.min_lr}")
    print(f"  Max LR: {args.lr_finder.max_lr}")
    print(f"  Num Training Steps: {args.lr_finder.num_training}")
    print(f"  Mode: {args.lr_finder.mode}")
    # print(f"  Plot save path: {args.lr_finder.plot_save_path}")
    print(f"  Plot save path: {PATH}")

    # --- FALLBACK: Instantiate Tuner separately and pass the Trainer ---
    try:
        explicit_tuner = Tuner(trainer_instance_for_tuner)
    except Exception as e:
        print(f"Error instantiating Tuner(trainer_instance_for_tuner): {e}")
        print("This suggests a critical issue with your PyTorch Lightning installation or the Tuner API.")
        return

    try:
        lr_finder_result = explicit_tuner.lr_find(
            model,
            datamodule=datamodule,  # Pass the datamodule here
            min_lr=args.lr_finder.min_lr,
            max_lr=args.lr_finder.max_lr,
            num_training=args.lr_finder.num_training,
            mode=args.lr_finder.mode,
            early_stop_threshold=args.lr_finder.early_stop_threshold,
            # update_attr_name=None # Default, or specify "lr" or "learning_rate"
        )
    except Exception as e:
        print(f"Error during explicit_tuner.lr_find(): {e}")
        return

    if lr_finder_result is None:
        print("LR Finder (explicit_tuner.lr_find) did not return a result object.")
        print("This can happen if the LR finding process itself failed or was interrupted.")
        print("Check console output above for any errors during the LR scan.")
        return

    # --- Process Results ---
    print(f"LR Finder Results object: {lr_finder_result}")  # See what this object contains

    try:
        suggested_lr = lr_finder_result.suggestion()
        print(f"Suggested LR: {suggested_lr if suggested_lr is not None else 'Not found'}")
    except Exception as e:
        print(f"Error getting suggestion from lr_finder_result: {e}")
        suggested_lr = None

    # Create directory for plot if it doesn't exist
    # plot_dir = os.path.dirname(args.lr_finder.plot_save_path)
    plot_dir = os.path.dirname(PATH)
    if plot_dir and not os.path.exists(plot_dir):  # Check if plot_dir is not empty string
        os.makedirs(plot_dir, exist_ok=True)

    try:
        fig = lr_finder_result.plot(suggest=True if suggested_lr is not None else False)
        if fig:
            # fig.savefig(args.lr_finder.plot_save_path)
            fig.savefig(PATH)
            # print(f"LR Finder plot saved to: {args.lr_finder.plot_save_path}")
            print(f"LR Finder plot saved to: {PATH}")
        else:
            print("LR Finder (lr_finder_result.plot) did not produce a plot object.")
    except Exception as e:
        print(f"Could not save LR finder plot: {e}")
        print("This might happen if the LR finding process did not complete successfully or if matplotlib has issues.")

    if suggested_lr:
        print(f"\n==> IMPORTANT: Consider updating your main training config with lr: {suggested_lr:.2e} <==")
    else:
        print("\nLR Finder could not confidently suggest a learning rate. Check the plot if available.")


if __name__ == "__main__":
    # Ensure CUDA settings are managed if using GPU
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():  # Check for BF16 support specifically for TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for CUDA matmul and cuDNN.")
    elif torch.cuda.is_available():
        print("CUDA available, but BF16 not supported for TF32 enabling on matmul/cuDNN by default.")
    else:
        print("CUDA not available.")

    main()