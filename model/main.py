# python main.py fit --config config/final_train_config.yaml

import os
import torch
import torch.backends.cuda
import torch.backends.cudnn
from jsonargparse import lazy_instance # Make sure this is installed if not already
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import CSVLogger # Or TensorBoardLogger, etc.

# load modules
from src.data import DataModule
from src.model import ClassificationModel


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        # Define default ModelCheckpoint behavior
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        parser.set_defaults(
            {
                "trainer.logger": lazy_instance(
                    CSVLogger, save_dir="output", name="run_logs"  # todo: logger.name
                ),
                "model_checkpoint.monitor": "val_acc",  # Set also in YAML config
                "model_checkpoint.mode": "max",
                "model_checkpoint.filename": "best-step-{step}-{val_acc:.4f}",
                "model_checkpoint.save_last": True,
                # Add a default for dirpath if not already present
                # "model_checkpoint.dirpath": "output/default_checkpoints" # CLI will create subfolder if save_dir used in logger
            }
        )

        # Link arguments as before
        parser.link_arguments("data.size", "model.image_size")
        # parser.link_arguments(
        #     "data.num_classes", "model.n_classes", apply_on="instantiate"
        # )
        parser.link_arguments("data.num_classes", "model.n_classes")

def cli_main():  # This function is the entry point for the CLI
    # These are good settings for performance with compatible GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cli = MyLightningCLI(
        ClassificationModel,
        DataModule,
        save_config_kwargs={"overwrite": True},
        # Can set trainer defaults here or via CLI/config file
        trainer_defaults={
            "check_val_every_n_epoch": 1,  # validate every epoch
            # "max_epochs": 10
        },
        # parser_kwargs={"parser_mode": "omegaconf"} # If using OmegaConf for config files
    )


if __name__ == "__main__":
    cli_main()


