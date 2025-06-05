import os
from typing import Optional, Sequence, List, Tuple

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset # Subset is not directly used anymore
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedKFold
from PIL import Image


# Helper Dataset for K-fold custom data to apply correct transforms
class CustomFoldDataset(Dataset):
    def __init__(self, all_samples: List[Tuple[str, int]], indices_for_this_subset: List[int], transform=None):
        """
        Args:
            all_samples (List[Tuple[str, int]]): The full list of (filepath, class_index) from ImageFolder.samples.
            indices_for_this_subset (List[int]): List of indices from all_samples that belong to this subset (fold).
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.samples_for_subset = [(all_samples[i][0], all_samples[i][1]) for i in indices_for_this_subset]
        self.transform = transform
        # ImageFolder.samples usually gives absolute paths or paths relative to its root.
        # Assuming paths are directly usable.

    def __len__(self):
        return len(self.samples_for_subset)

    def __getitem__(self, idx):
        img_path, target = self.samples_for_subset[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error opening image: {img_path}. Ensure paths in ImageFolder.samples are correct.")
            raise
        if self.transform:
            image = self.transform(image)
        return image, target


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir_train_val: str,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        data_dir_test: Optional[str] = None,
        # K-fold parameters
        k_fold_current_fold: int = 0,
        k_fold_num_splits: int = 1,  # Default to 1, indicating not typical K-fold
        k_fold_random_seed: int = 42,
        validation_split_ratio: Optional[float] = None,  # e.g., 0.1 for 10% validation split
        # Augmentation parameters (tuned for wounds)
        size: int = 224,
        resize_first: bool = False,  # If true, resize then RandomCrop, else RandomResizedCrop
        crop_scale_wound: Tuple[float, float] = (0.6, 1.0),  # Scale for RandomResizedCrop
        color_jitter_brightness: float = 0.2,
        color_jitter_contrast: float = 0.2,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        flip_prob: float = 0.5,
        rand_aug_n: int = 0,
        rand_aug_m: int = 9,
        use_trivial_aug: bool = False,
        erase_prob: float = 0.0,  # Default to 0.0 for wounds, can be > 0 if desired
        # Generic Dataloader parameters
        mean: Sequence[float] = (0.485, 0.456, 0.406),  # ImageNet means
        std: Sequence[float] = (0.229, 0.224, 0.225),  # ImageNet stds
        batch_size: int = 32,
        workers: int = 4,
    ):

        super().__init__()
        self.save_hyperparameters()

        # --- Load full dataset for splitting ---
        self.dataset_full_train_val = ImageFolder(root=self.hparams.data_dir_train_val)
        class_names: List[str] = self.dataset_full_train_val.classes  # Stores class names in order
        self.hparams.class_names = class_names
        self.class_names = class_names  # Save to link to model

        # --- Debug prints to confirm ---
        # print(f"DM_INIT End: self.hparams.num_classes from hparams: {self.hparams.get('num_classes', 'Not in hparams')}")
        # print(f"DM_INIT End: self.hparams.class_names from hparams: {self.hparams.get('class_names', 'Not in hparams')}")
        # print(f"DM_INIT End: self.class_names direct attribute: {getattr(self, 'class_names', 'Direct attribute not found')}")
        # try:
        #     val = getattr(self, "class_names")
        #     print(f"DM_INIT End: getattr(self, 'class_names') test: SUCCEEDED, value: {val}")
        # except AttributeError:
        #     print("DM_INIT End: getattr(self, 'class_names') test: FAILED!")
        # --- End Debug ---

        self.class_to_idx: dict = self.dataset_full_train_val.class_to_idx  # Maps class names to indices
        print(f"Discovered classes: {self.hparams.class_names}")
        # print(f"Class to index mapping: {self.class_to_idx}")

        # Check if test dataset is provided
        if self.hparams.data_dir_test:
            self.dataset_test_custom_root = ImageFolder(root=self.hparams.data_dir_test)
            # Info: if test classes match training/validation classes
            if self.dataset_test_custom_root.classes != self.hparams.class_names:
                print("WARNING: Class names or order in test set differ from train/val set!")
                print(f"Train/Val classes: {self.hparams.class_names}")
                print(f"Test classes: {self.dataset_test_custom_root.classes}")
        else:
            self.dataset_test_custom_root = None

        # Check if num_classes matches the number of classes found in ImageFolder
        if self.hparams.num_classes != len(self.hparams.class_names):
            print(f"WARNING: hparam num_classes ({self.hparams.num_classes}) "
                  f"does not match number of classes found by ImageFolder ({len(self.hparams.class_names)}: {self.hparams.class_names}). "
                  f"Using {len(self.hparams.class_names)} classes from ImageFolder.")
        self.num_classes = len(self.hparams.class_names)  # Override hparam if necessary, or ensure consistency

        print(f"Using custom dataset from {self.hparams.data_dir_train_val} for K-fold CV.")
        print(f"Target image size: {self.hparams.size}x{self.hparams.size}")
        print(f"Fold {self.hparams.k_fold_current_fold + 1}/{self.hparams.k_fold_num_splits}")

        # --- Define transformations (wound-specific) ---
        train_augs = []
        if self.hparams.resize_first:
            train_augs.extend([
                transforms.Resize(self.hparams.size + 32),  # Resize to slightly larger
                transforms.RandomCrop(self.hparams.size),
            ])
        else:
            train_augs.append(
                transforms.RandomResizedCrop(
                    (self.hparams.size, self.hparams.size),
                    scale=self.hparams.crop_scale_wound,
                    ratio=(0.75, 1.33)  # todo: Standard aspect ratio
                )
            )
        train_augs.extend([
            transforms.RandomHorizontalFlip(p=self.hparams.flip_prob),
            # transforms.TrivialAugmentWide()
            # if self.hparams.use_trivial_aug
            # else transforms.RandAugment(self.hparams.rand_aug_n, self.hparams.rand_aug_m),
            transforms.ColorJitter(
                brightness=self.hparams.color_jitter_brightness,
                contrast=self.hparams.color_jitter_contrast,
                saturation=self.hparams.color_jitter_saturation,
                hue=self.hparams.color_jitter_hue,
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std),
        ])
        if self.hparams.erase_prob > 0:
            train_augs.append(transforms.RandomErasing(p=self.hparams.erase_prob))

        self.transforms_train = transforms.Compose(train_augs)

        self.transforms_test = transforms.Compose(
            [
                transforms.Resize((self.hparams.size, self.hparams.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.hparams.mean, std=self.hparams.std),
            ]
        )

    def prepare_data(self):
        # if there's any one-time setup needed (e.g., checking data integrity).
        pass

    def setup(self, stage: Optional[str] = None):
        if self.hparams.k_fold_num_splits > 1:  # K-Fold CV Mode
            # K-fold splitting logic
            all_indices = np.arange(len(self.dataset_full_train_val))
            all_labels = [s[1] for s in self.dataset_full_train_val.samples]
            skf = StratifiedKFold(n_splits=self.hparams.k_fold_num_splits, shuffle=True,
                                  random_state=self.hparams.k_fold_random_seed)
            current_fold_indices = list(skf.split(all_indices, all_labels))[self.hparams.k_fold_current_fold]
            train_indices, val_indices = current_fold_indices[0], current_fold_indices[1]
            if stage == "fit" or stage is None:
                self.train_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                       indices_for_this_subset=train_indices,
                                                       transform=self.transforms_train)
                self.val_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                     indices_for_this_subset=val_indices,
                                                     transform=self.transforms_test)

        elif self.hparams.validation_split_ratio is not None and 0 < self.hparams.validation_split_ratio < 1:  # Standard Train/Val Split Mode
            from sklearn.model_selection import \
                train_test_split  # todo: StratifiedShuffleSplit for imbalanced datasets??

            all_indices = np.arange(len(self.dataset_full_train_val))
            all_labels = [s[1] for s in self.dataset_full_train_val.samples]

            # Stratified split
            train_indices, val_indices = train_test_split(
                all_indices,
                test_size=self.hparams.validation_split_ratio,
                stratify=all_labels,
                random_state=self.hparams.k_fold_random_seed  # Can reuse seed
            )
            if stage == "fit" or stage is None:
                self.train_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                       indices_for_this_subset=train_indices,
                                                       transform=self.transforms_train)
                self.val_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                     indices_for_this_subset=val_indices,
                                                     transform=self.transforms_test)
                print(f"Final train mode: Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}")
        else:  # Fallback: use all data for training, and also for validation (not ideal for monitoring)
            if stage == "fit" or stage is None:
                all_idxs = list(range(len(self.dataset_full_train_val)))
                self.train_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                       indices_for_this_subset=all_idxs,
                                                       transform=self.transforms_train)
                self.val_dataset = CustomFoldDataset(all_samples=self.dataset_full_train_val.samples,
                                                     indices_for_this_subset=all_idxs, transform=self.transforms_test)
                print("Final train mode: Using all data for train and val.")

        # Test set loading (common for all modes if data_dir_test is provided)
        if (stage == "test" or stage is None) and self.hparams.data_dir_test and self.dataset_test_custom_root:
            self.test_dataset = CustomFoldDataset(
                all_samples=self.dataset_test_custom_root.samples,
                indices_for_this_subset=list(range(len(self.dataset_test_custom_root.samples))),
                transform=self.transforms_test
            )
        elif stage == "test" and hasattr(self, 'val_dataset'):  # Fallback for test if no dedicated test set
            self.test_dataset = self.val_dataset  # Only if test specific setup requested

    def train_dataloader(self):  # PL hook, to create train dataloader
        if not hasattr(self, 'train_dataset'):
            raise RuntimeError("train_dataset not initialized. Call setup('fit') first or ensure trainer calls it.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'):
            # This might happen if setup was called with stage='test' and fit was skipped
            # or if validate is called standalone. Attempt to set it up for current fold.
            print("val_dataset not found, attempting to create it for val_dataloader call.")
            self.setup(stage='fit')  # Minimal setup to get val_dataset
            if not hasattr(self, 'val_dataset'):  # Still not there
                raise RuntimeError(
                    "val_dataset could not be initialized. Ensure data paths and fold parameters are correct.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            # This can happen if setup was not called with stage='test' or if no dedicated test set
            # and val_dataset (as fallback) wasn't created.
            print("test_dataset not found, attempting to create it for test_dataloader call.")
            self.setup(stage='test')  # Attempt to set up the test_dataset
            if not hasattr(self, 'test_dataset'):  # Still not there
                raise RuntimeError(
                    "test_dataset could not be initialized. Ensure data paths are correct or a fallback is available.")

        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            pin_memory=True,
        )