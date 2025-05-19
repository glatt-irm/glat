import torch
from torch.utils.data import DataLoader, Dataset, Subset
import h5py
import numpy as np
from sklearn.model_selection import KFold
from dataset_tcga import TCGADataset
from dataset_sicap import SICAPDataset

class WSI_PatchDataset(Dataset):
    def __init__(self, hdf5_path):
        """
        Initializes the dataset by loading the HDF5 file and extracting patches.
        """
        self.hdf5_path = hdf5_path
        with h5py.File(hdf5_path, 'r') as f:
            self.patches = np.array(f['patches'])  # Load into memory

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        
        # Ensure patches have 3 channels
        if patch.ndim == 2:  # If grayscale, add a channel dimension
            patch = np.expand_dims(patch, axis=0)
        elif patch.ndim == 3 and patch.shape[0] != 3:  # If RGB, transpose properly
            patch = np.transpose(patch, (2, 0, 1))

        patch = torch.tensor(patch, dtype=torch.float32)
        return patch

def get_kfold_dataloaders(dataset, batch_size=16, num_folds=5):
    """
    Performs k-fold cross-validation split and returns DataLoaders for each fold.
    """
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    dataloaders = []

    indices = list(range(len(dataset)))
    for train_idx, val_idx in kfold.split(indices):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        dataloaders.append((train_loader, val_loader))

    return dataloaders

def get_dataloader(batch_size=16, hdf5_path=None, num_folds=5):
    """
    Creates and returns DataLoader instances for five-fold cross-validation.
    If hdf5_path is provided, it loads pre-extracted patches instead of dataset objects.
    """
    if hdf5_path:
        dataset = WSI_PatchDataset(hdf5_path)
        return get_kfold_dataloaders(dataset, batch_size=batch_size, num_folds=num_folds)

    # Load TCGA and SICAP datasets
    tcga_dataset = TCGADataset(train=True)
    sicap_dataset = SICAPDataset(train=True)

    tcga_folds = get_kfold_dataloaders(tcga_dataset, batch_size=batch_size, num_folds=num_folds)
    sicap_folds = get_kfold_dataloaders(sicap_dataset, batch_size=batch_size, num_folds=num_folds)

    return tcga_folds, sicap_folds

