# data/dataloaders.py

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from project_root.config import PIN_MEMORY


def make_dataloaders(data_root, img_size=224, batch_size=32, num_workers=4):
    """
    Creates train/val/test DataLoaders for the specified input size.

    Parameters
    ----------
    data_root : str or Path
     Root folder with train/val/test subdirectories.
    img_size : int
      Input size (224 for ResNeXt/DenseNet, 299 for Inception).
    batch_size : int
      Batch size.
    num_workers : int
      Number of threads for loading data.

    Returns
    -------
    dls : dict
       {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    classes : list[str]
      List of classes.
    """

    data_root = Path(data_root)

    train_ds = ImageFolder(data_root / "train", transform=get_train_transforms(img_size))
    val_ds   = ImageFolder(data_root / "val",   transform=get_val_transforms(img_size))
    test_ds  = ImageFolder(data_root / "test",  transform=get_test_transforms(img_size))

    classes = train_ds.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=PIN_MEMORY)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=PIN_MEMORY)

    dls = {"train": train_dl, "val": val_dl, "test": test_dl}
    return dls, classes
