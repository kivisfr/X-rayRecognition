# data/transforms.py

from torchvision import transforms

from project_root.config import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(img_size=224):
    """
    Augmentations for the training set.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # гарантируем 3 канала
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])


def get_val_transforms(img_size=224):
    """
    Transformations for the validation set.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # гарантируем 3 канала
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])


def get_test_transforms(img_size=224):
    """
    Transformations for the test set.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),   # гарантируем 3 канала
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN,
                             std=IMAGENET_STD)
    ])
