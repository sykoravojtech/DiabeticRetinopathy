import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 8
CHECKPOINT_FILE = "b3_27_0.78.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

RESIZE_SIZE = 800
CROP_SIZE = 728

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=RESIZE_SIZE, height=RESIZE_SIZE),
        A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=CROP_SIZE, width=CROP_SIZE),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)