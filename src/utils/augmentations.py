# src/utils/augmentations.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_ocr_transforms(height=32, width=320, is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=height, width=width),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.7),
                A.CLAHE(clip_limit=4.0, p=0.3),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=1, p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.3),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
