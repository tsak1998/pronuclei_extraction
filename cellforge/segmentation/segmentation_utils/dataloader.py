import os
import random
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from PIL.ImageFile import ImageFile
import cv2
from pathlib import Path

def apply_clahe(pil_img):
    # Convert PIL image to NumPy array
    img = np.array(pil_img)
    if len(img.shape) == 2:  # Grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
    elif len(img.shape) == 3:  # RGB
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        img_lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError("Unsupported image format")
    return Image.fromarray(img_clahe)

class ImageDataset(data.Dataset):

    def __init__(self,
                 images: list[ImageFile],
                 masks: list[ImageFile],
                 transform: bool = False,
                 image_size: int = 224):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image = self.images[index]
        mask = self.masks[index]

        # Resize both image and mask
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Binary threshold for mask
        binary_threshold = 100
        mask = mask.point(lambda p: 255 if p > binary_threshold else 0)

        # Random CLAHE on image
        if np.random.rand() > 0.5:
            image = apply_clahe(image)

        # Random Gaussian blur on image with random kernel size
        if np.random.rand() > 0.5:
            kernel = np.random.choice([3, 5, 7, 9, 15,21])
            image = T.GaussianBlur(kernel)(image)

        normalize_tensor = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.ToTensor(),
            # Uncomment if normalization is desired:
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225]),
        ])

        if self.transform:
            # Random horizontal and vertical flips
            if np.random.rand() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            if np.random.rand() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)
            # Random rotation, shear, and resize (scale)
            angle = random.uniform(-90, 90)
            shear = random.uniform(-20, 20)
            scale = random.uniform(0.5, 1.0)
            image = F.affine(image, angle=angle, translate=(0, 0), scale=scale, shear=shear, interpolation=Image.BILINEAR)
            mask = F.affine(mask, angle=angle, translate=(0, 0), scale=scale, shear=shear, interpolation=Image.NEAREST)

        image = normalize_tensor(image)
        mask = T.ToTensor()(mask)
        return image, mask
