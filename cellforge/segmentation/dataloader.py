import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from PIL.ImageFile import ImageFile

from pathlib import Path


class TransformWrapper:

    def __init__(self):
        self.image_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=[30, 60, 90, 120, 150]),
            T.ToTensor(),
        ])

        self.mask_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=45),
            T.ToTensor(),
        ])

    def __call__(self, image, mask):
        # Apply the same random seed to ensure consistent transformations
        seed = torch.randint(0, 2**32, (1, )).item()
        torch.manual_seed(seed)
        image = self.image_transforms(image)

        torch.manual_seed(seed)
        mask = self.mask_transforms(mask)

        return image, mask


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

    def __getitem__(self,
                    index) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image = self.images[index]
        mask = self.masks[index]

        image = image.resize((self.image_size, self.image_size),
                             Image.Resampling.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size),
                           Image.Resampling.LANCZOS)

        mask = mask.convert("L")  # Ensure mask is in grayscale
        binary_threshold = 100  # Adjust this threshold as needed
        mask = mask.point(lambda p: 255 if p > binary_threshold else 0)
        mask = mask.convert('1')

        normalize_tensor = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.Lambda(lambda x: x),
        ])
        if self.transform:
            angle = random.uniform(-90, 90)
            image = F.rotate(image, angle)
            image = normalize_tensor(image)
            mask = T.ToTensor()(F.rotate(mask, angle))
            return image, mask

        return normalize_tensor(image), T.ToTensor()(mask)
