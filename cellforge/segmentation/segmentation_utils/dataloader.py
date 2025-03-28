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
import cv2

from pathlib import Path
def apply_clahe(pil_img):
    # Convert PIL image to NumPy array
    img = np.array(pil_img)

    # If grayscale
    if len(img.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
    # If RGB
    elif len(img.shape) == 3:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        img_lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)
    else:
        raise ValueError("Unsupported image format")

    # Convert back to PIL image
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

    def __getitem__(self,
                    index) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image = self.images[index]
        mask = self.masks[index]

        image = image.resize((self.image_size, self.image_size),
                             Image.Resampling.LANCZOS)
        mask = mask.resize((self.image_size, self.image_size),
                           Image.Resampling.LANCZOS)

        # mask = mask.convert("L")  # Ensure mask is in grayscale
        binary_threshold = 100  # Adjust this threshold as needed
        mask = mask.point(lambda p: 255 if p > binary_threshold else 0)

        image = apply_clahe(image)
        image = T.GaussianBlur(3)(image)
        
        # mask = mask.convert('1')

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
