import os
import random
from typing import Literal
import PIL
import numpy as np
import torch
import torch

torch.set_float32_matmul_precision("high")
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

    def __init__(
        self,
        images: list[ImageFile],
        masks: list[ImageFile],
        transform: bool = False,
        image_size: int = 224,
        problem_type: Literal["multiclass", "multilabel"] = "multilabel",
    ):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.transform = transform
        self.type = problem_type

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        image = self.images[index]
        mask = self.masks[index]

        # Resize both image and mask
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        mask = mask.resize((self.image_size, self.image_size))

        # Binary threshold for mask
        if self.type == "multilabel":
            binary_threshold = 100
            mask = mask.point(lambda p: 255 if p > binary_threshold else 0)

        # Random CLAHE on image
        # if np.random.rand() > 0.5:
        image = apply_clahe(image)

        # Random Gaussian blur on image with random kernel size
        if np.random.rand() > 0.5 and self.transform:
            kernel = np.random.choice([3, 5, 7])
            image = T.GaussianBlur(kernel)(image)

        normalize_tensor = T.Compose(
            [
                T.Lambda(lambda x: x.convert("RGB")),
                T.ToTensor(),
                # Uncomment if normalization is desired:
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if self.transform:
            # Random horizontal and vertical flips
            angle = random.uniform(-90, 90)
            image = F.rotate(image, angle)
            image = normalize_tensor(image)
            mask = T.ToTensor()(F.rotate(mask, angle))
            return image, mask

        image = normalize_tensor(image)
        mask = T.ToTensor()(mask)
        return image, mask


import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageCircleDatasetSeperate(data.Dataset):
    def __init__(
        self,
        images: list[Image],
        circles: list[dict],
        whole_embryo_masks: list[Image],

        transform: bool = False,
        image_size: int = 224,
        problem_type: Literal["multiclass", "multilabel"] = "multilabel",
        predict: bool = False
    ):
        assert len(images)==len(circles)==len(whole_embryo_masks)
        self.images = images
        self.circles = circles
        self.whole_embryo_masks = whole_embryo_masks
        self.image_size = image_size
        self.use_transform = transform
        self.type = problem_type
        self.predict = predict

        # we’ll stack: [pn1_circle, pn2_circle, whole_embryo_mask, pn_model_mask]
        self.tf = A.Compose([
                A.Rotate(limit=90, p=0.5),
                A.ElasticTransform(p=0.2, alpha=1, sigma=50),
                A.Resize(self.image_size, self.image_size),
                ],
            additional_targets={
                "mask": "mask",
                
            },
            is_check_shapes=False
        )

        # normalization to tensor
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.images)

    def _make_circle_mask(self, circle: dict) -> Image.Image:

    
        # draw pn1/pn2 circles into a 3‑ch PIL.Image and return
        H = W = 500  # original
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        y, x = np.ogrid[:H, :W]

        # pn1
        c1 = (int(circle["pn1"]["x"]), int(circle["pn1"]["y"]))
        r1 = int(circle["pn1"]["r"])
        blob1 = (x - c1[0])**2 + (y - c1[1])**2 <= (r1)**2
        mask[...,1][blob1] = 255

        # pn2 if exists
        if circle.get("pn2"):
            c2 = (int(circle["pn2"]["x"]), int(circle["pn2"]["y"]))
            r2 = int(circle["pn2"]["r"])
            blob2 = (x - c2[0])**2 + (y - c2[1])**2 <= (r2)**2
            mask[...,2][blob2] = 255

        return Image.fromarray(mask)

    def __getitem__(self, idx):
        # 1) load data
        img = self.images[idx].resize((self.image_size, self.image_size), Image.LANCZOS)
        img = apply_clahe(img)
        # 2) np arrays for albumentations
        normalize_tensor = T.Compose(
            [
                T.Lambda(lambda x: x.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_np   = normalize_tensor(img).numpy()

        img_np = img_np.transpose(1,2,0)

        if self.predict:
            
            out = {}
            out["image"] = self.to_tensor(image=img_np)["image"]

            return out["image"], out["image"]


        whole_mask = self.whole_embryo_masks[idx].resize((self.image_size, self.image_size), Image.NEAREST)


        if self.circles[idx] is None:

            circ_np = np.zeros((self.image_size,self.image_size,3))

            whole_np = (np.array(whole_mask)[:,:,0]!=0).astype(int)

        else:
            

            circ_mask = self._make_circle_mask(self.circles[idx]).resize((self.image_size, self.image_size), Image.NEAREST)
            circ_np  = np.array(circ_mask)

           
        
            whole_np = np.array(whole_mask)


        mask_np = np.zeros((self.image_size, self.image_size))
        # breakpoint()
        mask_np[whole_np!=0] = 2
        

        # intersection_pn = circ_np[:,:,1]&circ_np[:,:,2]

        mask_np[circ_np[:,:,2]!=0] = 1
        mask_np[circ_np[:,:,1]!=0] = 1

        # 3) optional augment
        if self.use_transform:
            aug = self.tf(
                image=img_np,
                mask = mask_np
            )
            img_np   = aug["image"]
            mask_np  = aug["mask"]


        else:
            img_np = img_np#.transpose(1,2,0)
            


        # 5) to tensor & normalize
        out = {}
        out["image"] = self.to_tensor(image=img_np)["image"]
        # binary masks: 0 or 255 → 0.0 or 1.0
        out["mask"] = torch.from_numpy(mask_np.astype(np.int32))

        return out["image"].float(), out["mask"].long()#[:-1,...]

import numpy as np
import torch
from torch.utils import data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Literal

class ImageCircleDatasetV2(data.Dataset):
    def __init__(
        self,
        images: list[Image],
        circles: list[dict],
        whole_embryo_masks: list[Image],
        transform: bool = False,
        image_size: int = 224,
        problem_type: Literal["multiclass", "multilabel"] = "multilabel",
        predict: bool = False
    ):
        assert len(images)==len(circles)==len(whole_embryo_masks)
        self.images = images
        self.circles = circles
        self.whole_embryo_masks = whole_embryo_masks
        self.image_size = image_size
        self.use_transform = transform
        self.type = problem_type
        self.predict = predict

        # we’ll stack: [pn1_circle, pn2_circle, whole_embryo_mask, pn_model_mask]
        self.tf = A.Compose([
                A.Rotate(limit=90, p=0.5),
                A.ElasticTransform(p=0.2, alpha=1, sigma=50, alpha_affine=50),
                A.Resize(self.image_size, self.image_size),
                ],
            additional_targets={
                "pn_mask": "pn_mask",
                "whole_embryo": "whole_embryo",
                
            },
            is_check_shapes=False
        )

        # normalization to tensor
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.images)

    def _make_circle_mask(self, circle: dict) -> Image.Image:

    
        # draw pn1/pn2 circles into a 3‑ch PIL.Image and return
        H = W = 500  # original
        mask = np.zeros((H, W, 3), dtype=np.uint8)
        y, x = np.ogrid[:H, :W]

        # pn1
        c1 = (int(circle["pn1"]["x"]), int(circle["pn1"]["y"]))
        r1 = int(circle["pn1"]["r"])
        blob1 = (x - c1[0])**2 + (y - c1[1])**2 <= r1**2
        mask[...,1][blob1] = 255

        # pn2 if exists
        if circle.get("pn2"):
            c2 = (int(circle["pn2"]["x"]), int(circle["pn2"]["y"]))
            r2 = int(circle["pn2"]["r"])
            blob2 = (x - c2[0])**2 + (y - c2[1])**2 <= r2**2
            mask[...,2][blob2] = 255

        return Image.fromarray(mask)

    def __getitem__(self, idx):
        # 1) load data
        img = self.images[idx].resize((self.image_size, self.image_size), Image.LANCZOS)
        img = apply_clahe(img)
        # 2) np arrays for albumentations
        normalize_tensor = T.Compose(
            [
                T.Lambda(lambda x: x.convert("RGB")),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_np   = normalize_tensor(img).numpy()

        img_np = img_np.transpose(1,2,0)

        if self.predict:
            
            out = {}
            out["image"] = self.to_tensor(image=img_np)["image"]

            return out["image"], out["image"]

        circ_mask = self._make_circle_mask(self.circles[idx]).resize((self.image_size, self.image_size), Image.NEAREST)
        whole_mask = self.whole_embryo_masks[idx].resize((self.image_size, self.image_size), Image.NEAREST)


        circ_np  = np.array(circ_mask)
        whole_np = np.array(whole_mask)

        final_masks = np.zeros((self.image_size, self.image_size))




        # breakpoint()
        # 3) optional augment
        if self.use_transform:
            aug = self.tf(
                image=img_np,
                pn_mask=circ_np,
                whole_embryo=whole_np,

            )
            img_np   = aug["image"]
            circ_np  = aug["pn_mask"]
            whole_np = aug["whole_embryo"]


        else:
            img_np = img_np#.transpose(1,2,0)
            

        # breakpoint()
        # 4) stack into a 4‑channel mask
        #   circ_np.shape = (H, W, 3) → circle channels at indices 1 and 2
        #   whole_np, pn_np are (H, W) or (H, W, 1)
        # ensure single‑channel
        if whole_np.ndim==3: whole_np = whole_np[...,0]
        

        mask_stack = np.stack([
            circ_np[...,1],        # pn1 circle
            circ_np[...,2],        # pn2 circle
            whole_np,              # whole‑embryo prediction
            
        ], axis=0)  # shape (4, H, W)

        # 5) to tensor & normalize
        out = {}
        out["image"] = self.to_tensor(image=img_np)["image"]
        # binary masks: 0 or 255 → 0.0 or 1.0
        out["mask"] = torch.from_numpy(mask_stack.astype(np.float32) / 255.0)

        return out["image"].float(), out["mask"]#[:-1,...]
