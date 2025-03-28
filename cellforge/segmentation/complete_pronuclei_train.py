import enum
from pathlib import Path
from random import shuffle
from typing import Literal
import torch
import numpy as np
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import models
from tqdm import tqdm
from .segmentation_utils.train_unet import train
from .segmentation_utils.dataloader import ImageDataset
from PIL import Image

from enum import StrEnum


class InferencePrecision(StrEnum):
    FULL = "full"
    HALF = "half"
    MIXED = "mixed"


data_path = Path("/home/tsakalis/ntua/phd/cellforge/cellforge/data/segmentation_data")


def load_image_folder(
    folder_data_pth: Path, image_type: Literal["jpg", "png"] = "jpg"
) -> list:
    image_file_paths = sorted(
        list((folder_data_pth).glob(f"*.{image_type}")), key=lambda x: x.stem
    )

    progress_bar = tqdm(image_file_paths)
    progress_bar.set_description("Loading Images...")

    return [Image.open(img_path) for img_path in progress_bar]


def inference(model, X, precision: InferencePrecision = InferencePrecision.FULL, *args):

    match precision:
        case InferencePrecision.MIXED:
            with autocast():
                return model(X, *args)

        case InferencePrecision.FULL:
            with torch.no_grad():
                return model(X, *args)

    return model(X, *args)


def create_all_masks(whole_embryo_segmentation_model: nn.Module):
    """
    This function will take the pronuclei masks and create the other 2 classes.

    We will need some extra samples to be used as counter examples
    (when pronuclei are not showing).

    """
    DEVICE = "cuda"
    BATCH_SIZE = 32
    MASK_THRESHOLD = 0.9
    IMAGE_SIZE = 224

    counter_pronuclei_examples_pth = data_path / "negative_pn"

    counter_example_images = load_image_folder(counter_pronuclei_examples_pth)

    dataset = ImageDataset(images=counter_example_images, masks=counter_example_images)

    dataloader = DataLoader(dataset, BATCH_SIZE)

    counter_example_masks = []
    model.eval()
    model.to(DEVICE)

    # CALCULATE THE MASKS FOR THE COUNTER EXAMPLE IMAGES
    for img_batch, _ in dataloader:
        img_batch = img_batch.to(DEVICE)
        model_output = (
            torch.sigmoid(
                inference(
                    whole_embryo_segmentation_model,
                    img_batch,
                    precision=InferencePrecision.MIXED,
                )
            )
            > MASK_THRESHOLD
        )

        model_output = model_output.cpu().numpy()

        for embryo_msk in model_output:
            embryo_binary_np = embryo_msk.astype(np.uint8) * 255

            three_class_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2)).astype(np.uint8)

            three_class_mask[:, :, 0] = embryo_binary_np[0]
            correct_mask = Image.fromarray(three_class_mask)

            counter_example_masks.append(correct_mask)

    counter_example_dataset = ImageDataset(
        images=counter_example_images, masks=counter_example_masks
    )

    dataloader = DataLoader(counter_example_dataset, BATCH_SIZE)

    pronuclei_sample_images = load_image_folder(data_path / "pronuclei/images")

    pronuclei_sample_masks = load_image_folder(
        data_path / "pronuclei/masks", image_type="png"
    )

    pronuclei_simple_dataset = ImageDataset(
        images=pronuclei_sample_images, masks=pronuclei_sample_masks
    )
    dataloader = DataLoader(pronuclei_simple_dataset, BATCH_SIZE)

    pronuclei_sample_masks_3_class = []
    for img_batch, mask_batch in dataloader:
        img_batch = img_batch.to(DEVICE)
        model_output = torch.sigmoid(inference(model, img_batch)) > 0.95

        model_output = model_output.cpu().numpy()

        for mask_idx, embryo_msk in enumerate(model_output):

            pn_mask = mask_batch[mask_idx] > 0.5

            embryo_binary_np = embryo_msk.astype(bool)
            pn_mask = pn_mask.cpu().numpy()

            three_class_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2)).astype(bool)

            three_class_mask[:, :, 0] = embryo_binary_np[
                0
            ]  # (embryo_binary_np[0] & (~pn_mask[0])).astype(bool)
            three_class_mask[:, :, 1] = pn_mask[0]

            # three_class_mask[embryo_binary_np[0]]=1
            # three_class_mask[pn_mask[0] ]=2

            correct_mask = Image.fromarray(three_class_mask.astype(np.uint8) * 255)

            pronuclei_sample_masks_3_class.append(correct_mask)

    full_images = pronuclei_sample_images + counter_example_images
    full_masks = pronuclei_sample_masks_3_class + counter_example_masks

    return full_images, full_masks

    # carefulll with the aligment of masks and images. is not guaranteed here.
    # counter_examples_dataset = ImageDataset(images=counter_example_images, masks=counter_example_masks, transform=True)


model_weights = Path("/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights")

if __name__ == "__main__":

    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet152",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    model_pronuclei = smp.Unet(
        encoder_name="resnext101_32x48d",
        encoder_weights="instagram",
        in_channels=3,
        classes=2,
    )

    model.load_state_dict(
        torch.load(model_weights / "inner_embryo.pt", weights_only=True)
    )
    model.eval()

    full_images, full_masks = create_all_masks(model)

    c = list(zip(full_images, full_masks))
    import random

    random.shuffle(c)

    full_images, full_masks = zip(*c)

    # full_dataset = ImageDataset(images=full_images, masks=full_masks)
    # generator = torch.Generator().manual_seed(42)/

    train_dataset = ImageDataset(full_images[:800], full_masks[:800], transform=True)

    val_dataset = ImageDataset(full_images[800:], full_masks[800:])
    # orch.utils.data.random_split(
    #     full_dataset, [0.8, 0.2], generator=generator
    # )

    type_of_problem = "multilabel"
    # type_of_problem = 'multiclass'
    from segmentation_models_pytorch.losses import DiceLoss

    loss_fn = DiceLoss(mode=type_of_problem, log_loss=True, from_logits=True)
    train(
        train_dataset,
        val_dataset,
        "pronuclei",
        lr=1e-5,
        n_epochs=50,
        batch_size=32,
        model=model_pronuclei,
        loss_fn=loss_fn,
        output_last_fn=lambda x: x,
        precision="mixed",
    )

