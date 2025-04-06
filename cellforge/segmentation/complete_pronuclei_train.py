import enum
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from random import shuffle
from typing import Callable, Literal
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

from typing import Callable


def load_image_folder(
    folder_data_pth: Path,
    image_type: Literal["jpg", "png"] = "jpg",
    sort_fn: Callable = lambda x: x.stem,
) -> list:
    image_file_paths = sorted(
        list((folder_data_pth).glob(f"*.{image_type}")), key=sort_fn
    )

    progress_bar = tqdm(image_file_paths)
    progress_bar.set_description("Loading Images...")

    return [Image.open(img_path) for img_path in progress_bar]


def inference(model, X, precision: InferencePrecision = InferencePrecision.FULL, *args):

    match precision:
        case InferencePrecision.MIXED:
            with torch.no_grad():
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

    dataloader = DataLoader(counter_example_datasbet, BATCH_SIZE)

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

            three_class_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(bool)

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


def create_all_masks_separate(whole_embryo_segmentation_model: nn.Module):
    """
    This function will take the pronuclei masks and create the other 2 classes.

    We will need some extra samples to be used as counter examples
    (when pronuclei are not showing).

    """
    DEVICE = "cuda"
    BATCH_SIZE = 32
    MASK_THRESHOLD = 0.9
    IMAGE_SIZE = 224

    base_cicle_pth = Path("/media/tsakalis/STORAGE/phd/pronuclei_tracking")
    timelapse_pth = Path(
        "/home/tsakalis/ntua/phd/cellforge/cellforge/data/raw_timelapses"
    )

    all_circle_data = list((base_cicle_pth / "fitted_circles_samples").glob("*.json"))
    import json

    images = []
    masks = []
    for circle_file_pth in tqdm(all_circle_data):
        slide_id = str(circle_file_pth).split("/")[-1][:-5]
        with open(circle_file_pth) as f:
            circles = json.load(f)

        for circle in circles:
            full_frame_pth = timelapse_pth / f"{slide_id}/{circle['frame']}_0.jpg"
            frame_img = Image.open(full_frame_pth)
            mask = np.zeros((500, 500, 3), dtype=np.uint8)

            y_grid, x_grid = np.ogrid[:500, :500]

            # Full blob for pn1 on channel 0
            center1 = (int(circle["pn1"]["x"]), int(circle["pn1"]["y"]))
            radius1 = int(circle["pn1"]["r"])
            blob1 = (x_grid - center1[0]) ** 2 + (
                y_grid - center1[1]
            ) ** 2 <= radius1**2
            mask[..., 1][blob1] = 255

            # Full blob for pn2 on channel 1, if available
            if circle["pn2"]:
                center2 = (int(circle["pn2"]["x"]), int(circle["pn2"]["y"]))
                radius2 = int(circle["pn2"]["r"])
                blob2 = (x_grid - center2[0]) ** 2 + (
                    y_grid - center2[1]
                ) ** 2 <= radius2**2
                mask[..., 2][blob2] = 255

            mask_image = Image.fromarray(mask)
            images.append(frame_img)
            masks.append(mask_image)

    dataset = ImageDataset(images=images, masks=masks, transform=False)

    dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=BATCH_SIZE)

    whole_embryo_segmentation_model.eval()
    whole_embryo_segmentation_model.to(DEVICE)

    complete_masks = []
    for batch_im, batch_mask in tqdm(dataloader):

        batch_im = batch_im.to(DEVICE)
        with torch.no_grad():
            pred_masks = (
                torch.sigmoid(
                    inference(
                        whole_embryo_segmentation_model,
                        batch_im,
                        precision=InferencePrecision.FULL,
                    )
                )
                > 0.1
            )

        pred_masks = pred_masks.cpu().numpy()

        for prd_msk, msk in zip(pred_masks, batch_mask.cpu().numpy()):

            # breakpoint()

            msk *= 255

            msk[0, ...] = prd_msk.astype(int) * 255

            complete_masks.append(
                Image.fromarray(msk.astype(np.uint8).transpose(1, 2, 0))
            )
            # breakpoint()/
            # breakpoint()

    final_images = []
    final_masks = []

    for im, msk in zip(images, complete_masks):

        msk_ar = np.array(msk)[:, :, 0]

        if msk_ar.sum() < 190190:
            continue

        final_images.append(im)
        final_masks.append(msk)

    return final_images, final_masks


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
        encoder_name="resnext101_32x48d",  # "resnext101_32x48d",
        encoder_weights="instagram",
        in_channels=3,
        classes=3,
    )

    model.load_state_dict(
        torch.load(model_weights / "inner_embryo.pt", weights_only=True)
    )
    model.eval()

    full_images, full_masks = create_all_masks_separate(model)

    c = list(zip(full_images, full_masks))
    import random

    random.shuffle(c)

    full_images, full_masks = zip(*c)

    # full_dataset = ImageDataset(images=full_images, masks=full_masks)
    # generator = torch.Generator().manual_seed(42)/

    train_dataset = ImageDataset(full_images[:5000], full_masks[:5000], transform=True)

    val_dataset = ImageDataset(full_images[5000:], full_masks[5000:])
    # orch.utils.data.random_split(
    #     full_dataset, [0.8, 0.2], generator=generator
    # )

    type_of_problem = "multilabel"
    # type_of_problem = 'multiclass'
    from segmentation_models_pytorch.losses import DiceLoss
    del model
    torch.cuda.empty_cache()

    loss_fn = DiceLoss(mode=type_of_problem, log_loss=True, from_logits=True)
    train(
        train_dataset,
        val_dataset,
        "pronuclei_seperate_xl",
        lr=1e-5,
        n_epochs=200,
        batch_size=32,
        model=model_pronuclei,
        loss_fn=loss_fn,
        output_last_fn=lambda x: x,
        precision="mixed",
    )
