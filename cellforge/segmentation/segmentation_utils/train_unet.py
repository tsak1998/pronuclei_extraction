import argparse
from typing import Callable, Literal
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm

from segmentation.segmentation_utils.dataloader import ImageDataset

# Global settings
device = 'cuda' if torch.cuda.is_available() else 'mps'
smooth = 1e-15
base_model_weight_dir = Path(
    '/Users/tsakalis/ntua/cellforge/cellforge/segmentation/model_weights')


def dice_coef(y_pred, y_true):
    intersection = torch.sum(y_true.flatten() * y_pred.flatten())
    return (2. * intersection + smooth) / (
        torch.sum(y_true).flatten() + torch.sum(y_pred).flatten() + smooth)


def dice_loss(y_pred, y_true):
    return 1.0 - dice_coef(y_pred, y_true)


def validate(model, val_dataloader, output_last_fn: Callable):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader, total=len(val_dataloader))
        for img_batch, gt_msk_batch in progress_bar:
            img_batch = img_batch.to(device)
            gt_msk_batch = gt_msk_batch.to(device)
            pred_mask = model(img_batch)
            loss = dice_loss(output_last_fn(pred_mask), gt_msk_batch)
            val_loss += loss.item()
    return val_loss / len(val_dataloader)


def train(train_dataset: ImageDataset,
          val_dataset: ImageDataset,
          task_name: Literal['full_embryo', 'inner_embryo', 'pronuclei',
                             'blastocyst'],
          lr: float,
          n_epochs: int,
          batch_size: int,
          model: nn.Module,
          loss_fn: nn.Module,
          output_last_fn: Callable = lambda x: x,
          precision: Literal['full', 'mixed'] = 'full',
          weights_path: Path | None = None):

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2)

    model.to(device)
    if weights_path is not None:
        model.load_state_dict(
            torch.load(base_model_weight_dir / weights_path,
                       weights_only=True))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if precision == 'mixed':
        scaler = GradScaler()

    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        progress_bar.set_description("Validating...")
        for img_batch, gt_msk_batch in progress_bar:
            img_batch = img_batch.to(device)
            gt_msk_batch = gt_msk_batch.to(device)
            optimizer.zero_grad()

            match precision:
                case 'full':
                    pred_mask = model(img_batch)
                    loss = loss_fn(output_last_fn(pred_mask), gt_msk_batch)
                    loss.backward()
                    optimizer.step()
                case 'mixed':
                    with autocast():
                        pred_mask = model(img_batch)
                        loss = loss_fn(output_last_fn(pred_mask), gt_msk_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            train_loss += loss.item()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")

        val_loss = validate(model, val_dataloader, output_last_fn)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       base_model_weight_dir / f"{task_name}.pt")
        print(f'Epoch {epoch + 1} | TrainLoss:'
              f' {train_loss / len(train_dataloader):.4f} '
              f'| ValLoss: {val_loss:.4f}')


class DiceLossModule(nn.Module):

    def forward(self, y_pred, y_true):
        return dice_loss(y_pred, y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Segmentation training pipeline")
    parser.add_argument(
        "--segmentation_task",
        help="Segmentation task (full_embryo, inner_embryo, pronuclei)",
        type=str,
        required=True)
    parser.add_argument("--lr", help="Learning rate", default=1e-4, type=float)
    parser.add_argument("--n_epochs",
                        help="Number of epochs",
                        default=5,
                        type=int)
    parser.add_argument("--batch_size", help="Batch size", default=8, type=int)
    parser.add_argument("--pretrained_weights",
                        help="Path to pretrained weights",
                        type=str,
                        default=None)
    args = parser.parse_args()

    lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    task_name = args.segmentation_task
    pretrained_weights = args.pretrained_weights

    data_pth = Path(
        '/Users/tsakalis/ntua/cellforge/data/segmentation_data') / task_name

    print("... Loading images ...")
    images = []
    masks = []
    match task_name:
        case 'full_embryo':
            for embry_pth in (data_pth / 'masks').glob('*'):
                for mask_pth in embry_pth.glob('*.png'):
                    msk_img = Image.open(mask_pth)
                    try:
                        raw_img = Image.open(
                            (data_pth / 'images') /
                            f"{embry_pth.name.upper()}/{mask_pth.stem}.jpg")
                        images.append(raw_img)
                        masks.append(msk_img)
                    except Exception as e:
                        print(e)
                        continue
        case 'inner_embryo':
            image_file_paths = sorted(list(
                (data_pth / "images").glob('*.jpg')),
                                      key=lambda x: x.stem)
            mask_file_paths = sorted(list((data_pth / "masks").glob('*.png')),
                                     key=lambda x: x.stem)
            for img_path, msk_path in tqdm(zip(image_file_paths,
                                               mask_file_paths),
                                           total=len(image_file_paths)):
                images.append(Image.open(img_path))
                masks.append(Image.open(msk_path))
        case _:
            raise ValueError("Unsupported segmentation task provided.")

    # Simple 80/20 train-validation split
    split_idx = int(len(images) * 0.8)
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]

    # Create dataset instances
    train_dataset = ImageDataset(train_images, train_masks, transform=True)
    val_dataset = ImageDataset(val_images, val_masks, transform=True)

    # Initialize model and loss function
    model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1)
    loss_fn = DiceLossModule()

    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_name=task_name,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        output_last_fn=torch.sigmoid,
        model=model,
        loss_fn=loss_fn,
        weights_path=Path(pretrained_weights) if pretrained_weights else None)
