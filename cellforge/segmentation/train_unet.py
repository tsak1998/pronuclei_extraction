import argparse
from typing import Literal
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from PIL import Image
from PIL.ImageFile import ImageFile

from tqdm import tqdm

from dataloader import ImageDataset, TransformWrapper

data_pth = Path('/Users/tsakalis/ntua/cellforge/data/Unet_train')

smooth = 1e-15

device = 'mps'
base_model_weight_dir = Path(
    '/Users/tsakalis/ntua/cellforge/cellforge/segmentation/model_weights')


def dice_coef(y_pred, y_true):

    intersection = torch.sum(y_true.flatten() * y_pred.flatten())
    return (2. * intersection + smooth) / (
        torch.sum(y_true).flatten() + torch.sum(y_pred).flatten() + smooth)


def dice_loss(y_pred, y_true):

    return 1.0 - dice_coef(y_true, y_pred)


def validate(model, val_dataloader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img_batch, gt_msk_batch in val_dataloader:

            img_batch = img_batch.to(device)
            gt_msk_batch = gt_msk_batch.to(device)

            pred_mask = model(img_batch)

            loss = dice_loss(torch.sigmoid(pred_mask), gt_msk_batch)

            val_loss += loss.item()

    mean_val_loss = val_loss / len(val_dataloader)
    return mean_val_loss


def train(images: list[ImageFile],
          masks: list[ImageFile],
          task_name: Literal['full_embryo', 'inner_embryo'],
          lr: float,
          n_epochs: int,
          batch_size: int,
          encoder: str = 'resnet34',
          weights_path: Path | None = None):

    train_size = int(len(images) * 0.75)
    train_dataset = ImageDataset(images=images[:train_size],
                                 masks=masks[:train_size],
                                 transform=True)
    val_dataset = ImageDataset(images=images[train_size:],
                               masks=masks[train_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size * 2)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )
    model.to(device)
    if weights_path is not None:
        model.load_state_dict(
            torch.load(base_model_weight_dir / weights_path,
                       weights_only=True))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = 1
    for epoch in range(n_epochs):
        train_loss: float = 0
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for img_batch, gt_msk_batch in progress_bar:

            optimizer.zero_grad()
            pred_mask = model(img_batch.to(device))

            loss = dice_loss(torch.sigmoid(pred_mask), gt_msk_batch.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            progress_bar.set_description(str(loss.item()))

        val_loss = validate(model, val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       base_model_weight_dir / f"{task_name}.pt")

        print(
            f'Epoch {epoch+1} | TrainLoss: {train_loss/len(train_dataloader)} ValLoss: {val_loss}'
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="A description of your script")

    parser.add_argument("--segmentation_task",
                        help="To build the index from scratch",
                        default=False,
                        type=str)

    parser.add_argument("--lr",
                        help="To build the index from scratch",
                        default=1e-4,
                        type=float)

    parser.add_argument("--n_epochs",
                        help="To build the index from scratch",
                        default=5,
                        type=int)

    parser.add_argument("--batch_size",
                        help="To build the index from scratch",
                        default=8,
                        type=int)

    parser.add_argument("--pretrained_weights",
                        help="To build the index from scratch",
                        type=str)

    # Parse the arguments
    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    task_name = args.segmentation_task
    pretrained_weights = args.pretrained_weights

    data_pth = Path(
        '/Users/tsakalis/ntua/cellforge/data/segmentation_data') / task_name

    print("... Loading images ...")
    match task_name:
        case 'full_embryo':
            images = []
            masks = []
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

            images = [
                Image.open(img_path) for img_path in tqdm(image_file_paths)
            ]
            masks = [Image.open(msk_pth) for msk_pth in tqdm(mask_file_paths)]
    train(images,
          masks,
          task_name=task_name,
          lr=lr,
          n_epochs=n_epochs,
          batch_size=batch_size,
          weights_path=pretrained_weights)
