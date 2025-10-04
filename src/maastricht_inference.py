import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
from PIL import Image
import torch

import numpy as np

from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

from .segmentation_utils.dataloader import (
    ImageCircleDatasetV2,
    ImageCircleDatasetSeperate,
)

import cv2
import numpy as np
import shutil
from pathlib import Path



device = "cuda"



def inference_whole_slide(model, slide_pth: Path, max_frame: int):
    # Get sample ID from the path
    sample_id = slide_pth.name

    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[
        :max_frame
    ]

    breakpoint()

    images = [Image.open(img_path) for img_path in tqdm(image_file_paths)]
    # Store original filenames for later use when saving masks
    image_filenames = [img_path.stem for img_path in image_file_paths]

    val_dataset = ImageCircleDatasetV2(images, images, images, images, predict=True)

    val_dataloader = DataLoader(val_dataset, batch_size=32)

    model.eval()
    from torch.cuda.amp import autocast

    all_masks = []
    for inpt_images, _ in val_dataloader:
        with torch.no_grad():
            # with autocast():

            pred_mask = model(inpt_images.to(device))
            #
            masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.05
            # masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            
            all_masks.extend([msk for msk in masks])
            # breakpoint()

    pn_size = []
    final_images = []
    upscaled_masks = []
    isolated_pns = []
    for pil_img, mask in zip(images[:], all_masks[:]):
        # Ensure the mask is 2D by removing extra dimensions
        # pil_img = pil_img.resize((224, 224), Image.Resampling.LANCZOS)
        image_ar = np.stack(3 * [np.array(pil_img)])

        upscaled_mask1 = cv2.resize(
            mask[1].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask2 = cv2.resize(
            mask[0].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )
        # upscaled_mask3 = cv2.resize(
        #     mask[3].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        # )

        # pn_size.append(upscaled_mask.sum())

        upscaled_masks.append((upscaled_mask1, upscaled_mask2))
        image_pn_isolated = image_ar.copy()
        image_pn_isolated[:, ~upscaled_mask1.astype(bool)] = 0
        isolated_pns.append(image_pn_isolated.transpose(1, 2, 0))
        image_ar[0, upscaled_mask1.astype(bool)] = 1
        image_ar[1, upscaled_mask2.astype(bool)] = 1
        # image_ar[2, upscaled_mask3.astype(bool)] = 1

        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))

    return (
        final_images,
        upscaled_masks,
        sample_id,
        image_filenames,
    )




if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Pronuclei inference on embryo images")
    parser.add_argument("--model_weights", type=str,
                         default="/home/tsakalis/Desktop/final_pn_weights/multiclass_dpt-vit_base_patch16_224.augreg_in21k_3_classes_WHOLE_SINGLE_MASK_FINAL2.pt",
                        help="Path to model weights file")

    parser.add_argument("--data_path", type=str,
                        default="/home/tsakalis/ntua/phd/maastricht/pronuclei_extraction/data/videoframe",
                        help="Primary path to look for samples")

    parser.add_argument("--output_dir", type=str, default="/home/tsakalis/pn_samples_all",
                        help="Directory to save output videos")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Maximum number of frames to process per sample")
    
    args = parser.parse_args()

    # Create the directory for saving masks
    masks_output_dir = Path("data/extracted_masks")
    masks_output_dir.mkdir(parents=True, exist_ok=True)

    model_pronuclei = smp.DPT(
        encoder_name="tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )
    # model_pronuclei = smp.Unet(
    #     encoder_name="mit_b5",
    #     encoder_weights="imagenet",
    #     in_channels=4,
    #     classes=4,
    # )


    type_of_problem = "multilabel"
    model_name = "multiclass_dpt-vit_base_patch16_224.augreg_in21k_3_classes_WHOLE_SINGLE_MASK_FINAL2"#"multiclass_ub5"-mit_
    
    
    data_path = Path(args.data_path)
    model_pronuclei.load_state_dict(
        torch.load(
            args.model_weights,
            weights_only=True,
        )
    )
    model_pronuclei.eval()

    model_pronuclei.to("cuda")

    all_pn_areas = []
    breakpoint()

    for sample_idx, sample_pth in enumerate(data_path.glob('*')):


            for well_pth in sample_pth.glob('*'):
                # Get inference results
                slide_images, slide_masks, sample_id, image_filenames = inference_whole_slide(
                    model_pronuclei, well_pth, args.max_frames
                )
                
                # Create directory for this sample ID
                sample_mask_dir = masks_output_dir / sample_id
                sample_mask_dir.mkdir(exist_ok=True)
                
                # Save masks for this sample
                print(f"Saving masks for sample {sample_id}")
                for i, ((mask1, mask2), filename) in enumerate(zip(slide_masks, image_filenames)):
                    # Save first pronucleus mask
                    mask1_path = sample_mask_dir / f"{filename}_pn1.png"
                    cv2.imwrite(str(mask1_path), mask1 * 255)
                    
                    # Save second pronucleus mask
                    mask2_path = sample_mask_dir / f"{filename}_pn2.png"
                    cv2.imwrite(str(mask2_path), mask2 * 255)
                    
                    # Optional: Save combined mask
                    combined_mask = np.zeros((500, 500, 3), dtype=np.uint8)
                    combined_mask[mask1.astype(bool), 0] = 255  # Red channel for first mask
                    combined_mask[mask2.astype(bool), 1] = 255  # Green channel for second mask
                    combined_path = sample_mask_dir / f"{filename}_combined.png"
                    cv2.imwrite(str(combined_path), combined_mask)
                
                print(f"Saved {len(slide_masks)} masks for sample {sample_id}")


