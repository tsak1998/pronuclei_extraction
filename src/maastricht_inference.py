import os

from concurrent.futures import ProcessPoolExecutor


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
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

from scipy.stats import skew, kurtosis


AVERAGE_TIMESTEP = 1.0  # or whatever your dt is



import numpy as np
from skimage.measure import label, regionprops_table

def extract_shape_geometry_features(img: np.ndarray, min_area: int = 5):
    """
    Return centroid_row, centroid_col, and area of the largest blob in a binary mask.
    Uses regionprops_table to compute only 'area' and 'centroid'.
    """
    binary = img > 0
    if not np.any(binary):
        return {"centroid_row": None, "centroid_col": None, "area": None}

    labeled = label(binary)
    tbl = regionprops_table(labeled, properties=('area', 'centroid'))
    areas = np.asarray(tbl['area'])
    if areas.size == 0:
        return {"centroid_row": None, "centroid_col": None, "area": None}

    i = int(np.argmax(areas))
    area = int(areas[i])
    if area < min_area:
        return {"centroid_row": None, "centroid_col": None, "area": None}

    return {
        "centroid_row": float(tbl['centroid-0'][i]),
        "centroid_col": float(tbl['centroid-1'][i]),
        "area": area,
    }


def inference_whole_slide(model, slide_pth: Path, max_frame: int):
    # Get sample ID from the path
    sample_id = slide_pth.name

    image_file_paths = sorted(
        list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem.split("frame")[1])
    )[:max_frame]

    # images = [Image.open(img_path) for img_path in image_file_paths]
    images = [Image.open(img_path).rotate(-90) for img_path in image_file_paths]
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
            # masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.5
            masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            all_masks.extend([msk for msk in masks])
            # breakpoint()

    pn_size = []
    final_images = []
    upscaled_masks = []
    isolated_pns = []
    img_dim = 500
    for pil_img, mask in zip(images[:], all_masks[:]):

        # Ensure the mask is 2D by removing extra dimensions
        pil_img = pil_img.resize((img_dim, img_dim), Image.Resampling.LANCZOS)
        # image_ar = np.stack(3 * [np.array(pil_img)])
        image_ar = np.stack(3 * [np.array(pil_img)[:,:,0]])

        upscaled_mask1 = cv2.resize(
            mask[1].astype(np.uint8), (img_dim, img_dim), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask2 = cv2.resize(
            mask[0].astype(np.uint8), (img_dim, img_dim), interpolation=cv2.INTER_NEAREST
        )
        upscaled_mask3 = cv2.resize(
            mask[2].astype(np.uint8), (img_dim, img_dim), interpolation=cv2.INTER_NEAREST
        )

        # pn_size.append(upscaled_mask.sum())
        # breakpoint()
        upscaled_masks.append((upscaled_mask1, upscaled_mask2, upscaled_mask3))
        image_pn_isolated = image_ar.copy()
        image_pn_isolated[:, ~upscaled_mask1.astype(bool)] = 0
        isolated_pns.append(image_pn_isolated.transpose(1, 2, 0))
        image_ar[0, upscaled_mask1.astype(bool)] = 1
        image_ar[1, upscaled_mask2.astype(bool)] = 1
        image_ar[2, upscaled_mask3.astype(bool)] = 1

        final_images.append(Image.fromarray(image_ar.transpose(1, 2, 0)))

    return (
        final_images,
        upscaled_masks,
        sample_id,
        image_filenames,
    )


def extract_all(msk_triplet):
    pn1 = extract_shape_geometry_features(msk_triplet[0])
    pn2 = extract_shape_geometry_features(msk_triplet[1])
    whole = extract_shape_geometry_features(msk_triplet[2])
    return pn1, pn2, whole  # dicts/Series preferred


def to_df(records):
    return pd.DataFrame.from_records(records)
from contextlib import contextmanager
import cv2
import numpy as np
# import matplotlib.pyplot as plt

@contextmanager
def video_writer_context(output_path, frame_height, frame_width, fps=15, fourcc="mp4v"):
    # Output is [frame | live-plot], so width doubles
    output_size = (frame_width * 2, frame_height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc),  # 'mp4v' for .mp4; try 'avc1' if available
        fps,
        output_size,
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try a different path/fourcc.")
    try:
        yield writer
    finally:
        writer.release()  # no destroyAllWindows() in headless

def generate_video(slide_images, slide_masks, output_path, frame_height=500, frame_width=500, fps=15):
    pn_size1, pn_size2 = [], []

    with video_writer_context(output_path, frame_height, frame_width, fps=fps) as writer:
        for frame_idx, frame in enumerate(slide_images):
            # --- accumulate PN areas (expect (pn1, pn2, whole)) ---
            m = slide_masks[frame_idx]
            if len(m) >= 2:
                pn_size1.append(int(m[0].sum()))
                pn_size2.append(int(m[1].sum()))
            else:
                pn_size1.append(0)
                pn_size2.append(0)

            # --- make plot as RGB array ---
            x = np.arange(1, len(pn_size1) + 1)
            fig, ax = plt.subplots()
            ax.plot(x, pn_size1)
            ax.plot(x, pn_size2)
            ax.legend(["PN 1", "PN 2"])
            ax.set_title(f"Accumulated PN Size (Frame {frame_idx})")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Accumulated Area")
            fig.tight_layout()
            fig.canvas.draw()
            plot_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)

            # --- prep left frame (BGR, resized HxW) ---
            fr = np.array(frame)  # PIL->ndarray (RGB)
            if fr.ndim == 2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            else:
                fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            fr = cv2.resize(fr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

            # --- prep right plot (BGR, same HxW) ---
            plot_bgr = cv2.cvtColor(plot_rgb, cv2.COLOR_RGB2BGR)
            plot_bgr = cv2.resize(plot_bgr, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

            # --- concatenate and write ---
            combined = np.hstack((fr, plot_bgr))
            writer.write(combined)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Pronuclei inference on embryo images")
    parser.add_argument(
        "--model_weights",
        type=str,
        default="/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights/multilabel_dpt-vit_base_patch16_224.augreg_in21k_3_classes_SEPARATE_MASK_FINAL.pt",
        help="Path to model weights file",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/tsakalis/ntua/phd/maastricht/pronuclei_extraction/data",
        help="Primary path to look for samples",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/tsakalis/ntua/phd/maastricht/pronuclei_extraction/data/extracted_data",
        help="Directory to save output videos",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=230,
        help="Maximum number of frames to process per sample",
    )

    args = parser.parse_args()

    # Create the directory for saving masks
    output_dir = Path(args.output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    model_pronuclei = smp.DPT(
        encoder_name="tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )

    SINGLE_MASK_MODEL_WEIGHTS = 'multiclass_dpt-vit_base_patch16_224.augreg_in21k_3_classes_WHOLE_SINGLE_MASK_FINAL.pt'
    model_pronuclei_single = smp.DPT(
        encoder_name="tu-vit_base_patch16_224.augreg_in21k",
        encoder_weights="imagenet",
        in_channels=3,
        classes=3,
    )


    type_of_problem = "multilabel"

    data_path = Path(args.data_path)

    # slide_info_df = pd.read_csv(data_path / "full_pth_data.csv")
    slide_info_df = pd.read_csv(data_path / "embryo_video_abnormality_202509.csv")

    ##### seperatate pronuclei model
    # model_pronuclei.load_state_dict(
    #     torch.load(
    #         args.model_weights, weights_only=True, map_location=torch.device(device)
    #     )
    # )
    # model_pronuclei.eval()

    # model_pronuclei.to(device)

    ##### Unified mask pronuclei model
    model_pronuclei.load_state_dict(
        torch.load(
            args.model_weights, weights_only=True, map_location=torch.device(device)
        )
    )
    model_pronuclei.eval()

    model_pronuclei.to(device)

    all_pn_areas = []

    pn1_features_all = []
    pn2_features_all = []

    # pn1_features_intens =[]
    # pn2_features_intens = []
    whole_emb_all = []

    # model_pronuclei.compile

    for _, row in tqdm(slide_info_df.iterrows(), total=len(slide_info_df)):

        try:
            sample_pth = data_path / f"videoframe/{row['embryoID']}"#Path(row['pth'])#

            print(sample_pth)

            slide_images, slide_masks, sample_id, image_filenames = (
                inference_whole_slide(model_pronuclei, sample_pth, args.max_frames)
            )

            # output_path = Path(
            # f"/home/tsakalis/pn_samples_all/seperate_pn_maas.mp4"
            # )
            # # breakpoint()
            # generate_video(slide_images, slide_masks, output_path) 


            n_workers = int(
                os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() // 4 or 4)
            )
            chunksize = max(1, len(slide_masks) // (n_workers * 4))

            # pn1_features = pd.DataFrame([extract_shape_geometry_features(msk[0]) for msk in slide_masks])
            # pn2_features = pd.DataFrame([extract_shape_geometry_features(msk[1]) for msk in slide_masks])

            # whole_emb = pd.DataFrame([extract_shape_geometry_features(msk[2]) for msk in slide_masks])
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                results = list(ex.map(extract_all, slide_masks, chunksize=chunksize))

            pn1_features = to_df([r[0] for r in results])
            pn2_features = to_df([r[1] for r in results])
            whole_emb = to_df([r[2] for r in results])

            pn1_features["embryo_id"] = row["embryoID"]
            pn2_features["embryo_id"] = row["embryoID"]
            whole_emb["embryo_id"] = row["embryoID"]

            # pn1_features["y"] = row["abnormality"]
            # pn2_features["y"] = row["abnormality"]
            # whole_emb["y"] = row["abnormality"]

            pn1_features_all.append(pn1_features)
            pn2_features_all.append(pn2_features)
            whole_emb_all.append(whole_emb)
            # break
        except Exception as e:
            print(e)
            print(row)
            break
        

    full_pn1_df = pd.concat(pn1_features_all).reset_index(drop=True)
    # full_pn2_df = pd.concat(pn2_features_all).reset_index(drop=True)
    full_emb_df = pd.concat(whole_emb_all).reset_index(drop=True)

    full_pn1_df.to_csv(output_dir / "full_pns_df_single.csv", index=False)
    # full_pn2_df.to_csv(output_dir / "full_pn2_df.csv", index=False)
    full_emb_df.to_csv(output_dir / "full_emb_df_single.csv", index=False)
