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
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
 
import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops

from scipy.stats import skew, kurtosis

AVERAGE_TIMESTEP = 1.0  # or whatever your dt is

def extract_shape_geometry_features(img: np.ndarray):
    """
    Given a 2D uint8 array (binary mask), threshold at >0, find connected components,
    and return a dict of geometric features for the largest (or only) blob:
      - centroid (row, col)
      - area
      - filled_area
      - perimeter
      - bbox (min_row, min_col, max_row, max_col)
      - bounding_box_area
      - extent (area / bounding_box_area)
      - aspect_ratio (width/height)
      - equivalent_diameter
      - major_axis_length
      - minor_axis_length
      - orientation (radians)
      - convex_area
      - convex_hull_coords (Nx2 array of (row, col))
      - convex_perimeter
      - solidity (area / convex_area)
      - eccentricity (from regionprops)
      - euler_number
      - extent (area / bbox_area)
      - circularity (4π·area / perimeter²)
      - feret_diameter_max (maximum caliper distance via convex hull)
      - hu_moments (7,)
      - skeleton_length (# of pixels in skeleton)
      - endpoints (# of skeleton endpoints)
      - dt (average timestep between frames)
    """
    # 1) Binarize & label
    binary = img > 0
    labeled = label(binary)
    props = regionprops(labeled)

    # If no blobs at all, return a dict full of Nones
    if not props:
        return {
            'centroid_row': None,
            'centroid_col': None,
            'area': None,
            'filled_area': None,
            'perimeter': None,
            'bbox': None,
            'bounding_box_area': None,
            'extent': None,
            'aspect_ratio': None,
            'equivalent_diameter': None,
            'major_axis_length': None,
            'minor_axis_length': None,
            'orientation': None,
            'convex_area': None,
            'convex_hull': None,
            'convex_perimeter': None,
            'solidity': None,
            'eccentricity': None,
            'euler_number': None,
            'circularity': None,
            'feret_diameter_max': None,
            'hu_moments': None,
            'skeleton_length': None,
            'endpoints': None,
            'dt': AVERAGE_TIMESTEP
        }

    # Find the largest region by area (in case of multiple small blobs)
    region = max(props, key=lambda r: r.area)

    # If the largest blob is too small, treat as "no meaningful region"
    if region.area < 5:
        return {
            'centroid_row': None,
            'centroid_col': None,
            'area': None,
            'filled_area': None,
            'perimeter': None,
            'bbox': None,
            'bounding_box_area': None,
            'extent': None,
            'aspect_ratio': None,
            'equivalent_diameter': None,
            'major_axis_length': None,
            'minor_axis_length': None,
            'orientation': None,
            'convex_area': None,
            'convex_hull': None,
            'convex_perimeter': None,
            'solidity': None,
            'eccentricity': None,
            'euler_number': None,
            'circularity': None,
            'feret_diameter_max': None,
            'hu_moments': None,
            'skeleton_length': None,
            'endpoints': None,
            'dt': AVERAGE_TIMESTEP
        }

    # Basic shape features from regionprops
    area = region.area
    filled_area = region.filled_area
    perimeter = region.perimeter
    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc
    bbox_area = width * height if (width > 0 and height > 0) else np.nan
    extent = area / bbox_area if bbox_area and not np.isnan(bbox_area) else np.nan
    aspect_ratio = width / float(height) if height > 0 else np.nan
    equiv_diameter = region.equivalent_diameter
    major_axis_length = region.major_axis_length
    minor_axis_length = region.minor_axis_length
    orientation = region.orientation  # in radians
    convex_area = region.convex_area
    eccentricity = region.eccentricity
    solidity = region.solidity  # area / convex_area
    euler_number = region.euler_number

    # Circularity: 4*pi*area / (perimeter^2)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else np.nan

    # Convex hull coords via OpenCV (to compute convex perimeter and Feret diameter)
    mask = region.image.astype(np.uint8)  # region-local mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(cnts[0])
    hull_pts = hull.squeeze()
    # Convert to global coordinates
    hull_global = np.column_stack([
        hull_pts[:, 1] + minr,  # row
        hull_pts[:, 0] + minc   # col
    ])
    # Convex perimeter (length of hull contour)
    convex_perimeter = cv2.arcLength(hull, True)
    # Compute maximum caliper distance (Feret diameter) from hull points
    # Brute‐force: pairwise distances
    if hull_pts.ndim == 2 and hull_pts.shape[0] > 1:
        # hull_pts are local coords [ [col, row], … ]
        pts = hull_pts[:, ::-1]  # convert to (row, col) if needed, but distances same regardless of ordering
        dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        feret_diameter_max = np.nanmax(dists)
    else:
        feret_diameter_max = 0.0

    # Hu moments
    m = cv2.moments(mask)
    hu = cv2.HuMoments(m).flatten()

    # Skeletonize to compute skeleton length and endpoints
    from skimage.morphology import skeletonize, medial_axis
    from scipy import ndimage as ndi

    # skeleton (binary) of the region
    skeleton = skeletonize(mask > 0)
    skeleton_length = np.count_nonzero(skeleton)

    # Count endpoints: pixels in skeleton with only one neighbor
    # Compute neighbor count via convolution
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    neighbor_map = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3)), mode='constant', cval=0)
    # For each skeleton pixel, count adjacent skeleton pixels
    endpoints = 0
    for (r, c), val in np.ndenumerate(skeleton):
        if val:
            # Count neighbors in 8‐connectivity
            nbr_count = np.sum(skeleton[max(r-1, 0):r+2, max(c-1, 0):c+2]) - 1
            if nbr_count == 1:
                endpoints += 1

    return {
        'centroid_row': float(region.centroid[0]),
        'centroid_col': float(region.centroid[1]),
        'area': area,
        'filled_area': filled_area,
        'perimeter': perimeter,
        'bbox': (minr, minc, maxr, maxc),
        'bounding_box_area': bbox_area,
        'extent': extent,
        'aspect_ratio': aspect_ratio,
        'equivalent_diameter': equiv_diameter,
        'major_axis_length': major_axis_length,
        'minor_axis_length': minor_axis_length,
        'orientation': orientation,
        'convex_area': convex_area,
        'convex_hull': hull_global,        # Nx2 array of (row, col)
        'convex_perimeter': convex_perimeter,
        'solidity': solidity,
        'eccentricity': eccentricity,
        'euler_number': euler_number,
        'circularity': circularity,
        'feret_diameter_max': feret_diameter_max,
        'hu_moments': hu,                  # length-7 array
        'skeleton_length': skeleton_length,
        'endpoints': endpoints,
        'dt': AVERAGE_TIMESTEP
    }


def extract_intensity_features(gray_img: np.ndarray, mask: np.ndarray, distances=[1], angles=[0]):
    """
    Given a 2D grayscale image and a binary mask (same dimensions), compute
    intensity-based features over the region where mask>0. Returns:
      - mean_intensity
      - median_intensity
      - std_intensity
      - min_intensity
      - max_intensity
      - skewness
      - kurtosis
      - entropy (Shannon)
      - percentiles (10th, 25th, 75th, 90th)
      - GLCM texture features: contrast, dissimilarity, homogeneity, ASM, energy, correlation
        (averaged over specified distances and angles)
      - dt (not meaningful here, but kept for consistency)
    """
    # Extract pixel values under mask
    pixels = gray_img[mask > 0].ravel().astype(np.float64)
    if pixels.size == 0:
        return {
            'mean_intensity': None,
            'median_intensity': None,
            'std_intensity': None,
            'min_intensity': None,
            'max_intensity': None,
            'skewness': None,
            'kurtosis': None,
            'entropy': None,
            'percentile_10': None,
            'percentile_25': None,
            'percentile_75': None,
            'percentile_90': None,
            'glcm_contrast': None,
            'glcm_dissimilarity': None,
            'glcm_homogeneity': None,
            'glcm_ASM': None,
            'glcm_energy': None,
            'glcm_correlation': None,
            'dt': None
        }

    # Basic statistics
    mean_intensity = float(np.mean(pixels))
    median_intensity = float(np.median(pixels))
    std_intensity = float(np.std(pixels))
    min_intensity = float(np.min(pixels))
    max_intensity = float(np.max(pixels))
    skewness = float(skew(pixels))
    kurt = float(kurtosis(pixels))

    # Shannon entropy over pixel histogram (256 bins)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 255), density=True)
    # avoid log(0) by masking
    hist_nonzero = hist[hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero)))

    # Percentiles
    p10 = float(np.percentile(pixels, 10))
    p25 = float(np.percentile(pixels, 25))
    p75 = float(np.percentile(pixels, 75))
    p90 = float(np.percentile(pixels, 90))

    # GLCM texture features: compute on masked region by cropping to bounding box
    coords = np.column_stack(np.where(mask > 0))
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    roi = gray_img[minr:maxr+1, minc:maxc+1]
    roi_mask = mask[minr:maxr+1, minc:maxc+1]

    # Quantize ROI to 8 gray levels (0–7)
    roi_quant = np.floor(roi / 32).astype(np.uint8)
    roi_quant[roi_mask == 0] = 0  # force background to zero

    glcm = graycomatrix(
        roi_quant,
        distances=distances,
        angles=angles,
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    ASM = float(np.mean(graycoprops(glcm, 'ASM')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))

    return {
        'mean_intensity': mean_intensity,
        'median_intensity': median_intensity,
        'std_intensity': std_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity,
        'skewness': skewness,
        'kurtosis': kurt,
        'entropy': entropy,
        'percentile_10': p10,
        'percentile_25': p25,
        'percentile_75': p75,
        'percentile_90': p90,
        'glcm_contrast': contrast,
        'glcm_dissimilarity': dissimilarity,
        'glcm_homogeneity': homogeneity,
        'glcm_ASM': ASM,
        'glcm_energy': energy,
        'glcm_correlation': correlation,
        'dt': AVERAGE_TIMESTEP
    }



def inference_whole_slide(model, slide_pth: Path, max_frame: int):
    # Get sample ID from the path
    sample_id = slide_pth.name

    image_file_paths = sorted(list(slide_pth.glob("*.jpg")), key=lambda x: int(x.stem))[
        :max_frame
    ]


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
            # masks = torch.softmax(pred_mask,axis=1).cpu().numpy()>0.5
            masks = torch.sigmoid(pred_mask).cpu().numpy() > 0.05

            
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
        upscaled_mask3 = cv2.resize(
            mask[2].astype(np.uint8), (500, 500), interpolation=cv2.INTER_NEAREST
        )

        # pn_size.append(upscaled_mask.sum())

        upscaled_masks.append((upscaled_mask1, upscaled_mask2, upscaled_mask3))
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
                         default="/home/tsakalis/ntua/phd/cellforge/cellforge/model_weights/multilabel_dpt-vit_base_patch16_224.augreg_in21k_3_classes_SEPARATE_MASK_FINAL.pt",
                        help="Path to model weights file")

    parser.add_argument("--data_path", type=str,
                        default="/home/tsakalis/ntua/phd/maastricht/pronuclei_extraction/data",
                        help="Primary path to look for samples")

    parser.add_argument("--output_dir", type=str, default="data/extracted_signals",
                        help="Directory to save output videos")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Maximum number of frames to process per sample")
    
    args = parser.parse_args()
    

    



    # Create the directory for saving masks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    
    data_path = Path(args.data_path)

    slide_info_df = pd.read_csv(data_path/"embryo_video_abnormality_202509.csv")


    model_pronuclei.load_state_dict(
        torch.load(
            args.model_weights,
            weights_only=True,
            map_location=torch.device(device)
        )
    )
    model_pronuclei.eval()

    model_pronuclei.to(device)

    all_pn_areas = []
    
    pn1_features_all =[]
    pn2_features_all = []

    # pn1_features_intens =[]
    # pn2_features_intens = []
    whole_emb_all = []
    
    for _, row in slide_info_df.iterrows():
        
        try: 
            sample_pth= data_path/f"videoframe/{row['embryoID']}"

            slide_images, slide_masks, sample_id, image_filenames = inference_whole_slide(
                model_pronuclei, sample_pth, args.max_frames
            )

            pn1_features = pd.DataFrame([extract_shape_geometry_features(msk[0]) for msk in slide_masks])
            pn2_features = pd.DataFrame([extract_shape_geometry_features(msk[1]) for msk in slide_masks])
            
            whole_emb = pd.DataFrame([extract_shape_geometry_features(msk[2]) for msk in slide_masks])

            pn1_features['embryo_id'] = row['embryoID']
            pn2_features['embryo_id'] = row['embryoID']
            whole_emb['embryo_id'] = row['embryoID']

            pn1_features['y'] = row['abnormality']
            pn2_features['y'] = row['abnormality']
            whole_emb['y'] = row['abnormality']
            
            pn1_features_all.append(pn1_features)
            pn2_features_all.append(pn2_features)
            whole_emb_all.append(whole_emb)

        except Exception as e:
            print(row)

    full_pn1_df = pd.concat(pn1_features_all).reset_index(drop=True)
    full_pn2_df = pd.concat(pn2_features_all).reset_index(drop=True)
    full_emb_df = pd.concat(whole_emb_all).reset_index(drop=True)

    full_pn1_df.to_csv(output_dir/'full_pn1_df.csv',index=False)
    full_pn2_df.to_csv(output_dir/'full_pn2_df.csv',index=False)
    full_emb_df.to_csv(output_dir/'full_emb_df.csv',index=False)
    

    