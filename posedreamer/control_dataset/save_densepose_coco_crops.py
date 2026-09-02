"""Render DensePose-COCO CSE annotations into control images + image crops
for control-LoRA training (paper 3.1)."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage.io import imread
import cv2
from detectron2.utils.file_io import PathManager
from scipy.interpolate import griddata
from pycocotools import mask as mask_utils
import os
from tqdm import tqdm
from math import ceil 
from posedreamer.utils.paths import WEIGHTS_DIR

def get_smpl_vertex_embedding(device=torch.device("cpu")):
    embed_url = "https://dl.fbaipublicfiles.com/densepose/data/cse/mds_d=256.npy"
    embed_path = PathManager.get_local_path(embed_url)
    embed_map, _ = np.load(embed_path, allow_pickle=True)  
    embed_map = torch.tensor(embed_map).float()[:, 0].to(device)
    embed_map -= embed_map.min()
    embed_map /= embed_map.max()
    return embed_map.cpu().numpy() 

def map_vertices():
    min_dist = np.load(str(WEIGHTS_DIR / "min_dist.npy"))
    return min_dist

import argparse

parser = argparse.ArgumentParser(description="Render DensePose-COCO CSE annotations into control images + crops")
parser.add_argument("--coco_json", default="DensePose_COCO/densepose_train2014_cse.json",
                    help="DensePose-COCO CSE annotation file (fetch with ./get_data.sh)")
parser.add_argument("--output_dir", required=True,
                    help="Output root; writes densepose_changes/ and image_crops/ inside it")
args = parser.parse_args()

coco = COCO(args.coco_json)

save_densepose_path = os.path.join(args.output_dir, "densepose_changes")
os.makedirs(save_densepose_path, exist_ok=True)
save_crops_path = os.path.join(args.output_dir, "image_crops")
os.makedirs(save_crops_path, exist_ok=True)
embedding = get_smpl_vertex_embedding()
print(embedding.shape)
mapping = map_vertices()
print(mapping.shape)

# DensePose-COCO annotates vertices in SMPL coordinates (min_dist.npy maps the
# CSE vertex space to SMPL's 6890 vertices), so colors are looked up in the SMPL
# colormap here. The colors are identical to the SMPL-X colormap (which
# make_colormaps.py derives from this one via deformation transfer), so a
# control model trained on these crops generalises to SMPL-X-rendered poses.
colormap_rgb = np.ascontiguousarray(np.load(str(WEIGHTS_DIR / "new_colormap_smpl.npy")))

def crop_bounding_box(bbox, image, expansion=0.0):
    x, y, w, h = bbox
    ih, iw = image.shape[:2]

    # desired square side (random 0..19% expansion)
    side = int(ceil(max(w, h) * (1 + expansion)))
    side = max(1, side)
    half = side / 2.0

    # original center
    cx, cy = x + w / 2.0, y + h / 2.0

    # helper to compute required clamping/pad for one dimension
    def calc_dim(c, dim):
        """
        Return (center_after_clamp, pad_left, pad_right)
        - If side <= dim: clamp center into [half, dim-half] (no pad)
        - If side > dim: clamp center into [0, dim] (minimize asymmetric pad)
          then compute minimal left/right pad so a window of length `side` around center fits.
        """
        if side <= dim:
            c_clamped = float(np.clip(c, half, dim - half))
            return c_clamped, 0, 0
        # side > dim: center can be anywhere inside original image [0, dim]
        c_clamped = float(np.clip(c, 0.0, float(dim)))
        pad_left = max(0.0, half - c_clamped)
        pad_right = max(0.0, c_clamped + half - dim)
        return c_clamped, int(ceil(pad_left)), int(ceil(pad_right))

    cx2, pad_l, pad_r = calc_dim(cx, iw)
    cy2, pad_t, pad_b = calc_dim(cy, ih)

    # if any padding needed, apply minimal zero-padding (symmetric as computed)
    if any((pad_l, pad_r, pad_t, pad_b)):
        if image.ndim == 3:
            pad_cfg = ((pad_t, pad_b), (pad_l, pad_r), (0, 0))
        else:
            pad_cfg = ((pad_t, pad_b), (pad_l, pad_r))
        img = np.pad(image, pad_cfg, mode='constant', constant_values=0)
        # adjust center to new coordinates
        cx2 += pad_l
        cy2 += pad_t
        H, W = img.shape[:2]
    else:
        img = image
        H, W = ih, iw

    # final integer crop coords (clip to be safe)
    x0 = int(round(cx2 - half))
    y0 = int(round(cy2 - half))
    x0 = max(0, min(x0, W - side))
    y0 = max(0, min(y0, H - side))
    side = int(side)

    return img[y0:y0 + side, x0:x0 + side, :].copy()

img_ids = list(coco.imgs.keys())
for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info["coco_url"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
    for i, ann in enumerate(anns):
        try: 
            img_rgb = imread(img_url)
            if img_rgb.ndim == 2:
                img_rgb = np.stack([img_rgb]*3, axis=-1)
            if "dp_masks" not in ann.keys():
                continue
            mask = mask_utils.decode(mask_utils.merge([i for i in ann["dp_masks"] if isinstance(i, dict)])) 
            bbox = ann["bbox"]
            x, y, w, h = map(int, ann["bbox"])
            img_rgb = img_rgb[y:y+h, x:x+w]

            dp_x = np.array(ann["dp_x"]).astype(int)
            dp_y = np.array(ann["dp_y"]).astype(int)
            dp_vertex = np.array(ann["dp_vertex"])
            dp_vertex = mapping[dp_vertex]
            colors = colormap_rgb[dp_vertex] 

            h, w = mask.shape
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            overlay_interp = np.zeros((h, w, 3), dtype=np.uint8)

            points = np.stack((dp_x, dp_y), axis=-1) 

            for c in range(3):
                channel_vals = colors[:, c]
                interp_linear = griddata(points, channel_vals, (grid_x, grid_y), method='linear', fill_value=np.nan)
                interp_nearest = griddata(points, channel_vals, (grid_x, grid_y), method='nearest')
                interp_combined = np.where(np.isnan(interp_linear), interp_nearest, interp_linear)
                overlay_interp[:, :, c] = np.clip(interp_combined, 0, 255).astype(np.uint8)

            overlay_interp[mask == 0] = 0

            target_h, target_w = img_rgb.shape[:2]
            rendered = cv2.resize(overlay_interp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            coords = cv2.findNonZero(mask)
            bbox = cv2.boundingRect(coords)
            expansion = 0.0
            rendered_crop = crop_bounding_box(bbox, rendered, expansion=expansion)
            original_crop = crop_bounding_box(bbox, img_rgb, expansion=expansion)


            filename = f"densepose_coco_{img_id}_{i}"
            cv2.imwrite(os.path.join(save_densepose_path, f"{filename}.png"), rendered_crop[:, :, ::-1])
            cv2.imwrite(os.path.join(save_crops_path, f"{filename}.jpg"), original_crop[:, :, ::-1])
        except Exception as e:
            print(f"Skipping {img_id} annotation {i} with exception {e}")
