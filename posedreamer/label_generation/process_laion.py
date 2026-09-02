"""Process LAION images into the LAION pose source (paper 3.1): YOLO person
detection, square crops, DensePose predictions, and per-crop metadata.
SMPL-X parameters for the crops are fitted externally (SMPLer-X + TokenHMR).
"""
import json
from typing import Optional, Dict, Any, Tuple
import uuid
import webdataset as wds
import tqdm
import glob
import io
import os
from PIL import Image
from fire import Fire
import numpy as np
import cv2
import detectron2
import kornia.geometry as KG
import torch
from densepose import add_densepose_config
from densepose.vis.extractor import (
    DensePoseResultExtractor,
)
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from ultralytics import YOLO

from posedreamer.label_generation.image_captioning import ImageCaptioner
from posedreamer.label_generation.predict_densepose_crops import crop_bbox, detect_humans

MAX_ASPECT_RATIO = 3
CROP_TARGET_SIZE = 512  # size of square crop around human
CROP_BBOX_SCALE = 1.1  # detected bbox size multiplier
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def generate_unique_filename(metadata: Dict, prefix: Optional[str] = None) -> str:
    unique_filename = metadata["sha256"]
    if prefix is not None:
        unique_filename = prefix + '_' + unique_filename
    return unique_filename


def valid_image(image: np.ndarray, metadata: Dict[str, Any]) -> bool:
    if image.shape[0] / image.shape[1] > MAX_ASPECT_RATIO or image.shape[1] / image.shape[0] > MAX_ASPECT_RATIO:
        return False
    if metadata["NSFW"] == "NSFW" or metadata["NSFW"] == "UNSURE":
        return False
    return True


def predict_visible_keypoints(crops, pose_model):
    results = pose_model(crops)
    results = [r.keypoints.data[0, :, 2] > 0.8 for r in results]
    results = [torch.sum(r).item() / 17 for r in results]
    return results


def resize_keep_aspect(img: np.ndarray, min_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_size:
        scale = min_size / min(h, w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_img
    else:
        return img


def process_image(image: np.ndarray, metadata: Dict[str, Any], captioner, detection_model, pose_model, predictor, extractor, save_dir):
    image = resize_keep_aspect(image, min_size=640)
    # detect humans in the image and filter confident and large predictions
    filename = generate_unique_filename(metadata=metadata, prefix="laion")
    if os.path.exists(os.path.join(save_dir, "densepose", f"{filename}.png")):
        print(f"File {filename} already exists, skipping...")
        return
    bboxes_t, instance_masks_t = detect_humans(image[..., ::-1], detection_model)
    if bboxes_t.shape[0] > 1:
        areas = (bboxes_t[:, 2] - bboxes_t[:, 0]) * (bboxes_t[:, 3] - bboxes_t[:, 1])
        bboxes_t = bboxes_t[areas.argmax()][None]
    bboxes = bboxes_t.cpu().numpy().astype(np.int64)

    # if no valid detections -> write empty file and skip
    if bboxes.shape[0] > 0:

        # prepare crops around detected humans
        image_t = torch.tensor(image, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)[None]

        image_crops, tforms, (H_crop, W_crop), new_bboxes = crop_bbox(
            image_t.expand(bboxes_t.shape[0], -1, -1, -1), bboxes_t,
            target_min_size=768, target_max_size=1024,
            scale=CROP_BBOX_SCALE, mode='bilinear', pad_div=32)

        instance_masks_crops = KG.warp_perspective(
            instance_masks_t[:, None],
            tforms, dsize=(H_crop, W_crop), mode='nearest')[:, 0].bool()
        
        visible_keypoints_ratios = predict_visible_keypoints(image_crops, pose_model) 

        # predict densepose in individual crops
        with torch.inference_mode():
            inputs = [{"image": im, "height": H_crop, "width": W_crop} for im in image_crops]
            detected_instances = []
            bbox = bboxes_t.new_tensor([[0., 0., W_crop, H_crop]])
            for i in range(image_crops.shape[0]):
                det_bbox = detectron2.structures.Instances(image_size=(H_crop, W_crop))
                det_bbox.set('pred_boxes', detectron2.structures.boxes.Boxes(bbox))
                det_bbox.set('pred_classes', bboxes_t.new_zeros(1))
                detected_instances.append(det_bbox)

            predictions = predictor.model.inference(inputs, detected_instances, do_postprocess=True)

        # postprocess predictions and save results
        for i in range(image_crops.shape[0]):
            output_dp = extractor(predictions[i]['instances'])[0][0].labels.byte().cpu().numpy()
            # filter by instance segm mask
            output_dp[(~instance_masks_crops[i]).cpu().numpy()] = 0

            image_crop = image_crops[i].permute(1, 2, 0).byte().cpu().numpy()

            num_unique = np.unique(output_dp).shape[0]
            if num_unique > 10 and visible_keypoints_ratios[i] > 0.5:
                cv2.imwrite(os.path.join(save_dir, "densepose", f"{filename}.png"), output_dp)
                cv2.imwrite(os.path.join(save_dir, "image_crops", f"{filename}.jpg"), image_crop)
                generated_metadata = {
                    "caption": captioner.generate_caption(image_crop[..., ::-1]),
                    "original_caption": metadata["caption"],
                    "bbox": bboxes[i].astype(int).tolist(),
                    "crop_bbox": new_bboxes,
                }
                with open(os.path.join(save_dir, "metadata", f"{filename}.json"), "w") as file:
                    json.dump(generated_metadata, file)


def dataset(laion_path: str, save_dir: str, split: str = "split_00000"):
    print(f"Processing {laion_path} at {split} split. Saving to {save_dir}")
    detection_model = YOLO("yolov8x-seg.pt", task="segment", verbose=False)
    detection_model.to(DEVICE)
    pose_model = YOLO("yolov8x-pose.pt", task="pose", verbose=False)
    pose_model.to("cuda")

    def setup_config(config_fpath: str, model_fpath: str, opts, device):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        return cfg

    cfg = setup_config(
        config_fpath=os.path.join("densepose_rcnn_R_101_FPN_DL_s1x.yaml"),
        model_fpath=os.path.join("model_final_844d15.pkl"),
        opts=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(0.3)], device=DEVICE
    )

    predictor = DefaultPredictor(cfg)
    extractor = DensePoseResultExtractor()

    os.makedirs(os.path.join(save_dir, "densepose"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "image_crops"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "metadata"), exist_ok=True)
    image_captioner = ImageCaptioner("gemma-3-4b-it")
    images_paths = glob.glob(f"{laion_path}/{split}/*.tar")
    print(f"Found {len(images_paths)} tar files in {laion_path}/{split}. Starting processing...")
    dataset = wds.WebDataset(images_paths)
    for sample in tqdm.tqdm(dataset):
        image_bytes = sample["jpg"] 
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        metadata = json.load(io.BytesIO(sample["json"]))
        if not valid_image(image, metadata):
            continue
        process_image(image, metadata, image_captioner, detection_model, pose_model, predictor, extractor, save_dir)


if __name__ == "__main__":
    Fire(dataset)
