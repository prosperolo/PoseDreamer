"""YOLO human detection and DensePose crop helpers used by process_laion.py."""
import os
from typing import Tuple

import math
import cv2
import detectron2
import kornia.geometry as KG
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.vis.extractor import (
    DensePoseResultExtractor,
)
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from ultralytics import YOLO

BBOX_MIN_FILTER_SIZE = 192  # at least one side has length larger than this value
BBOX_MIN_AREA = 192 * 96  # area of bbox larger than this value
BBOX_MIN_CONF = 0.7  # minimal detection confidence


def elwise_maximum(a: torch.Tensor, b: torch.Tensor):
    return torch.stack([a, b], dim=0).max(0).values


def transform_points2d(points2d: torch.Tensor, tform: torch.Tensor):
    b, n, _ = points2d.shape
    points2d_hom = torch.cat([points2d, points2d.new_ones(b, n, 1)], dim=-1)
    points2d_hom = torch.matmul(points2d_hom, tform.permute(0, 2, 1))
    points2d = points2d_hom[..., :2] / points2d_hom[..., 2:]
    return points2d


def bbox_transform(bboxes: torch.Tensor, target_size: int, scale: float = 1.0):
    batch_size = bboxes.size(0)

    left, top, right, bottom = bboxes.unbind(dim=1)
    width, height = right - left, bottom - top
    size = 0.5 * scale * elwise_maximum(width, height)
    center_x, center_y = 0.5 * (right + left), 0.5 * (bottom + top)

    src_pts = torch.stack([
        torch.stack([center_x - size, center_y - size], dim=-1),
        torch.stack([center_x - size, center_y + size], dim=-1),
        torch.stack([center_x + size, center_y - size], dim=-1),
        torch.stack([center_x + size, center_y + size], dim=-1),
    ], dim=1)

    # crop square around person and resize to low res
    dst_pts = bboxes.new_tensor([
        [0., 0.],
        [0., target_size],
        [target_size, 0.],
        [target_size, target_size],
    ]).unsqueeze(0).expand(batch_size, -1, -1)

    tform = KG.get_perspective_transform(src_pts, dst_pts)

    # aspect ratio like in the original bbox but scaled
    dst_bbox = transform_points2d(
        torch.stack([
            center_x - 0.5 * scale * width,
            center_y - 0.5 * scale * height,
            center_x + 0.5 * scale * width,
            center_y + 0.5 * scale * height,
        ], dim=-1).view(-1, 2, 2),
        tform).view(-1, 4)

    return tform, dst_bbox


def crop_bbox(images_batch: torch.Tensor, bboxes: torch.Tensor,
              target_min_size: int, target_max_size: int, scale: float = 1.0, mode: str = "bilinear", pad_div: int = 1,
              padding_mode: str = 'zeros', fill_value: torch.Tensor = torch.zeros(3)):
    batch_size = images_batch.size(0)
    assert batch_size == 1, 'Only single image is currently supported'
    assert (target_min_size % pad_div == 0) and (
            target_max_size % pad_div == 0), 'Target size must be divisible by pad_div'

    left, top, right, bottom = bboxes.unbind(dim=1)
    width, height = (right - left).item(), (bottom - top).item()
    max_side = max(width, height) * scale  
    size = max_side / 2.0  

    set_bigger_size_flag = max(width, height) / min(width, height) * target_min_size > target_max_size
    target_size = target_max_size if set_bigger_size_flag else target_min_size

    center_x, center_y = 0.5 * (right + left), 0.5 * (bottom + top)
    src_pts = torch.stack([
        torch.stack([center_x - size, center_y - size], dim=-1),
        torch.stack([center_x - size, center_y + size], dim=-1),
        torch.stack([center_x + size, center_y - size], dim=-1),
        torch.stack([center_x + size, center_y + size], dim=-1),
    ], dim=1)

    dst_pts = images_batch.new_tensor([
        [0., 0.],
        [0., target_size],
        [target_size, 0.],
        [target_size, target_size],
    ]).unsqueeze(0).expand(batch_size, -1, -1)

    tform = KG.get_perspective_transform(src_pts, dst_pts)
    dst_image = KG.warp_perspective(images_batch, tform, dsize=(target_size, target_size), mode=mode,
                                    padding_mode=padding_mode, fill_value=fill_value)
    
    bbox_corners = torch.stack([
    torch.stack([left, top], dim=-1),
    torch.stack([left, bottom], dim=-1),
    torch.stack([right, top], dim=-1),
    torch.stack([right, bottom], dim=-1),
    ], dim=1).unsqueeze(0) 

    transformed_bbox = KG.transform_points(tform, bbox_corners).squeeze(0)  

    new_left, new_top = transformed_bbox[0, 0]
    new_right, new_bottom = transformed_bbox[0, -1]

    new_bboxes = [new_left.item(), new_top.item(), new_right.item(), new_bottom.item()]
    new_bboxes = [round(x) for x in new_bboxes]

    return dst_image, tform, (target_size, target_size), new_bboxes


def detect_humans(image, model):
    H, W, *_ = image.shape
    results = model(image, retina_masks=True, verbose=False)[0]
    detection_results = results.boxes

    cls = detection_results.cls
    conf = detection_results.conf
    xyxy = detection_results.xyxy
    xywh = detection_results.xywh
    size_filter = ((xywh[:, 2] > BBOX_MIN_FILTER_SIZE) | (xywh[:, 3] > BBOX_MIN_FILTER_SIZE)) & \
                  (xywh[:, 2] * xywh[:, 3] > BBOX_MIN_AREA)

    filtering_mask = (cls == 0) & (conf > BBOX_MIN_CONF) & size_filter

    xyxy = xyxy[filtering_mask]
    if cls.shape[0] > 0:

        instance_masks = results.masks.data[filtering_mask]
        instance_masks = torch.nn.functional.interpolate(instance_masks[:, None], (H, W), mode='nearest')[:, 0]
    else:
        instance_masks = None

    return torch.round(xyxy), instance_masks
