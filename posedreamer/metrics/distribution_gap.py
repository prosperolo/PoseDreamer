"""DINO/CLIP feature extraction and t-SNE / UMAP / PRD comparisons between
datasets (paper 4.4). Point the DATASETS dict at your local copies."""
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import linalg
import warnings

from posedreamer.metrics.features_extractors import DINOExtractor, CLIPExtractor
from posedreamer.metrics.features_extractors import load_all_features
import os


warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATASETS = {
    'AGORA': '/path/to/AGORA/train_images',
    'SynBody': '/path/to/SynBody/synbody_v1_0',
    'BEDLAM': '/path/to/BEDLAM/png',
    'PoseDreamer': '/path/to/posedreamer-filtered/images',
    'LAION-Face': '/path/to/laion-annotations/image_crops'
}

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


dino_model = DINOExtractor().to(device)
dino_img_features, dino_img_labels = load_all_features(
    DATASETS, dino_model, 'images'
)
print(f"\n  ✓ Shape: {dino_img_features.shape}")

print("\n[1/2] Extracting CLIP features from images...")
clip_model = CLIPExtractor().to(device)
clip_img_features, clip_img_labels = load_all_features(
    DATASETS, clip_model, 'images'
)
print(f"\n  ✓ Shape: {clip_img_features.shape}")
