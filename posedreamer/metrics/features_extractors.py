import os
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

from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

REAL_DATASETS = ['LAION-Face']  # Traditional SOD datasets
SYNTHETIC_DATASETS = ['AGORA', 'SynBody', 'BEDLAM', 'PoseDreamer']  # DIS-5K treated as comparison
BATCH_SIZE = 32
SEED = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CLIPExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()
        self.use_clip_preprocess = True
    
    def forward(self, x):
        with torch.no_grad():
            return self.model.encode_image(x).float()


class DINOExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        self.use_clip_preprocess = False
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
        
def group_features_by_type(features, labels):
    """Group features into SOD (real) and comparison datasets (DIS-5K + synthetics)"""
    real_mask = np.isin(labels, REAL_DATASETS)
    agora_mask = labels == 'AGORA'
    bedlam_mask = labels == 'BEDLAM'
    sybody_mask = labels == 'SynBody'
    posedreamer_mask = labels == 'PoseDreamer'

    grouped_features = []
    grouped_labels = []
    
    if real_mask.any():
        grouped_features.append(features[real_mask])
        grouped_labels.extend(['LAION-Faces'] * real_mask.sum())
    
    if agora_mask.any():
        grouped_features.append(features[agora_mask])
        grouped_labels.extend(['AGORA'] * agora_mask.sum())
    
    if bedlam_mask.any():
        grouped_features.append(features[bedlam_mask])
        grouped_labels.extend(['BEDLAM'] * bedlam_mask.sum())
    
    if sybody_mask.any():
        grouped_features.append(features[sybody_mask])
        grouped_labels.extend(['SynBody'] * sybody_mask.sum())

    if posedreamer_mask.any():
        grouped_features.append(features[posedreamer_mask])
        grouped_labels.extend(['PoseDreamer'] * posedreamer_mask.sum())
    
    return np.vstack(grouped_features), np.array(grouped_labels)

def load_image(path, size=448, use_clip=False):
    """Load and preprocess image"""
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if use_clip:
        from PIL import Image
        import torchvision.transforms as T
        img_pil = Image.fromarray(img)
        transform = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), 
                       (0.26862954, 0.26130258, 0.27577711))
        ])
        return transform(img_pil)
    else:
        img = cv2.resize(img, (size, size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


def get_image_paths(dataset_path, dataset_name):
    """Get image or mask paths from dataset"""
    if dataset_name == 'SynBody':
        paths = []
        for dir, _, files in os.walk(dataset_path):
            if dir.endswith("rgb"):
                files = [f for f in files if f.endswith("jpeg")]
                files = [os.path.join(dir, f) for f in files]
                paths += files
    elif dataset_name == 'BEDLAM':
        paths = []
        for dir, _, files in os.walk(dataset_path):
            files = [f for f in files if f.endswith("png")]
            files = [os.path.join(dir, f) for f in files]
            if "suburb" in dir:
                random = np.random.rand(len(files))
                files = [f for f in files if random[files.index(f)] < 0.3]
            paths += files
    else:
        paths = []
        image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp')
        files = os.listdir(dataset_path)
        files = [f for f in files if f.lower().endswith(image_exts)]
        files = [f for f in files if not f.lower().startswith("densepose_coco")]
        files = [os.path.join(dataset_path, f) for f in files]
        paths += files


    return paths


def load_all_features(datasets_dict, model, data_type='images'):
    """Load features from all datasets"""
    all_features = []
    all_labels = []
    
    use_clip = model.use_clip_preprocess
    
    for dataset_name, dataset_path in datasets_dict.items():
        n_samples = 300 #SAMPLES_PER_DATASET[dataset_name]
        print(f"\n  Loading {dataset_name} ({data_type}) - {n_samples} samples...")
        
        image_paths = get_image_paths(dataset_path, dataset_name)
        
        if not image_paths:
            print(f"    ⚠ No {data_type} found, skipping...")
            continue
        else:
            print(f"    found {len(image_paths)} image_paths")
        
        # Sample the required number
        if len(image_paths) > n_samples:
            image_paths = np.random.choice(image_paths, n_samples, replace=False)
        
        print(f"    Processing {len(image_paths)} {data_type}")
        
        dataset_features = []
        
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"    Extracting"):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_images = []
            
            for path in batch_paths:
                img = load_image(path, use_clip=use_clip)
                if img is not None:
                    batch_images.append(img)
            
            if not batch_images:
                continue
            
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                features = model(batch_tensor)
                dataset_features.append(features.cpu().numpy())
        
        if dataset_features:
            dataset_features = np.vstack(dataset_features)
            all_features.append(dataset_features)
            all_labels.extend([dataset_name] * len(dataset_features))
            print(f"    ✓ Extracted {len(dataset_features)} features")
    
    return np.vstack(all_features), np.array(all_labels)



def calculate_fid(feat1, feat2):
    """Fréchet Inception Distance"""
    mu1, sigma1 = feat1.mean(axis=0), np.cov(feat1, rowvar=False)
    mu2, sigma2 = feat2.mean(axis=0), np.cov(feat2, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def calculate_mmd(feat1, feat2):
    """Maximum Mean Discrepancy"""
    # Use subset for efficiency
    n_sample = min(1000, len(feat1), len(feat2))
    feat1_sub = feat1[np.random.choice(len(feat1), n_sample, replace=False)]
    feat2_sub = feat2[np.random.choice(len(feat2), n_sample, replace=False)]
    
    sigma = np.median(cdist(feat1_sub[:200], feat1_sub[:200], 'euclidean'))
    
    def gaussian_kernel(x, y, sigma):
        x_size, y_size, dim = x.shape[0], y.shape[0], x.shape[1]
        tiled_x = np.tile(x.reshape(x_size, 1, dim), (1, y_size, 1))
        tiled_y = np.tile(y.reshape(1, y_size, dim), (x_size, 1, 1))
        return np.exp(-np.mean((tiled_x - tiled_y) ** 2, axis=2) / (2 * sigma))
    
    xx = gaussian_kernel(feat1_sub, feat1_sub, sigma).mean()
    yy = gaussian_kernel(feat2_sub, feat2_sub, sigma).mean()
    xy = gaussian_kernel(feat1_sub, feat2_sub, sigma).mean()
    
    return max(0, xx + yy - 2 * xy)


def calculate_kl_divergence(feat1, feat2):
    """
    KL divergence between two multivariate Gaussians
    KL(P||Q) where P=feat1, Q=feat2
    """
    mu1 = feat1.mean(axis=0)
    mu2 = feat2.mean(axis=0)
    
    sigma1 = np.cov(feat1, rowvar=False)
    sigma2 = np.cov(feat2, rowvar=False)
    
    # Add small regularization for numerical stability
    reg = 1e-6
    sigma1 += np.eye(sigma1.shape[0]) * reg
    sigma2 += np.eye(sigma2.shape[0]) * reg
    
    k = feat1.shape[1]  # dimensionality
    
    # KL(P||Q) = 0.5 * [tr(Σ2^-1 * Σ1) + (μ2-μ1)^T * Σ2^-1 * (μ2-μ1) - k + ln(det(Σ2)/det(Σ1))]
    try:
        sigma2_inv = np.linalg.inv(sigma2)
        
        term1 = np.trace(sigma2_inv @ sigma1)
        mu_diff = mu2 - mu1
        term2 = mu_diff.T @ sigma2_inv @ mu_diff
        
        sign1, logdet1 = np.linalg.slogdet(sigma1)
        sign2, logdet2 = np.linalg.slogdet(sigma2)
        term3 = logdet2 - logdet1
        
        kl = 0.5 * (term1 + term2 - k + term3)
        
        return kl
    except np.linalg.LinAlgError:
        return np.nan


def compute_comprehensive_metrics(features, labels):
    """Compute FID, MMD, Coverage, Density, Authenticity"""
    print(f"\n{'Dataset':<15} | {'FID':>8} | {'MMD':>10} | {'Coverage':>9} | {'Density':>8} | {'Authen':>7}")
    print("-" * 80)
    
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    sod_feat = features_norm[labels == 'SOD']
    results = {}
    
    for dataset in ['MaskFactory', 'S3OD', 'DIS-5K']:
        mask = labels == dataset
        if not mask.any():
            continue
        
        dataset_feat = features_norm[mask]
        
        # Distance-based metrics
        fid = calculate_fid(sod_feat, dataset_feat)
        mmd = calculate_mmd(sod_feat, dataset_feat)
        
        # Distribution metrics
        coverage, density = calculate_coverage_density(sod_feat, dataset_feat)
        authenticity, avg_dist = calculate_authenticity(sod_feat, dataset_feat)
        
        results[dataset] = {
            'fid': fid,
            'mmd': mmd,
            'coverage': coverage,
            'density': density,
            'authenticity': authenticity
        }
        
        print(f"{dataset:<15} | {fid:8.2f} | {mmd:10.6f} | {coverage:9.3f} | {density:8.2f} | {authenticity:7.3f}")
    
    return results