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
import umap
import warnings
from posedreamer.metrics.func import compute_prd_from_embedding, prd_to_max_f_beta_pair

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def visualize_distribution(features, labels, title='', filename='', method='tsne', SEED=42):
    """Visualize distribution using all available samples"""
    print(f"\n  Using all samples for visualization...")
    
    # Normalize features
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    # Count samples per group
    for dataset_name in ['LAION-Faces', 'AGORA', 'SynBody', 'BEDLAM', 'PoseDreamer']:
        mask = labels == dataset_name
        if mask.any():
            print(f"    {dataset_name}: {mask.sum()} samples")
    
    print(f"  Total samples: {len(features_norm)}")
    print(f"  Computing {method.upper()}...")
    
    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            random_state=SEED,
            max_iter=2000,
            learning_rate=200
        )
    else:  # umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.3,
            random_state=SEED
        )
    
    embedding = reducer.fit_transform(features_norm)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'LAION-Faces': "#648fff", 
        'AGORA': '#dc267f',         
        'SynBody': '#785ef0', 
        'BEDLAM': '#fe6100',        
        'PoseDreamer': "#ffb000"       
    }
    
    markers = {
        'LAION-Faces': 'X',
        'AGORA': 'o',
        'SynBody': 's',
        'BEDLAM': 'D',
        'PoseDreamer': '^'
    }

    def dataset_label_map(dataset_name):
        if dataset_name == 'LAION-Faces':
            return 'Real Images (LAION-Faces)'
        elif dataset_name == 'PoseDreamer':
            return 'PoseDreamer (Ours)'
        else:
            return dataset_name
    
    # Plot in order: SOD, MaskFactory, S3OD, DIS-5K
    for dataset_name in ['LAION-Faces', 'AGORA', 'SynBody', 'BEDLAM', 'PoseDreamer']:
        mask = labels == dataset_name
        if not mask.any():
            continue
        
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=80,
            c=colors[dataset_name],
            label=dataset_label_map(dataset_name),
            alpha=1.0,
            marker=markers[dataset_name],
            edgecolors='black',
            linewidth=1.0
        )
    
    # Legend with proper spacing
    legend = ax.legend(
        fontsize=12,
        markerscale=1.5,
        framealpha=0.95,
        loc='best',
        handletextpad=0.5,
        borderpad=1.0,
        labelspacing=0.8,
        handlelength=2.0
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel(f'{method.upper()} 1', fontsize=13)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  ✓ Saved: {filename}")

def compute_and_plot_prd(features, labels, filename, title=''):
    """Compute and plot PRD curves for all synthetic datasets vs Real (SOD + DIS-5K)"""
    print(f"\n  Computing PRD curves for {title}...")
    
    features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    real_mask = labels == 'LAION-Faces'
    real_features = features_norm[real_mask]
    
    print(f"    Real (LAION-Faces) samples: {len(real_features)}")
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 8))

    def dataset_label_map(dataset_name):
        if dataset_name == 'LAION-Faces':
            return 'Real Images (LAION-Faces)'
        elif dataset_name == 'PoseDreamer':
            return 'PoseDreamer (Ours)'
        else:
            return dataset_name
    
    colors = {
        'AGORA': '#dc267f',         
        'SynBody': '#785ef0', 
        'BEDLAM': '#fe6100',        
        'PoseDreamer': "#ffb000"         
    }
    
    prd_results = {}
    
    # Compute PRD for each synthetic dataset
    for dataset_name in ['AGORA', 'SynBody', 'BEDLAM', 'PoseDreamer']:
        synth_mask = labels == dataset_name
        if not synth_mask.any():
            print(f"    ⚠ No data for {dataset_name}")
            continue
        
        synth_features = features_norm[synth_mask]
        print(f"    {dataset_name}: {len(synth_features)} samples")
        
        # Balance datasets
        min_samples = min(len(real_features), len(synth_features))
        real_subset = real_features[np.random.choice(len(real_features), min_samples, replace=False)]
        synth_subset = synth_features[np.random.choice(len(synth_features), min_samples, replace=False)]
        
        # Compute PRD
        precision, recall = compute_prd_from_embedding(
            synth_subset,
            real_subset,
            num_clusters=20,
            num_angles=1001,
            num_runs=10,
            enforce_balance=True
        )
        
        # Compute F_beta scores
        f_beta_8, f_beta_0125 = prd_to_max_f_beta_pair(precision, recall, beta=8)
        print(f"      F_β(8): {f_beta_8:.4f}, F_β(1/8): {f_beta_0125:.4f}")
        
        # Plot curve
        ax.plot(recall, precision,
               label=f'{dataset_label_map(dataset_name)} (F={f_beta_8:.3f})',
               alpha=1.0,
               linewidth=2.5,
               color=colors[dataset_name])
        
        prd_results[dataset_name] = {
            'precision': precision,
            'recall': recall,
            'f_beta_8': f_beta_8,
            'f_beta_0125': f_beta_0125
        }
    
    # Finalize plot
    # ax.set_xlim([0, 1])
    # ax.set_ylim([0, 1])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  ✓ Saved: {filename}")
    
    return prd_results


def create_comparison_table(dino_img_res, dino_mask_res, clip_img_res, clip_mask_res):
    """Create comprehensive comparison table"""
    print("\n" + "="*130)
    print("COMPREHENSIVE METRICS TABLE (vs SOD Datasets)")
    print("="*130)
    
    print(f"\n{'Metric':<12} | {'Encoder':<6} | {'Type':<6} | {'MaskFactory':>12} | {'S3OD':>12} | {'DIS-5K':>12}")
    print("-" * 130)
    
    # FID
    print(f"{'FID ↓':<12} | {'DINO':<6} | {'Image':<6} | {dino_img_res['MaskFactory']['fid']:>12.2f} | {dino_img_res['S3OD']['fid']:>12.2f} | {dino_img_res['DIS-5K']['fid']:>12.2f}")
    print(f"{'FID ↓':<12} | {'DINO':<6} | {'Mask':<6} | {dino_mask_res['MaskFactory']['fid']:>12.2f} | {dino_mask_res['S3OD']['fid']:>12.2f} | {dino_mask_res['DIS-5K']['fid']:>12.2f}")
    print(f"{'FID ↓':<12} | {'CLIP':<6} | {'Image':<6} | {clip_img_res['MaskFactory']['fid']:>12.2f} | {clip_img_res['S3OD']['fid']:>12.2f} | {clip_img_res['DIS-5K']['fid']:>12.2f}")
    print(f"{'FID ↓':<12} | {'CLIP':<6} | {'Mask':<6} | {clip_mask_res['MaskFactory']['fid']:>12.2f} | {clip_mask_res['S3OD']['fid']:>12.2f} | {clip_mask_res['DIS-5K']['fid']:>12.2f}")
    print()
    
    # MMD
    print(f"{'MMD ↓':<12} | {'DINO':<6} | {'Image':<6} | {dino_img_res['MaskFactory']['mmd']:>12.6f} | {dino_img_res['S3OD']['mmd']:>12.6f} | {dino_img_res['DIS-5K']['mmd']:>12.6f}")
    print(f"{'MMD ↓':<12} | {'DINO':<6} | {'Mask':<6} | {dino_mask_res['MaskFactory']['mmd']:>12.6f} | {dino_mask_res['S3OD']['mmd']:>12.6f} | {dino_mask_res['DIS-5K']['mmd']:>12.6f}")
    print(f"{'MMD ↓':<12} | {'CLIP':<6} | {'Image':<6} | {clip_img_res['MaskFactory']['mmd']:>12.6f} | {clip_img_res['S3OD']['mmd']:>12.6f} | {clip_img_res['DIS-5K']['mmd']:>12.6f}")
    print(f"{'MMD ↓':<12} | {'CLIP':<6} | {'Mask':<6} | {clip_mask_res['MaskFactory']['mmd']:>12.6f} | {clip_mask_res['S3OD']['mmd']:>12.6f} | {clip_mask_res['DIS-5K']['mmd']:>12.6f}")
    print()
    
    # Coverage
    print(f"{'Coverage ↑':<12} | {'DINO':<6} | {'Image':<6} | {dino_img_res['MaskFactory']['coverage']:>12.3f} | {dino_img_res['S3OD']['coverage']:>12.3f} | {dino_img_res['DIS-5K']['coverage']:>12.3f}")
    print(f"{'Coverage ↑':<12} | {'DINO':<6} | {'Mask':<6} | {dino_mask_res['MaskFactory']['coverage']:>12.3f} | {dino_mask_res['S3OD']['coverage']:>12.3f} | {dino_mask_res['DIS-5K']['coverage']:>12.3f}")
    print(f"{'Coverage ↑':<12} | {'CLIP':<6} | {'Image':<6} | {clip_img_res['MaskFactory']['coverage']:>12.3f} | {clip_img_res['S3OD']['coverage']:>12.3f} | {clip_img_res['DIS-5K']['coverage']:>12.3f}")
    print(f"{'Coverage ↑':<12} | {'CLIP':<6} | {'Mask':<6} | {clip_mask_res['MaskFactory']['coverage']:>12.3f} | {clip_mask_res['S3OD']['coverage']:>12.3f} | {clip_mask_res['DIS-5K']['coverage']:>12.3f}")
    print()
    
    # Authenticity
    print(f"{'Authen ↑':<12} | {'DINO':<6} | {'Image':<6} | {dino_img_res['MaskFactory']['authenticity']:>12.3f} | {dino_img_res['S3OD']['authenticity']:>12.3f} | {dino_img_res['DIS-5K']['authenticity']:>12.3f}")
    print(f"{'Authen ↑':<12} | {'DINO':<6} | {'Mask':<6} | {dino_mask_res['MaskFactory']['authenticity']:>12.3f} | {dino_mask_res['S3OD']['authenticity']:>12.3f} | {dino_mask_res['DIS-5K']['authenticity']:>12.3f}")
    print(f"{'Authen ↑':<12} | {'CLIP':<6} | {'Image':<6} | {clip_img_res['MaskFactory']['authenticity']:>12.3f} | {clip_img_res['S3OD']['authenticity']:>12.3f} | {clip_img_res['DIS-5K']['authenticity']:>12.3f}")
    print(f"{'Authen ↑':<12} | {'CLIP':<6} | {'Mask':<6} | {clip_mask_res['MaskFactory']['authenticity']:>12.3f} | {clip_mask_res['S3OD']['authenticity']:>12.3f} | {clip_mask_res['DIS-5K']['authenticity']:>12.3f}")
    
    print("\n" + "="*130)
    
    # LaTeX table
    print("\n\\begin{table*}[t]")
    print("\\centering")
    print("\\caption{Distribution Metrics comparing datasets to traditional SOD benchmarks (UHRSD-TR, DUTS-TR, HRSOD-TR). DIS-5K represents dichotomous image segmentation as a comparison point. Lower is better for FID/MMD, higher is better for Coverage/Authenticity.}")
    print("\\label{tab:distribution_metrics}")
    print("\\begin{tabular}{llcrrr}")
    print("\\toprule")
    print("Metric & Encoder & Type & MaskFactory & S3OD & DIS-5K \\\\")
    print("\\midrule")
    print(f"FID $\\downarrow$ & DINO & Image & {dino_img_res['MaskFactory']['fid']:.2f} & {dino_img_res['S3OD']['fid']:.2f} & {dino_img_res['DIS-5K']['fid']:.2f} \\\\")
    print(f"               & DINO & Mask  & {dino_mask_res['MaskFactory']['fid']:.2f} & {dino_mask_res['S3OD']['fid']:.2f} & {dino_mask_res['DIS-5K']['fid']:.2f} \\\\")
    print(f"               & CLIP & Image & {clip_img_res['MaskFactory']['fid']:.2f} & {clip_img_res['S3OD']['fid']:.2f} & {clip_img_res['DIS-5K']['fid']:.2f} \\\\")
    print(f"               & CLIP & Mask  & {clip_mask_res['MaskFactory']['fid']:.2f} & {clip_mask_res['S3OD']['fid']:.2f} & {clip_mask_res['DIS-5K']['fid']:.2f} \\\\")
    print("\\midrule")
    print(f"MMD $\\downarrow$ & DINO & Image & {dino_img_res['MaskFactory']['mmd']:.6f} & {dino_img_res['S3OD']['mmd']:.6f} & {dino_img_res['DIS-5K']['mmd']:.6f} \\\\")
    print(f"               & DINO & Mask  & {dino_mask_res['MaskFactory']['mmd']:.6f} & {dino_mask_res['S3OD']['mmd']:.6f} & {dino_mask_res['DIS-5K']['mmd']:.6f} \\\\")
    print(f"               & CLIP & Image & {clip_img_res['MaskFactory']['mmd']:.6f} & {clip_img_res['S3OD']['mmd']:.6f} & {clip_img_res['DIS-5K']['mmd']:.6f} \\\\")
    print(f"               & CLIP & Mask  & {clip_mask_res['MaskFactory']['mmd']:.6f} & {clip_mask_res['S3OD']['mmd']:.6f} & {clip_mask_res['DIS-5K']['mmd']:.6f} \\\\")
    print("\\midrule")
    print(f"Coverage $\\uparrow$ & DINO & Image & {dino_img_res['MaskFactory']['coverage']:.3f} & {dino_img_res['S3OD']['coverage']:.3f} & {dino_img_res['DIS-5K']['coverage']:.3f} \\\\")
    print(f"               & DINO & Mask  & {dino_mask_res['MaskFactory']['coverage']:.3f} & {dino_mask_res['S3OD']['coverage']:.3f} & {dino_mask_res['DIS-5K']['coverage']:.3f} \\\\")
    print(f"               & CLIP & Image & {clip_img_res['MaskFactory']['coverage']:.3f} & {clip_img_res['S3OD']['coverage']:.3f} & {clip_img_res['DIS-5K']['coverage']:.3f} \\\\")
    print(f"               & CLIP & Mask  & {clip_mask_res['MaskFactory']['coverage']:.3f} & {clip_mask_res['S3OD']['coverage']:.3f} & {clip_mask_res['DIS-5K']['coverage']:.3f} \\\\")
    print("\\midrule")
    print(f"Authenticity $\\uparrow$ & DINO & Image & {dino_img_res['MaskFactory']['authenticity']:.3f} & {dino_img_res['S3OD']['authenticity']:.3f} & {dino_img_res['DIS-5K']['authenticity']:.3f} \\\\")
    print(f"               & DINO & Mask  & {dino_mask_res['MaskFactory']['authenticity']:.3f} & {dino_mask_res['S3OD']['authenticity']:.3f} & {dino_mask_res['DIS-5K']['authenticity']:.3f} \\\\")
    print(f"               & CLIP & Image & {clip_img_res['MaskFactory']['authenticity']:.3f} & {clip_img_res['S3OD']['authenticity']:.3f} & {clip_img_res['DIS-5K']['authenticity']:.3f} \\\\")
    print(f"               & CLIP & Mask  & {clip_mask_res['MaskFactory']['authenticity']:.3f} & {clip_mask_res['S3OD']['authenticity']:.3f} & {clip_mask_res['DIS-5K']['authenticity']:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")