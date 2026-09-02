import json
import logging
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class DPOEasyControlDataset(Dataset):
    """
    DPO dataset for EasyControl that creates winner/loser pairs based on OKS metrics.
    Groups samples by caption and selects best/worst performing samples.
    """
    
    def __init__(
        self,
        metadata_path: str,
        metric_name: str = "oks_metric",
        metric_mode: str = "max",
        image_size: int = 1024,
        cond_size: int = 512,
        min_gap: float = 0.05,
        min_win_metric: float = 0.5,
    ):
        """
        Args:
            metadata_path: Path to merged JSON file or folder containing individual JSON files
            metric_name: Name of metric field in metadata (e.g., "oks_metric")
            metric_mode: "max" if higher metric is better, "min" if lower is better
            image_size: Target image size for resizing
            cond_size: Target condition size for resizing
            min_gap: Minimum gap between winner and loser metrics
            min_win_metric: Minimum metric value for winner
        """
        self.metadata_path = Path(metadata_path)
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.image_size = image_size
        self.cond_size = cond_size
        self.min_gap = min_gap
        self.min_win_metric = min_win_metric
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.transform_cond = transforms.Compose([
            transforms.Resize((cond_size, cond_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Load and group metadata
        self.pairs = self._load_and_group_metadata()
        logger.info(f"Created {len(self.pairs)} DPO pairs from {metadata_path}")
    
    def _load_and_group_metadata(self) -> List[Tuple[Dict, Dict]]:
        """Load metadata files and create winner/loser pairs."""
        all_metadata = []
        
        if self.metadata_path.is_file():
            # Load from single merged JSON file
            logger.info(f"Loading merged metadata from {self.metadata_path}")
            with open(self.metadata_path, 'r') as f:
                merged_data = json.load(f)
                all_metadata = merged_data.get("metadata", [])
            logger.info(f"Loaded {len(all_metadata)} samples from merged file")
            
        elif self.metadata_path.is_dir():
            # Load from individual JSON files in folder
            metadata_files = list(self.metadata_path.glob("*.json"))
            logger.info(f"Loading from {len(metadata_files)} individual JSON files")
            
            for file_path in metadata_files:
                with open(file_path, 'r') as f:
                    metadata = json.load(f)
                    if self.metric_name in metadata and metadata[self.metric_name] != -1:
                        all_metadata.append(metadata)
            
            logger.info(f"Loaded {len(all_metadata)} valid samples with {self.metric_name}")
        else:
            raise ValueError(f"Metadata path must be a file or directory: {self.metadata_path}")
        
        # Filter valid samples (in case merged file contains invalid ones)
        valid_metadata = []
        for metadata in all_metadata:
            if self.metric_name in metadata and metadata[self.metric_name] != -1:
                valid_metadata.append(metadata)
        
        logger.info(f"Using {len(valid_metadata)} valid samples with {self.metric_name}")
        
        condition_groups = {}
        for metadata in valid_metadata:
            condition_path = metadata["condition_path"]
            if condition_path not in condition_groups:
                condition_groups[condition_path] = []
            condition_groups[condition_path].append(metadata)
        
        pairs = []
        for condition_path, samples in condition_groups.items():
            if len(samples) >= 2:
                samples.sort(
                    key=lambda x: x[self.metric_name], 
                    reverse=(self.metric_mode == "max")
                )
                if abs(samples[0][self.metric_name] - samples[-1][self.metric_name]) >= self.min_gap and samples[0][self.metric_name] >= self.min_win_metric:
                    pairs.append((samples[0], samples[-1]))
        logger.info(f"Created {len(pairs)} winner/loser pairs from {len(condition_groups)} condition groups")
        return pairs
    
    def _load_image(self, image_path: str, transform: transforms.Compose = None) -> torch.Tensor:
        """Load and transform image."""
        image = Image.open(image_path).convert("RGB")
        if transform is not None:
            image = transform(image)
        return image
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a DPO training sample with winner and loser data.
        
        Returns:
            Dict containing:
                - x_w: Winner image tensor
                - x_l: Loser image tensor
                - cond: Condition tensor (same for both winner and loser)
                - caption: Caption string (same for both winner and loser)
                - winner_metric: Winner metric value
                - loser_metric: Loser metric value
        """
        winner_metadata, loser_metadata = self.pairs[idx]
        
        # Load winner and loser images
        x_w = self._load_image(winner_metadata["generated_image_path"], self.transform)
        x_l = self._load_image(loser_metadata["generated_image_path"], self.transform)
        
        # Load condition (same for both since grouped by condition_path)
        cond = self._load_image(winner_metadata["condition_path"], self.transform_cond)
        caption = winner_metadata["caption"]
        
        return {
            "x_w": x_w,
            "x_l": x_l,
            "cond": cond,
            "caption": caption,
            "winner_metric": float(winner_metadata[self.metric_name]),
            "loser_metric": float(loser_metadata[self.metric_name])
        }


def collate_dpo_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DPO batch processing."""
    # Stack tensors
    x_w = torch.stack([item["x_w"] for item in batch])
    x_l = torch.stack([item["x_l"] for item in batch])
    cond = torch.stack([item["cond"] for item in batch])
    
    # Collect captions and metrics
    captions = [item["caption"] for item in batch]
    winner_metrics = torch.tensor([item["winner_metric"] for item in batch])
    loser_metrics = torch.tensor([item["loser_metric"] for item in batch])
    
    return {
        "x_w": x_w,
        "x_l": x_l,
        "cond": cond,
        "captions": captions,
        "winner_metrics": winner_metrics,
        "loser_metrics": loser_metrics
    }
