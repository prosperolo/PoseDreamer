from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class Sample:
    """Represents a single data sample with image, SMPLX annotations, and densepose render."""
    
    image_path: str
    smplx_path: Optional[str]
    densepose_path: str
    sample_id: str
    smpl_path: Optional[str] = None

    # Optional loaded data - populated on demand
    image_data: Optional[np.ndarray] = None
    smplx_data: Optional[Dict[str, Any]] = None
    densepose_data: Optional[np.ndarray] = None
    smpl_data: Optional[Dict[str, Any]] = None
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def filename(self) -> str:
        """Get the base filename without extension."""
        import os
        return os.path.splitext(os.path.basename(self.image_path))[0] 