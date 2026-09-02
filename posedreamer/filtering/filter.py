"""Hydra entry point of the filtering pipeline (paper 3.4): stream generated
samples through the configured filters and apply the copy/move action to
those that pass."""
import os
from typing import List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from posedreamer.filtering.pipeline import FilteringPipeline
from posedreamer.filtering.filters.base_filter import BaseFilter
from posedreamer.filtering.actions import BaseAction


def instantiate_stages(cfg: DictConfig) -> Tuple[List[BaseFilter], BaseAction]:
    """Instantiate filters and action from Hydra configuration."""
    filters = []
    for filter_cfg in cfg.filters:
        filter_obj = instantiate(filter_cfg)
        filters.append(filter_obj)

    action = instantiate(cfg.action)
    return filters, action


def validate_inputs(input_dirs: List[dict]) -> None:
    """Validate input directories."""
    for entry in input_dirs:
        for key in ('images', 'densepose', 'smplx', 'smpl'):
            path = entry.get(key)
            if path is not None and not os.path.exists(path):
                print(f"Warning: {key} directory does not exist: {path}")


@hydra.main(version_base=None, config_path="config", config_name="filter")
def main(cfg: DictConfig) -> None:
    """Main function to run the filtering pipeline with Hydra configuration."""
    input_dirs = OmegaConf.to_container(cfg.input_directories, resolve=True)
    settings = cfg.get('settings', {})
    
    validate_inputs(input_dirs)
    filters, action = instantiate_stages(cfg)
    
    verbose = settings.get('verbose', True)
    pipeline = FilteringPipeline(
        filters=filters,
        action=action,
        verbose=verbose,
        max_samples=settings.get('max_samples', 0)
    )
    pipeline.process_directory(input_dirs)

if __name__ == "__main__":
    main() 
