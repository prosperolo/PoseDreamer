import json
import fire
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_metadata_files(
    input_folder: str, 
    output_file: str, 
    metric_name: str = "oks_metric",
    verbose: bool = False
) -> None:
    """
    Merge multiple JSON metadata files into a single JSON file.
    
    Args:
        input_folder: Path to folder containing individual JSON metadata files
        output_file: Path to output merged JSON file
        metric_name: Name of metric field to filter valid samples (default: oks_metric)
        verbose: Enable verbose logging (default: False)
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(input_folder)
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Find all JSON files
    json_files = list(input_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {input_folder}")
    
    if len(json_files) == 0:
        logger.warning("No JSON files found in input folder")
        return
    
    # Load and merge all metadata with progress bar
    all_metadata = []
    valid_count = 0
    invalid_count = 0
    
    logger.info("Processing JSON files...")
    for file_path in tqdm(json_files, desc="Loading files", unit="file"):
        try:
            with open(file_path, 'r') as f:
                metadata = json.load(f)
                
            # Check if metadata has valid metric
            if metric_name in metadata and metadata[metric_name] != -1:
                all_metadata.append(metadata)
                valid_count += 1
            else:
                invalid_count += 1
                logger.debug(f"Skipping {file_path.name} - invalid metric")
                
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            invalid_count += 1
    
    logger.info(f"Processed {len(json_files)} files:")
    logger.info(f"  Valid samples: {valid_count}")
    logger.info(f"  Invalid/skipped: {invalid_count}")
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged metadata
    merged_data = {
        "metadata": all_metadata,
        "total_samples": len(all_metadata),
        "metric_name": metric_name,
        "source_folder": str(input_path),
        "source_files_count": len(json_files)
    }
    
    logger.info("Saving merged metadata...")
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    logger.info(f"Merged {len(all_metadata)} samples into {output_file}")
    logger.info(f"Output file size: {output_path.stat().st_size / (1024*1024):.1f} MB")
    logger.info("Merge completed successfully!")


if __name__ == "__main__":
    fire.Fire(merge_metadata_files)
