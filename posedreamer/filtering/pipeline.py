from typing import List, Dict, Any, Optional, Tuple
import os
import glob
from posedreamer.filtering.sample import Sample
from posedreamer.filtering.filters.base_filter import BaseFilter
from posedreamer.filtering.actions import BaseAction, NoOpAction


MAX_TASKS = 12


class FilteringPipeline:
    """Main filtering pipeline that processes samples through multiple filters."""
    
    def __init__(self, 
                 filters: List[BaseFilter], 
                 action: Optional[BaseAction] = None,
                 verbose: bool = True,
                 max_samples: int = 0):
        """
        Args:
            filters: List of filters to apply
            action: Action to perform on validated samples (default: NoOpAction)
            verbose: Whether to print progress information
        """
        self.filters = filters
        self.action = action or NoOpAction()
        self.verbose = verbose
        self.max_samples = max_samples
        print(f"Max samples: {self.max_samples}")
        
        # Pipeline statistics
        self.stats = {
            'total_samples': 0,
            'samples_passed': 0,
            'samples_failed': 0,
            'action_successes': 0,
            'action_failures': 0
        }
    
    def create_sample_from_image_path(self, image_path: str, mapping: Dict[str, Any], idx: int) -> Sample:
        """
        Create a Sample object from an image path, following the naming convention.

        Args:
            image_path: Path to the generated image.
            mapping: Input-directory entry from the config. Keys: 'images' (dir the
                image lives in), 'densepose' (dir of control renders), optional
                'smplx' / 'smpl' (dirs of .h5 label files; either may be absent),
                optional 'filecount_suffix' (see below, default True).
        """
        filename = os.path.basename(image_path)
        filename_base = filename[:-4]  # remove extension
        filename_nocount = filename_base[:-2]  # remove last two characters (e.g., "_0" or "_1")

        image_folder_path = mapping['images']
        densepose_folder_path = mapping['densepose']
        smplx_folder_path = mapping.get('smplx')
        smpl_folder_path = mapping.get('smpl')
        has_filecount_suffix = mapping.get('filecount_suffix', True)

        densepose_path = image_path.replace(image_folder_path, densepose_folder_path).replace('.jpg', '.png')
        smplx_path = (image_path.replace(image_folder_path, smplx_folder_path).replace('.jpg', '.h5')
                      if smplx_folder_path is not None else None)
        smpl_path = (image_path.replace(image_folder_path, smpl_folder_path).replace('.jpg', '.h5')
                     if smpl_folder_path is not None else None)

        # Source images with a per-image count suffix (e.g. "_0", "_1") share a single
        # annotation file keyed by the un-suffixed filename. Sources flagged with
        # filecount=False already have 1:1 names so no stripping is needed.
        if has_filecount_suffix:
            densepose_path = densepose_path.replace(filename_base, filename_nocount)
            if smplx_path is not None:
                smplx_path = smplx_path.replace(filename_base, filename_nocount)
            if smpl_path is not None:
                smpl_path = smpl_path.replace(filename_base, filename_nocount)

        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
        return Sample(
            image_path=image_path,
            smplx_path=smplx_path,
            densepose_path=densepose_path,
            smpl_path=smpl_path,
            sample_id = f'task_{task_id}_idx_{idx}'
        )
    
    def validate_sample(self, sample: Sample) -> bool:
        """
        Run a sample through all filters.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample passes all filters, False otherwise
        """
        for filter_obj in self.filters:
            if not filter_obj(sample):
                return False
        return True
    
    def process_sample(self, sample: Sample) -> bool:
        """
        Process a single sample through the pipeline.
        
        Args:
            sample: Sample to process
            
        Returns:
            True if sample was successfully processed and action executed
        """
        self.stats['total_samples'] += 1
        
        # Validate sample through all filters
        if self.validate_sample(sample):
            self.stats['samples_passed'] += 1
            
            # Execute action on validated sample
            if self.action.execute(sample):
                self.stats['action_successes'] += 1
                return True
            else:
                self.stats['action_failures'] += 1
                return False
        else:
            self.stats['samples_failed'] += 1
            return False
    
    def process_directory(self, input_dirs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process all images in the given input-directory entries.

        Args:
            input_dirs: List of config entries, each a dict with the image dir
                ('images'), the control-render dir ('densepose') and optional
                'smplx' / 'smpl' label dirs (see create_sample_from_image_path).

        Returns:
            Dictionary with processing statistics
        """
        # Reset statistics
        self.reset_stats()

        # Collect all image files, remembering which entry each came from
        image_files = []
        for entry in input_dirs:
            pattern = os.path.join(entry['images'], '*.jpg')
            image_files.extend((f, entry) for f in glob.glob(pattern))
        
        if self.verbose:
            print(f"Found {len(image_files)} image files to process")
            print(f"Using {len(self.filters)} filters: {[f.name for f in self.filters]}")
        
        if self.max_samples > 0:
            image_files = image_files[:self.max_samples]
        
        # Process each image
        image_files = sorted(image_files, key=lambda pair: pair[0])
        start_idx, end_idx = self._get_start_end_idx(image_files)
        print(f"Processing {len(image_files[start_idx:end_idx])} samples")
        for i, (image_path, entry) in enumerate(image_files[start_idx:end_idx]):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} samples")
            sample = self.create_sample_from_image_path(image_path, entry, idx=i)
            self.process_sample(sample)
    
        if self.verbose:
            self.print_summary()
        
        return self.get_stats()
    
    def _get_start_end_idx(self, image_files: List[str]) -> Tuple[int, int]:
        """Get start and end indices for the current task."""
        if 'SLURM_ARRAY_TASK_ID' in os.environ:
            task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
            # TOTAL_TASKS overrides SLURM_ARRAY_TASK_COUNT so a single-index resubmit
            # (e.g. `sbatch --array=2 --export=ALL,TOTAL_TASKS=8`) still produces
            # the same shard as the original full array.
            num_tasks = int(os.environ.get('TOTAL_TASKS', os.environ['SLURM_ARRAY_TASK_COUNT']))
            samples_per_task = len(image_files) // num_tasks
            start_idx = task_id * samples_per_task
            end_idx = start_idx + samples_per_task
        else:
            start_idx = 0
            end_idx = len(image_files)
        return start_idx, end_idx
    
    def reset_stats(self):
        """Reset pipeline and filter statistics."""
        self.stats = {
            'total_samples': 0,
            'samples_passed': 0,
            'samples_failed': 0,
            'action_successes': 0,
            'action_failures': 0
        }
        
        for filter_obj in self.filters:
            filter_obj.reset_stats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including filter-specific stats."""
        stats = self.stats.copy()
        stats['filter_stats'] = {f.name: f.get_stats() for f in self.filters}
        return stats
    
    def print_summary(self):
        """Print a summary of processing results."""
        print("\n=== Filtering Pipeline Summary ===")
        print(f"Total samples processed: {self.stats['total_samples']}")
        print(f"Samples passed: {self.stats['samples_passed']}")
        print(f"Samples failed: {self.stats['samples_failed']}")
        print(f"Action successes: {self.stats['action_successes']}")
        print(f"Action failures: {self.stats['action_failures']}")
        
        if self.stats['total_samples'] > 0:
            pass_rate = self.stats['samples_passed'] / self.stats['total_samples'] * 100
            print(f"Pass rate: {pass_rate:.2f}%")
        
        print("\n=== Filter Statistics ===")
        for filter_obj in self.filters:
            filter_obj.print_detailed_stats()
