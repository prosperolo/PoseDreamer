from abc import ABC, abstractmethod
from typing import List
import os
import shutil
from posedreamer.filtering.sample import Sample


class BaseAction(ABC):
    """Abstract base class for actions to perform on validated samples."""
    
    @abstractmethod
    def execute(self, sample: Sample) -> bool:
        """
        Execute action on a validated sample.
        
        Args:
            sample: Validated sample
            
        Returns:
            True if action succeeded, False otherwise
        """
        pass


def _build_dest_metadata(sample: Sample, dest_base_dir: str, copy_smpl: bool, rename_files: bool) -> List[tuple]:
    """Build the (src, dst) pairs for a sample's files, honoring rename_files and copy_smpl."""
    def _name(src_path: str, ext: str) -> str:
        if rename_files:
            return f'{sample.sample_id}{ext}'
        return os.path.basename(src_path)

    pairs = [
        (sample.image_path, os.path.join(dest_base_dir, 'images', _name(sample.image_path, '.jpg'))),
        (sample.densepose_path, os.path.join(dest_base_dir, 'densepose', _name(sample.densepose_path, '.png'))),
    ]
    if sample.smplx_path is not None:
        pairs.append((sample.smplx_path, os.path.join(dest_base_dir, 'smplx', _name(sample.smplx_path, '.h5'))))
    if copy_smpl and sample.smpl_path is not None and sample.smpl_path != sample.smplx_path:
        pairs.append(
            (sample.smpl_path, os.path.join(dest_base_dir, 'smpl', _name(sample.smpl_path, '.h5')))
        )
    return pairs


class MoveFilesAction(BaseAction):
    """Action that moves sample files to a destination directory."""

    def __init__(self, dest_base_dir: str, preserve_structure: bool = True,
                 copy_smpl: bool = False, rename_files: bool = True):
        """
        Args:
            dest_base_dir: Base destination directory
            preserve_structure: Whether to preserve the original directory structure
            copy_smpl: Whether to also move SMPL params (when available on the sample)
            rename_files: If True, destination filenames use sample_id; if False, keep source basenames
        """
        self.dest_base_dir = dest_base_dir
        self.preserve_structure = preserve_structure
        self.copy_smpl = copy_smpl
        self.rename_files = rename_files

    def execute(self, sample: Sample) -> bool:
        os.makedirs(self.dest_base_dir, exist_ok=True)
        """Move all files associated with the sample."""
        move_metadata = _build_dest_metadata(sample, self.dest_base_dir, self.copy_smpl, self.rename_files)

        for file_path, dest_path in move_metadata:
            if os.path.exists(file_path):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(file_path, dest_path)

        return True


class CopyFilesAction(BaseAction):
    """Action that copies (or symlinks) sample files to a destination directory."""

    def __init__(self, dest_base_dir: str, preserve_structure: bool = True,
                 copy_smpl: bool = False, rename_files: bool = True,
                 symlink: bool = False):
        """
        Args:
            symlink: If True, create absolute symlinks to the source files instead
                of copying bytes. Near-zero storage — ideal for many overlapping
                subsets on the same filesystem — but the output is NOT self-contained
                (breaks if the source moves) and cannot be tar'd/shared as-is.
        """
        self.dest_base_dir = dest_base_dir
        self.preserve_structure = preserve_structure
        self.copy_smpl = copy_smpl
        self.rename_files = rename_files
        self.symlink = symlink

    def execute(self, sample: Sample) -> bool:
        os.makedirs(self.dest_base_dir, exist_ok=True)
        os.makedirs(os.path.join(self.dest_base_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.dest_base_dir, 'smplx'), exist_ok=True)
        os.makedirs(os.path.join(self.dest_base_dir, 'densepose'), exist_ok=True)
        if self.copy_smpl:
            os.makedirs(os.path.join(self.dest_base_dir, 'smpl'), exist_ok=True)
        """Copy (or symlink) all files associated with the sample."""
        copy_metadata = _build_dest_metadata(sample, self.dest_base_dir, self.copy_smpl, self.rename_files)
        for file_path, dest_path in copy_metadata:
            if os.path.exists(file_path):
                if self.symlink:
                    # Absolute target so the link resolves from the dest dir;
                    # replace any stale link/file so re-runs are idempotent.
                    if os.path.islink(dest_path) or os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.symlink(os.path.abspath(file_path), dest_path)
                else:
                    shutil.copy2(file_path, dest_path)

        return True


class NoOpAction(BaseAction):
    """Action that does nothing - useful for testing."""
    
    def execute(self, sample: Sample) -> bool:
        """Do nothing."""
        return True 
