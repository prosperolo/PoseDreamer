from abc import ABC, abstractmethod
from typing import Dict, Any
from posedreamer.filtering.sample import Sample


class BaseFilter(ABC):
    """Abstract base class for all filtering methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {}
        self.reset_stats()
    
    @abstractmethod
    def validate(self, sample: Sample) -> bool:
        """
        Validate a sample.
        
        Args:
            sample: Sample to validate
            
        Returns:
            True if sample passes validation, False otherwise
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collected statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics to initial state."""
        self.stats = {
            'total_processed': 0,
            'total_passed': 0,
            'total_failed': 0
        }
    
    def _update_stats(self, passed: bool):
        """Update internal statistics."""
        self.stats['total_processed'] += 1
        if passed:
            self.stats['total_passed'] += 1
        else:
            self.stats['total_failed'] += 1
    
    def __call__(self, sample: Sample) -> bool:
        """Make the filter callable."""
        result = self.validate(sample)
        self._update_stats(result)
        return result
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')" 