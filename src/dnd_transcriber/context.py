from typing import List, Tuple, Any


class ContextWindowManager:
    """Manages context windows for LLM validation of transcription segments."""
    
    def __init__(self, window_size: int, overlap: int):
        """Initialize context window manager.
        
        Args:
            window_size: Number of segments to include in context
            overlap: Number of overlapping segments between windows
        """
        self.window_size = window_size
        self.overlap = overlap
        
        if overlap >= window_size:
            raise ValueError("Overlap must be less than window size")
    
    def create_windows(self, segments: List[Any]) -> List[Tuple[Any, List[Any], List[Any]]]:
        """Create context windows for segments.
        
        Args:
            segments: List of segments to create windows for
            
        Returns:
            List of tuples: (current_segment, context_before, context_after)
        """
        if not segments:
            return []
        
        windows = []
        
        for i, segment in enumerate(segments):
            # Calculate context before
            before_start = max(0, i - self.window_size // 2)
            context_before = segments[before_start:i]
            
            # Calculate context after
            after_end = min(len(segments), i + 1 + self.window_size // 2)
            context_after = segments[i + 1:after_end]
            
            windows.append((segment, context_before, context_after))
        
        return windows
    
    def create_overlapping_windows(self, segments: List[Any]) -> List[List[Any]]:
        """Create overlapping windows of segments.
        
        Args:
            segments: List of segments to create windows for
            
        Returns:
            List of segment windows with overlap
        """
        if not segments:
            return []
        
        windows = []
        step = self.window_size - self.overlap
        
        for i in range(0, len(segments), step):
            window_end = min(i + self.window_size, len(segments))
            window = segments[i:window_end]
            
            if window:  # Only add non-empty windows
                windows.append(window)
            
            # Break if we've reached the end
            if window_end >= len(segments):
                break
        
        return windows