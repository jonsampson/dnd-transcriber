from typing import List


def identify_low_confidence_segments(segments: list, threshold: float) -> List[int]:
    """Identify segments with confidence below threshold.
    
    Args:
        segments: List of segments with confidence scores
        threshold: Confidence threshold (0.0 to 1.0)
        
    Returns:
        List of segment indices with low confidence
    """
    low_confidence_indices = []
    
    for i, segment in enumerate(segments):
        confidence = getattr(segment, 'confidence', None)
        if confidence is not None and confidence < threshold:
            low_confidence_indices.append(i)
    
    return low_confidence_indices


def group_adjacent_segments(segments: List[int]) -> List[List[int]]:
    """Group adjacent segment indices into contiguous ranges.
    
    Args:
        segments: List of segment indices
        
    Returns:
        List of groups, where each group is a list of adjacent indices
    """
    if not segments:
        return []
    
    # Sort indices to ensure proper grouping
    sorted_segments = sorted(segments)
    groups = []
    current_group = [sorted_segments[0]]
    
    for i in range(1, len(sorted_segments)):
        current_idx = sorted_segments[i]
        previous_idx = sorted_segments[i - 1]
        
        # If adjacent (difference of 1), add to current group
        if current_idx == previous_idx + 1:
            current_group.append(current_idx)
        else:
            # Start new group
            groups.append(current_group)
            current_group = [current_idx]
    
    # Add the last group
    groups.append(current_group)
    
    return groups