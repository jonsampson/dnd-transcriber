import json
import re
from pathlib import Path
from typing import Dict, List, Optional


class CharacterRoster:
    """Manages character names for D&D session transcription."""
    
    def __init__(self):
        self.characters: Dict[str, str] = {}  # canonical_name -> variations
        self.player_names: List[str] = []
    
    def load_from_file(self, path: Path) -> None:
        """Load character roster from JSON file."""
        if not path.exists():
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.characters = data.get('characters', {})
        self.player_names = data.get('players', [])
    
    def find_closest_match(self, name: str) -> str:
        """Find closest character name using simple edit distance."""
        name = name.strip().lower()
        if not name:
            return name
        
        # Check exact matches first
        all_names = list(self.characters.keys()) + self.player_names
        for char_name in all_names:
            if name == char_name.lower():
                return char_name
        
        # Simple edit distance check
        best_match = name
        min_distance = float('inf')
        
        for char_name in all_names:
            distance = self._edit_distance(name, char_name.lower())
            if distance < min_distance and distance <= 2:  # Max 2 edits
                min_distance = distance
                best_match = char_name
        
        return best_match
    
    def correct_names_in_text(self, text: str) -> str:
        """Correct character names in transcribed text."""
        words = re.findall(r'\b\w+\b', text)
        corrected_text = text
        
        for word in words:
            if len(word) > 2:  # Only check words longer than 2 chars
                corrected = self.find_closest_match(word)
                if corrected != word and corrected.lower() != word.lower():
                    # Use word boundaries to avoid partial replacements
                    pattern = r'\b' + re.escape(word) + r'\b'
                    corrected_text = re.sub(pattern, corrected, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate simple edit distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]