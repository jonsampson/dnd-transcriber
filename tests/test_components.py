from dnd_transcriber.confidence import (
    group_adjacent_segments,
    identify_low_confidence_segments,
)
from dnd_transcriber.context import ContextWindowManager
from dnd_transcriber.models import Segment
from dnd_transcriber.roster import CharacterRoster


class TestCharacterRoster:
    """Test CharacterRoster functionality."""

    def test_find_closest_match_exact(self):
        """Test exact name matching."""
        roster = CharacterRoster()
        roster.characters = {"Gandalf": "wizard", "Frodo": "hobbit"}

        assert roster.find_closest_match("Gandalf") == "Gandalf"
        assert roster.find_closest_match("gandalf") == "Gandalf"

    def test_find_closest_match_fuzzy(self):
        """Test fuzzy name matching."""
        roster = CharacterRoster()
        roster.characters = {"Legolas": "elf", "Gimli": "dwarf"}

        assert roster.find_closest_match("Legols") == "Legolas"
        assert roster.find_closest_match("Gimly") == "Gimli"

    def test_find_closest_match_no_match(self):
        """Test when no close match exists."""
        roster = CharacterRoster()
        roster.characters = {"Aragorn": "ranger"}

        # Should return original if distance too large
        assert roster.find_closest_match("xyz") == "xyz"

    def test_correct_names_in_text(self):
        """Test text correction with character names."""
        roster = CharacterRoster()
        roster.characters = {"Thorin": "dwarf king"}

        corrected = roster.correct_names_in_text("Thoron attacks the orc")
        assert "Thorin" in corrected


class TestContextWindowManager:
    """Test ContextWindowManager functionality."""

    def test_create_windows_basic(self):
        """Test basic window creation."""
        manager = ContextWindowManager(window_size=3, overlap=1)
        segments = ["A", "B", "C", "D", "E"]

        windows = manager.create_windows(segments)
        assert len(windows) == 5

        # Check first window
        current, before, after = windows[0]
        assert current == "A"
        assert before == []
        assert after == ["B"]

    def test_create_windows_middle(self):
        """Test window creation for middle segments."""
        manager = ContextWindowManager(window_size=4, overlap=1)
        segments = ["A", "B", "C", "D", "E"]

        windows = manager.create_windows(segments)
        current, before, after = windows[2]  # Middle segment "C"

        assert current == "C"
        assert before == ["A", "B"]
        assert after == ["D", "E"]

    def test_create_overlapping_windows(self):
        """Test overlapping window creation."""
        manager = ContextWindowManager(window_size=3, overlap=1)
        segments = ["A", "B", "C", "D", "E"]

        windows = manager.create_overlapping_windows(segments)
        assert len(windows) == 2  # With step=2 (3-1), we get 2 windows: [0:3] and [2:5]
        assert windows[0] == ["A", "B", "C"]
        assert windows[1] == ["C", "D", "E"]

    def test_empty_segments(self):
        """Test with empty segment list."""
        manager = ContextWindowManager(window_size=3, overlap=1)
        windows = manager.create_windows([])
        assert windows == []


class TestConfidenceAnalyzer:
    """Test confidence analysis functions."""

    def test_identify_low_confidence_segments(self, sample_segments):
        """Test identification of low confidence segments."""
        # Modify confidence scores
        sample_segments[0].confidence = 0.9
        sample_segments[1].confidence = 0.5  # Low confidence

        low_indices = identify_low_confidence_segments(sample_segments, threshold=0.7)
        assert low_indices == [1]

    def test_identify_low_confidence_no_confidence(self):
        """Test with segments without confidence scores."""
        segments = [
            Segment(text="Test", speaker="A", start_time=0, end_time=1),
            Segment(text="Test2", speaker="B", start_time=1, end_time=2),
        ]

        low_indices = identify_low_confidence_segments(segments, threshold=0.7)
        assert low_indices == []

    def test_group_adjacent_segments(self):
        """Test grouping of adjacent segment indices."""
        indices = [1, 2, 3, 5, 7, 8, 10]
        groups = group_adjacent_segments(indices)

        assert len(groups) == 4
        assert groups[0] == [1, 2, 3]
        assert groups[1] == [5]
        assert groups[2] == [7, 8]
        assert groups[3] == [10]

    def test_group_adjacent_empty(self):
        """Test grouping with empty indices."""
        groups = group_adjacent_segments([])
        assert groups == []
