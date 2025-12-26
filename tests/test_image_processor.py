"""Tests for the image processor."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from PIL import Image

from src.image_processor import ImageProcessor


@pytest.fixture
def sample_image_path():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Create a 64x64 image with distinct quadrants
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :32] = [255, 0, 0]    # Top-left: red
        img[:32, 32:] = [0, 255, 0]    # Top-right: green
        img[32:, :32] = [0, 0, 255]    # Bottom-left: blue
        img[32:, 32:] = [255, 255, 0]  # Bottom-right: yellow
        
        Image.fromarray(img).save(f.name)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestImageProcessor:
    """Tests for ImageProcessor class."""
    
    def test_init(self, sample_image_path):
        """Test basic initialization."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        assert processor.grid_size == 4
        assert processor.total_pieces == 16
        # Note: Before shuffling, all pieces are in correct position, so is_solved() is True
    
    def test_initial_state_is_solved(self, sample_image_path):
        """Test that initial state (before shuffle) is solved."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        # Before shuffling, all pieces should be in correct position
        assert processor.count_correct_pieces() == 16
        assert processor.is_solved()
    
    def test_shuffle(self, sample_image_path):
        """Test that shuffling changes the arrangement."""
        processor = ImageProcessor(sample_image_path, grid_size=4, seed=42)
        
        initial_correct = processor.count_correct_pieces()
        assert initial_correct == 16
        
        processor.shuffle()
        
        # After shuffling, should have fewer correct pieces (with high probability)
        shuffled_correct = processor.count_correct_pieces()
        assert shuffled_correct < 16  # Very unlikely to shuffle to same arrangement
    
    def test_shuffle_reproducibility(self, sample_image_path):
        """Test that same seed produces same shuffle."""
        processor1 = ImageProcessor(sample_image_path, grid_size=4, seed=123)
        processor2 = ImageProcessor(sample_image_path, grid_size=4, seed=123)
        
        processor1.shuffle()
        processor2.shuffle()
        
        # Same seed should produce identical arrangements
        mapping1 = processor1.get_solution_mapping()
        mapping2 = processor2.get_solution_mapping()
        
        assert mapping1 == mapping2
    
    def test_parse_coordinate_valid(self, sample_image_path):
        """Test parsing valid coordinates."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        assert processor._parse_coordinate("1,1") == (0, 0)
        assert processor._parse_coordinate("4,4") == (3, 3)
        assert processor._parse_coordinate("2, 3") == (1, 2)  # With space
        assert processor._parse_coordinate(" 1,1 ") == (0, 0)  # With whitespace
    
    def test_parse_coordinate_invalid(self, sample_image_path):
        """Test parsing invalid coordinates."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        with pytest.raises(ValueError):
            processor._parse_coordinate("0,0")  # Out of bounds (too low)
        
        with pytest.raises(ValueError):
            processor._parse_coordinate("5,1")  # Out of bounds (too high)
        
        with pytest.raises(ValueError):
            processor._parse_coordinate("abc")  # Invalid format
        
        with pytest.raises(ValueError):
            processor._parse_coordinate("1,2,3")  # Too many parts
    
    def test_apply_swap(self, sample_image_path):
        """Test applying a single swap."""
        processor = ImageProcessor(sample_image_path, grid_size=4, seed=42)
        processor.shuffle()
        
        # Get initial state
        initial_mapping = processor.get_solution_mapping()
        
        # Apply a swap
        success = processor.apply_swap("1,1", "2,2")
        assert success
        
        # Verify the swap occurred
        new_mapping = processor.get_solution_mapping()
        assert initial_mapping["1,1"] == new_mapping["2,2"]
        assert initial_mapping["2,2"] == new_mapping["1,1"]
    
    def test_apply_swap_invalid(self, sample_image_path):
        """Test applying invalid swaps."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        # Invalid coordinates should return False
        assert processor.apply_swap("0,0", "1,1") == False
        assert processor.apply_swap("5,5", "1,1") == False
        assert processor.apply_swap("abc", "1,1") == False
    
    def test_apply_swaps_batch(self, sample_image_path):
        """Test applying multiple swaps."""
        processor = ImageProcessor(sample_image_path, grid_size=4, seed=42)
        processor.shuffle()
        
        swaps = [
            ("1,1", "2,2"),
            ("3,3", "4,4"),
            ("1,1", "invalid"),  # One invalid
        ]
        
        successful, failed = processor.apply_swaps(swaps)
        
        assert successful == 2
        assert failed == 1
    
    def test_get_current_state_dimensions(self, sample_image_path):
        """Test that current state has correct dimensions."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        state = processor.get_current_state()
        
        assert state.shape == (64, 64, 3)
    
    def test_piece_dimensions(self, sample_image_path):
        """Test piece dimension calculation."""
        processor = ImageProcessor(sample_image_path, grid_size=4)
        
        piece_h, piece_w = processor.get_piece_dimensions()
        
        assert piece_h == 16  # 64 / 4
        assert piece_w == 16
    
    def test_copy(self, sample_image_path):
        """Test copying processor state."""
        processor = ImageProcessor(sample_image_path, grid_size=4, seed=42)
        processor.shuffle()
        
        # Apply a swap
        processor.apply_swap("1,1", "2,2")
        
        # Copy
        copy = processor.copy()
        
        # Verify copy has same state
        assert processor.count_correct_pieces() == copy.count_correct_pieces()
        assert processor.get_solution_mapping() == copy.get_solution_mapping()
        
        # Modify original
        processor.apply_swap("3,3", "4,4")
        
        # Copy should be unchanged
        assert processor.get_solution_mapping() != copy.get_solution_mapping()
    
    def test_solve_by_swapping(self, sample_image_path):
        """Test that puzzle can be solved by swapping."""
        processor = ImageProcessor(sample_image_path, grid_size=2, seed=42)
        processor.shuffle()
        
        # Get solution mapping and solve
        mapping = processor.get_solution_mapping()
        
        # Simple solve: swap pieces until solved (brute force for small grid)
        max_iterations = 100
        for _ in range(max_iterations):
            if processor.is_solved():
                break
            
            # Find a misplaced piece and swap it to correct position
            mapping = processor.get_solution_mapping()
            for current, correct in mapping.items():
                if current != correct:
                    # Find which piece is at the correct position
                    for c2, cor2 in mapping.items():
                        if c2 == correct:
                            processor.apply_swap(current, c2)
                            break
                    break
        
        assert processor.is_solved()
