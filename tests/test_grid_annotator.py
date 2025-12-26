"""Tests for the grid annotator."""

import numpy as np
import pytest

from src.grid_annotator import GridAnnotator


class TestGridAnnotator:
    """Tests for GridAnnotator class."""
    
    def test_init(self):
        """Test basic initialization."""
        annotator = GridAnnotator(grid_size=8)
        
        assert annotator.grid_size == 8
    
    def test_generate_distinct_colors(self):
        """Test that colors are generated for all cells."""
        annotator = GridAnnotator(grid_size=4)
        
        assert len(annotator._colors) == 16  # 4x4 = 16 cells
        
        # Check colors are valid RGB tuples
        for color in annotator._colors:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_annotate_border_labels(self):
        """Test annotation with border labels."""
        annotator = GridAnnotator(grid_size=4)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        result = annotator.annotate(image, mode="border_labels")
        
        # Should add margin for labels
        assert result.shape[0] > 64
        assert result.shape[1] > 64
        assert result.shape[2] == 3
    
    def test_annotate_cell_labels(self):
        """Test annotation with cell labels."""
        annotator = GridAnnotator(grid_size=4)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        result = annotator.annotate(image, mode="cell_labels")
        
        # No margin added for cell labels only
        assert result.shape[0] == 64
        assert result.shape[1] == 64
    
    def test_annotate_both(self):
        """Test annotation with both label types."""
        annotator = GridAnnotator(grid_size=4)
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        result = annotator.annotate(image, mode="both")
        
        # Should add margin
        assert result.shape[0] > 64
        assert result.shape[1] > 64
    
    def test_annotate_colored_borders(self):
        """Test that colored borders are applied."""
        annotator = GridAnnotator(grid_size=4)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128  # Gray image
        
        result_with = annotator.annotate(image, mode="cell_labels", colored_borders=True)
        result_without = annotator.annotate(image, mode="cell_labels", colored_borders=False)
        
        # Results should differ
        assert not np.array_equal(result_with, result_without)
    
    def test_annotate_preserves_image_content(self):
        """Test that annotation preserves original image content area."""
        annotator = GridAnnotator(grid_size=4)
        
        # Create a distinctive image
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:32, :32] = [255, 0, 0]  # Red quadrant
        
        result = annotator.annotate(image, mode="cell_labels", colored_borders=False)
        
        # Check that red is still present (borders might modify edges)
        assert np.any(result[:, :, 0] > 200)  # Red channel high somewhere
    
    def test_coordinate_format_description(self):
        """Test coordinate format description generation."""
        annotator = GridAnnotator(grid_size=8)
        
        desc = annotator.get_coordinate_format_description()
        
        assert "1,1" in desc
        assert "8,8" in desc
        assert "row" in desc.lower()
        assert "col" in desc.lower()
    
    def test_different_grid_sizes(self):
        """Test annotation works for different grid sizes."""
        for grid_size in [2, 4, 8, 16]:
            annotator = GridAnnotator(grid_size=grid_size)
            image_size = grid_size * 16  # 16 pixels per piece
            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            
            result = annotator.annotate(image, mode="both")
            
            assert result.shape[2] == 3
            assert result.shape[0] > image_size  # Margin added
            assert result.shape[1] > image_size


class TestGridAnnotatorEdgeCases:
    """Edge case tests for GridAnnotator."""
    
    def test_small_pieces(self):
        """Test with very small pieces."""
        annotator = GridAnnotator(grid_size=8)
        # 8x8 grid with 8x8 image = 1 pixel per piece
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        
        # Should not crash even with tiny pieces
        result = annotator.annotate(image, mode="cell_labels")
        assert result is not None
    
    def test_large_grid(self):
        """Test with large grid size."""
        annotator = GridAnnotator(grid_size=32)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        
        result = annotator.annotate(image, mode="both")
        
        assert result is not None
        assert len(annotator._colors) == 1024  # 32x32
