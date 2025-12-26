"""Image processing for jigsaw puzzle creation and manipulation."""

import numpy as np
from PIL import Image
from pathlib import Path
import random
from typing import Optional
import copy


class ImageProcessor:
    """Handles image slicing, shuffling, and state management for jigsaw puzzles."""
    
    def __init__(
        self, 
        image_path: str | Path, 
        grid_size: int, 
        seed: Optional[int] = None,
        resize_to: Optional[int | tuple[int, int]] = None,
    ):
        """
        Initialize the image processor.
        
        Args:
            image_path: Path to the source image
            grid_size: Size of the grid (e.g., 8 for 8x8)
            seed: Random seed for reproducible shuffling
            resize_to: Optional resize target. Can be:
                - int: Resize so the shorter side equals this value (maintains aspect ratio)
                - tuple[int, int]: Resize to exact (width, height)
                - None: No resizing (default)
        """
        self.image_path = Path(image_path)
        self.grid_size = grid_size
        self.seed = seed
        self.resize_to = resize_to
        
        # Load and prepare the image
        self._original_image = self._load_and_prepare_image()
        self._piece_height, self._piece_width = self._calculate_piece_dimensions()
        
        # Slice into pieces
        self._pieces = self._slice_image()
        
        # Current arrangement: maps current position to piece index
        # piece index corresponds to the original/correct position
        # Position format: (row, col) 0-indexed internally
        self._current_arrangement: list[list[int]] = [
            [row * grid_size + col for col in range(grid_size)]
            for row in range(grid_size)
        ]
        
        # Track if shuffled
        self._is_shuffled = False
    
    def _load_and_prepare_image(self) -> np.ndarray:
        """Load image, optionally resize, and ensure it can be evenly divided."""
        img = Image.open(self.image_path).convert("RGB")
        
        # Resize if requested
        if self.resize_to is not None:
            img = self._resize_image(img)
        
        img_array = np.array(img)
        
        # Crop to make dimensions divisible by grid_size
        h, w = img_array.shape[:2]
        new_h = (h // self.grid_size) * self.grid_size
        new_w = (w // self.grid_size) * self.grid_size
        
        # Center crop
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        
        return img_array[start_h:start_h + new_h, start_w:start_w + new_w]
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image according to resize_to parameter.
        
        Args:
            img: PIL Image to resize
            
        Returns:
            Resized PIL Image
        """
        if isinstance(self.resize_to, int):
            # Resize so shorter side equals resize_to, maintaining aspect ratio
            w, h = img.size
            if w < h:
                new_w = self.resize_to
                new_h = int(h * (self.resize_to / w))
            else:
                new_h = self.resize_to
                new_w = int(w * (self.resize_to / h))
            return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            # Resize to exact dimensions (width, height)
            return img.resize(self.resize_to, Image.Resampling.LANCZOS)
    
    def _calculate_piece_dimensions(self) -> tuple[int, int]:
        """Calculate the dimensions of each puzzle piece."""
        h, w = self._original_image.shape[:2]
        return h // self.grid_size, w // self.grid_size
    
    def _slice_image(self) -> list[np.ndarray]:
        """Slice the image into grid pieces."""
        pieces = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                y1 = row * self._piece_height
                y2 = y1 + self._piece_height
                x1 = col * self._piece_width
                x2 = x1 + self._piece_width
                piece = self._original_image[y1:y2, x1:x2].copy()
                pieces.append(piece)
        return pieces
    
    def shuffle(self) -> None:
        """Shuffle the puzzle pieces randomly."""
        if self.seed is not None:
            random.seed(self.seed)
        
        # Flatten, shuffle, reshape
        flat = [
            self._current_arrangement[r][c] 
            for r in range(self.grid_size) 
            for c in range(self.grid_size)
        ]
        random.shuffle(flat)
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self._current_arrangement[r][c] = flat[r * self.grid_size + c]
        
        self._is_shuffled = True
    
    def _parse_coordinate(self, coord: str) -> tuple[int, int]:
        """
        Parse a coordinate string to (row, col) 0-indexed.
        
        Args:
            coord: Coordinate in "row,col" format (1-indexed)
            
        Returns:
            Tuple of (row, col) 0-indexed
            
        Raises:
            ValueError: If coordinate format is invalid or out of bounds
        """
        try:
            parts = coord.strip().split(",")
            if len(parts) != 2:
                raise ValueError(f"Invalid coordinate format: {coord}")
            
            row = int(parts[0].strip()) - 1  # Convert to 0-indexed
            col = int(parts[1].strip()) - 1
            
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                raise ValueError(
                    f"Coordinate {coord} out of bounds for grid size {self.grid_size}"
                )
            
            return row, col
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse coordinate '{coord}': {e}")
    
    def apply_swap(self, coord_a: str, coord_b: str) -> bool:
        """
        Swap two pieces by their coordinates.
        
        Args:
            coord_a: First coordinate in "row,col" format (1-indexed)
            coord_b: Second coordinate in "row,col" format (1-indexed)
            
        Returns:
            True if swap was successful, False otherwise
        """
        try:
            row_a, col_a = self._parse_coordinate(coord_a)
            row_b, col_b = self._parse_coordinate(coord_b)
            
            # Perform the swap
            self._current_arrangement[row_a][col_a], self._current_arrangement[row_b][col_b] = \
                self._current_arrangement[row_b][col_b], self._current_arrangement[row_a][col_a]
            
            return True
        except ValueError:
            return False
    
    def apply_swaps(self, swaps: list[tuple[str, str]]) -> tuple[int, int]:
        """
        Apply multiple swaps.
        
        Args:
            swaps: List of (coord_a, coord_b) tuples
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        successful = 0
        failed = 0
        
        for coord_a, coord_b in swaps:
            if self.apply_swap(coord_a, coord_b):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def get_current_state(self) -> np.ndarray:
        """
        Get the current puzzle state as an image.
        
        Returns:
            Numpy array of the current puzzle arrangement
        """
        h = self._piece_height * self.grid_size
        w = self._piece_width * self.grid_size
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                piece_idx = self._current_arrangement[row][col]
                piece = self._pieces[piece_idx]
                
                y1 = row * self._piece_height
                y2 = y1 + self._piece_height
                x1 = col * self._piece_width
                x2 = x1 + self._piece_width
                
                result[y1:y2, x1:x2] = piece
        
        return result
    
    def get_original_image(self) -> np.ndarray:
        """Get the original (solved) image."""
        return self._original_image.copy()
    
    def get_solution_mapping(self) -> dict[str, str]:
        """
        Get mapping from current positions to correct positions.
        
        Returns:
            Dict mapping current "row,col" to correct "row,col" (1-indexed)
        """
        mapping = {}
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                piece_idx = self._current_arrangement[row][col]
                correct_row = piece_idx // self.grid_size
                correct_col = piece_idx % self.grid_size
                
                current_coord = f"{row + 1},{col + 1}"
                correct_coord = f"{correct_row + 1},{correct_col + 1}"
                
                mapping[current_coord] = correct_coord
        
        return mapping
    
    def count_correct_pieces(self) -> int:
        """Count how many pieces are in their correct position."""
        correct = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                piece_idx = self._current_arrangement[row][col]
                expected_idx = row * self.grid_size + col
                if piece_idx == expected_idx:
                    correct += 1
        return correct
    
    def is_solved(self) -> bool:
        """Check if the puzzle is completely solved."""
        return self.count_correct_pieces() == self.grid_size ** 2
    
    def get_piece_dimensions(self) -> tuple[int, int]:
        """Get the dimensions of each piece (height, width)."""
        return self._piece_height, self._piece_width
    
    def get_image_dimensions(self) -> tuple[int, int]:
        """Get the dimensions of the full image (height, width)."""
        return self._original_image.shape[:2]
    
    def copy(self) -> "ImageProcessor":
        """Create a deep copy of the current state."""
        new_processor = ImageProcessor.__new__(ImageProcessor)
        new_processor.image_path = self.image_path
        new_processor.grid_size = self.grid_size
        new_processor.seed = self.seed
        new_processor.resize_to = self.resize_to
        new_processor._original_image = self._original_image
        new_processor._piece_height = self._piece_height
        new_processor._piece_width = self._piece_width
        new_processor._pieces = self._pieces  # Pieces don't change, can share
        new_processor._current_arrangement = copy.deepcopy(self._current_arrangement)
        new_processor._is_shuffled = self._is_shuffled
        return new_processor
    
    @property
    def total_pieces(self) -> int:
        """Total number of pieces in the puzzle."""
        return self.grid_size ** 2
