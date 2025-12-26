"""Visual test script for image processor and grid annotator."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image

from image_processor import ImageProcessor
from grid_annotator import GridAnnotator


def create_sample_image(path: str = "images/sample.png", size: int = 512):
    """Create a sample test image with a gradient and shapes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create a gradient background
    for i in range(size):
        for j in range(size):
            img[i, j] = [
                int(255 * i / size),  # Red gradient top to bottom
                int(255 * j / size),  # Green gradient left to right
                128,  # Constant blue
            ]
    
    # Add some rectangles for visual reference
    # Top-left quadrant - red box
    img[50:150, 50:150] = [255, 0, 0]
    # Top-right quadrant - green box
    img[50:150, 362:462] = [0, 255, 0]
    # Bottom-left quadrant - blue box
    img[362:462, 50:150] = [0, 0, 255]
    # Bottom-right quadrant - yellow box
    img[362:462, 362:462] = [255, 255, 0]
    
    # Center - white circle-ish
    center = size // 2
    for i in range(size):
        for j in range(size):
            if (i - center) ** 2 + (j - center) ** 2 < 50 ** 2:
                img[i, j] = [255, 255, 255]
    
    # Save
    Path(path).parent.mkdir(exist_ok=True)
    Image.fromarray(img).save(path)
    print(f"Created sample image: {path}")
    return path


def test_image_processor(image_path: str, grid_size: int = 4):
    """Test the image processor."""
    print(f"\n{'='*50}")
    print(f"Testing ImageProcessor with grid_size={grid_size}")
    print('='*50)
    
    processor = ImageProcessor(image_path, grid_size=grid_size, resize_to=512, seed=42)
    
    piece_h, piece_w = processor.get_piece_dimensions()
    img_h, img_w = processor.get_image_dimensions()
    print(f"Original image shape: {img_h}x{img_w}")
    print(f"Piece size: {piece_h}x{piece_w}")
    print(f"Total pieces: {processor.total_pieces}")
    
    # Save original assembled
    original = processor.get_current_state()
    Image.fromarray(original).save("results/01_original.png")
    print("Saved: results/01_original.png")
    
    # Shuffle
    processor.shuffle()
    shuffled = processor.get_current_state()
    Image.fromarray(shuffled).save("results/02_shuffled.png")
    print("Saved: results/02_shuffled.png")
    
    print(f"Correct pieces after shuffle: {processor.count_correct_pieces()}/{processor.total_pieces}")
    
    # Show the mapping
    print("\nShuffle mapping (current_pos -> original_pos):")
    solution = processor.get_solution_mapping()
    for current, original in list(solution.items())[:8]:
        status = "✓" if current == original else "✗"
        print(f"  {current} should have piece from {original} {status}")
    if len(solution) > 8:
        print(f"  ... and {len(solution) - 8} more")
    
    return processor


def test_grid_annotator(processor: ImageProcessor):
    """Test the grid annotator."""
    print(f"\n{'='*50}")
    print("Testing GridAnnotator")
    print('='*50)
    
    annotator = GridAnnotator(processor.grid_size)
    image = processor.get_current_state()
    
    # Test different annotation modes
    modes = ["border_labels", "cell_labels", "both"]
    
    for mode in modes:
        # Without colored borders
        annotated = annotator.annotate(image, mode=mode, colored_borders=False)
        path = f"results/03_annotated_{mode}.png"
        Image.fromarray(annotated).save(path)
        print(f"Saved: {path}")
        
        # With colored borders
        annotated_colored = annotator.annotate(image, mode=mode, colored_borders=True)
        path = f"results/04_annotated_{mode}_colored.png"
        Image.fromarray(annotated_colored).save(path)
        print(f"Saved: {path}")


def test_swaps(processor: ImageProcessor):
    """Test applying swaps."""
    print(f"\n{'='*50}")
    print("Testing Swaps")
    print('='*50)
    
    annotator = GridAnnotator(processor.grid_size)
    
    # Get solution and make some correct swaps
    solution = processor.get_solution_mapping()
    correct_before = processor.count_correct_pieces()
    print(f"Correct before swaps: {correct_before}")
    
    # Find a piece that needs to move and swap it
    swaps_to_make = []
    for current, original in solution.items():
        if current != original and len(swaps_to_make) < 2:
            # This piece at 'current' should be at 'original'
            swaps_to_make.append((current, original))
    
    if swaps_to_make:
        print(f"Making swaps: {swaps_to_make}")
        processor.apply_swaps(swaps_to_make)
        
        correct_after = processor.count_correct_pieces()
        print(f"Correct after swaps: {correct_after}")
        
        # Save annotated result
        image = processor.get_current_state()
        annotated = annotator.annotate(image, mode="both", colored_borders=True)
        Image.fromarray(annotated).save("results/05_after_swaps.png")
        print("Saved: results/05_after_swaps.png")


def main():
    """Run all visual tests."""
    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # Create or use sample image
    sample_path = "images/lena.jpg"
    if not Path(sample_path).exists():
        create_sample_image(sample_path)
    
    # Test with different grid sizes
    for grid_size in [4, 8]:
        processor = test_image_processor(sample_path, grid_size)
        
        if grid_size == 8:  # Only do detailed tests for 4x4
            test_grid_annotator(processor)
            test_swaps(processor)
    
    print(f"\n{'='*50}")
    print("All visual tests complete! Check the 'results/' directory.")
    print('='*50)


if __name__ == "__main__":
    main()