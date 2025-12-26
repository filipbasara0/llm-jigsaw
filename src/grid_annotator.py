"""Grid annotation for puzzle visualization."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Literal, Optional
import colorsys


class GridAnnotator:
    """Adds visual annotations to puzzle images for LLM consumption."""
    
    def __init__(self, grid_size: int):
        """
        Initialize the grid annotator.
        
        Args:
            grid_size: Size of the grid (e.g., 8 for 8x8)
        """
        self.grid_size = grid_size
        self._colors = self._generate_distinct_colors()
    
    def _generate_distinct_colors(self) -> list[tuple[int, int, int]]:
        """Generate visually distinct colors for grid borders."""
        colors = []
        # Use HSV color space for better distribution
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Create a unique hue based on position
                hue = ((i * self.grid_size + j) * 0.618033988749895) % 1.0
                saturation = 0.7
                value = 0.9
                r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
                colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Get a font for drawing text, with fallback."""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except (OSError, IOError):
            try:
                return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
            except (OSError, IOError):
                try:
                    return ImageFont.truetype("arial.ttf", size)
                except (OSError, IOError):
                    return ImageFont.load_default()
    
    def annotate(
        self,
        image: np.ndarray,
        mode: Literal["border_labels", "cell_labels", "both"] = "both",
        colored_borders: bool = True,
        border_width: int = 2,
        label_color: tuple[int, int, int] = (255, 255, 255),
        label_bg_color: tuple[int, int, int] = (0, 0, 0),
    ) -> np.ndarray:
        """
        Add grid annotations to an image.
        
        Args:
            image: Input image as numpy array (H, W, 3)
            mode: Type of labels to add
            colored_borders: Whether to add colored borders to cells
            border_width: Width of grid lines/borders
            label_color: RGB color for label text
            label_bg_color: RGB color for label background
            
        Returns:
            Annotated image as numpy array
        """
        h, w = image.shape[:2]
        piece_h = h // self.grid_size
        piece_w = w // self.grid_size
        
        # Add margin for border labels
        margin = 30 if mode in ["border_labels", "both"] else 0
        
        # Create output image with margin
        output_h = h + margin
        output_w = w + margin
        output = np.ones((output_h, output_w, 3), dtype=np.uint8) * 40  # Dark gray background
        
        # Place the puzzle image
        output[margin:, margin:] = image
        
        # Convert to PIL for drawing
        pil_image = Image.fromarray(output)
        draw = ImageDraw.Draw(pil_image)
        
        # Draw colored borders if enabled
        if colored_borders:
            self._draw_colored_borders(draw, margin, piece_h, piece_w, border_width)
        else:
            self._draw_grid_lines(draw, margin, piece_h, piece_w, h, w, border_width)
        
        # Draw labels based on mode
        if mode in ["border_labels", "both"]:
            self._draw_border_labels(draw, margin, piece_h, piece_w)
        
        if mode in ["cell_labels", "both"]:
            self._draw_cell_labels(
                draw, margin, piece_h, piece_w, 
                label_color, label_bg_color
            )
        
        return np.array(pil_image)
    
    def _draw_grid_lines(
        self,
        draw: ImageDraw.ImageDraw,
        margin: int,
        piece_h: int,
        piece_w: int,
        total_h: int,
        total_w: int,
        border_width: int
    ) -> None:
        """Draw simple grid lines."""
        line_color = (128, 128, 128)
        
        # Horizontal lines
        for i in range(self.grid_size + 1):
            y = margin + i * piece_h
            draw.line([(margin, y), (margin + total_w, y)], fill=line_color, width=border_width)
        
        # Vertical lines
        for j in range(self.grid_size + 1):
            x = margin + j * piece_w
            draw.line([(x, margin), (x, margin + total_h)], fill=line_color, width=border_width)
    
    def _draw_colored_borders(
        self,
        draw: ImageDraw.ImageDraw,
        margin: int,
        piece_h: int,
        piece_w: int,
        border_width: int
    ) -> None:
        """Draw colored borders around each cell."""
        # Adjust border width for very small pieces
        effective_border = min(border_width, piece_h // 2, piece_w // 2)
        if effective_border < 1:
            effective_border = 1
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                color_idx = row * self.grid_size + col
                color = self._colors[color_idx]
                
                x1 = margin + col * piece_w
                y1 = margin + row * piece_h
                x2 = x1 + piece_w
                y2 = y1 + piece_h
                
                # Draw rectangle border (skip if piece is too small)
                for i in range(effective_border):
                    if x1 + i < x2 - i and y1 + i < y2 - i:
                        draw.rectangle(
                            [x1 + i, y1 + i, x2 - i, y2 - i],
                            outline=color
                        )
    
    def _draw_border_labels(
        self,
        draw: ImageDraw.ImageDraw,
        margin: int,
        piece_h: int,
        piece_w: int
    ) -> None:
        """Draw row and column labels on the borders."""
        font_size = max(10, min(margin - 4, 20))
        font = self._get_font(font_size)
        text_color = (200, 200, 200)
        
        # Column labels (top)
        for col in range(self.grid_size):
            x = margin + col * piece_w + piece_w // 2
            y = margin // 2
            label = str(col + 1)
            
            # Get text size for centering
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.text(
                (x - text_w // 2, y - text_h // 2),
                label,
                fill=text_color,
                font=font
            )
        
        # Row labels (left)
        for row in range(self.grid_size):
            x = margin // 2
            y = margin + row * piece_h + piece_h // 2
            label = str(row + 1)
            
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            draw.text(
                (x - text_w // 2, y - text_h // 2),
                label,
                fill=text_color,
                font=font
            )
    
    def _draw_cell_labels(
        self,
        draw: ImageDraw.ImageDraw,
        margin: int,
        piece_h: int,
        piece_w: int,
        text_color: tuple[int, int, int],
        bg_color: tuple[int, int, int]
    ) -> None:
        """Draw coordinate labels in each cell corner."""
        # Adjust font size based on piece size
        font_size = max(8, min(piece_h // 6, piece_w // 4, 16))
        font = self._get_font(font_size)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                label = f"{row + 1},{col + 1}"
                
                x = margin + col * piece_w + 2
                y = margin + row * piece_h + 2
                
                # Get text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                
                # Draw background rectangle
                padding = 2
                draw.rectangle(
                    [x, y, x + text_w + padding * 2, y + text_h + padding * 2],
                    fill=bg_color + (180,)  # Semi-transparent
                )
                
                # Draw text
                draw.text(
                    (x + padding, y + padding),
                    label,
                    fill=text_color,
                    font=font
                )
    
    def get_coordinate_format_description(self) -> str:
        """Get a description of the coordinate format for prompts."""
        return (
            f'Coordinates are in "row,col" format where row and column are 1-indexed. '
            f'Top-left is "1,1", bottom-right is "{self.grid_size},{self.grid_size}". '
            f"Rows increase downward, columns increase rightward."
        )
