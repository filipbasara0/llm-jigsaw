#!/usr/bin/env python3
"""Streamlit app for human players to solve jigsaw puzzles."""

import streamlit as st
from PIL import Image
import numpy as np
from pathlib import Path
import io
import re
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_processor import ImageProcessor
from src.grid_annotator import GridAnnotator

# Page config
st.set_page_config(
    page_title="Jigsaw Puzzle Solver",
    page_icon="🧩",
    layout="wide",
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_available_images() -> list[Path]:
    """Get list of available images from the images directory."""
    images_dir = PROJECT_ROOT / "images"
    if not images_dir.exists():
        return []
    
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    images = [f for f in images_dir.iterdir() if f.suffix.lower() in extensions]
    return sorted(images)


def create_puzzle(
    image_source: str | Path | Image.Image,
    grid_rows: int,
    grid_cols: int,
    seed: Optional[int] = None,
) -> tuple[ImageProcessor, GridAnnotator]:
    """Create a new puzzle from an image."""
    # Handle uploaded image
    if isinstance(image_source, Image.Image):
        # Save to temp location
        temp_path = Path("/tmp/uploaded_puzzle_image.png")
        image_source.save(temp_path)
        image_source = temp_path
    
    processor = ImageProcessor(
        image_path=image_source,
        grid_size=(grid_rows, grid_cols),
        seed=seed,
        resize_to=800,  # Resize for consistent display
    )
    processor.shuffle()
    
    annotator = GridAnnotator(grid_size=(grid_rows, grid_cols))
    
    return processor, annotator


def get_annotated_image(processor: ImageProcessor, annotator: GridAnnotator) -> np.ndarray:
    """Get the current puzzle state with annotations."""
    current_state = processor.get_current_state()
    return annotator.annotate(
        image=current_state,
        mode="both",
        colored_borders=False,
    )


def get_reference_image(processor: ImageProcessor) -> np.ndarray:
    """Get the reference (solved) image without annotations."""
    return processor.get_original_image()


def parse_moves(move_text: str) -> list[tuple[str, str]]:
    """
    Parse move text into a list of swap pairs.
    
    Accepts formats like:
    - "1,1 2,2" (single swap)
    - "1,1 <-> 2,2" (with arrow)
    - "1,1 2,2; 3,3 4,4" (multiple swaps)
    - Multiple lines
    """
    moves = []
    
    # Split by newlines and semicolons
    lines = re.split(r'[;\n]', move_text)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove arrow notation
        line = re.sub(r'\s*<->\s*', ' ', line)
        line = re.sub(r'\s*->\s*', ' ', line)
        line = re.sub(r'\s+', ' ', line)  # Normalize spaces
        
        # Find all coordinates
        coords = re.findall(r'(\d+)\s*,\s*(\d+)', line)
        
        # Pair them up
        for i in range(0, len(coords) - 1, 2):
            coord_a = f"{coords[i][0]},{coords[i][1]}"
            coord_b = f"{coords[i+1][0]},{coords[i+1][1]}"
            moves.append((coord_a, coord_b))
    
    return moves


def initialize_session_state():
    """Initialize session state variables."""
    if "processor" not in st.session_state:
        st.session_state.processor = None
    if "annotator" not in st.session_state:
        st.session_state.annotator = None
    if "turn_number" not in st.session_state:
        st.session_state.turn_number = 0
    if "total_moves" not in st.session_state:
        st.session_state.total_moves = 0
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "last_message" not in st.session_state:
        st.session_state.last_message = None
    if "move_history" not in st.session_state:
        st.session_state.move_history = []
    if "form_key" not in st.session_state:
        st.session_state.form_key = 0


def reset_game():
    """Reset the game state."""
    st.session_state.processor = None
    st.session_state.annotator = None
    st.session_state.turn_number = 0
    st.session_state.total_moves = 0
    st.session_state.game_started = False
    st.session_state.game_over = False
    st.session_state.last_message = None
    st.session_state.move_history = []
    st.session_state.form_key = 0


def main():
    """Main app entry point."""
    initialize_session_state()
    
    st.title("🧩 Jigsaw Puzzle Solver")
    st.markdown(
        "Try to solve the jigsaw puzzle by swapping pieces! "
        "This is the same interface that LLMs see when solving puzzles."
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Game Settings")
        
        # Image source selection
        image_source_type = st.radio(
            "Image Source",
            ["Gallery", "Upload"],
            horizontal=True,
        )
        
        selected_image = None
        uploaded_image = None
        
        if image_source_type == "Gallery":
            available_images = get_available_images()
            if available_images:
                image_names = [img.name for img in available_images]
                selected_name = st.selectbox(
                    "Select Image",
                    image_names,
                    index=0,
                )
                selected_image = PROJECT_ROOT / "images" / selected_name
                
                # Preview
                with st.expander("Preview", expanded=True):
                    preview = Image.open(selected_image)
                    st.image(preview, use_container_width=True)
            else:
                st.warning("No images found in the 'images' directory.")
        else:
            uploaded_file = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file:
                uploaded_image = Image.open(uploaded_file).convert("RGB")
                st.image(uploaded_image, use_container_width=True)
        
        st.divider()
        
        # Grid settings
        st.subheader("Grid Size")
        col1, col2 = st.columns(2)
        with col1:
            grid_rows = st.number_input("Rows", min_value=2, max_value=10, value=3)
        with col2:
            grid_cols = st.number_input("Columns", min_value=2, max_value=10, value=3)
        
        # Game limits
        st.subheader("Game Limits")
        max_turns = st.number_input(
            "Max Turns",
            min_value=1,
            max_value=1000,
            value=100,
            help="Maximum number of turns before the game ends",
        )
        max_moves_per_turn = st.number_input(
            "Max Moves per Turn",
            min_value=1,
            max_value=50,
            value=16,
            help="Maximum swaps you can make in each turn",
        )
        
        # Optional seed
        use_seed = st.checkbox("Use fixed seed (reproducible shuffle)")
        seed = None
        if use_seed:
            seed = st.number_input("Seed", min_value=0, value=42)
        
        st.divider()
        
        # Start/Reset buttons
        can_start = selected_image is not None or uploaded_image is not None
        
        if st.button(
            "🎮 Start New Game",
            type="primary",
            disabled=not can_start,
            use_container_width=True,
        ):
            reset_game()
            
            image_to_use = selected_image if image_source_type == "Gallery" else uploaded_image
            
            try:
                processor, annotator = create_puzzle(
                    image_to_use,
                    grid_rows,
                    grid_cols,
                    seed,
                )
                st.session_state.processor = processor
                st.session_state.annotator = annotator
                st.session_state.game_started = True
                st.session_state.max_turns = max_turns
                st.session_state.max_moves_per_turn = max_moves_per_turn
                st.session_state.last_message = "Game started! Make your moves below."
                st.rerun()
            except Exception as e:
                st.error(f"Error creating puzzle: {e}")
        
        if st.session_state.game_started:
            if st.button("🔄 Reset Game", use_container_width=True):
                reset_game()
                st.rerun()
    
    # Main game area
    if not st.session_state.game_started:
        st.info("👈 Select an image and click 'Start New Game' to begin!")
        
        # Show instructions
        with st.expander("📖 How to Play", expanded=True):
            st.markdown("""
            ### Welcome to the Jigsaw Puzzle Solver!
            
            This app lets you solve jigsaw puzzles the same way AI models do:
            
            1. **Select an image** from the gallery or upload your own
            2. **Choose grid size** (e.g., 3×3 for 9 pieces)
            3. **Set game limits** for turns and moves per turn
            4. **Start the game** and try to solve the puzzle!
            
            ### Making Moves
            
            Enter your swaps in the format: `row,col row,col`
            
            For example, to swap piece at row 1, column 2 with piece at row 3, column 1:
            ```
            1,2 3,1
            ```
            
            You can make multiple swaps per turn (separated by `;` or newlines):
            ```
            1,2 3,1
            2,2 2,3
            ```
            
            ### Coordinate System
            
            - Coordinates are in **row,col** format (1-indexed)
            - Top-left is **1,1**, rows increase downward, columns increase rightward
            - Each cell is labeled with its coordinates
            
            ### Goal
            
            Rearrange all pieces to match the reference image. The game ends when:
            - ✅ You solve the puzzle (all pieces in correct positions)
            - ⏱️ You run out of turns
            """)
        return
    
    # Game in progress
    processor = st.session_state.processor
    annotator = st.session_state.annotator
    
    if processor is None:
        st.error("Game state error. Please start a new game.")
        return
    
    # Check if solved
    if processor.is_solved():
        st.session_state.game_over = True
    
    # Game status
    if st.session_state.game_over:
        if processor.is_solved():
            st.success(f"🎉 **Congratulations!** You solved the puzzle in {st.session_state.turn_number} turns with {st.session_state.total_moves} total moves!")
            st.balloons()
        else:
            st.error(f"⏱️ **Game Over!** You ran out of turns. The puzzle was not solved.")
    
    # Display message
    if st.session_state.last_message:
        if "error" in st.session_state.last_message.lower():
            st.warning(st.session_state.last_message)
        elif "success" in st.session_state.last_message.lower() or "applied" in st.session_state.last_message.lower():
            st.success(st.session_state.last_message)
        else:
            st.info(st.session_state.last_message)
    
    # Stats row
    col_stats = st.columns(4)
    with col_stats[0]:
        st.metric("Turn", f"{st.session_state.turn_number}/{st.session_state.max_turns}")
    with col_stats[1]:
        st.metric("Total Moves", st.session_state.total_moves)
    with col_stats[2]:
        st.metric("Grid Size", f"{processor.grid_rows}×{processor.grid_cols}")
    with col_stats[3]:
        st.metric("Total Pieces", processor.total_pieces)
    
    st.divider()
    
    # Image display
    col_puzzle, col_reference = st.columns(2)
    
    with col_puzzle:
        st.subheader("🧩 Current Puzzle State")
        puzzle_image = get_annotated_image(processor, annotator)
        st.image(puzzle_image, use_container_width=True)
    
    with col_reference:
        st.subheader("🖼️ Reference (Goal)")
        reference_image = get_reference_image(processor)
        st.image(reference_image, use_container_width=True)
    
    st.divider()
    
    # Move input (only if game not over)
    if not st.session_state.game_over:
        st.subheader("🎯 Make Your Move")
        
        with st.form(f"move_form_{st.session_state.form_key}"):
            st.markdown(
                f"Enter up to **{st.session_state.max_moves_per_turn}** swaps. "
                "Format: `row,col row,col` (one swap per line or separated by `;`)"
            )
            
            move_text = st.text_area(
                "Swaps",
                placeholder="1,2 3,1\n2,2 2,3",
                height=100,
                label_visibility="collapsed",
                key=f"move_input_{st.session_state.form_key}",
            )
            
            submitted = st.form_submit_button(
                "▶️ Apply Moves",
                type="primary",
                use_container_width=True,
            )
            
            if submitted:
                if not move_text.strip():
                    st.session_state.last_message = "⚠️ Please enter at least one move."
                else:
                    # Parse moves
                    moves = parse_moves(move_text)
                    
                    if not moves:
                        st.session_state.last_message = "⚠️ Could not parse any valid moves. Use format: row,col row,col"
                    else:
                        # Limit moves
                        moves = moves[:st.session_state.max_moves_per_turn]
                        
                        # Apply moves
                        successful, failed = processor.apply_swaps(moves)
                        
                        st.session_state.turn_number += 1
                        st.session_state.total_moves += successful
                        
                        # Add to history
                        st.session_state.move_history.append({
                            "turn": st.session_state.turn_number,
                            "moves": moves[:successful] if failed else moves,
                            "successful": successful,
                            "failed": failed,
                        })
                        
                        if failed > 0:
                            st.session_state.last_message = f"⚠️ Applied {successful} swaps, {failed} invalid moves skipped."
                        else:
                            st.session_state.last_message = f"✅ Successfully applied {successful} swap(s)."
                        
                        # Increment form key to clear the text area
                        st.session_state.form_key += 1
                        
                        # Check for end conditions
                        if processor.is_solved():
                            st.session_state.game_over = True
                        elif st.session_state.turn_number >= st.session_state.max_turns:
                            st.session_state.game_over = True
                        
                        st.rerun()
    
    # Move history
    if st.session_state.move_history:
        with st.expander("📜 Move History", expanded=False):
            for entry in reversed(st.session_state.move_history):
                moves_str = " | ".join([f"{a} ↔ {b}" for a, b in entry["moves"]])
                status = "✅" if entry["failed"] == 0 else f"⚠️ ({entry['failed']} failed)"
                st.markdown(f"**Turn {entry['turn']}:** {moves_str} {status}")


if __name__ == "__main__":
    main()
