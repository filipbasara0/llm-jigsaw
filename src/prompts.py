"""Prompt templates for the jigsaw puzzle solver."""

from typing import Optional


SYSTEM_PROMPT_TEMPLATE = """You are an expert puzzle solver. Your task is to solve a jigsaw puzzle by identifying misplaced pieces and swapping them to restore the original image.

**Coordinate System:**
{coordinate_description}

**Rules:**
- You may make up to {max_moves} swaps per turn
- Each swap exchanges two pieces at the specified coordinates
- You must respond with valid JSON only

**Response Format:**
```json
{{
  "reasoning": "Brief explanation of your analysis and moves",
  "moves": [
    {{"op": "swap", "a": "row,col", "b": "row,col"}},
    {{"op": "swap", "a": "row,col", "b": "row,col"}}
  ]
}}
```
"""

USER_PROMPT_TEMPLATE = """Here is the current state of the {grid_description} puzzle.

**Input Image:**
The attached image shows the puzzle grid with {grid_description} pieces. Each piece is in a cell of the grid, and pieces may be in incorrect positions. Your goal is to identify which pieces are misplaced and swap them to restore the original image.

{history_section}
{hints_section}

Analyze the image and provide your next moves to solve the puzzle."""


def build_system_prompt(
    coordinate_description: str,
    max_moves: int,
) -> str:
    """
    Build the system prompt for the LLM.

    Args:
        coordinate_description: Description of the coordinate system
        max_moves: Maximum moves allowed per turn

    Returns:
        Formatted system prompt
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        coordinate_description=coordinate_description,
        max_moves=max_moves,
    )


def build_user_prompt(
    grid_rows: int,
    grid_cols: int,
    move_history: Optional[list[dict]] = None,
    show_correct_count: bool = False,
    correct_count: Optional[int] = None,
    total_pieces: Optional[int] = None,
    has_reference_image: bool = False,
) -> str:
    """
    Build the user prompt for the LLM.

    Args:
        grid_rows: Number of rows in the puzzle grid
        grid_cols: Number of columns in the puzzle grid
        move_history: List of previous turns with moves
        show_correct_count: Whether to show how many pieces are correct
        correct_count: Number of correctly placed pieces
        total_pieces: Total number of pieces
        has_reference_image: Whether a reference image is provided

    Returns:
        Formatted user prompt
    """
    # Create grid description
    if grid_rows == grid_cols:
        grid_description = f"{grid_rows}×{grid_cols}"
    else:
        grid_description = f"{grid_rows}×{grid_cols} (rows×columns)"

    # Build history section
    history_section = ""
    if move_history:
        history_lines = ["**Previous Moves:**"]
        for i, turn in enumerate(move_history, 1):
            moves_str = ", ".join(f"{a} <-> {b}" for a, b in turn.get("moves", []))
            if moves_str:
                line = f"Turn {i}: {moves_str}"
                if "correct_after" in turn:
                    line += f" (pieces correct after: {turn['correct_after']})"
                history_lines.append(line)
        history_section = "\n".join(history_lines) + "\n"

    # Build hints section
    hints_section = ""
    hints = []

    if show_correct_count and correct_count is not None:
        incorrect_count = total_pieces - correct_count
        hints.append(
            f"Currently {correct_count}/{total_pieces} pieces are in the correct position. "
            f"There are still {incorrect_count} pieces that need to be swapped, provide moves to fix them."
        )

    if has_reference_image:
        hints.append("A reference image of the solved puzzle is provided as the second image.")

    if hints:
        hints_section = "**Hints:**\n" + "\n".join(f"- {h}" for h in hints) + "\n"

    return USER_PROMPT_TEMPLATE.format(
        grid_description=grid_description,
        history_section=history_section,
        hints_section=hints_section,
    )


def format_move_history_entry(
    turn_number: int,
    moves: list[tuple[str, str]],
    correct_after: Optional[int] = None,
) -> dict:
    """
    Format a move history entry.

    Args:
        turn_number: The turn number
        moves: List of (coord_a, coord_b) swaps made
        correct_after: Number of correct pieces after the turn

    Returns:
        Dict with turn information
    """
    entry = {
        "turn": turn_number,
        "moves": moves,
    }
    if correct_after is not None:
        entry["correct_after"] = correct_after
    return entry
