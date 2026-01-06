"""Prompt templates for the jigsaw puzzle solver."""

from typing import Optional


SYSTEM_PROMPT_TEMPLATE = """You are an expert puzzle solver. Your task is to solve a jigsaw puzzle by identifying misplaced pieces and swapping them to restore the original image.

**Iterative Approach:**
This puzzle is solved step-by-step over multiple turns. You do NOT need to solve it all at once!
- Focus on making steady progress each turn
- Analyze the current state carefully before making moves
- After each turn, you will see the updated puzzle state and can refine your approach
- Aim to get the puzzle into a better shape with each turn

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
{turn_info}
**Input Image:**
The attached image shows the puzzle grid with {grid_description} pieces. Each piece is in a cell of the grid, and pieces may be in incorrect positions. Your goal is to identify which pieces are misplaced and swap them to restore the original image.

{history_section}
{hints_section}

Analyze the image carefully and provide your next moves. Remember: you have multiple turns to solve this puzzle, so focus on making confident improvements this turn rather than trying to solve everything at once."""


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
    max_history_turns: int = 2,
    current_turn: Optional[int] = None,
    max_turns: Optional[int] = None,
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
        max_history_turns: Maximum number of recent turns to show in detail
        current_turn: Current turn number (1-indexed)
        max_turns: Maximum number of turns allowed

    Returns:
        Formatted user prompt
    """
    # Create grid description
    if grid_rows == grid_cols:
        grid_description = f"{grid_rows}×{grid_cols}"
    else:
        grid_description = f"{grid_rows}×{grid_cols} (rows×columns)"

    # Build turn info
    turn_info = ""
    if current_turn is not None and max_turns is not None:
        turns_remaining = max_turns - current_turn
        turn_info = f"\n**Turn {current_turn} of {max_turns}** ({turns_remaining} turn{'s' if turns_remaining != 1 else ''} remaining)\n"

    # Build history section
    history_section = ""
    if move_history:
        total_turns = len(move_history)
        recent_history = move_history[-max_history_turns:]
        start_turn = total_turns - len(recent_history) + 1

        history_lines = [f"**Progress (Turn {total_turns}):**"]

        # Add summary if we have enough history
        if total_turns >= 2:
            # Calculate progress trend
            scores = [
                t.get("correct_after") for t in move_history if t.get("correct_after") is not None
            ]
            if len(scores) >= 2:
                recent_change = scores[-1] - scores[-2]
                if recent_change > 0:
                    trend = f"📈 Last move improved by {recent_change} piece(s)"
                elif recent_change < 0:
                    trend = f"📉 Last move went backwards by {abs(recent_change)} piece(s) — reconsider your approach"
                else:
                    trend = "➡️ No change from last move — try a different strategy"
                history_lines.append(trend)

                if total_turns > max_history_turns:
                    history_lines.append(
                        f"(Overall: started at {scores[0]}, now at {scores[-1]} correct pieces)"
                    )

        # Show recent moves
        history_lines.append("")
        history_lines.append(f"**Recent Moves (last {len(recent_history)} turns):**")
        for i, turn in enumerate(recent_history):
            turn_num = start_turn + i
            moves = turn.get("moves", [])
            num_moves = len(moves)
            correct_str = f" → {turn['correct_after']} correct" if "correct_after" in turn else ""

            if num_moves == 0:
                continue
            elif num_moves <= 3:
                # Compact format for few moves
                moves_str = " | ".join(f"{a} <-> {b}" for a, b in moves)
                history_lines.append(f"  Turn {turn_num}: {moves_str}{correct_str}")
            else:
                # Multi-line format for many moves
                history_lines.append(f"  Turn {turn_num} ({num_moves} swaps){correct_str}:")
                for a, b in moves:
                    history_lines.append(f"    • {a} <-> {b}")

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
        turn_info=turn_info,
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
