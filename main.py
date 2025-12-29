#!/usr/bin/env python3
"""CLI entry point for the LLM Jigsaw Puzzle Solver."""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from src.game import PuzzleGame, GameConfig


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def get_api_key(provider: str, api_key: str | None) -> str:
    """Get API key from argument or environment."""
    if api_key:
        return api_key

    env_vars = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }

    env_var = env_vars.get(provider.lower())
    if env_var and os.environ.get(env_var):
        return os.environ[env_var]

    raise ValueError(f"No API key provided. Set --api-key or {env_var} environment variable.")


def parse_grid_size(value: str) -> int | tuple[int, int]:
    """
    Parse grid size argument.

    Accepts:
        - Single integer: "4" -> 4 (4x4 square grid)
        - Two integers with 'x': "3x5" -> (3, 5) (3 rows x 5 columns)

    Returns:
        int for square grids, tuple[int, int] for rectangular grids
    """
    value = value.strip().lower()

    if "x" in value:
        parts = value.split("x")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid grid size format: {value}. Use 'NxM' (e.g., '3x5') or single number (e.g., '4')"
            )
        try:
            rows = int(parts[0].strip())
            cols = int(parts[1].strip())
            if rows < 2 or cols < 2:
                raise argparse.ArgumentTypeError(
                    f"Grid dimensions must be at least 2. Got: {rows}x{cols}"
                )
            if rows == cols:
                return rows  # Return int for square grids
            return (rows, cols)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid grid size: {value}. Dimensions must be integers."
            )
    else:
        try:
            size = int(value)
            if size < 2:
                raise argparse.ArgumentTypeError(f"Grid size must be at least 2. Got: {size}")
            return size
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid grid size: {value}. Must be an integer or 'NxM' format."
            )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Jigsaw Puzzle Solver - Test multimodal LLM spatial reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with 4x4 grid
  python main.py --image puzzle.jpg --grid-size 4 --model gpt-4o

  # Rectangular 3x5 grid (3 rows, 5 columns)
  python main.py --image puzzle.jpg --grid-size 3x5 --model gpt-4o

  # Use Claude with hints enabled
  python main.py --image puzzle.jpg --grid-size 4 \\
    --provider anthropic --model claude-3-5-sonnet-20241022 \\
    --show-correct-count

  # Full options with 6x8 grid
  python main.py --image puzzle.jpg --grid-size 6x8 \\
    --provider openai --model gpt-4o \\
    --max-moves-per-turn 16 --max-turns 50 \\
    --annotation-mode both --colored-borders \\
    --show-correct-count --show-reference \\
    --seed 42 --output results/run1/
        """,
    )

    # Required arguments
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to the image file to use as puzzle"
    )
    parser.add_argument(
        "--grid-size",
        "-g",
        type=parse_grid_size,
        required=True,
        help="Size of the puzzle grid. Use single number for square (e.g., '4' for 4x4) "
        "or 'NxM' for rectangular (e.g., '3x5' for 3 rows, 5 columns)",
    )

    # LLM settings
    parser.add_argument(
        "--provider",
        "-p",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "google"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--model", "-m", type=str, default="gpt-4o", help="Model name (default: gpt-4o)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key (or set via environment variable)"
    )
    parser.add_argument(
        "--base-url", type=str, default=None, help="Base URL for OpenAI-compatible APIs"
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for OpenAI reasoning models (default: none)",
    )

    # Game limits
    parser.add_argument(
        "--max-moves-per-turn", type=int, default=16, help="Maximum swaps per turn (default: 16)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=50, help="Maximum number of turns (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible shuffling"
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize image so shorter side equals this value (e.g., 512, 1024)",
    )

    # Annotation settings
    parser.add_argument(
        "--annotation-mode",
        type=str,
        default="both",
        choices=["border_labels", "cell_labels", "both"],
        help="Type of grid annotations (default: both)",
    )
    parser.add_argument(
        "--colored-borders", action="store_true", help="Enable colored cell borders"
    )

    # Hint settings
    parser.add_argument(
        "--show-correct-count", action="store_true", help="Show how many pieces are correct"
    )
    parser.add_argument(
        "--show-reference", action="store_true", help="Provide the original image as reference"
    )
    parser.add_argument(
        "--annotate-reference",
        action="store_true",
        help="Add grid lines/coordinates to the reference image (makes puzzle easier)",
    )
    parser.add_argument(
        "--no-history", action="store_true", help="Don't include move history in prompts"
    )

    # Output settings
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save intermediate puzzle images"
    )
    parser.add_argument(
        "--no-gif", action="store_true", help="Don't save an animated GIF showing game evolution"
    )
    parser.add_argument(
        "--gif-duration",
        type=int,
        default=500,
        help="Duration of each frame in the GIF in milliseconds (default: 500)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Setup
    setup_logging(not args.quiet)
    logger = logging.getLogger(__name__)

    # Validate image path
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    # Get API key
    try:
        api_key = get_api_key(args.provider, args.api_key)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Default output directory
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(args.grid_size, tuple):
            grid_str = f"{args.grid_size[0]}x{args.grid_size[1]}"
        else:
            grid_str = f"{args.grid_size}x{args.grid_size}"
        output_dir = f"results/{args.model}_{grid_str}_{timestamp}"

    # Create config
    config = GameConfig(
        image_path=str(image_path),
        grid_size=args.grid_size,
        resize_to=args.resize,
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_moves_per_turn=args.max_moves_per_turn,
        max_turns=args.max_turns,
        shuffle_seed=args.seed,
        annotation_mode=args.annotation_mode,
        colored_borders=args.colored_borders,
        show_correct_count=args.show_correct_count,
        show_reference_image=args.show_reference,
        annotate_reference_image=args.annotate_reference,
        include_move_history=not args.no_history,
        output_dir=output_dir,
        save_intermediate_images=args.save_images,
        save_gif=not args.no_gif,
        gif_frame_duration=args.gif_duration,
        verbose=not args.quiet,
    )

    # Run the game
    try:
        game = PuzzleGame(config)
        result = game.run()

        # Print summary
        print("\n" + "=" * 50)
        print("PUZZLE GAME COMPLETE")
        print("=" * 50)
        print(f"Solved: {'Yes' if result.solved else 'No'}")
        print(f"Turns: {result.total_turns}")
        print(f"Final Score: {result.final_correct}/{result.total_pieces} ({result.accuracy:.1f}%)")
        print(
            f"Max Score: {result.max_correct_achieved}/{result.total_pieces} ({result.max_accuracy:.1f}%)"
        )
        print(f"Total Moves: {result.total_moves}")
        print(f"Invalid Moves: {result.total_invalid_moves}")
        print(f"Duration: {result.duration_seconds:.1f}s")
        print(f"Tokens Used: {result.total_tokens:,}")
        print(f"Results saved to: {output_dir}")
        print("=" * 50)

        # Exit code based on success
        sys.exit(0 if result.solved else 1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Game failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
