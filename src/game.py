"""Main game controller for the jigsaw puzzle solver."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from PIL import Image
import numpy as np

from .image_processor import ImageProcessor
from .grid_annotator import GridAnnotator
from .llm_interface import LLMInterface
from .metrics import GameResult, TurnResult, MetricsTracker
from .prompts import (
    build_system_prompt,
    build_user_prompt,
    format_move_history_entry,
)

logger = logging.getLogger(__name__)


@dataclass
class GameConfig:
    """Configuration for a puzzle game."""

    # Required
    image_path: str
    grid_size: (
        int | tuple[int, int]
    )  # Single int for square (e.g., 4), tuple for rectangular (e.g., (3, 5))

    # Image settings
    resize_to: Optional[int | tuple[int, int]] = None  # Resize image before processing

    # LLM settings
    provider: str = "openai"
    model: str = "gpt-5.2"
    api_key: str = ""
    base_url: Optional[str] = None
    reasoning_effort: str = "none"  # "none", "low", "medium", "high" for reasoning models

    # Game limits
    max_moves_per_turn: int = 16
    max_turns: int = 100
    shuffle_seed: Optional[int] = None

    # Annotation settings
    annotation_mode: Literal["border_labels", "cell_labels", "both"] = "both"
    colored_borders: bool = False

    # Hint settings
    show_correct_count: bool = True
    show_reference_image: bool = False
    annotate_reference_image: bool = (
        False  # If True, reference image includes grid lines/coordinates
    )
    include_move_history: bool = False

    # Output settings
    output_dir: Optional[str] = None
    save_intermediate_images: bool = False
    save_gif: bool = False
    gif_frame_duration: int = 500  # milliseconds per frame
    verbose: bool = True


class PuzzleGame:
    """Orchestrates the puzzle-solving game loop."""

    def __init__(self, config: GameConfig):
        """
        Initialize the puzzle game.

        Args:
            config: Game configuration
        """
        self.config = config

        # Initialize components
        self.processor = ImageProcessor(
            image_path=config.image_path,
            grid_size=config.grid_size,
            seed=config.shuffle_seed,
            resize_to=config.resize_to,
        )
        self.annotator = GridAnnotator(grid_size=config.grid_size)
        self.llm = LLMInterface(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
        )

        # Shuffle the puzzle
        self.processor.shuffle()

        # Initialize tracking
        self.metrics = MetricsTracker(total_pieces=self.processor.total_pieces)
        self.move_history: list[dict] = []
        self.turn_number = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._gif_frames: list[Image.Image] = []  # Frames for GIF animation

        # Output directory
        if config.output_dir:
            self.output_dir = Path(config.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

    def _get_annotated_image(self) -> np.ndarray:
        """Get the current puzzle state with annotations."""
        current_state = self.processor.get_current_state()
        return self.annotator.annotate(
            image=current_state,
            mode=self.config.annotation_mode,
            colored_borders=self.config.colored_borders,
        )

    def _get_reference_image(self) -> Optional[np.ndarray]:
        """Get the reference (solved) image if enabled."""
        if not self.config.show_reference_image:
            return None

        original = self.processor.get_original_image()

        if self.config.annotate_reference_image:
            return self.annotator.annotate(
                image=original,
                mode=self.config.annotation_mode,
                colored_borders=self.config.colored_borders,
            )
        else:
            # Return plain image without grid lines or coordinates
            return original

    def _save_image(self, image: np.ndarray, name: str) -> None:
        """Save an image to the output directory."""
        if self.output_dir is None:
            return

        path = self.output_dir / name
        Image.fromarray(image).save(path)

    def _capture_gif_frame(self, image: np.ndarray) -> None:
        """Capture a frame for the GIF animation."""
        if not self.config.save_gif:
            return

        # Convert to PIL Image and store
        pil_image = Image.fromarray(image)
        # Convert to palette mode for smaller GIF size
        self._gif_frames.append(pil_image.convert("P", palette=Image.ADAPTIVE, colors=256))

    def _save_gif(self) -> None:
        """Save the captured frames as an animated GIF."""
        if not self.config.save_gif or not self._gif_frames or self.output_dir is None:
            return

        gif_path = self.output_dir / "game_evolution.gif"

        # Save with the first frame, appending the rest
        self._gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=self._gif_frames[1:],
            duration=self.config.gif_frame_duration,
            loop=0,  # Loop forever
            optimize=True,
        )

        if self.config.verbose:
            logger.info(f"GIF saved to: {gif_path} ({len(self._gif_frames)} frames)")

    def step(self) -> TurnResult:
        """
        Execute a single turn of the puzzle game.

        Returns:
            TurnResult with details of the turn
        """
        self.turn_number += 1
        correct_before = self.processor.count_correct_pieces()

        if self.config.verbose:
            logger.info(
                f"Turn {self.turn_number}: {correct_before}/{self.processor.total_pieces} correct"
            )

        # Get images
        annotated_image = self._get_annotated_image()
        reference_image = self._get_reference_image()

        # Save intermediate image if enabled
        if self.config.save_intermediate_images:
            self._save_image(annotated_image, f"turn_{self.turn_number:03d}_before.png")

        # Build prompts
        system_prompt = build_system_prompt(
            coordinate_description=self.annotator.get_coordinate_format_description(),
            max_moves=self.config.max_moves_per_turn,
        )

        user_prompt = build_user_prompt(
            grid_rows=self.processor.grid_rows,
            grid_cols=self.processor.grid_cols,
            move_history=self.move_history if self.config.include_move_history else None,
            show_correct_count=self.config.show_correct_count,
            correct_count=correct_before if self.config.show_correct_count else None,
            total_pieces=self.processor.total_pieces,
            has_reference_image=self.config.show_reference_image,
            current_turn=self.turn_number,
            max_turns=self.config.max_turns,
        )

        # Request moves from LLM
        response = self.llm.request_moves(
            image=annotated_image,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            reference_image=reference_image,
        )

        # Parse and apply moves
        moves_requested = response.get("moves", [])
        moves_to_apply = moves_requested[: self.config.max_moves_per_turn]  # Enforce limit

        moves_applied = []
        moves_invalid = []

        for coord_a, coord_b in moves_to_apply:
            if self.processor.apply_swap(coord_a, coord_b):
                moves_applied.append((coord_a, coord_b))
            else:
                moves_invalid.append((coord_a, coord_b))
                logger.warning(f"Invalid move skipped: {coord_a} <-> {coord_b}")

        correct_after = self.processor.count_correct_pieces()

        # Create turn result
        turn_result = TurnResult(
            turn_number=self.turn_number,
            moves_requested=moves_requested,
            moves_applied=moves_applied,
            moves_invalid=moves_invalid,
            correct_before=correct_before,
            correct_after=correct_after,
            reasoning=response.get("reasoning"),
            raw_response=response.get("raw_response"),
            usage=response.get("usage", {}),
            error=response.get("error"),
        )

        # Record metrics
        self.metrics.record_turn(turn_result)

        # Update move history
        if self.config.include_move_history:
            self.move_history.append(
                format_move_history_entry(
                    turn_number=self.turn_number,
                    moves=moves_applied,
                    correct_after=correct_after if self.config.show_correct_count else None,
                )
            )

        if self.config.verbose:
            progress = correct_after - correct_before
            sign = "+" if progress >= 0 else ""
            logger.info(
                f"  Applied {len(moves_applied)} moves, "
                f"skipped {len(moves_invalid)} invalid. "
                f"Progress: {sign}{progress} ({correct_after}/{self.processor.total_pieces})"
            )

        return turn_result

    def run(self) -> GameResult:
        """
        Run the complete puzzle game until solved or max turns reached.

        Returns:
            GameResult with complete game metrics
        """
        self._start_time = datetime.now()

        if self.config.verbose:
            logger.info(f"Starting puzzle game: {self.config.grid_size}x{self.config.grid_size}")
            logger.info(f"Image: {self.config.image_path}")
            logger.info(f"Max moves per turn: {self.config.max_moves_per_turn}")
            logger.info(f"Max turns: {self.config.max_turns}")

        # Save initial state
        initial_image = self._get_annotated_image()
        if self.config.save_intermediate_images and self.output_dir:
            self._save_image(initial_image, "initial_state.png")
            if self.config.show_reference_image:
                ref = self._get_reference_image()
                if ref is not None:
                    self._save_image(ref, "reference.png")

        # Capture initial frame for GIF
        self._capture_gif_frame(initial_image)

        solve_turn = None

        while self.turn_number < self.config.max_turns:
            if self.processor.is_solved():
                solve_turn = self.turn_number
                if self.config.verbose:
                    logger.info(f"Puzzle solved in {self.turn_number} turns!")
                break

            self.step()

            # Capture frame after each turn for GIF
            self._capture_gif_frame(self._get_annotated_image())

        # Check final state
        solved = self.processor.is_solved()
        if solved and solve_turn is None:
            solve_turn = self.turn_number

        self._end_time = datetime.now()
        duration = (self._end_time - self._start_time).total_seconds()

        # Save final state
        if self.output_dir:
            self._save_image(self._get_annotated_image(), "final_state.png")

        # Save GIF animation
        self._save_gif()

        # Build result
        usage = self.llm.get_total_usage()

        result = GameResult(
            image_path=str(self.config.image_path),
            grid_size=self.config.grid_size,
            max_moves_per_turn=self.config.max_moves_per_turn,
            max_turns=self.config.max_turns,
            shuffle_seed=self.config.shuffle_seed,
            provider=self.config.provider,
            model=self.config.model,
            annotation_mode=self.config.annotation_mode,
            colored_borders=self.config.colored_borders,
            show_correct_count=self.config.show_correct_count,
            show_reference_image=self.config.show_reference_image,
            include_move_history=self.config.include_move_history,
            solved=solved,
            total_turns=self.turn_number,
            solve_turn=solve_turn,
            total_moves=self.metrics.total_moves,
            total_invalid_moves=self.metrics.total_invalid_moves,
            max_correct_achieved=self.metrics.max_correct,
            final_correct=self.processor.count_correct_pieces(),
            total_pieces=self.processor.total_pieces,
            total_prompt_tokens=usage.get("prompt_tokens", 0),
            total_completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            start_time=self._start_time.isoformat(),
            end_time=self._end_time.isoformat(),
            duration_seconds=duration,
            turn_history=self.metrics.turns,
        )

        # Save result
        if self.output_dir:
            result.save(self.output_dir / "result.json")

        if self.config.verbose:
            logger.info(f"Game complete. Solved: {solved}")
            logger.info(
                f"Final: {result.final_correct}/{result.total_pieces} ({result.accuracy:.1f}%)"
            )
            logger.info(
                f"Max achieved: {result.max_correct_achieved}/{result.total_pieces} ({result.max_accuracy:.1f}%)"
            )
            logger.info(f"Total moves: {result.total_moves}, Invalid: {result.total_invalid_moves}")
            logger.info(f"Duration: {duration:.1f}s, Tokens: {result.total_tokens}")

        return result

    def get_current_image(self) -> np.ndarray:
        """Get the current annotated puzzle image."""
        return self._get_annotated_image()

    def is_solved(self) -> bool:
        """Check if the puzzle is currently solved."""
        return self.processor.is_solved()
