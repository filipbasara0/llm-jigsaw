"""Metrics and result tracking for puzzle games."""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class TurnResult:
    """Result of a single puzzle-solving turn."""

    turn_number: int
    moves_requested: list[tuple[str, str]]  # Moves the LLM requested
    moves_applied: list[tuple[str, str]]  # Moves actually applied (valid ones)
    moves_invalid: list[tuple[str, str]]  # Invalid moves that were skipped
    correct_before: int
    correct_after: int
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    usage: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def moves_successful(self) -> int:
        """Number of moves successfully applied."""
        return len(self.moves_applied)

    @property
    def moves_failed(self) -> int:
        """Number of moves that failed."""
        return len(self.moves_invalid)

    @property
    def progress(self) -> int:
        """Net change in correct pieces."""
        return self.correct_after - self.correct_before

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "turn_number": self.turn_number,
            "moves_requested": self.moves_requested,
            "moves_applied": self.moves_applied,
            "moves_invalid": self.moves_invalid,
            "correct_before": self.correct_before,
            "correct_after": self.correct_after,
            "reasoning": self.reasoning,
            "usage": self.usage,
            "error": self.error,
        }


@dataclass
class GameResult:
    """Complete result of a puzzle game."""

    # Configuration
    image_path: str
    grid_size: int
    max_moves_per_turn: int
    max_turns: int
    shuffle_seed: Optional[int]

    # LLM info
    provider: str
    model: str

    # Settings
    annotation_mode: str
    colored_borders: bool
    show_correct_count: bool
    show_reference_image: bool
    include_move_history: bool

    # Results
    solved: bool
    total_turns: int
    solve_turn: Optional[int]  # Turn when solved, if applicable

    # Metrics
    total_moves: int
    total_invalid_moves: int
    max_correct_achieved: int
    final_correct: int
    total_pieces: int

    # Token usage
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int

    # Timing
    start_time: str
    end_time: str
    duration_seconds: float

    # History
    turn_history: list[TurnResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Final accuracy as percentage."""
        return (self.final_correct / self.total_pieces) * 100 if self.total_pieces > 0 else 0

    @property
    def max_accuracy(self) -> float:
        """Maximum accuracy achieved as percentage."""
        return (self.max_correct_achieved / self.total_pieces) * 100 if self.total_pieces > 0 else 0

    @property
    def moves_per_turn(self) -> float:
        """Average moves per turn."""
        return self.total_moves / self.total_turns if self.total_turns > 0 else 0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "config": {
                "image_path": self.image_path,
                "grid_size": self.grid_size,
                "max_moves_per_turn": self.max_moves_per_turn,
                "max_turns": self.max_turns,
                "shuffle_seed": self.shuffle_seed,
                "provider": self.provider,
                "model": self.model,
                "annotation_mode": self.annotation_mode,
                "colored_borders": self.colored_borders,
                "show_correct_count": self.show_correct_count,
                "show_reference_image": self.show_reference_image,
                "include_move_history": self.include_move_history,
            },
            "results": {
                "solved": self.solved,
                "total_turns": self.total_turns,
                "solve_turn": self.solve_turn,
                "total_moves": self.total_moves,
                "total_invalid_moves": self.total_invalid_moves,
                "max_correct_achieved": self.max_correct_achieved,
                "final_correct": self.final_correct,
                "total_pieces": self.total_pieces,
                "accuracy": self.accuracy,
                "max_accuracy": self.max_accuracy,
            },
            "usage": {
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "timing": {
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration_seconds": self.duration_seconds,
            },
            "turn_history": [t.to_dict() for t in self.turn_history],
        }

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "GameResult":
        """Load from dictionary."""
        config = data["config"]
        results = data["results"]
        usage = data["usage"]
        timing = data["timing"]

        turn_history = [
            TurnResult(
                turn_number=t["turn_number"],
                moves_requested=[tuple(m) for m in t["moves_requested"]],
                moves_applied=[tuple(m) for m in t["moves_applied"]],
                moves_invalid=[tuple(m) for m in t["moves_invalid"]],
                correct_before=t["correct_before"],
                correct_after=t["correct_after"],
                reasoning=t.get("reasoning"),
                usage=t.get("usage", {}),
                error=t.get("error"),
            )
            for t in data.get("turn_history", [])
        ]

        return cls(
            image_path=config["image_path"],
            grid_size=config["grid_size"],
            max_moves_per_turn=config["max_moves_per_turn"],
            max_turns=config["max_turns"],
            shuffle_seed=config.get("shuffle_seed"),
            provider=config["provider"],
            model=config["model"],
            annotation_mode=config["annotation_mode"],
            colored_borders=config["colored_borders"],
            show_correct_count=config["show_correct_count"],
            show_reference_image=config["show_reference_image"],
            include_move_history=config["include_move_history"],
            solved=results["solved"],
            total_turns=results["total_turns"],
            solve_turn=results.get("solve_turn"),
            total_moves=results["total_moves"],
            total_invalid_moves=results["total_invalid_moves"],
            max_correct_achieved=results["max_correct_achieved"],
            final_correct=results["final_correct"],
            total_pieces=results["total_pieces"],
            total_prompt_tokens=usage["total_prompt_tokens"],
            total_completion_tokens=usage["total_completion_tokens"],
            total_tokens=usage["total_tokens"],
            start_time=timing["start_time"],
            end_time=timing["end_time"],
            duration_seconds=timing["duration_seconds"],
            turn_history=turn_history,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GameResult":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


class MetricsTracker:
    """Tracks metrics during a game."""

    def __init__(self, total_pieces: int):
        self.total_pieces = total_pieces
        self.turns: list[TurnResult] = []
        self.max_correct = 0
        self.total_moves = 0
        self.total_invalid_moves = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def record_turn(self, turn_result: TurnResult) -> None:
        """Record a turn result."""
        self.turns.append(turn_result)
        self.total_moves += turn_result.moves_successful
        self.total_invalid_moves += turn_result.moves_failed
        self.max_correct = max(self.max_correct, turn_result.correct_after)

        usage = turn_result.usage
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_correct_progression(self) -> list[int]:
        """Get list of correct counts after each turn."""
        return [t.correct_after for t in self.turns]

    def get_summary(self) -> dict:
        """Get a summary of current metrics."""
        return {
            "turns": len(self.turns),
            "total_moves": self.total_moves,
            "total_invalid_moves": self.total_invalid_moves,
            "max_correct": self.max_correct,
            "current_correct": self.turns[-1].correct_after if self.turns else 0,
            "total_tokens": self.total_tokens,
        }
