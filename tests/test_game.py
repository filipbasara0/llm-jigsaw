"""Tests for the game controller and related components."""

import numpy as np
import pytest
from pathlib import Path
import tempfile
from PIL import Image
from unittest.mock import Mock, patch

from src.game import PuzzleGame, GameConfig
from src.metrics import TurnResult, GameResult, MetricsTracker
from src.prompts import build_system_prompt, build_user_prompt, format_move_history_entry


@pytest.fixture
def sample_image_path():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :32] = [255, 0, 0]
        img[:32, 32:] = [0, 255, 0]
        img[32:, :32] = [0, 0, 255]
        img[32:, 32:] = [255, 255, 0]

        Image.fromarray(img).save(f.name)
        yield f.name

    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestGameConfig:
    """Tests for GameConfig dataclass."""

    def test_defaults(self, sample_image_path):
        """Test default configuration values."""
        config = GameConfig(image_path=sample_image_path, grid_size=4, api_key="test-key")

        assert config.max_moves_per_turn == 16
        assert config.max_turns == 100
        assert config.annotation_mode == "both"
        assert config.colored_borders == True
        assert config.show_correct_count == False
        assert config.include_move_history == True


class TestTurnResult:
    """Tests for TurnResult dataclass."""

    def test_properties(self):
        """Test computed properties."""
        result = TurnResult(
            turn_number=1,
            moves_requested=[("1,1", "2,2"), ("3,3", "4,4")],
            moves_applied=[("1,1", "2,2")],
            moves_invalid=[("3,3", "4,4")],
            correct_before=5,
            correct_after=7,
        )

        assert result.moves_successful == 1
        assert result.moves_failed == 1
        assert result.progress == 2

    def test_to_dict(self):
        """Test serialization."""
        result = TurnResult(
            turn_number=1,
            moves_requested=[("1,1", "2,2")],
            moves_applied=[("1,1", "2,2")],
            moves_invalid=[],
            correct_before=5,
            correct_after=6,
            reasoning="Test reasoning",
        )

        data = result.to_dict()

        assert data["turn_number"] == 1
        assert data["reasoning"] == "Test reasoning"
        assert len(data["moves_applied"]) == 1


class TestMetricsTracker:
    """Tests for MetricsTracker class."""

    def test_record_turn(self):
        """Test recording turn results."""
        tracker = MetricsTracker(total_pieces=16)

        turn = TurnResult(
            turn_number=1,
            moves_requested=[("1,1", "2,2")],
            moves_applied=[("1,1", "2,2")],
            moves_invalid=[],
            correct_before=5,
            correct_after=7,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        )

        tracker.record_turn(turn)

        assert len(tracker.turns) == 1
        assert tracker.total_moves == 1
        assert tracker.max_correct == 7
        assert tracker.total_prompt_tokens == 100
        assert tracker.total_completion_tokens == 50

    def test_max_correct_tracking(self):
        """Test that max correct is tracked across turns."""
        tracker = MetricsTracker(total_pieces=16)

        # Turn 1: improve
        tracker.record_turn(
            TurnResult(
                turn_number=1,
                moves_requested=[],
                moves_applied=[],
                moves_invalid=[],
                correct_before=5,
                correct_after=8,
            )
        )

        # Turn 2: regress
        tracker.record_turn(
            TurnResult(
                turn_number=2,
                moves_requested=[],
                moves_applied=[],
                moves_invalid=[],
                correct_before=8,
                correct_after=6,
            )
        )

        assert tracker.max_correct == 8


class TestPrompts:
    """Tests for prompt building functions."""

    def test_build_system_prompt(self):
        """Test system prompt generation."""
        prompt = build_system_prompt(
            coordinate_description="Coords are row,col from 1,1 to 8,8",
            max_moves=16,
        )

        assert "16" in prompt
        assert "swap" in prompt.lower()
        assert "json" in prompt.lower()

    def test_build_user_prompt_basic(self):
        """Test basic user prompt."""
        prompt = build_user_prompt(grid_rows=8, grid_cols=8)

        assert "8×8" in prompt

    def test_build_user_prompt_rectangular(self):
        """Test user prompt with rectangular grid."""
        prompt = build_user_prompt(grid_rows=3, grid_cols=5)

        assert "3×5" in prompt
        assert "rows×columns" in prompt

    def test_build_user_prompt_with_history(self):
        """Test user prompt with move history."""
        history = [
            {"turn": 1, "moves": [("1,1", "2,2")], "correct_after": 10},
            {"turn": 2, "moves": [("3,3", "4,4")], "correct_after": 12},
        ]

        prompt = build_user_prompt(
            grid_rows=8,
            grid_cols=8,
            move_history=history,
        )

        assert "Previous Moves" in prompt
        assert "Turn 1" in prompt

    def test_build_user_prompt_with_hints(self):
        """Test user prompt with hints enabled."""
        prompt = build_user_prompt(
            grid_rows=8,
            grid_cols=8,
            show_correct_count=True,
            correct_count=30,
            total_pieces=64,
        )

        assert "30/64" in prompt

    def test_format_move_history_entry(self):
        """Test move history entry formatting."""
        entry = format_move_history_entry(
            turn_number=3,
            moves=[("1,1", "2,2"), ("3,3", "4,4")],
            correct_after=15,
        )

        assert entry["turn"] == 3
        assert len(entry["moves"]) == 2
        assert entry["correct_after"] == 15


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_properties(self):
        """Test computed properties."""
        result = GameResult(
            image_path="test.png",
            grid_size=4,
            max_moves_per_turn=16,
            max_turns=100,
            shuffle_seed=42,
            provider="openai",
            model="gpt-5.2",
            annotation_mode="both",
            colored_borders=True,
            show_correct_count=False,
            show_reference_image=False,
            include_move_history=True,
            solved=False,
            total_turns=10,
            solve_turn=None,
            total_moves=50,
            total_invalid_moves=5,
            max_correct_achieved=12,
            final_correct=10,
            total_pieces=16,
            total_prompt_tokens=5000,
            total_completion_tokens=1000,
            total_tokens=6000,
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:01:00",
            duration_seconds=60.0,
        )

        assert result.accuracy == 62.5  # 10/16 * 100
        assert result.max_accuracy == 75.0  # 12/16 * 100
        assert result.moves_per_turn == 5.0  # 50/10

    def test_save_and_load(self, temp_output_dir):
        """Test saving and loading results."""
        result = GameResult(
            image_path="test.png",
            grid_size=4,
            max_moves_per_turn=16,
            max_turns=100,
            shuffle_seed=42,
            provider="openai",
            model="gpt-5.2",
            annotation_mode="both",
            colored_borders=True,
            show_correct_count=False,
            show_reference_image=False,
            include_move_history=True,
            solved=True,
            total_turns=5,
            solve_turn=5,
            total_moves=20,
            total_invalid_moves=2,
            max_correct_achieved=16,
            final_correct=16,
            total_pieces=16,
            total_prompt_tokens=3000,
            total_completion_tokens=500,
            total_tokens=3500,
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:00:30",
            duration_seconds=30.0,
        )

        path = Path(temp_output_dir) / "test_result.json"
        result.save(path)

        loaded = GameResult.load(path)

        assert loaded.solved == result.solved
        assert loaded.total_turns == result.total_turns
        assert loaded.accuracy == result.accuracy


class TestPuzzleGame:
    """Tests for PuzzleGame class."""

    def test_initialization(self, sample_image_path):
        """Test game initialization."""
        config = GameConfig(
            image_path=sample_image_path,
            grid_size=4,
            api_key="test-key",
            shuffle_seed=42,
        )

        game = PuzzleGame(config)

        assert game.turn_number == 0
        assert not game.is_solved()  # Should be shuffled

    def test_get_current_image(self, sample_image_path):
        """Test getting annotated image."""
        config = GameConfig(
            image_path=sample_image_path,
            grid_size=4,
            api_key="test-key",
        )

        game = PuzzleGame(config)
        image = game.get_current_image()

        assert image is not None
        assert len(image.shape) == 3
        assert image.shape[2] == 3

    @patch("src.game.LLMInterface")
    def test_step_with_mock_llm(self, mock_llm_class, sample_image_path):
        """Test a single step with mocked LLM."""
        # Setup mock
        mock_llm = Mock()
        mock_llm.request_moves.return_value = {
            "moves": [("1,1", "2,2")],
            "reasoning": "Test move",
            "raw_response": '{"moves": [{"op": "swap", "a": "1,1", "b": "2,2"}]}',
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        mock_llm_class.return_value = mock_llm

        config = GameConfig(
            image_path=sample_image_path,
            grid_size=4,
            api_key="test-key",
            shuffle_seed=42,
        )

        game = PuzzleGame(config)
        result = game.step()

        assert result.turn_number == 1
        assert len(result.moves_applied) == 1
        assert mock_llm.request_moves.called
