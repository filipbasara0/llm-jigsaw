"""LLM Jigsaw Puzzle Solver - A benchmark for multimodal LLM spatial reasoning."""

from .image_processor import ImageProcessor
from .grid_annotator import GridAnnotator
from .llm_interface import LLMInterface
from .game import PuzzleGame, GameConfig
from .metrics import GameResult, TurnResult

__version__ = "0.1.0"

__all__ = [
    "ImageProcessor",
    "GridAnnotator", 
    "LLMInterface",
    "PuzzleGame",
    "GameConfig",
    "GameResult",
    "TurnResult",
]
