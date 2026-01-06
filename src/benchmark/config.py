"""Configuration classes for benchmarking."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import json
import os


@dataclass
class ModelSpec:
    """Specification for a model to benchmark."""

    provider: str
    model_name: str
    api_key: Optional[str] = None  # If None, uses environment variable
    base_url: Optional[str] = None
    reasoning_effort: str = "none"

    @classmethod
    def from_string(cls, model_string: str) -> "ModelSpec":
        """
        Parse model string in provider/model-name format.

        Args:
            model_string: String in format "provider/model-name"

        Returns:
            ModelSpec instance
        """
        if "/" not in model_string:
            raise ValueError(
                f"Invalid model format: {model_string}. Use 'provider/model-name' format "
                "(e.g., 'openai/gpt-5.2', 'google/gemini-3-pro-preview', 'anthropic/claude-opus-4-5')"
            )

        parts = model_string.split("/", 1)
        provider = parts[0].lower()
        model_name = parts[1]

        valid_providers = ["openai", "anthropic", "google"]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of: {', '.join(valid_providers)}"
            )

        return cls(provider=provider, model_name=model_name)

    def get_api_key(self) -> str:
        """Get the API key from instance or environment."""
        if self.api_key:
            return self.api_key

        env_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        env_var = env_vars.get(self.provider.lower())
        if env_var and os.environ.get(env_var):
            return os.environ[env_var]

        raise ValueError(f"No API key for {self.provider}. Set {env_var} environment variable.")

    @property
    def full_name(self) -> str:
        """Get full model name as provider/model."""
        return f"{self.provider}/{self.model_name}"

    @property
    def safe_name(self) -> str:
        """Get a filesystem-safe name for the model."""
        return f"{self.provider}_{self.model_name}".replace("/", "_").replace(":", "_")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "reasoning_effort": self.reasoning_effort,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelSpec":
        """Create from dictionary."""
        return cls(
            provider=data["provider"],
            model_name=data["model_name"],
            base_url=data.get("base_url"),
            reasoning_effort=data.get("reasoning_effort", "none"),
        )


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""

    image_path: str
    model: ModelSpec
    grid_size: int | tuple[int, int]
    seed: int
    run_index: int  # Which run (0, 1, 2, ...) for this image/model combination

    # Game settings
    max_moves_per_turn: int = 16
    max_turns: int = 50
    resize_to: Optional[int] = None

    # Annotation settings
    annotation_mode: Literal["border_labels", "cell_labels", "both"] = "both"
    colored_borders: bool = False

    # Hint settings
    show_correct_count: bool = True
    show_reference_image: bool = False
    annotate_reference_image: bool = False
    include_move_history: bool = False

    @property
    def run_id(self) -> str:
        """Generate a unique ID for this run."""
        image_name = Path(self.image_path).stem
        grid_str = (
            f"{self.grid_size}x{self.grid_size}"
            if isinstance(self.grid_size, int)
            else f"{self.grid_size[0]}x{self.grid_size[1]}"
        )
        return f"{self.model.safe_name}_{image_name}_{grid_str}_seed{self.seed}_run{self.run_index}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "image_path": self.image_path,
            "model": self.model.to_dict(),
            "grid_size": self.grid_size,
            "seed": self.seed,
            "run_index": self.run_index,
            "max_moves_per_turn": self.max_moves_per_turn,
            "max_turns": self.max_turns,
            "resize_to": self.resize_to,
            "annotation_mode": self.annotation_mode,
            "colored_borders": self.colored_borders,
            "show_correct_count": self.show_correct_count,
            "show_reference_image": self.show_reference_image,
            "annotate_reference_image": self.annotate_reference_image,
            "include_move_history": self.include_move_history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """Create from dictionary."""
        return cls(
            image_path=data["image_path"],
            model=ModelSpec.from_dict(data["model"]),
            grid_size=tuple(data["grid_size"])
            if isinstance(data["grid_size"], list)
            else data["grid_size"],
            seed=data["seed"],
            run_index=data["run_index"],
            max_moves_per_turn=data.get("max_moves_per_turn", 16),
            max_turns=data.get("max_turns", 50),
            resize_to=data.get("resize_to"),
            annotation_mode=data.get("annotation_mode", "both"),
            colored_borders=data.get("colored_borders", False),
            show_correct_count=data.get("show_correct_count", True),
            show_reference_image=data.get("show_reference_image", False),
            annotate_reference_image=data.get("annotate_reference_image", False),
            include_move_history=data.get("include_move_history", False),
        )


@dataclass
class BenchmarkConfig:
    """Configuration for the entire benchmark suite."""

    # Models to benchmark
    models: list[ModelSpec] = field(default_factory=list)

    # Images
    image_folder: str = "images"
    image_patterns: list[str] = field(
        default_factory=lambda: ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    )

    # Benchmark settings
    grid_size: int | tuple[int, int] = 4
    runs_per_image: int = 3
    base_seed: int = 42  # Seeds will be base_seed, base_seed+1, base_seed+2, etc.

    # Game settings
    max_moves_per_turn: int = 16
    max_turns: int = 50
    resize_to: Optional[int] = None

    # Annotation settings
    annotation_mode: Literal["border_labels", "cell_labels", "both"] = "both"
    colored_borders: bool = False

    # Hint settings
    show_correct_count: bool = True
    show_reference_image: bool = False
    annotate_reference_image: bool = False
    include_move_history: bool = False

    # Output settings
    output_dir: str = "benchmark_results"
    save_intermediate_images: bool = False
    save_gifs: bool = False

    # Parallel execution settings
    parallel_providers: bool = False  # Run different providers in parallel for the same puzzle
    max_parallel_workers: int = 4  # Maximum number of parallel workers

    def get_images(self) -> list[Path]:
        """Get all matching images from the image folder."""
        folder = Path(self.image_folder)
        if not folder.exists():
            raise ValueError(f"Image folder does not exist: {folder}")

        images = []
        for pattern in self.image_patterns:
            images.extend(folder.glob(pattern))

        return sorted(set(images))

    def generate_runs(self) -> list[RunConfig]:
        """Generate all run configurations for the benchmark."""
        images = self.get_images()
        runs = []

        for image_path in images:
            for run_idx in range(self.runs_per_image):
                seed = self.base_seed + run_idx

                for model in self.models:
                    run = RunConfig(
                        image_path=str(image_path),
                        model=model,
                        grid_size=self.grid_size,
                        seed=seed,
                        run_index=run_idx,
                        max_moves_per_turn=self.max_moves_per_turn,
                        max_turns=self.max_turns,
                        resize_to=self.resize_to,
                        annotation_mode=self.annotation_mode,
                        colored_borders=self.colored_borders,
                        show_correct_count=self.show_correct_count,
                        show_reference_image=self.show_reference_image,
                        annotate_reference_image=self.annotate_reference_image,
                        include_move_history=self.include_move_history,
                    )
                    runs.append(run)

        return runs

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "models": [m.to_dict() for m in self.models],
            "image_folder": self.image_folder,
            "image_patterns": self.image_patterns,
            "grid_size": self.grid_size,
            "runs_per_image": self.runs_per_image,
            "base_seed": self.base_seed,
            "max_moves_per_turn": self.max_moves_per_turn,
            "max_turns": self.max_turns,
            "resize_to": self.resize_to,
            "annotation_mode": self.annotation_mode,
            "colored_borders": self.colored_borders,
            "show_correct_count": self.show_correct_count,
            "show_reference_image": self.show_reference_image,
            "annotate_reference_image": self.annotate_reference_image,
            "include_move_history": self.include_move_history,
            "output_dir": self.output_dir,
            "save_intermediate_images": self.save_intermediate_images,
            "save_gifs": self.save_gifs,
            "parallel_providers": self.parallel_providers,
            "max_parallel_workers": self.max_parallel_workers,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkConfig":
        """Create from dictionary."""
        return cls(
            models=[ModelSpec.from_dict(m) for m in data.get("models", [])],
            image_folder=data.get("image_folder", "images"),
            image_patterns=data.get("image_patterns", ["*.jpg", "*.jpeg", "*.png", "*.webp"]),
            grid_size=tuple(data["grid_size"])
            if isinstance(data.get("grid_size"), list)
            else data.get("grid_size", 4),
            runs_per_image=data.get("runs_per_image", 3),
            base_seed=data.get("base_seed", 42),
            max_moves_per_turn=data.get("max_moves_per_turn", 16),
            max_turns=data.get("max_turns", 50),
            resize_to=data.get("resize_to"),
            annotation_mode=data.get("annotation_mode", "both"),
            colored_borders=data.get("colored_borders", False),
            show_correct_count=data.get("show_correct_count", True),
            show_reference_image=data.get("show_reference_image", False),
            annotate_reference_image=data.get("annotate_reference_image", False),
            include_move_history=data.get("include_move_history", False),
            output_dir=data.get("output_dir", "benchmark_results"),
            save_intermediate_images=data.get("save_intermediate_images", False),
            save_gifs=data.get("save_gifs", False),
            parallel_providers=data.get("parallel_providers", False),
            max_parallel_workers=data.get("max_parallel_workers", 4),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)
