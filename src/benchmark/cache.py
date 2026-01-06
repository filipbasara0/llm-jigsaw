"""Cache manager for benchmark results to enable resumable runs."""

import json
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import RunConfig, BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class CachedResult:
    """A cached result for a single benchmark run."""

    run_id: str
    run_config: dict
    result: dict
    completed_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "run_config": self.run_config,
            "result": self.result,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CachedResult":
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            run_config=data["run_config"],
            result=data["result"],
            completed_at=data["completed_at"],
        )


class BenchmarkCache:
    """
    Manages caching of benchmark results for resumable runs.

    The cache stores:
    - Individual run results keyed by run_id
    - Benchmark configuration for validation
    - Progress tracking
    """

    CACHE_FILE = "benchmark_cache.json"
    CONFIG_FILE = "benchmark_config.json"
    RESULTS_DIR = "runs"

    def __init__(self, output_dir: str | Path):
        """
        Initialize the cache manager.

        Args:
            output_dir: Directory to store cache and results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cache_file = self.output_dir / self.CACHE_FILE
        self.config_file = self.output_dir / self.CONFIG_FILE
        self.results_dir = self.output_dir / self.RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache or create new
        self._cache: dict[str, CachedResult] = {}
        self._config_hash: Optional[str] = None
        self._load_cache()

    def _compute_config_hash(self, config: BenchmarkConfig) -> str:
        """Compute a hash of the benchmark configuration."""
        # Hash relevant config fields that affect reproducibility
        config_dict = {
            "models": [m.to_dict() for m in config.models],
            "image_folder": config.image_folder,
            "image_patterns": config.image_patterns,
            "grid_size": config.grid_size,
            "runs_per_image": config.runs_per_image,
            "base_seed": config.base_seed,
            "max_moves_per_turn": config.max_moves_per_turn,
            "max_turns": config.max_turns,
            "resize_to": config.resize_to,
            "annotation_mode": config.annotation_mode,
            "colored_borders": config.colored_borders,
            "show_correct_count": config.show_correct_count,
            "show_reference_image": config.show_reference_image,
            "annotate_reference_image": config.annotate_reference_image,
            "include_move_history": config.include_move_history,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load cache from disk if it exists."""
        if not self.cache_file.exists():
            logger.debug("No existing cache found")
            return

        try:
            with open(self.cache_file) as f:
                data = json.load(f)

            self._config_hash = data.get("config_hash")

            for run_id, cached_data in data.get("results", {}).items():
                self._cache[run_id] = CachedResult.from_dict(cached_data)

            logger.info(f"Loaded {len(self._cache)} cached results from {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        data = {
            "config_hash": self._config_hash,
            "results": {run_id: cr.to_dict() for run_id, cr in self._cache.items()},
            "last_updated": datetime.now().isoformat(),
        }

        with open(self.cache_file, "w") as f:
            json.dump(data, f, indent=2)

    def initialize(self, config: BenchmarkConfig) -> bool:
        """
        Initialize or validate the cache for a benchmark config.

        Args:
            config: The benchmark configuration

        Returns:
            True if resuming from existing cache, False if starting fresh
        """
        new_hash = self._compute_config_hash(config)

        # Save the config
        config.save(self.config_file)

        if self._config_hash is None:
            # New benchmark
            self._config_hash = new_hash
            self._save_cache()
            return False

        if self._config_hash != new_hash:
            logger.warning(
                "Benchmark configuration has changed. "
                "Cached results will be invalidated for runs that don't match."
            )
            self._config_hash = new_hash
            self._save_cache()
            return False

        return len(self._cache) > 0

    def is_completed(self, run_config: RunConfig) -> bool:
        """Check if a run has already been completed."""
        return run_config.run_id in self._cache

    def get_result(self, run_config: RunConfig) -> Optional[dict]:
        """Get a cached result if it exists."""
        cached = self._cache.get(run_config.run_id)
        return cached.result if cached else None

    def save_result(self, run_config: RunConfig, result: dict) -> None:
        """
        Save a run result to the cache.

        Args:
            run_config: The run configuration
            result: The game result dictionary
        """
        cached = CachedResult(
            run_id=run_config.run_id,
            run_config=run_config.to_dict(),
            result=result,
            completed_at=datetime.now().isoformat(),
        )

        self._cache[run_config.run_id] = cached

        # Also save individual result file
        result_file = self.results_dir / f"{run_config.run_id}.json"
        with open(result_file, "w") as f:
            json.dump(cached.to_dict(), f, indent=2)

        # Update main cache file
        self._save_cache()

        logger.debug(f"Saved result for {run_config.run_id}")

    def get_all_results(self) -> dict[str, dict]:
        """Get all cached results."""
        return {run_id: cr.result for run_id, cr in self._cache.items()}

    def get_completed_count(self) -> int:
        """Get the number of completed runs."""
        return len(self._cache)

    def get_pending_runs(self, all_runs: list[RunConfig]) -> list[RunConfig]:
        """
        Get the list of runs that haven't been completed yet.

        Args:
            all_runs: All run configurations

        Returns:
            List of pending run configurations
        """
        return [run for run in all_runs if not self.is_completed(run)]

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache = {}
        self._config_hash = None

        if self.cache_file.exists():
            self.cache_file.unlink()

        # Clear individual result files
        for result_file in self.results_dir.glob("*.json"):
            result_file.unlink()

        logger.info("Cache cleared")

    def export_results(self, output_file: str | Path) -> None:
        """
        Export all results to a single JSON file.

        Args:
            output_file: Path to the output file
        """
        output_file = Path(output_file)

        data = {
            "config_hash": self._config_hash,
            "exported_at": datetime.now().isoformat(),
            "results": [cr.to_dict() for cr in self._cache.values()],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._cache)} results to {output_file}")
