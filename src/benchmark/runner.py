"""Benchmark runner for orchestrating puzzle-solving evaluations."""

import logging
import time
import traceback
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..game import PuzzleGame, GameConfig
from ..metrics import GameResult
from .config import BenchmarkConfig, RunConfig
from .cache import BenchmarkCache

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Orchestrates benchmark runs across multiple models and images.

    Features:
    - Runs puzzles for multiple models on all images in a folder
    - Supports multiple runs per image with consistent seeds across models
    - Caches intermediate results for resumable execution
    - Provides progress callbacks
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Initialize the benchmark runner.

        Args:
            config: Benchmark configuration
            progress_callback: Optional callback(completed, total, message) for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback

        # Initialize cache
        self.cache = BenchmarkCache(config.output_dir)

        # Generate all run configurations
        self.all_runs = config.generate_runs()

        # Track progress
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0

        # Thread-safe lock for progress updates
        self._progress_lock = threading.Lock()

    def _report_progress(self, message: str) -> None:
        """Report progress to callback if available."""
        if self.progress_callback:
            total = len(self.all_runs)
            self.progress_callback(self.completed_count, total, message)

    def _run_single(self, run_config: RunConfig) -> Optional[GameResult]:
        """
        Execute a single benchmark run.

        Args:
            run_config: Configuration for this run

        Returns:
            GameResult or None if failed
        """
        image_name = Path(run_config.image_path).stem
        model_name = run_config.model.full_name

        logger.info(
            f"Running: {model_name} on {image_name} "
            f"(seed={run_config.seed}, run={run_config.run_index})"
        )

        # Create output directory for this run
        run_output_dir = Path(self.config.output_dir) / "runs" / run_config.run_id

        # Create game config
        game_config = GameConfig(
            image_path=run_config.image_path,
            grid_size=run_config.grid_size,
            resize_to=run_config.resize_to,
            provider=run_config.model.provider,
            model=run_config.model.model_name,
            api_key=run_config.model.get_api_key(),
            base_url=run_config.model.base_url,
            reasoning_effort=run_config.model.reasoning_effort,
            max_moves_per_turn=run_config.max_moves_per_turn,
            max_turns=run_config.max_turns,
            shuffle_seed=run_config.seed,
            annotation_mode=run_config.annotation_mode,
            colored_borders=run_config.colored_borders,
            show_correct_count=run_config.show_correct_count,
            show_reference_image=run_config.show_reference_image,
            annotate_reference_image=run_config.annotate_reference_image,
            include_move_history=run_config.include_move_history,
            output_dir=str(run_output_dir),
            save_intermediate_images=self.config.save_intermediate_images,
            save_gif=self.config.save_gifs,
            verbose=True,
        )

        try:
            game = PuzzleGame(game_config)
            result = game.run()
            return result
        except Exception as e:
            logger.error(f"Run failed: {e}")
            logger.debug(traceback.format_exc())
            return None

    def run(self, force_rerun: bool = False) -> dict:
        """
        Execute the full benchmark suite.

        Args:
            force_rerun: If True, ignore cache and rerun all

        Returns:
            Dictionary with benchmark statistics
        """
        start_time = datetime.now()

        # Initialize cache
        is_resuming = self.cache.initialize(self.config)

        if force_rerun:
            logger.info("Force rerun enabled - clearing cache")
            self.cache.clear()
            pending_runs = self.all_runs
        else:
            pending_runs = self.cache.get_pending_runs(self.all_runs)
            if is_resuming:
                self.completed_count = self.cache.get_completed_count()
                self.skipped_count = self.completed_count
                logger.info(f"Resuming benchmark - {self.completed_count} runs already completed")

        total_runs = len(self.all_runs)
        logger.info(f"Benchmark: {total_runs} total runs, {len(pending_runs)} pending")

        # Log configuration
        images = self.config.get_images()
        logger.info(f"  Models: {[m.full_name for m in self.config.models]}")
        logger.info(f"  Images: {len(images)} images from {self.config.image_folder}")
        logger.info(f"  Runs per image: {self.config.runs_per_image}")
        logger.info(f"  Grid size: {self.config.grid_size}")

        if self.config.parallel_providers:
            logger.info(
                f"  Parallel execution: enabled (max {self.config.max_parallel_workers} workers)"
            )
        else:
            logger.info("  Parallel execution: disabled")

        self._report_progress(f"Starting benchmark with {len(pending_runs)} pending runs")

        # Execute runs
        if self.config.parallel_providers and len(self.config.models) > 1:
            self._run_parallel(pending_runs, total_runs)
        else:
            self._run_sequential(pending_runs, total_runs)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Generate summary
        summary = {
            "total_runs": total_runs,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "skipped_cached": self.skipped_count,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "parallel_execution": self.config.parallel_providers,
        }

        logger.info("=" * 50)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Completed: {self.completed_count}")
        logger.info(f"Failed: {self.failed_count}")
        logger.info(f"Duration: {duration:.1f}s")

        return summary

    def _run_sequential(self, pending_runs: list[RunConfig], total_runs: int) -> None:
        """Execute runs sequentially."""
        for i, run_config in enumerate(pending_runs):
            run_num = self.skipped_count + i + 1

            self._report_progress(
                f"[{run_num}/{total_runs}] {run_config.model.full_name} - "
                f"{Path(run_config.image_path).stem} (run {run_config.run_index})"
            )

            result = self._run_single(run_config)

            if result is not None:
                # Save to cache
                self.cache.save_result(run_config, result.to_dict())
                self.completed_count += 1

                logger.info(
                    f"  Result: {'Solved' if result.solved else 'Not solved'} - "
                    f"{result.final_correct}/{result.total_pieces} "
                    f"({result.accuracy:.1f}%) in {result.total_turns} turns"
                )
            else:
                self.failed_count += 1
                logger.error(f"  Run failed for {run_config.run_id}")

            # Small delay between runs to avoid rate limiting
            if i < len(pending_runs) - 1:
                time.sleep(1)

    def _run_parallel(self, pending_runs: list[RunConfig], total_runs: int) -> None:
        """
        Execute runs in parallel, grouping by image+seed to run different providers concurrently.

        This groups runs by (image_path, seed) so that different providers/models can be
        executed in parallel for the same puzzle configuration.
        """
        # Group runs by (image_path, seed) - these can run in parallel across providers
        run_groups = defaultdict(list)
        for run_config in pending_runs:
            key = (run_config.image_path, run_config.seed)
            run_groups[key].append(run_config)

        logger.info(
            f"Parallel execution: {len(run_groups)} puzzle groups, "
            f"max {self.config.max_parallel_workers} concurrent workers"
        )

        runs_started = self.skipped_count

        # Process each group - within a group, different providers run in parallel
        for group_key, group_runs in run_groups.items():
            image_path, seed = group_key
            image_name = Path(image_path).stem

            logger.info(f"Processing: {image_name} (seed={seed}) - {len(group_runs)} providers")

            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                # Submit all runs in this group to the executor
                future_to_config = {}
                for run_config in group_runs:
                    runs_started += 1
                    self._report_progress(
                        f"[{runs_started}/{total_runs}] Starting {run_config.model.full_name} - "
                        f"{image_name} (run {run_config.run_index})"
                    )
                    future = executor.submit(self._run_single, run_config)
                    future_to_config[future] = run_config

                # Collect results as they complete
                for future in as_completed(future_to_config):
                    run_config = future_to_config[future]

                    try:
                        result = future.result()

                        with self._progress_lock:
                            if result is not None:
                                # Save to cache (thread-safe via lock)
                                self.cache.save_result(run_config, result.to_dict())
                                self.completed_count += 1

                                logger.info(
                                    f"  {run_config.model.full_name}: "
                                    f"{'Solved' if result.solved else 'Not solved'} - "
                                    f"{result.final_correct}/{result.total_pieces} "
                                    f"({result.accuracy:.1f}%) in {result.total_turns} turns"
                                )
                            else:
                                self.failed_count += 1
                                logger.error(f"  Run failed for {run_config.run_id}")
                    except Exception as e:
                        with self._progress_lock:
                            self.failed_count += 1
                        logger.error(f"  Exception for {run_config.run_id}: {e}")
                        logger.debug(traceback.format_exc())

            # Small delay between groups to avoid overwhelming APIs
            time.sleep(0.5)

    def get_results(self) -> dict[str, dict]:
        """Get all cached results."""
        return self.cache.get_all_results()

    def get_progress(self) -> dict:
        """Get current progress statistics."""
        return {
            "total": len(self.all_runs),
            "completed": self.cache.get_completed_count(),
            "pending": len(self.cache.get_pending_runs(self.all_runs)),
        }
