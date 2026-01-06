"""Aggregates benchmark results and generates reports."""

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelStats:
    """Aggregated statistics for a single model."""

    model_name: str
    provider: str
    num_runs: int = 0
    num_solved: int = 0

    # Accuracy metrics
    accuracies: list[float] = field(default_factory=list)
    max_accuracies: list[float] = field(default_factory=list)

    # Solve metrics
    solve_turns: list[int] = field(default_factory=list)
    total_turns: list[int] = field(default_factory=list)

    # Move metrics
    total_moves: list[int] = field(default_factory=list)
    invalid_moves: list[int] = field(default_factory=list)

    # Token usage
    total_tokens: list[int] = field(default_factory=list)

    # Timing
    durations: list[float] = field(default_factory=list)

    def add_result(self, result: dict) -> None:
        """Add a result to the statistics."""
        results = result.get("results", result)
        config = result.get("config", result)
        timing = result.get("timing", result)
        usage = result.get("usage", result)

        self.num_runs += 1

        if results.get("solved", False):
            self.num_solved += 1
            if results.get("solve_turn"):
                self.solve_turns.append(results["solve_turn"])

        self.accuracies.append(results.get("accuracy", 0))
        self.max_accuracies.append(results.get("max_accuracy", 0))
        self.total_turns.append(results.get("total_turns", 0))
        self.total_moves.append(results.get("total_moves", 0))
        self.invalid_moves.append(results.get("total_invalid_moves", 0))
        self.total_tokens.append(usage.get("total_tokens", 0))
        self.durations.append(timing.get("duration_seconds", 0))

    @property
    def solve_rate(self) -> float:
        """Percentage of puzzles solved."""
        return (self.num_solved / self.num_runs * 100) if self.num_runs > 0 else 0

    @property
    def mean_accuracy(self) -> float:
        """Mean final accuracy."""
        return statistics.mean(self.accuracies) if self.accuracies else 0

    @property
    def std_accuracy(self) -> float:
        """Standard deviation of final accuracy."""
        return statistics.stdev(self.accuracies) if len(self.accuracies) > 1 else 0

    @property
    def mean_max_accuracy(self) -> float:
        """Mean maximum accuracy achieved."""
        return statistics.mean(self.max_accuracies) if self.max_accuracies else 0

    @property
    def mean_solve_turns(self) -> float:
        """Mean turns to solve (only for solved puzzles)."""
        return statistics.mean(self.solve_turns) if self.solve_turns else 0

    @property
    def mean_total_turns(self) -> float:
        """Mean total turns taken."""
        return statistics.mean(self.total_turns) if self.total_turns else 0

    @property
    def mean_moves(self) -> float:
        """Mean total moves."""
        return statistics.mean(self.total_moves) if self.total_moves else 0

    @property
    def mean_invalid_rate(self) -> float:
        """Mean percentage of invalid moves."""
        if not self.total_moves or not self.invalid_moves:
            return 0
        rates = [
            (inv / (tot + inv) * 100) if (tot + inv) > 0 else 0
            for tot, inv in zip(self.total_moves, self.invalid_moves)
        ]
        return statistics.mean(rates)

    @property
    def mean_tokens(self) -> float:
        """Mean tokens used."""
        return statistics.mean(self.total_tokens) if self.total_tokens else 0

    @property
    def total_tokens_sum(self) -> int:
        """Total tokens used across all runs."""
        return sum(self.total_tokens)

    @property
    def mean_duration(self) -> float:
        """Mean duration in seconds."""
        return statistics.mean(self.durations) if self.durations else 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "provider": self.provider,
            "full_name": f"{self.provider}/{self.model_name}",
            "num_runs": self.num_runs,
            "num_solved": self.num_solved,
            "solve_rate": round(self.solve_rate, 2),
            "mean_accuracy": round(self.mean_accuracy, 2),
            "std_accuracy": round(self.std_accuracy, 2),
            "mean_max_accuracy": round(self.mean_max_accuracy, 2),
            "mean_solve_turns": round(self.mean_solve_turns, 2),
            "mean_total_turns": round(self.mean_total_turns, 2),
            "mean_moves": round(self.mean_moves, 2),
            "mean_invalid_rate": round(self.mean_invalid_rate, 2),
            "mean_tokens": round(self.mean_tokens, 0),
            "total_tokens": self.total_tokens_sum,
            "mean_duration": round(self.mean_duration, 2),
        }


@dataclass
class ImageStats:
    """Aggregated statistics for a single image across all models."""

    image_name: str
    image_path: str
    results_by_model: dict[str, list[dict]] = field(default_factory=dict)

    def add_result(self, model_name: str, result: dict) -> None:
        """Add a result for a model."""
        if model_name not in self.results_by_model:
            self.results_by_model[model_name] = []
        self.results_by_model[model_name].append(result)

    def get_model_stats(self, model_name: str) -> dict:
        """Get statistics for a specific model on this image."""
        results = self.results_by_model.get(model_name, [])
        if not results:
            return {}

        accuracies = [r.get("results", r).get("accuracy", 0) for r in results]
        solved = [r.get("results", r).get("solved", False) for r in results]

        return {
            "num_runs": len(results),
            "num_solved": sum(solved),
            "mean_accuracy": statistics.mean(accuracies) if accuracies else 0,
            "std_accuracy": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_name": self.image_name,
            "image_path": self.image_path,
            "models": {model: self.get_model_stats(model) for model in self.results_by_model},
        }


class ResultsAggregator:
    """
    Aggregates benchmark results across multiple runs, models, and images.

    Provides:
    - Per-model statistics
    - Per-image statistics
    - Cross-model comparisons
    - Report generation
    """

    def __init__(self):
        """Initialize the aggregator."""
        self.results: list[dict] = []
        self.model_stats: dict[str, ModelStats] = {}
        self.image_stats: dict[str, ImageStats] = {}

    def add_result(self, result: dict) -> None:
        """
        Add a single result to the aggregator.

        Args:
            result: Result dictionary from a benchmark run
        """
        self.results.append(result)

        config = result.get("config", result)
        provider = config.get("provider", "unknown")
        model_name = config.get("model", "unknown")
        full_name = f"{provider}/{model_name}"

        image_path = config.get("image_path", "unknown")
        image_name = Path(image_path).stem

        # Update model stats
        if full_name not in self.model_stats:
            self.model_stats[full_name] = ModelStats(
                model_name=model_name,
                provider=provider,
            )
        self.model_stats[full_name].add_result(result)

        # Update image stats
        if image_name not in self.image_stats:
            self.image_stats[image_name] = ImageStats(
                image_name=image_name,
                image_path=image_path,
            )
        self.image_stats[image_name].add_result(full_name, result)

    def add_results(self, results: dict[str, dict]) -> None:
        """
        Add multiple results from cache.

        Args:
            results: Dictionary of run_id -> result
        """
        for run_id, result in results.items():
            self.add_result(result)

    def get_model_comparison(self) -> list[dict]:
        """Get comparison statistics across all models."""
        return sorted(
            [stats.to_dict() for stats in self.model_stats.values()],
            key=lambda x: x["mean_accuracy"],
            reverse=True,
        )

    def get_image_difficulty_ranking(self) -> list[dict]:
        """
        Rank images by difficulty (based on mean accuracy across all models).

        Returns:
            List of image stats sorted by difficulty (hardest first)
        """
        rankings = []

        for image_name, stats in self.image_stats.items():
            all_accuracies = []
            for model_results in stats.results_by_model.values():
                for result in model_results:
                    acc = result.get("results", result).get("accuracy", 0)
                    all_accuracies.append(acc)

            if all_accuracies:
                rankings.append(
                    {
                        "image_name": image_name,
                        "mean_accuracy": statistics.mean(all_accuracies),
                        "std_accuracy": statistics.stdev(all_accuracies)
                        if len(all_accuracies) > 1
                        else 0,
                        "num_runs": len(all_accuracies),
                    }
                )

        return sorted(rankings, key=lambda x: x["mean_accuracy"])

    def get_leaderboard(self) -> list[dict]:
        """
        Generate a leaderboard ranking models by performance.

        Returns:
            Sorted list of model performance dictionaries
        """
        return self.get_model_comparison()

    def generate_report(self) -> dict:
        """
        Generate a comprehensive benchmark report.

        Returns:
            Dictionary containing all aggregated statistics
        """
        return {
            "generated_at": datetime.now().isoformat(),
            "total_runs": len(self.results),
            "num_models": len(self.model_stats),
            "num_images": len(self.image_stats),
            "leaderboard": self.get_leaderboard(),
            "image_difficulty": self.get_image_difficulty_ranking(),
            "per_image_results": {
                name: stats.to_dict() for name, stats in self.image_stats.items()
            },
        }

    def save_report(self, output_path: str | Path) -> None:
        """
        Save the benchmark report to a JSON file.

        Args:
            output_path: Path to save the report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to {output_path}")

    def print_summary(self) -> None:
        """Print a summary of the benchmark results to console."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        print(f"\nTotal runs: {len(self.results)}")
        print(f"Models evaluated: {len(self.model_stats)}")
        print(f"Images tested: {len(self.image_stats)}")

        # Leaderboard
        print("\n" + "-" * 70)
        print("MODEL LEADERBOARD (sorted by accuracy)")
        print("-" * 70)
        print(f"{'Model':<40} {'Solved':<10} {'Accuracy':<15} {'Tokens':<12}")
        print("-" * 70)

        for entry in self.get_leaderboard():
            solve_str = f"{entry['num_solved']}/{entry['num_runs']} ({entry['solve_rate']:.0f}%)"
            acc_str = f"{entry['mean_accuracy']:.1f}% ± {entry['std_accuracy']:.1f}"
            tokens_str = f"{entry['mean_tokens']:.0f}"
            print(f"{entry['full_name']:<40} {solve_str:<10} {acc_str:<15} {tokens_str:<12}")

        # Hardest images
        print("\n" + "-" * 70)
        print("IMAGE DIFFICULTY (sorted by mean accuracy, hardest first)")
        print("-" * 70)

        difficulty = self.get_image_difficulty_ranking()[:10]  # Top 10 hardest
        for entry in difficulty:
            print(
                f"  {entry['image_name']:<30} "
                f"Accuracy: {entry['mean_accuracy']:.1f}% ± {entry['std_accuracy']:.1f}"
            )

        print("\n" + "=" * 70)

    def generate_csv(self, output_path: str | Path) -> None:
        """
        Generate a CSV file with all results.

        Args:
            output_path: Path to save the CSV
        """
        import csv

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "model",
            "provider",
            "image",
            "grid_size",
            "seed",
            "run_index",
            "solved",
            "accuracy",
            "max_accuracy",
            "total_turns",
            "solve_turn",
            "total_moves",
            "invalid_moves",
            "total_tokens",
            "duration_seconds",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                config = result.get("config", result)
                results = result.get("results", result)
                timing = result.get("timing", result)
                usage = result.get("usage", result)

                writer.writerow(
                    {
                        "model": config.get("model"),
                        "provider": config.get("provider"),
                        "image": Path(config.get("image_path", "")).stem,
                        "grid_size": config.get("grid_size"),
                        "seed": config.get("shuffle_seed"),
                        "run_index": result.get("run_index", 0),
                        "solved": results.get("solved"),
                        "accuracy": results.get("accuracy"),
                        "max_accuracy": results.get("max_accuracy"),
                        "total_turns": results.get("total_turns"),
                        "solve_turn": results.get("solve_turn"),
                        "total_moves": results.get("total_moves"),
                        "invalid_moves": results.get("total_invalid_moves"),
                        "total_tokens": usage.get("total_tokens"),
                        "duration_seconds": timing.get("duration_seconds"),
                    }
                )

        logger.info(f"CSV saved to {output_path}")
