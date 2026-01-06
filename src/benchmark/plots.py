"""Plotting utilities for benchmark results visualization."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed. Plotting features will be disabled.")


class BenchmarkPlotter:
    """
    Generates visualization plots for benchmark results.

    Available plots:
    - Model comparison bar chart
    - Accuracy distribution box plots
    - Per-image heatmap
    - Progress curves
    - Token usage comparison
    """

    # Color palette for different models
    COLORS = [
        "#4C72B0",  # Blue
        "#55A868",  # Green
        "#C44E52",  # Red
        "#8172B3",  # Purple
        "#CCB974",  # Yellow
        "#64B5CD",  # Cyan
        "#E377C2",  # Pink
        "#7F7F7F",  # Gray
    ]

    def __init__(self, output_dir: str | Path):
        """
        Initialize the plotter.

        Args:
            output_dir: Directory to save plots
        """
        if not HAS_MATPLOTLIB:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
        plt.rcParams["font.size"] = 12

    def _get_color(self, index: int) -> str:
        """Get a color from the palette."""
        return self.COLORS[index % len(self.COLORS)]

    def plot_model_comparison(
        self,
        leaderboard: list[dict],
        metric: str = "mean_accuracy",
        title: Optional[str] = None,
        filename: str = "model_comparison.png",
    ) -> Path:
        """
        Create a bar chart comparing models on a specific metric.

        Args:
            leaderboard: List of model statistics dictionaries
            metric: Metric to compare (e.g., 'mean_accuracy', 'solve_rate', 'mean_tokens')
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        models = [entry["full_name"] for entry in leaderboard]
        values = [entry.get(metric, 0) for entry in leaderboard]

        # Create bars
        bars = ax.barh(
            range(len(models)), values, color=[self._get_color(i) for i in range(len(models))]
        )

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}" if isinstance(val, float) else str(val),
                va="center",
                fontsize=10,
            )

        # Formatting
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        ax.invert_yaxis()

        metric_labels = {
            "mean_accuracy": "Mean Accuracy (%)",
            "solve_rate": "Solve Rate (%)",
            "mean_tokens": "Mean Tokens",
            "mean_duration": "Mean Duration (s)",
            "mean_max_accuracy": "Mean Max Accuracy (%)",
        }
        ax.set_xlabel(metric_labels.get(metric, metric))
        ax.set_title(title or f"Model Comparison: {metric_labels.get(metric, metric)}")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_path}")
        return output_path

    def plot_accuracy_distribution(
        self,
        results: list[dict],
        title: Optional[str] = None,
        filename: str = "accuracy_distribution.png",
    ) -> Path:
        """
        Create box plots showing accuracy distribution per model.

        Args:
            results: List of all result dictionaries
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        # Group accuracies by model
        model_accuracies = {}
        for result in results:
            config = result.get("config", result)
            results_data = result.get("results", result)

            model = f"{config.get('provider')}/{config.get('model')}"
            accuracy = results_data.get("accuracy", 0)

            if model not in model_accuracies:
                model_accuracies[model] = []
            model_accuracies[model].append(accuracy)

        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(model_accuracies.keys())
        data = [model_accuracies[m] for m in models]

        # Create box plot
        bp = ax.boxplot(data, patch_artist=True, labels=models)

        # Color the boxes
        for i, (box, median) in enumerate(zip(bp["boxes"], bp["medians"])):
            box.set_facecolor(self._get_color(i))
            box.set_alpha(0.7)
            median.set_color("black")
            median.set_linewidth(2)

        # Add scatter points for individual results
        for i, model_data in enumerate(data):
            x = np.random.normal(i + 1, 0.04, len(model_data))
            ax.scatter(x, model_data, alpha=0.4, s=20, color=self._get_color(i))

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title or "Accuracy Distribution by Model")
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_path}")
        return output_path

    def plot_image_heatmap(
        self,
        per_image_results: dict[str, dict],
        title: Optional[str] = None,
        filename: str = "image_heatmap.png",
    ) -> Path:
        """
        Create a heatmap showing accuracy per image per model.

        Args:
            per_image_results: Dictionary of image_name -> {models: {model: stats}}
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        # Extract all models and images
        all_models = set()
        for img_data in per_image_results.values():
            all_models.update(img_data.get("models", {}).keys())

        models = sorted(all_models)
        images = sorted(per_image_results.keys())

        # Build matrix
        matrix = np.zeros((len(images), len(models)))

        for i, image in enumerate(images):
            for j, model in enumerate(models):
                stats = per_image_results.get(image, {}).get("models", {}).get(model, {})
                matrix[i, j] = stats.get("mean_accuracy", 0)

        fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), max(8, len(images) * 0.4)))

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        # Labels
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=45, ha="right")
        ax.set_yticks(range(len(images)))
        ax.set_yticklabels(images)

        # Add text annotations
        for i in range(len(images)):
            for j in range(len(models)):
                val = matrix[i, j]
                text_color = "white" if val < 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", color=text_color, fontsize=8)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Accuracy (%)")

        ax.set_title(title or "Accuracy by Image and Model")
        ax.set_xlabel("Model")
        ax.set_ylabel("Image")

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_path}")
        return output_path

    def plot_solve_rate_comparison(
        self,
        leaderboard: list[dict],
        title: Optional[str] = None,
        filename: str = "solve_rate_comparison.png",
    ) -> Path:
        """
        Create a bar chart comparing solve rates across models.

        Args:
            leaderboard: List of model statistics dictionaries
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        return self.plot_model_comparison(
            leaderboard,
            metric="solve_rate",
            title=title or "Puzzle Solve Rate by Model",
            filename=filename,
        )

    def plot_token_usage(
        self,
        leaderboard: list[dict],
        title: Optional[str] = None,
        filename: str = "token_usage.png",
    ) -> Path:
        """
        Create a bar chart comparing token usage across models.

        Args:
            leaderboard: List of model statistics dictionaries
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        return self.plot_model_comparison(
            leaderboard,
            metric="mean_tokens",
            title=title or "Mean Token Usage by Model",
            filename=filename,
        )

    def plot_accuracy_vs_tokens(
        self,
        leaderboard: list[dict],
        title: Optional[str] = None,
        filename: str = "accuracy_vs_tokens.png",
    ) -> Path:
        """
        Create a scatter plot of accuracy vs token usage.

        Args:
            leaderboard: List of model statistics dictionaries
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, entry in enumerate(leaderboard):
            ax.scatter(
                entry["mean_tokens"],
                entry["mean_accuracy"],
                s=200,
                c=self._get_color(i),
                label=entry["full_name"],
                alpha=0.7,
                edgecolors="black",
                linewidths=1,
            )

            # Add label
            ax.annotate(
                entry["full_name"].split("/")[-1],
                (entry["mean_tokens"], entry["mean_accuracy"]),
                xytext=(10, 5),
                textcoords="offset points",
                fontsize=9,
            )

        ax.set_xlabel("Mean Tokens Used")
        ax.set_ylabel("Mean Accuracy (%)")
        ax.set_title(title or "Accuracy vs Token Usage")

        # Add grid
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_path}")
        return output_path

    def plot_difficulty_ranking(
        self,
        image_difficulty: list[dict],
        title: Optional[str] = None,
        filename: str = "difficulty_ranking.png",
    ) -> Path:
        """
        Create a bar chart showing image difficulty ranking.

        Args:
            image_difficulty: List of image difficulty dictionaries
            title: Plot title
            filename: Output filename

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, max(6, len(image_difficulty) * 0.3)))

        images = [entry["image_name"] for entry in image_difficulty]
        accuracies = [entry["mean_accuracy"] for entry in image_difficulty]
        errors = [entry.get("std_accuracy", 0) for entry in image_difficulty]

        # Color gradient from red (hard) to green (easy)
        colors = plt.cm.RdYlGn([acc / 100 for acc in accuracies])

        bars = ax.barh(
            range(len(images)), accuracies, xerr=errors, color=colors, capsize=3, alpha=0.8
        )

        ax.set_yticks(range(len(images)))
        ax.set_yticklabels(images)
        ax.set_xlabel("Mean Accuracy (%)")
        ax.set_title(title or "Image Difficulty Ranking (hardest first)")
        ax.set_xlim(0, 105)

        # Add vertical line at 50%
        ax.axvline(x=50, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved plot: {output_path}")
        return output_path

    def generate_all_plots(
        self,
        report: dict,
        results: list[dict],
    ) -> list[Path]:
        """
        Generate all available plots from a benchmark report.

        Args:
            report: The benchmark report dictionary
            results: List of all result dictionaries

        Returns:
            List of paths to generated plots
        """
        plots = []

        leaderboard = report.get("leaderboard", [])
        per_image = report.get("per_image_results", {})
        difficulty = report.get("image_difficulty", [])

        if leaderboard:
            plots.append(self.plot_model_comparison(leaderboard))
            plots.append(self.plot_solve_rate_comparison(leaderboard))
            plots.append(self.plot_token_usage(leaderboard))
            plots.append(self.plot_accuracy_vs_tokens(leaderboard))

        if results:
            plots.append(self.plot_accuracy_distribution(results))

        if per_image:
            plots.append(self.plot_image_heatmap(per_image))

        if difficulty:
            plots.append(self.plot_difficulty_ranking(difficulty))

        logger.info(f"Generated {len(plots)} plots in {self.output_dir}")
        return plots
