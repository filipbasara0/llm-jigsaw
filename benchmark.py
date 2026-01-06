#!/usr/bin/env python3
"""
Benchmark CLI for running jigsaw puzzle evaluations across multiple models and images.

This script provides a comprehensive benchmarking framework with:
- Multi-model evaluation with consistent shuffles
- Resumable execution with result caching
- Detailed metrics and reports
- Visualization plots

Examples:
    # Run benchmark with multiple models
    python benchmark.py \\
        --models google/gemini-3-pro-preview openai/gpt-5.2 anthropic/claude-opus-4-5 \\
        --image-folder images \\
        --grid-size 4 \\
        --runs-per-image 3 \\
        --output benchmark_results

    # Resume a previous benchmark run
    python benchmark.py --output benchmark_results --resume

    # Generate plots from existing results
    python benchmark.py --output benchmark_results --plots-only

    # Use a config file
    python benchmark.py --config benchmark_config.json
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.benchmark import (
    BenchmarkConfig,
    ModelSpec,
    BenchmarkRunner,
    ResultsAggregator,
    BenchmarkPlotter,
)


def setup_logging(verbose: bool, log_file: str | None = None) -> None:
    """Configure logging."""
    level = logging.INFO if verbose else logging.WARNING

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def parse_grid_size(value: str) -> int | tuple[int, int]:
    """
    Parse grid size argument.

    Accepts:
        - Single integer: "4" -> 4 (4x4 square grid)
        - Two integers with 'x': "3x5" -> (3, 5) (3 rows x 5 columns)
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
                return rows
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


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="LLM Jigsaw Puzzle Benchmark - Evaluate multiple models on spatial reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with two models, 3 runs per image
  python benchmark.py \\
      --models google/gemini-3-pro-preview openai/gpt-5.2 \\
      --image-folder images \\
      --grid-size 4 \\
      --runs-per-image 3

  # Resume an interrupted benchmark
  python benchmark.py --output benchmark_results --resume

  # Generate report and plots from existing results
  python benchmark.py --output benchmark_results --report-only

  # Use custom settings
  python benchmark.py \\
      --models google/gemini-3-pro-preview \\
      --image-folder images \\
      --grid-size 5x5 \\
      --max-turns 100 \\
      --runs-per-image 5 \\
      --seed 42
        """,
    )

    # Model specification
    parser.add_argument(
        "--models",
        "-m",
        type=str,
        nargs="+",
        help="Models to benchmark in provider/model-name format "
        "(e.g., 'google/gemini-3-pro-preview', 'openai/gpt-5.2')",
    )

    # Image settings
    parser.add_argument(
        "--image-folder",
        "-i",
        type=str,
        default="images",
        help="Folder containing images to use (default: images)",
    )
    parser.add_argument(
        "--image-patterns",
        type=str,
        nargs="+",
        default=["*.jpg", "*.jpeg", "*.png", "*.webp"],
        help="Glob patterns for image files (default: *.jpg *.jpeg *.png *.webp)",
    )

    # Benchmark settings
    parser.add_argument(
        "--grid-size",
        "-g",
        type=parse_grid_size,
        default=4,
        help="Puzzle grid size (e.g., '4' for 4x4, '3x5' for rectangular) (default: 4)",
    )
    parser.add_argument(
        "--runs-per-image",
        "-r",
        type=int,
        default=1,
        help="Number of runs per image with different shuffles (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible shuffling (default: 42)",
    )

    # Game settings
    parser.add_argument(
        "--max-moves-per-turn",
        type=int,
        default=16,
        help="Maximum swaps per turn (default: 16)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="Maximum number of turns per game (default: 12)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        choices=["none", "low", "medium", "high"],
        help="Reasoning effort for reasoning models (default: none)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize images so shorter side equals this value",
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
        "--colored-borders",
        action="store_true",
        help="Enable colored cell borders",
    )

    # Hint settings
    parser.add_argument(
        "--no-correct-count",
        action="store_true",
        help="Don't show how many pieces are correct",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Don't provide the original image as reference",
    )
    parser.add_argument(
        "--annotate-reference",
        action="store_true",
        help="Add grid lines/coordinates to the reference image",
    )
    parser.add_argument(
        "--no-move-history",
        action="store_true",
        help="Don't include move history in prompts",
    )

    # Output settings
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results (default: benchmark_TIMESTAMP)",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save intermediate puzzle images for each run",
    )
    parser.add_argument(
        "--save-gifs",
        action="store_true",
        help="Save GIF animations for each run",
    )

    # Execution control
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous benchmark run from cache",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Force rerun of all benchmarks, ignoring cache",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run different providers in parallel for the same puzzle (speeds up multi-model benchmarks)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers when --parallel is enabled (default: 4)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report and plots from existing results without running new benchmarks",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate plots only from existing report",
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save configuration to JSON file and exit",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress most output",
    )

    return parser


def build_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Build BenchmarkConfig from command line arguments."""
    # Parse models
    models = []
    if args.models:
        for model_str in args.models:
            try:
                model = ModelSpec.from_string(model_str)
                model.reasoning_effort = args.reasoning_effort
                models.append(model)
            except ValueError as e:
                print(f"Error parsing model: {e}", file=sys.stderr)
                sys.exit(1)

    # Determine output directory
    output_dir = args.output
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"benchmark_{timestamp}"

    return BenchmarkConfig(
        models=models,
        image_folder=args.image_folder,
        image_patterns=args.image_patterns,
        grid_size=args.grid_size,
        runs_per_image=args.runs_per_image,
        base_seed=args.seed,
        max_moves_per_turn=args.max_moves_per_turn,
        max_turns=args.max_turns,
        resize_to=args.resize,
        annotation_mode=args.annotation_mode,
        colored_borders=args.colored_borders,
        show_correct_count=not args.no_correct_count,
        show_reference_image=not args.no_reference,
        annotate_reference_image=args.annotate_reference,
        include_move_history=not args.no_move_history,
        output_dir=output_dir,
        save_intermediate_images=args.save_images,
        save_gifs=args.save_gifs,
        parallel_providers=args.parallel,
        max_parallel_workers=args.max_workers,
    )


def generate_report_and_plots(output_dir: Path, verbose: bool = True) -> None:
    """Generate report and plots from existing results."""
    from src.benchmark.cache import BenchmarkCache

    cache = BenchmarkCache(output_dir)
    results = cache.get_all_results()

    if not results:
        print("No results found in cache. Run benchmarks first.", file=sys.stderr)
        sys.exit(1)

    # Aggregate results
    aggregator = ResultsAggregator()
    aggregator.add_results(results)

    # Save report
    report_path = output_dir / "benchmark_report.json"
    aggregator.save_report(report_path)

    # Save CSV
    csv_path = output_dir / "benchmark_results.csv"
    aggregator.generate_csv(csv_path)

    # Print summary
    if verbose:
        aggregator.print_summary()

    # Generate plots
    try:
        plotter = BenchmarkPlotter(output_dir / "plots")
        report = aggregator.generate_report()
        plotter.generate_all_plots(report, list(results.values()))
        print(f"\nPlots saved to: {output_dir / 'plots'}")
    except ImportError:
        print("\nWarning: matplotlib not installed. Skipping plot generation.")
        print("Install with: pip install matplotlib")

    print(f"\nReport saved to: {report_path}")
    print(f"CSV saved to: {csv_path}")


def run_benchmark(config: BenchmarkConfig, force_rerun: bool = False) -> None:
    """Run the benchmark suite."""
    # Validate configuration
    if not config.models:
        print("Error: No models specified. Use --models to specify models.", file=sys.stderr)
        sys.exit(1)

    try:
        images = config.get_images()
        if not images:
            print(f"Error: No images found in {config.image_folder}", file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("LLM JIGSAW PUZZLE BENCHMARK")
    print("=" * 60)
    print(f"Models: {[m.full_name for m in config.models]}")
    print(f"Images: {len(images)} from {config.image_folder}")
    print(f"Grid size: {config.grid_size}")
    print(f"Runs per image: {config.runs_per_image}")
    print(f"Total runs: {len(config.models) * len(images) * config.runs_per_image}")
    if config.parallel_providers:
        print(f"Parallel execution: enabled (max {config.max_parallel_workers} workers)")
    else:
        print("Parallel execution: disabled (use --parallel to enable)")
    print(f"Output: {config.output_dir}")
    print("=" * 60)

    # Progress callback
    def progress_callback(completed: int, total: int, message: str) -> None:
        pct = completed / total * 100 if total > 0 else 0
        print(f"[{pct:5.1f}%] {message}")

    # Run benchmark
    runner = BenchmarkRunner(config, progress_callback=progress_callback)

    try:
        summary = runner.run(force_rerun=force_rerun)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted. Results have been cached.")
        print(f"Resume with: python benchmark.py --output {config.output_dir} --resume")
        sys.exit(130)

    # Generate report and plots
    output_dir = Path(config.output_dir)
    generate_report_and_plots(output_dir, verbose=True)


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Handle config file
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        config = BenchmarkConfig.load(config_path)

        # Override with command line arguments if provided
        if args.output:
            config.output_dir = args.output
    else:
        config = build_config_from_args(args)

    # Save config if requested
    if args.save_config:
        config.save(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
        return

    # Setup logging with output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(output_dir / "benchmark.log")
    setup_logging(not args.quiet, log_file)

    # Handle different execution modes
    if args.plots_only:
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)

        # Just regenerate plots from existing report
        report_path = output_dir / "benchmark_report.json"
        if not report_path.exists():
            print("No existing report found. Generating from cache...")
            generate_report_and_plots(output_dir, verbose=not args.quiet)
        else:
            import json

            with open(report_path) as f:
                report = json.load(f)

            from src.benchmark.cache import BenchmarkCache

            cache = BenchmarkCache(output_dir)
            results = list(cache.get_all_results().values())

            try:
                plotter = BenchmarkPlotter(output_dir / "plots")
                plotter.generate_all_plots(report, results)
                print(f"Plots saved to: {output_dir / 'plots'}")
            except ImportError:
                print("Error: matplotlib not installed. Install with: pip install matplotlib")
                sys.exit(1)
        return

    if args.report_only:
        if not output_dir.exists():
            print(f"Error: Output directory not found: {output_dir}", file=sys.stderr)
            sys.exit(1)
        generate_report_and_plots(output_dir, verbose=not args.quiet)
        return

    # Run benchmark
    run_benchmark(config, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
