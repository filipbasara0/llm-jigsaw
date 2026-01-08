# Benchmark Guide

Run systematic evaluations across multiple models and images.

---

## Quick Start

```bash
# Set API keys
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"

# Run a benchmark
python benchmark.py \
  --models openai/gpt-5.2 google/gemini-3-pro-preview \
  --image-folder images \
  --grid-size 4 \
  --runs-per-image 1
```

---

## Example: Full 5×5 Benchmark

```bash
python benchmark.py \
  --models google/gemini-3-pro-preview openai/gpt-5.2 \
  --image-folder images \
  --grid-size 5 \
  --runs-per-image 1 \
  --save-images \
  --save-gifs \
  --reasoning-effort low \
  --resize 768 \
  --parallel
```

---

## Key Options

| Option | Description |
|--------|-------------|
| `--models` | Models to test (e.g., `openai/gpt-5.2 google/gemini-3-pro-preview`) |
| `--grid-size` | Puzzle size (`3`, `4`, `5`, or `3x5` for rectangular) |
| `--runs-per-image` | Runs per image with different shuffles |
| `--reasoning-effort` | `none`, `low`, `medium`, `high` (for reasoning models) |
| `--resize` | Resize images (e.g., `768`) to reduce tokens |
| `--parallel` | Run different providers in parallel |
| `--save-gifs` | Save animated GIFs of solving process |
| `--resume` | Resume interrupted benchmark |

---

## Output Structure

```
benchmark_TIMESTAMP/
├── benchmark_config.json   # Run configuration
├── benchmark_cache.json    # Results cache (for resume)
├── benchmark_results.csv   # Aggregated metrics
├── benchmark_report.json   # Summary statistics
├── plots/                  # Visualization charts
└── runs/                   # Individual run outputs
    └── model_image_seed/
        ├── result.json
        └── game.gif
```

---

## Resuming & Re-running

=== "Resume interrupted"

    ```bash
    python benchmark.py --output benchmark_20260104_101347 --resume
    ```

=== "Generate plots only"

    ```bash
    python benchmark.py --output benchmark_20260104_101347 --plots-only
    ```

=== "Force rerun"

    ```bash
    python benchmark.py --output benchmark_20260104_101347 --force-rerun
    ```

---

## Using Config Files

Save settings to JSON and reuse:

```bash
# Run with config file
python benchmark.py --config benchmark_config.json
```

---

## All Options

Run `python benchmark.py --help` for complete CLI reference.
