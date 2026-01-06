# Installation & Usage Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-jigsaw.git
cd llm-jigsaw

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# Run a simple puzzle
python main.py --image images/sample.jpg --resize 512 --grid-size 3 --model openai/gpt-5.2
```

## Tips for Best Results

1. **Start Small**: Begin with 3×3 to verify setup, then increase difficulty
2. **Use Seeds**: Set `--seed` for reproducible experiments
3. **Correct Count**: Correct count is shown by default; this helps models converge
4. **Resize Images**: Use eg. `--resize 512` to reduce token usage and speed up API calls
5. **Use Reasoning**: For grids larger than 3×3, enable `--reasoning-effort low` or `medium/high` for better results

## Command Line Reference

### Basic Usage

```bash
python main.py --image <path> --grid-size <n> [options]
```

## Examples

### Easy Puzzle with GPT-5.2
```bash
python main.py \
  --image images/landscape.jpg \
  --resize 512 \
  --grid-size 3 \
  --model openai/gpt-5.2 \
  --seed 42
```

### Rectangular Grid (3 rows × 5 columns)
```bash
python main.py \
  --image images/panorama.jpg \
  --resize 512 \
  --grid-size 3x5 \
  --model openai/gpt-5.2
```

### Medium Puzzle with Claude
```bash
python main.py \
  --image images/artwork.jpg \
  --resize 512 \
  --grid-size 4 \
  --model anthropic/claude-opus-4-5 \
  --max-turns 50
```

### Hard Puzzle with Full Logging
```bash
python main.py \
  --image images/photo.jpg \
  --resize 512 \
  --grid-size 5 \
  --model openai/gpt-5.2 \
  --save-images \
  --output results/hard_run/
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--image`, `-i` | Path to the puzzle image |
| `--grid-size`, `-g` | Grid size: single number for square (e.g., `4`) or `NxM` for rectangular (e.g., `3x5`) |

### LLM Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model`, `-m` | (required) | Model in `provider/model-name` format (e.g., `openai/gpt-5.2`, `google/gemini-3-pro-preview`, `anthropic/claude-opus-4-5`) |
| `--api-key` | env var | API key (or use environment variable) |
| `--base-url` | None | Custom base URL for OpenAI-compatible APIs |
| `--reasoning-effort` | none | Reasoning effort for OpenAI reasoning models (none, low, medium, high) |

### Game Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-moves-per-turn` | 16 | Maximum swaps per turn |
| `--max-turns` | 50 | Maximum number of turns |
| `--seed` | None | Random seed for reproducible shuffling |
| `--resize` | None | Resize image so shorter side equals this value (e.g., 512, 1024) |

### Annotation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--annotation-mode` | both | Label style (border_labels, cell_labels, both) |
| `--colored-borders` | False | Enable colored cell borders |

### Hint Options

| Argument | Description |
|----------|-------------|
| `--no-correct-count` | Don't show how many pieces are correctly placed (shown by default) |
| `--no-reference` | Don't provide the solved image as reference (shown by default) |
| `--annotate-reference` | Add grid lines/coordinates to the reference image |
| `--show-move-history` | Include move history in prompts (off by default) |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output`, `-o` | auto-generated | Output directory for results |
| `--save-images` | False | Save intermediate puzzle states |
| `--no-gif` | False | Don't save an animated GIF |
| `--gif-duration` | 500 | Duration of each frame in the GIF (ms) |
| `--quiet`, `-q` | False | Suppress progress output |

## Move Format

The LLM responds with JSON specifying swaps:

```json
{
  "reasoning": "The sky piece at 1,3 belongs at 1,1 based on color continuity",
  "moves": [
    {"op": "swap", "a": "1,1", "b": "1,3"},
    {"op": "swap", "a": "2,4", "b": "4,2"}
  ]
}
```

Results are saved to the output directory:

```
results/run_name/
├── result.json       # Complete metrics and move history
├── initial_state.png # Shuffled puzzle at start
├── final_state.png   # Puzzle state at end
├── game.gif          # Animated solving process
├── reference.png     # Original image (if --show-reference)
└── turn_*.png        # Intermediate states (if --save-images)
```

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `solved` | Whether puzzle was completed |
| `total_turns` | Number of LLM calls made |
| `solve_turn` | Turn when solved (if applicable) |
| `total_moves` | Total swaps made |
| `total_invalid_moves` | Moves that failed (bad coordinates) |
| `max_correct_achieved` | Best score reached |
| `final_correct` | Pieces correct at end |
| `accuracy` | Final percentage correct |
| `total_tokens` | Total API tokens used |
| `duration_seconds` | Wall-clock time |

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_image_processor.py -v
```

## Project Structure

```
llm-jigsaw/
├── src/
│   ├── __init__.py
│   ├── image_processor.py   # Image slicing and state management
│   ├── grid_annotator.py    # Visual annotations
│   ├── llm_interface.py     # LLM API abstraction
│   ├── game.py              # Game controller
│   ├── metrics.py           # Result tracking
│   └── prompts.py           # Prompt templates
├── tests/
│   ├── test_image_processor.py
│   ├── test_grid_annotator.py
│   └── test_game.py
├── images/                   # Test images
├── results/                  # Output directory
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```
