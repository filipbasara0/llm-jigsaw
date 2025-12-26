# LLM Jigsaw Puzzle Solver

A benchmark for testing multimodal LLM spatial reasoning capabilities through iterative jigsaw puzzle solving.

## Overview

This project shuffles an image into an N×N grid and challenges an LLM to restore the original image by iteratively swapping pieces. The task tests:

- **Visual understanding**: Recognizing piece content and how pieces fit together
- **Spatial reasoning**: Understanding grid coordinates and piece relationships
- **Iterative problem solving**: Making progress across multiple turns
- **Memory/context**: Tracking previous moves and learning from them

## Features

- **Configurable difficulty**: Grid sizes from 4×4 (easy) to 32×32 (hard)
- **Multiple LLM providers**: OpenAI, Anthropic, Google
- **Visual annotations**: Grid labels, colored borders for easy piece identification
- **Comprehensive metrics**: Tracks moves, accuracy, tokens, timing
- **Reproducible**: Seed-based shuffling for consistent benchmarks
- **Optional hints**: Show correct count, provide reference image

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

# Run a simple puzzle
python main.py --image images/sample.jpg --grid-size 4 --model gpt-4o
```

## Usage

### Basic Usage

```bash
python main.py --image <path> --grid-size <n> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--image`, `-i` | Path to the puzzle image |
| `--grid-size`, `-g` | Grid size (2, 4, 8, 16, or 32) |

### LLM Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--provider`, `-p` | openai | LLM provider (openai, anthropic, google) |
| `--model`, `-m` | gpt-4o | Model name |
| `--api-key` | env var | API key (or use environment variable) |
| `--base-url` | None | Custom base URL for OpenAI-compatible APIs |

### Game Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-moves-per-turn` | 16 | Maximum swaps per turn |
| `--max-turns` | 100 | Maximum number of turns |
| `--seed` | None | Random seed for reproducible shuffling |

### Annotation Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--annotation-mode` | both | Label style (border_labels, cell_labels, both) |
| `--no-colored-borders` | False | Disable colored cell borders |

### Hint Options

| Argument | Description |
|----------|-------------|
| `--show-correct-count` | Show how many pieces are correctly placed |
| `--show-reference` | Provide the solved image as reference |
| `--no-history` | Don't include move history in prompts |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--output`, `-o` | auto-generated | Output directory for results |
| `--save-images` | False | Save intermediate puzzle states |
| `--quiet`, `-q` | False | Suppress progress output |

## Examples

### Easy Puzzle with GPT-4o
```bash
python main.py \
  --image images/landscape.jpg \
  --grid-size 4 \
  --model gpt-4o \
  --seed 42
```

### Medium Puzzle with Claude, Hints Enabled
```bash
python main.py \
  --image images/artwork.jpg \
  --grid-size 8 \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --show-correct-count \
  --max-turns 50
```

### Hard Puzzle with Full Logging
```bash
python main.py \
  --image images/photo.jpg \
  --grid-size 16 \
  --model gpt-4o \
  --save-images \
  --output results/hard_run/
```

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

### Coordinate System

- Format: `"row,col"` (1-indexed)
- Origin: Top-left is `"1,1"`
- Example for 8×8: Top-left `"1,1"`, bottom-right `"8,8"`

## Output

Results are saved to the output directory:

```
results/run_name/
├── result.json       # Complete metrics and move history
├── initial_state.png # Shuffled puzzle at start
├── final_state.png   # Puzzle state at end
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

## Tips for Best Results

1. **Image Selection**: Choose images with clear structure and distinct regions
2. **Start Small**: Begin with 4×4 to verify setup, then increase difficulty
3. **Use Seeds**: Set `--seed` for reproducible experiments
4. **Enable History**: Move history helps the model avoid repeating mistakes
5. **Try Hints**: `--show-correct-count` can help struggling models

## Recommended Test Images

- **Landscapes**: Clear horizon lines and distinct sky/ground
- **Architecture**: Strong geometric patterns
- **Artwork**: Paintings with clear composition
- **Avoid**: Uniform textures (sky-only, walls), very busy patterns

## License

MIT License
