# Benchmark Results

Comprehensive benchmark results testing frontier multimodal LLMs on jigsaw puzzle solving across different grid sizes.

---

## Summary

| Grid Size | Pieces | GPT-5.2 Accuracy | GPT-5.2 Solve Rate | Gemini 3 Pro Accuracy | Gemini 3 Pro Solve Rate |
|-----------|--------|------------------|--------------------|-----------------------|-------------------------|
| 3×3       | 9      | 96.7%            | **95%**            | 93.3%                 | 85%                     |
| 4×3/3×4   | 12     | 93.5%            | **85%**            | 85.8%                 | 62.5%                   |
| 4×4       | 16     | 76.9%            | **40%**            | 71.9%                 | 25%                     |
| 5×5       | 25     | 46.4%            | 0%                 | 49.2%                 | **10%**                 |

!!! note "Test configuration"
    Results averaged across 20 images per model per grid size. All models received the **reference image**, **correct piece count**, and **last 3 moves** as context. Claude Opus 4.5 tested only on 3×3 (20% solve rate, 47.2% piece accuracy). Gemini-3.0 Pro and GPT-5.2 were using low reasoning effort, while Opus 4.5 was using high reasoning effort.

---

## Performance vs Puzzle Complexity

![Performance vs Grid Size](assets/grid_size_analysis.png)

### Key Findings

<div class="grid cards" markdown>

- 📉 **Steep Difficulty Scaling**
  
    Solve rates drop dramatically as puzzle complexity increases
    
    - GPT-5.2: 95% → 0% solve rate from 3×3 to 5×5
    - Gemini 3 Pro: 85% → 10% solve rate from 3×3 to 5×5

- 🪙 **Token Usage Increases**
  
    Models require significantly more tokens for larger puzzles
    
    - GPT-5.2: ~15K → ~116K tokens (3×3 to 5×5)
    - Gemini 3 Pro: ~55K → ~345K tokens

- ❌ **5×5 Remains Unsolved**
  
    No model reliably solves 5×5 puzzles – even frontier models struggle with 25 pieces

- 📊 **Partial Progress Common**
  
    Piece accuracy remains reasonable (50-80%) even when puzzles aren't fully solved

</div>

---

## Detailed Results by Grid Size

### 3×3 Grid (9 pieces)

| Model | Piece Accuracy | Solve Rate | Avg Turns | Avg Tokens |
|-------|----------------|------------|-----------|------------|
| GPT-5.2 | 96.7% ± 14.9% | **95%** | 2.9 | 14,487 |
| Gemini 3 Pro | 93.3% ± 17.4% | 85% | 3.8 | 54,770 |
| Claude Opus 4.5 | 47.2% ± 33.2% | 20% | 11.3 | 33,822 |

!!! success "Best performance"
    Both GPT-5.2 and Gemini 3 Pro solve 3×3 puzzles reliably with 85%+ solve rates.

---

### 4×4 Grid (16 pieces)

| Model | Piece Accuracy | Solve Rate | Avg Turns | Avg Tokens |
|-------|----------------|------------|-----------|------------|
| GPT-5.2 | 76.9% ± 21.9% | **40%** | 13.6 | 76,936 |
| Gemini 3 Pro | 71.9% ± 24.3% | 25% | 14.8 | 281,648 |

!!! warning "Degraded performance"
    Performance drops significantly – only 25-40% of puzzles are solved completely. Using higher reasoning would probably have a high positive impact on solve rate.

---

### 5×5 Grid (25 pieces)

| Model | Piece Accuracy | Solve Rate | Avg Turns | Avg Tokens |
|-------|----------------|------------|-----------|------------|
| GPT-5.2 | 46.4% ± 13.9% | 0% | 20.0 | 115,918 |
| Gemini 3 Pro | 49.2% ± 27.3% | **10%** | 18.6 | 345,060 |

!!! failure "Spatial reasoning limit"
    Neither model can reliably solve 25-piece puzzles. Models consistently hit a wall around 50% accuracy. Higher reasoning effort marginally improves accuracy but has minimal impact on solve rate (conclusions from limited testing on several images).
---

## Methodology

<div class="grid" markdown>

| Parameter | Value |
|-----------|-------|
| **Images** | 20 diverse test images |
| **Categories** | Landscapes, portraits, abstract art, photos |
| **Seed** | Fixed seed (42) for reproducibility |
| **Max turns** | 12 (3×3), 17 (4×4), 20 (5×5) |
| **Hints** | **Reference image**, **correct count**, **last 3 moves** |
| **Image size** | Resized to 512px shortest side |

</div>

### Reasoning Effort

!!! info "Reasoning configuration"
    GPT-5.2 and Gemini 3 Pro used `low` reasoning effort; Claude Opus 4.5 used `high` reasoning effort. Neither can solve puzzles without reasoning even for 3×3 grid sizes.

!!! warning "Higher reasoning tradeoffs"
    Informal testing with `high` reasoning effort for GPT-5.2 and Gemini 3 Pro showed slightly better performance (up to ~10%), but at significantly higher cost:
    
    - A single puzzle could consume ~1M tokens with Gemini
    - Much longer solving times
    - Requests would quite often time out for `medium` or `high` reasoning for GPT-5.2
    - Both GPT and Gemini would still be stuck at ~50%-80% piece accuracy on average
    
    We opted for `low` reasoning to keep the benchmark practical.

---

## Reproducing Results

```bash
# Run benchmark for a specific grid size
python benchmark.py \
  --grid-size 3 \
  --models openai/gpt-5.2 google/gemini-3-pro-preview \
  --images-dir images/ \
  --seed 42 \
  --resize 512
```

See the [CLI Usage Guide](usage.md) and [Benchmark Guide](benchmark.md) for full documentation.
