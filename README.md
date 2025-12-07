# The Shapes of Cinderella: Emotional Architecture and the Language of Moral Difference

**Katherine Elkins, Kenyon College**

*Humanities* 2025, 14(10), 198  
DOI: [10.3390/h14100198](https://doi.org/10.3390/h14100198)

---

## Overview

This repository contains data and code for reproducing the sentiment analysis visualizations in "The Shapes of Cinderella," which examines how four versions of the Cinderella tale encode different moral architectures through their emotional trajectories.

## Versions Analyzed

| Version | Date | Language | Clauses |
|---------|------|----------|---------|
| Ye Xian | c. 850 CE | Classical Chinese | 112 |
| Perrault's *Cendrillon* | 1697 | French | 257 |
| Grimm's *Aschenputtel* | 1812 | German | 275 |
| Grimm's *Aschenputtel* | 1857 | German | 315 |

## Repository Structure

```
├── data/                          # Clause-level sentiment scores
│   ├── ye_xian_sentiment_scores.csv
│   ├── perrault_sentiment_scores.csv
│   ├── grimm_1812_sentiment_scores.csv
│   └── grimm_1857_sentiment_scores.csv
├── images/                        # Publication figures
├── cinderella_sentiment_figures.py  # Replicable visualization code
└── README.md
```

## Methodology

Each text was segmented into clauses (not sentences) to capture emotional shifts within complex sentences across different linguistic traditions. Each clause was scored on a scale from -5 (most negative: death, murder) to +5 (most positive: transformation, recognition, marriage).

**Smoothing**: Savitzky-Golay filter with reflection padding  
- Medium smoothing: window = 7  
- Heavy smoothing: window = 15

**Cross-validation**: Three LLMs (Grok 3, Claude 4, Claude 4.1 Opus) scored independently; human adjudication resolved disagreements.

## Usage

```bash
# Install requirements
pip install numpy matplotlib scipy

# Generate all figures
python cinderella_sentiment_figures.py
```

This will produce:
- `figure1_ye_xian.png`
- `figure2_perrault.png`
- `figure3_grimm_1812.png`
- `figure4_grimm_1857.png`
- `figure5_comparative.png`

## Key Findings

- **Ye Xian**: Double valley with gradual decline; recognition through sacred reciprocity
- **Perrault**: Smooth hill; linguistic violence and material escape  
- **Grimm 1812**: W-shaped volatility; productive suffering
- **Grimm 1857**: Mountain with cliff; Christianization and divine retribution

All versions cluster transformation scenes at 41-54% of narrative progression.

## Citation

```bibtex
@article{elkins2025shapes,
  title={The Shapes of Cinderella: Emotional Architecture and the Language of Moral Difference},
  author={Elkins, Katherine},
  journal={Humanities},
  volume={14},
  number={10},
  pages={198},
  year={2025},
  publisher={MDPI},
  doi={10.3390/h14100198}
}
```

## License

MIT License
