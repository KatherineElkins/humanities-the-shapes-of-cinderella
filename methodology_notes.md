# Methodology Notes: AI-Assisted Sentiment Analysis

## Cross-Validation with Multiple AI Systems

This project employed multiple AI systems (Claude, Grok) to cross-validate sentiment scoring. **Every sentiment score was reviewed clause-by-clause**, with Grok providing detailed critiques and suggested adjustments that were incorporated into the final dataset.

## Cross-Validation Process

1. **Initial scoring:** Claude generated clause-by-clause sentiment scores (-5 to +5)
2. **Cross-validation:** Grok reviewed each score against narrative context and cultural tone
3. **Adjudication:** Researcher (K. Elkins) evaluated suggested changes and incorporated appropriate adjustments
4. **Final verification:** Scores verified against source texts

## Clause Counts by Version

| Version | Clauses | Language | Original Text Source |
|---------|---------|----------|---------------------|
| Ye Xian | 112 | Classical Chinese | Youyang Zazu (酉陽雜俎), c. 850 CE |
| Perrault | 257 | French | Histoires ou Contes du Temps Passé, 1697 |
| Grimm 1812 | 275 | German | Kinder- und Hausmärchen, 1st ed. |
| Grimm 1857 | 315 | German | Kinder- und Hausmärchen, 7th ed. |

## Why Clause-Level Analysis?

Clause-by-clause analysis was chosen because:
1. It allows for better context-informed scoring than single words
2. Different languages have different sentence-length "tolerances"
3. Classical Chinese lacks Western sentence structure
4. Long French sentences in Perrault contain multiple emotional shifts

## Smoothing Parameters

### Savitzky-Golay Polynomial Smoothing

We chose Savitzky-Golay over simple moving averages because it better preserves peaks and valleys while reducing noise.

**Parameters:**
- Window sizes: 7 (medium), 15 (heavy)
- Polynomial order: 2
- Edge handling: Reflection padding

## Grok's Shape Characterizations

| Version | Shape | Description |
|---------|-------|-------------|
| Ye Xian | Double valley with decline | Asymmetrical W, gradual depletion ending |
| Perrault | Smooth hill | Steady rise, positive resolution |
| Grimm 1812 | W-shaped curve | Multiple rises and falls, emotional volatility |
| Grimm 1857 | Mountain with cliff | Gradual rise, sharp punitive drop at end |

## Files for Replication

- `ye_xian_sentiment_scores.csv` (112 clauses)
- `perrault_sentiment_scores.csv` (257 clauses)
- `grimm_1812_sentiment_scores.csv` (275 clauses)
- `grimm_1857_sentiment_scores.csv` (315 clauses)

## Citation

Elkins, K. (2025). The Shapes of Cinderella: Emotional Architecture and the Language of Moral Difference. *Humanities*, 14(10), 198. https://doi.org/10.3390/h14100198
