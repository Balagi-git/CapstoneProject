from pathlib import Path

content = """
# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis (Titanic Dataset)

## Dataset
Titanic CSV dataset

## Sections Covered
1. Data Loading
2. Null Value Analysis
3. Duplicate Detection
4. Data Type Correction
5. Descriptive Statistics and Skewness
6. Outlier Detection (IQR)
7. Visualizations
8. Correlation Heat Map
9. Imputation Strategy Comparison
10. Spearman Correlation
11. Grouped Aggregation

## Key Decisions

### Missing Values
- Numeric columns under 20% nulls → median imputation
- Median chosen because skewed distributions make mean sensitive to outliers

### Dtype Optimization
- Repetitive text columns converted to category

### Outliers
- Identified using IQR
- Retained for later modeling decisions

### Correlation
- Pearson for linear relationships
- Spearman for monotonic / skewed relationships

## Outputs
- cleaned_data.csv
- line_plot.png
- bar_chart.png
- histogram.png
- scatter.png
- boxplot.png
- heatmap.png

## Final Note
Correlation does not imply causation.
"""

path = "/mnt/data/README_Part1_Titanic.md"
Path(path).write_text(content, encoding="utf-8")

print(path)
