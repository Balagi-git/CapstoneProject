import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD DATA
# =====================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DTYPES ==========")
print(df.dtypes)

print("\n========== SHAPE ==========")
print(df.shape)

# =====================================================
# NULL ANALYSIS
# =====================================================

print("\n========== NULL ANALYSIS ==========")

null_count = df.isnull().sum()

null_percent = (
    df.isnull().sum()
    / df.shape[0]
) * 100

null_df = pd.DataFrame({
    "Null_Count": null_count,
    "Null_Percent": null_percent
})

print(null_df)

print("\nColumns >20% null")

high_null = null_df[
    null_df["Null_Percent"] > 20
]

print(high_null)

# fill numeric columns (<20%)

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    if (
        null_percent[col] > 0
        and null_percent[col] < 20
    ):

        df[col] = df[col].fillna(
            df[col].median()
        )

print("\nRemaining Nulls")
print(df.isnull().sum())

# =====================================================
# DUPLICATES
# =====================================================

print("\n========== DUPLICATES ==========")

duplicates = df.duplicated().sum()

print(
    "Duplicate rows:",
    duplicates
)

rows_before = df.shape[0]

df = df.drop_duplicates()

rows_after = df.shape[0]

print(
    "Rows removed:",
    rows_before - rows_after
)

# =====================================================
# DATA TYPE CORRECTION
# =====================================================

print("\n========== DTYPE CORRECTION ==========")

memory_before = (
    df.memory_usage(
        deep=True
    ).sum()
)

# Example conversion

df["Embarked"] = (
    df["Embarked"]
    .astype("category")
)

df["Sex"] = (
    df["Sex"]
    .astype("category")
)

memory_after = (
    df.memory_usage(
        deep=True
    ).sum()
)

print(
    "Memory Before:",
    memory_before
)

print(
    "Memory After:",
    memory_after
)

# =====================================================
# DESCRIPTIVE STATS
# =====================================================

print("\n========== DESCRIBE ==========")

print(
    df.describe()
)

print("\n========== SKEWNESS ==========")

skew_values = {}

for col in numeric_cols:

    skew_values[col] = (
        df[col]
        .dropna()
        .skew()
    )

skew_df = pd.DataFrame({
    "Skew": skew_values
})

print(skew_df)

highest_skew = (
    skew_df["Skew"]
    .abs()
    .idxmax()
)

print(
    "\nMost skewed:",
    highest_skew
)

# =====================================================
# OUTLIERS
# =====================================================

print("\n========== IQR OUTLIERS ==========")

target_cols = [
    "Age",
    "Fare"
]

for col in target_cols:

    Q1 = (
        df[col]
        .quantile(0.25)
    ) 

    Q3 = (
        df[col]
        .quantile(0.75)
    )

    IQR = Q3 - Q1

    lower = (
        Q1
        - 1.5 * IQR
    )

    upper = (
        Q3
        + 1.5 * IQR
    )

    outliers = (
        (
            df[col]
            < lower
        )
        |
        (
            df[col]
            > upper
        )
    ).sum()

    print(
        col,
        "Outliers:",
        outliers
    )

# =====================================================
# VISUALIZATION
# =====================================================

print("\nGenerating Charts...")

# Line Plot

plt.figure()

plt.plot(
    df.index,
    df["Fare"]
)

plt.title(
    "Fare Trend"
)

plt.xlabel(
    "Passenger"
)

plt.ylabel(
    "Fare"
)

plt.savefig(
    "line_plot.png"
)

plt.close()

# Bar Chart

grouped = (
    df.groupby(
        "Sex"
    )["Fare"]
    .mean()
)

plt.figure()

grouped.plot.bar()

plt.title(
    "Mean Fare by Sex"
)

plt.xlabel(
    "Sex"
)

plt.ylabel(
    "Mean Fare"
)

plt.savefig(
    "bar_chart.png"
)

plt.close()

# Histogram

plt.figure()

sns.histplot(
    df[highest_skew],
    bins=20
)

plt.title(
    f"{highest_skew} Histogram"
)

plt.savefig(
    "histogram.png"
)

plt.close()

# Scatter

plt.figure()

sns.scatterplot(
    data=df,
    x="Age",
    y="Fare"
)

plt.title(
    "Age vs Fare"
)

plt.savefig(
    "scatter.png"
)

plt.close()

# Boxplot

plt.figure()

sns.boxplot(
    data=df,
    x="Sex",
    y="Fare"
)

plt.title(
    "Fare by Sex"
)

plt.savefig(
    "boxplot.png"
)

plt.close()

# =====================================================
# PEARSON
# =====================================================

print("\n========== PEARSON ==========")

pearson = (
    df.corr(
        numeric_only=True
    )
)

print(pearson)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    pearson,
    annot=True
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "heatmap.png"
)

plt.close()

# =====================================================
# IMPUTATION STRATEGY
# =====================================================

print(
    "\n========== MEAN VS MEDIAN =========="
)

top2 = (
    skew_df["Skew"]
    .abs()
    .sort_values(
        ascending=False
    )
    .head(2)
    .index
)

for col in top2:

    print(
        col,
        "Mean:",
        df[col].mean(),
        "Median:",
        df[col].median()
    )

# =====================================================
# SPEARMAN
# =====================================================

print(
    "\n========== SPEARMAN =========="
)

spearman = (
    df.corr(
        method="spearman",
        numeric_only=True
    )
)

print(
    spearman
)

difference = (
    spearman
    - pearson
).abs()

print(
    "\nDifference Matrix"
)

print(
    difference
)

# =====================================================
# GROUP AGGREGATION
# =====================================================

print(
    "\n========== GROUP AGG =========="
)

group = (
    df.groupby(
        "Sex"
    )["Fare"]
    .agg(
        [
            "mean",
            "std",
            "count"
        ]
    )
)

print(
    group
)

ratio = (
    group["mean"].max()
    /
    group["mean"].min()
)

print(
    "\nMean Ratio:",
    ratio
)

# =====================================================
# SAVE CLEAN DATA
# =====================================================

df.to_csv(
    "cleaned_data.csv",
    index=False
)

print(
    "\nDONE"
)

print(
    "Generated:"
)

print(
    "- cleaned_data.csv"
)

print(
    "- line_plot.png"
)

print(
    "- bar_chart.png"
)

print(
    "- histogram.png"
)

print(
    "- scatter.png"
)

print(
    "- boxplot.png"
)

print(
    "- heatmap.png"
)