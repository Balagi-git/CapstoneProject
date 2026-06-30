
# README Part 3

## Decision Tree
Unconstrained trees may overfit because they greedily split until leaves become highly specific.

max_depth:
Limits complexity and reduces variance.

min_samples_split:
Prevents unstable splits on tiny subsets.

## Gini vs Entropy

Gini:
1 − Σ(pi²)

Entropy:
−Σ(pi log2(pi))

Gini = 0:
Pure node.

## Random Forest

Feature importance:
Average reduction in Gini impurity across trees.

Bagging:
Bootstrap samples + random feature subsets reduce variance.

## Gradient Boosting
Sequentially corrects previous tree errors.

## Cross Validation
More reliable than one split because all observations participate in validation.

## Grid Search
Configurations:
3 × 3 × 2 × 5 folds = 90 fits

Grid Search vs Randomized Search:
Grid is exhaustive but slower.

## Learning Curve
Increasing data can reveal whether performance is data-limited.

## Serialization
Best model stored in best_model.pkl

## Recommendation
Choose highest CV AUC with stable variance.

