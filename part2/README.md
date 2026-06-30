
# Part 2 README

Regression target:
Fare

Classification target:
Survived

Encoding:
- Categorical columns → One-hot encoding
- First dummy dropped to avoid multicollinearity.

Leakage prevention:
Scaler fitted only on training data.

Regression:
Compare OLS and Ridge.

Coefficient meaning:
Positive → increases prediction.
Negative → decreases prediction.

Precision:
TP / (TP + FP)

Recall:
TP / (TP + FN)

Threshold study:
0.30–0.70 evaluated.

AUC:
Probability model ranks positive class above negative.

Regularization:
Smaller C → stronger penalty.

Bootstrap:
500 resamples.
If CI excludes zero → difference likely reliable.
