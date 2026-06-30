
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_data.csv")

# Targets
y_reg = df["Fare"]
y_clf = df["Survived"]

X = df.drop(columns=["Fare", "Survived"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

numeric = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", numeric, num_cols),
    ("cat", categorical, cat_cols)
])

X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

_, _, yc_train, yc_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

X_train_t = pre.fit_transform(X_train)
X_test_t = pre.transform(X_test)

# Regression
lr = LinearRegression()
lr.fit(X_train_t, yr_train)
pred = lr.predict(X_test_t)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_t, yr_train)
ridge_pred = ridge.predict(X_test_t)

print("OLS MSE", mean_squared_error(yr_test, pred))
print("OLS R2", r2_score(yr_test, pred))
print("RIDGE MSE", mean_squared_error(yr_test, ridge_pred))
print("RIDGE R2", r2_score(yr_test, ridge_pred))

# Classification
clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

clf.fit(X_train_t, yc_train)

proba = clf.predict_proba(X_test_t)[:,1]
yp = (proba >= 0.5).astype(int)

print(confusion_matrix(yc_test, yp))
print(classification_report(yc_test, yp))

auc = roc_auc_score(yc_test, proba)
print("AUC", auc)

fpr, tpr, _ = roc_curve(yc_test, proba)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC AUC={auc:.3f}")
plt.savefig("roc_curve.png")

thresholds = [0.30,0.40,0.50,0.60,0.70]

for t in thresholds:
    y=(proba>=t).astype(int)
    print(
        t,
        precision_score(yc_test,y),
        recall_score(yc_test,y),
        f1_score(yc_test,y)
    )

# Regularization
clf2 = LogisticRegression(
    C=0.01,
    max_iter=1000,
    class_weight="balanced"
)

clf2.fit(X_train_t, yc_train)

p1 = clf.predict_proba(X_test_t)[:,1]
p2 = clf2.predict_proba(X_test_t)[:,1]

diff=[]

for _ in range(500):
    idx=np.random.choice(
        len(yc_test),
        len(yc_test),
        replace=True
    )
    d=(
        roc_auc_score(yc_test.iloc[idx],p1[idx])
        -
        roc_auc_score(yc_test.iloc[idx],p2[idx])
    )
    diff.append(d)

print(
    np.mean(diff),
    np.percentile(diff,2.5),
    np.percentile(diff,97.5)
)
