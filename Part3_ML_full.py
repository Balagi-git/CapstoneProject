


# PART 3 - Tree Models, Ensembles, CV, Grid Search, Serialization
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("cleaned_data.csv")

y = df["Survived"]
X = df.drop(columns=["Survived"])

cat = X.select_dtypes(include=["object","category"]).columns
num = X.select_dtypes(exclude=["object","category"]).columns

pre = ColumnTransformer([
("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
OneHotEncoder(drop="first", handle_unknown="ignore")), cat)
])

X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=42,stratify=y
)

Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)

# Decision Tree baseline
dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xtr,y_train)

print("DT Train", accuracy_score(y_train, dt.predict(Xtr)))
print("DT Test", accuracy_score(y_test, dt.predict(Xte)))

# Controlled Tree
dt2 = DecisionTreeClassifier(
max_depth=5,
min_samples_split=20,
random_state=42
)
dt2.fit(Xtr,y_train)

print("Controlled Test", accuracy_score(y_test, dt2.predict(Xte)))

# Gini vs Entropy
for c in ["gini","entropy"]:
    m=DecisionTreeClassifier(max_depth=5,criterion=c,random_state=42)
    m.fit(Xtr,y_train)
    print(c, accuracy_score(y_test,m.predict(Xte)))

# Random Forest
rf = RandomForestClassifier(
n_estimators=100,
max_depth=10,
random_state=42
)

rf.fit(Xtr,y_train)

prob = rf.predict_proba(Xte)[:,1]

print("RF AUC", roc_auc_score(y_test,prob))

# Gradient Boosting
gb = GradientBoostingClassifier(
n_estimators=100,
learning_rate=0.1,
random_state=42
)

gb.fit(Xtr,y_train)

# CV
cv = StratifiedKFold(
n_splits=5,
shuffle=True,
random_state=42
)

models = {
"LR": LogisticRegression(max_iter=1000),
"DT": dt2,
"RF": rf,
"GB": gb
}

for n,m in models.items():
    pipe = make_pipeline(pre,m)
    scores = cross_val_score(
        pipe,
        X,
        y,
        scoring="roc_auc",
        cv=cv
    )
    print(n, scores.mean(), scores.std())
# Grid Search (fixed)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_leaf": [1, 5]
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    error_score="raise"
)

gs.fit(Xtr, y_train)

print("Best Params:")
print(gs.best_params_)

print("Best Score:")
print(gs.best_score_)

joblib.dump(
    gs.best_estimator_,
    "best_model.pkl"
)

loaded = joblib.load("best_model.pkl")

print(
loaded.predict(
Xtr[:2]
))


