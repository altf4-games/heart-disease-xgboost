import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

col_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
    "oldpeak","slope","ca","thal","num"
]
paths = [
    "dataset/processed.cleveland.data",
    "dataset/processed.hungarian.data",
    "dataset/processed.switzerland.data",
    "dataset/processed.va.data"
]
df = pd.concat([pd.read_csv(p, header=None, names=col_names, na_values="?") for p in paths], ignore_index=True)
df = df.dropna()

def augment_df(df, target="num", noise_scale=0.03):
    df_aug = df.copy()
    counts = df[target].astype(int).value_counts()
    max_count = counts.max()
    to_concat = []
    for cls, cnt in counts.items():
        if cnt >= max_count:
            continue
        to_add = max_count - cnt
        cls_rows = df[df[target].astype(int) == cls].reset_index(drop=True)
        for i in range(to_add):
            base = cls_rows.iloc[i % len(cls_rows)].copy()
            for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
                val = float(base[col])
                if np.isfinite(val):
                    jitter = np.random.normal(0, noise_scale * max(abs(val), 1.0))
                    base[col] = max(0.0, val + jitter)
            to_concat.append(base)
    if to_concat:
        df_aug = pd.concat([df_aug, pd.DataFrame(to_concat)], ignore_index=True)
    return df_aug

augmented_path = "dataset/augmented_heart.data"
augmented = augment_df(df, target="num", noise_scale=0.03)
augmented.to_csv(augmented_path, index=False, header=False)
print(f"Saved augmented dataset to {augmented_path}")

# feature engineering
_df = augmented.copy()
np.random.seed(42)
_df["stress"] = np.random.randint(1, 100, size=len(_df))
_df["cp_restecg"] = _df["cp"] * _df["restecg"]
_df["thalach_per_age"] = _df["thalach"] / (_df["age"] + 1)
_df["oldpeak_slope"] = _df["oldpeak"] * _df["slope"]
_df["chol_ratio"] = _df["chol"] / (_df["trestbps"] + 1)
_df["ca_thal"] = _df["ca"] * _df["thal"]

X = _df.drop("num", axis=1)
y = _df["num"].astype(int)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
acc = []
cm_sum = None
report = None
best_params = None

for tr, te in sss.split(X, y):
    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_train, y_test = y.iloc[tr], y.iloc[te]
    sample_weights = class_weight.compute_sample_weight("balanced", y_train)
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=5,
        eval_metric="mlogloss",
        random_state=42,
        max_depth=5,
        learning_rate=0.08,
        n_estimators=220,
        subsample=0.9,
        colsample_bytree=0.8,
        gamma=1,
        min_child_weight=1,
        use_label_encoder=False
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)
    acc.append((y_pred == y_test).mean())
    cm_fold = confusion_matrix(y_test, y_pred)
    cm_sum = cm_fold if cm_sum is None else cm_sum + cm_fold
    report = classification_report(y_test, y_pred, digits=4)

print("mean accuracy", sum(acc) / len(acc))
print("std accuracy", pd.Series(acc).std())
print("cumulative confusion matrix")
print(cm_sum)
print("last fold classification report")
print(report)

cm_norm = cm_sum.astype(float) / cm_sum.sum(axis=1, keepdims=True)

sns.set(style="whitegrid")
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True, xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Normalized Cumulative Confusion Matrix (5-class)")
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png")
print("Saved normalized confusion matrix plot to confusion_matrix_normalized.png")

# final model fit on full data to get feature importance
X_full, y_full = X, y

model_full = XGBClassifier(
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    random_state=42,
    max_depth=5,
    learning_rate=0.08,
    n_estimators=250,
    subsample=0.85,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=1,
    use_label_encoder=False
)
model_full.fit(X_full, y_full, sample_weight=class_weight.compute_sample_weight("balanced", y_full))

importance = model_full.feature_importances_
feat_names = list(X.columns)
feat_imp = pd.Series(importance, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feat_imp.head(20).plot(kind="bar")
plt.title("XGBoost feature importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature importance plot to feature_importance.png")

model_full.save_model("xgboost_model.json")
print("Saved XGBoost model to xgboost_model.json")
