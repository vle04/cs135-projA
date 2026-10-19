"""
Problem 1: BoW + LogisticRegression pipeline with cross-validated hyperparameter search.

Edit the CONFIG section and PARAM_GRID to try different settings.
Run:  python problem1.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

DATA_DIR = "data_readinglevel"
N_FOLDS = 5
RANDOM_STATE = 0
SCORING = "roc_auc"
GROUP_BY_AUTHOR = True

x_tr = pd.read_csv(f"{DATA_DIR}/x_train.csv")
y_tr = pd.read_csv(f"{DATA_DIR}/y_train.csv")
x_te = pd.read_csv(f"{DATA_DIR}/x_test.csv")

texts_tr = x_tr["text"].astype(str)
texts_te = x_te["text"].astype(str)
y = (y_tr["Coarse Label"] == "Key Stage 4-5").astype(int).values  # 1 = upper level
groups = x_tr["author"].values  # used to build CV folds, never as a feature

print(f"Train: {len(texts_tr)} passages, Test: {len(texts_te)} passages")
print(f"Positive rate (Key Stage 4-5): {y.mean():.3f}")

# Vectorizer is refit inside each CV fold -> vocabulary never leaks from the validation fold.
# Unseen test words are simply ignored (OOV -> dropped)
pipe = Pipeline([
    ("bow", CountVectorizer()),
    ("clf", LogisticRegression(solver="liblinear", max_iter=2000)),
])

# HYPERPARAMETER GRID  (edit freely; each dict is a sub-grid)
# Use "bow": [...] to swap whole vectorizers, or "bow__<param>" to tweak one.
C_GRID = np.logspace(-3, 2, 11)

PARAM_GRID = [
    # BoW preprocessing configurations
    {
        "bow": [CountVectorizer()],
        "bow__min_df": [1, 2, 5],
        "bow__stop_words": [None, "english"],
        "bow__binary": [False, True],
        "bow__ngram_range": [(1, 1)],
        "clf__penalty": ["l2"],
        "clf__C": [1.0],
    },
]

# CV
if GROUP_BY_AUTHOR:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fit_kwargs = {"groups": groups}
    splits = list(cv.split(texts_tr, y, groups))
else:
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fit_kwargs = {}
    splits = list(cv.split(texts_tr, y))

# Fold sizes (1B)
for i, (tr_idx, va_idx) in enumerate(splits):
    print(f"Fold {i}: train={len(tr_idx)}, valid={len(va_idx)}, "
          f"valid positive rate={y[va_idx].mean():.3f}")

search = GridSearchCV(
    pipe,
    PARAM_GRID,
    scoring=SCORING,
    cv=splits,            # fixed splits so every config sees identical folds
    refit=True,           # refits best config on ALL training data -> final model
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)
search.fit(texts_tr, y)

print("\nBest CV AUROC: %.4f" % search.best_score_)
print("Best params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_bow = search.best_estimator_.named_steps["bow"]
print("Final vocabulary size (fit on all training data):", len(best_bow.vocabulary_))

# Vocabulary size across CV folds for the best config (1A)
from sklearn.base import clone
sizes = []
for tr_idx, _ in splits:
    v = clone(best_bow).fit(texts_tr.iloc[tr_idx])
    sizes.append(len(v.vocabulary_))
print("Vocab size per fold:", sizes)

# CV AUROC vs. C (other hyperparameters fixed at their best values)
res = pd.DataFrame(search.cv_results_)
best = search.best_params_

mask = np.ones(len(res), dtype=bool)
for k, v in best.items():
    if k == "clf__C":
        continue
    if k == "bow":
        mask &= res["param_bow"].apply(lambda b: type(b) is type(v)).values
    else:
        mask &= (res[f"param_{k}"].astype(str) == str(v)).values
sub = res[mask].sort_values("param_clf__C")

Cs = sub["param_clf__C"].astype(float).values
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(Cs, sub["mean_test_score"], yerr=sub["std_test_score"],
            marker="o", capsize=3, label="validation (mean ± std over folds)")
ax.plot(Cs, sub["mean_train_score"], marker="s", linestyle="--", label="train (mean)")
ax.set_xscale("log")
ax.set_xlabel("C (inverse regularization strength)")
ax.set_ylabel("AUROC")
ax.set_title("CV performance vs. C")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("cv_auroc_vs_C.png", dpi=200)
print("Saved cv_auroc_vs_C.png")

# Per-fold scores of the best config (to show the range of performance)
best_idx = search.best_index_
fold_scores = [res.loc[best_idx, f"split{i}_test_score"] for i in range(N_FOLDS)]
print("Per-fold AUROC of best config:", np.round(fold_scores, 4))

# Test set predictions (probability of class 1 = Key Stage 4-5)
proba = search.best_estimator_.predict_proba(texts_te)[:, 1]
np.savetxt("yproba1_test.txt", proba, fmt="%.8f")
print(f"Saved yproba1_test.txt ({len(proba)} rows)")