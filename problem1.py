"""
Problem 1: BoW + LogisticRegression pipeline with cross-validated hyperparameter search.

Edit the Setup section and PARAM_GRID to try different settings.
Run:  python problem1.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict

# Setup
DATA_DIR = "data_readinglevel"
N_FOLDS = 5
RANDOM_STATE = 0
SCORING = "roc_auc"
GROUP_BY_AUTHOR = True

# Hyperparameter to put on the x-axis of the 1C figure (must appear in PARAM_GRID).
# Should control model complexity; clf__C is the natural choice.
PLOT_PARAM = "clf__C"

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

# HYPERPARAMETER GRID  (each dict is a sub-grid)
# Use "bow": [...] to swap whole vectorizers, or "bow__<param>" to tweak one
# Every key is "<pipeline step name>__<parameter name>"
C_GRID = np.logspace(-3, 2, 11)

PARAM_GRID = [
    # Binary counts, lowercase
    {
        "bow": [CountVectorizer()],
        "bow__lowercase": [True],
        "bow__binary": [True, False],
        "bow__stop_words": ["english"],
        "bow__min_df": [1, 2],
        "bow__max_df": [0.0, 0.5, 1.0],
        "bow__max_features": [5000, 20000, None],
        "bow__token_pattern": [None, r"(?u)\b\w+\b|[^\w\s]"],
        "clf__l1_ratio": [0.0, 1.0],
        "clf__class_weight": [None, "balanced"],
        "clf__C": C_GRID,
    },
    # TF-IDF
    # {
    #     "bow": [TfidfVectorizer()],
    #     "bow__lowercase": [True],
    #     "bow__sublinear_tf": [True],
    #     "bow__stop_words": [None],
    #     "bow__min_df": [2, 5, 10, 20],
    #     "bow__ngram_range": [(1, 1), (1, 2), (1, 3)],
    #     "clf__penalty": ["l2"],
    #     "clf__C": C_GRID,
    # },
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

# Restults Table (1A/1C)
res = pd.DataFrame(search.cv_results_)
best = search.best_params_
best_idx = search.best_index_

if "param_bow" in res.columns:
    res["param_bow"] = res["param_bow"].apply(lambda b: type(b).__name__)


def best_val_str(k, v):
    """String form of a best-param value, matching how res[param_*] is stored above."""
    return type(v).__name__ if k == "bow" else str(v)


res.drop(columns=["params"]).to_csv("cv_results_problem1.csv", index=False)
param_cols = [c for c in res.columns if c.startswith("param_")]  # union over all sub-grids

# Top 10 configurations
print("\nTop 10 configurations by mean validation AUROC:")
show_cols = param_cols + ["mean_train_score", "mean_test_score", "std_test_score"]
print(res.sort_values("rank_test_score")[show_cols].head(10).round(4).to_string(index=False))

# For each hyperparameter value: best validation AUROC achieved by any config using it.
# Shows which choices matter (big differences) and which don't (all about equal).
print("\nBest mean validation AUROC for each value of each hyperparameter:")
for col in param_cols:
    g = res.groupby(res[col].astype(str))["mean_test_score"].max().sort_values(ascending=False)
    print(f"  {col[len('param_'):]}:")
    for val, sc in g.items():
        print(f"      {val:>30s}  {sc:.4f}")

fold_scores = np.array([res.loc[best_idx, f"split{i}_test_score"] for i in range(N_FOLDS)])
print("\nPer-fold validation AUROC of best config:", np.round(fold_scores, 4),
      f"(mean {fold_scores.mean():.4f}, std {fold_scores.std():.4f})")

# FIGURE (1C rubric): train AND validation AUROC vs ONE complexity hyperparameter.
# All other hyperparameters fixed at their best values. One dot per fold + the mean.
assert PLOT_PARAM in best, f"{PLOT_PARAM} must be a key in the best sub-grid of PARAM_GRID"

mask = np.ones(len(res), dtype=bool)
for k, v in best.items():
    if k != PLOT_PARAM:
        mask &= (res[f"param_{k}"].astype(str) == best_val_str(k, v)).values
sub = res[mask].copy()

xs_num = pd.to_numeric(sub[f"param_{PLOT_PARAM}"], errors="coerce")
numeric_x = not xs_num.isna().any()
if numeric_x:
    sub["x"] = xs_num.astype(float)
    sub = sub.sort_values("x")
    xs = sub["x"].values
else:  # e.g. ngram_range tuples or None: plot as categories
    sub = sub.reset_index(drop=True)
    xs = np.arange(len(sub))
    xlabels = sub[f"param_{PLOT_PARAM}"].astype(str).tolist()

if len(xs) < 5:
    print(f"WARNING: only {len(xs)} values of {PLOT_PARAM}; the rubric wants at least 5.")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for kind, color, label in [("test", "tab:blue", "validation"),
                           ("train", "tab:orange", "train")]:
    per_fold = np.array([sub[f"split{i}_{kind}_score"].values for i in range(N_FOLDS)])
    ax.scatter(np.tile(xs, (N_FOLDS, 1)).ravel(), per_fold.ravel(),
               color=color, alpha=0.35, s=14, label=f"{label}: one dot per fold")
    ax.plot(xs, per_fold.mean(axis=0), color=color, marker="o", label=f"{label}: mean over folds")

best_x = best[PLOT_PARAM]
if numeric_x:
    ax.axvline(float(best_x), color="gray", linestyle="--", alpha=0.6, label="selected value")
    if xs.min() > 0 and xs.max() / xs.min() > 50:
        ax.set_xscale("log")
else:
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
ax.set_xlabel(PLOT_PARAM + (" (larger C = weaker regularization = more complex)"
                            if PLOT_PARAM == "clf__C" else ""))
ax.set_ylabel("AUROC")
ax.set_title(f"{N_FOLDS}-fold grouped CV: AUROC vs {PLOT_PARAM}")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("cv_auroc_vs_hparam.png", dpi=200)
print("Saved cv_auroc_vs_hparam.png")

# Numbers behind the figure + under/overfitting diagnostics for the caption
tbl = pd.DataFrame({
    PLOT_PARAM: sub[f"param_{PLOT_PARAM}"].astype(str).values,
    "mean_train": sub["mean_train_score"].values,
    "mean_valid": sub["mean_test_score"].values,
    "std_valid": sub["std_test_score"].values,
})
tbl["train-valid gap"] = tbl["mean_train"] - tbl["mean_valid"]
print("\nValues shown in the figure:")
print(tbl.round(4).to_string(index=False))

# Test set predictions (probability of class 1 = Key Stage 4-5)
proba = search.best_estimator_.predict_proba(texts_te)[:, 1]
np.savetxt("yproba1_test.txt", proba, fmt="%.8f")
print(f"Saved yproba1_test.txt ({len(proba)} rows)")

# HELDOUT PREDICTIONS FROM CV: most confidently correct / incorrect (for 2D reflection)
heldout = cross_val_predict(search.best_estimator_, texts_tr, y, cv=splits,
                            method="predict_proba", n_jobs=-1)[:, 1]
p_true = np.where(y == 1, heldout, 1 - heldout)  # probability given to the TRUE class
order = np.argsort(p_true)
meta = x_tr[["author", "title", "passage_id"]].assign(
    true_label=y, p_upper=heldout, p_true_class=p_true)
print("\nMost confidently CORRECT (heldout):")
print(meta.iloc[order[::-1][:3]].to_string())
print("\nMost confidently INCORRECT (heldout):")
print(meta.iloc[order[:3]].to_string())