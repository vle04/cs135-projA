"""
Problem 2: open-ended challenge. Compare several feature sets / classifiers with the
same author-grouped CV, make rubric-style figures (train + validation, one dot per fold),
find the most confidently correct/incorrect heldout predictions, and write test predictions.

Edit Setup and the EXPERIMENTS dict at the bottom of the setup section.
Run:  python problem2.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup
DATA_DIR = "data_readinglevel"
N_FOLDS = 5
RANDOM_STATE = 0
GROUP_BY_AUTHOR = True  # False = ordinary random folds (only to compare against grouped CV)

# Paths to the BERT embeddings
EMB_TRAIN = f"{DATA_DIR}/x_train_BERT_embeddings.npz"
EMB_TEST = f"{DATA_DIR}/x_test_BERT_embeddings.npz"

# LOAD DATA
x_tr = pd.read_csv(f"{DATA_DIR}/x_train.csv")
y_tr = pd.read_csv(f"{DATA_DIR}/y_train.csv")
x_te = pd.read_csv(f"{DATA_DIR}/x_test.csv")

y = (y_tr["Coarse Label"] == "Key Stage 4-5").astype(int).values
groups = x_tr["author"].values          # ONLY used to build CV folds, never a feature

# Numeric features = every numeric column except the keys (cols 1-3).
KEY_COLS = ["author", "title", "passage_id"]
NUM_COLS = [c for c in x_tr.columns
            if c not in KEY_COLS and c != "text"
            and pd.api.types.is_numeric_dtype(x_tr[c])]
print(f"{len(NUM_COLS)} numeric feature columns")


def load_emb(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        z = np.load(path)
        return z[z.files[0]]
    return pd.read_csv(path).select_dtypes("number").values


def build_frame(x, emb_path):
    """One DataFrame holding text, numeric columns and (optionally) embedding columns."""
    df = x[["text"] + NUM_COLS].copy()
    df["text"] = df["text"].astype(str)
    emb_cols = []
    if emb_path is not None:
        E = load_emb(emb_path)
        assert len(E) == len(x), "embedding rows must match x_*.csv rows"
        emb_cols = [f"emb_{i}" for i in range(E.shape[1])]
        df = pd.concat([df, pd.DataFrame(E, columns=emb_cols, index=df.index)], axis=1)
    return df, emb_cols


X_tr, EMB_COLS = build_frame(x_tr, EMB_TRAIN)
X_te, _ = build_frame(x_te, EMB_TEST)
HAVE_EMB = len(EMB_COLS) > 0

# CV SPLITS (shared by every experiment so results are directly comparable)
if GROUP_BY_AUTHOR:
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_tr, y, groups))
else:
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    splits = list(cv.split(X_tr, y))
for i, (a, b) in enumerate(splits):
    print(f"Fold {i}: train={len(a)}, valid={len(b)}, valid positive rate={y[b].mean():.3f}")


# PIPELINE BUILDER: choose which feature blocks to use and which classifier
def make_pipe(use_text, use_num, use_emb, clf):
    blocks = []
    if use_text:
        blocks.append(("text",
                       TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=5),
                       "text"))
    if use_num:
        blocks.append(("num",
                       Pipeline([("imp", SimpleImputer(strategy="median")),
                                 ("sc", StandardScaler())]),
                       NUM_COLS))
    if use_emb:
        blocks.append(("emb", StandardScaler(), EMB_COLS))
    return Pipeline([("pre", ColumnTransformer(blocks)), ("clf", clf)])


C_GRID = np.logspace(-4, 1, 11)

# name -> (pipeline, param grid, hyperparameter to plot on the x-axis)
# Parameter names: "pre__<block>__<param>" for features, "clf__<param>" for the classifier.
EXPERIMENTS = {
    # Problem 1-style baseline (text only) so you can compare on the same folds
    "text_lr": (
        make_pipe(True, False, False, LogisticRegression(max_iter=5000)),
        {"clf__C": np.logspace(-3, 1, 9)},
        "clf__C",
    ),
    # Provided numeric features only (counts, sentiment, readability, ...)
    "numeric_lr": (
        make_pipe(False, True, False, LogisticRegression(max_iter=5000)),
        {"clf__C": C_GRID},
        "clf__C",
    ),
    # Tree-based model on numeric features: captures nonlinear interactions
    "numeric_hgb": (
        make_pipe(False, True, False,
                  HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
        {"clf__max_iter": [5, 10, 25, 50, 100, 200, 400, 800],
         "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [3]},
        "clf__max_iter",
    ),
    # Text + numeric together
    "text_numeric_lr": (
        make_pipe(True, True, False, LogisticRegression(max_iter=5000)),
        {"clf__C": np.logspace(-3, 1, 9)},
        "clf__C",
    ),
}
if HAVE_EMB:
    EXPERIMENTS.update({
        "emb_lr": (
            make_pipe(False, False, True, LogisticRegression(max_iter=5000)),
            {"clf__C": C_GRID},
            "clf__C",
        ),
        "all_lr": (
            make_pipe(True, True, True, LogisticRegression(max_iter=5000)),
            {"clf__C": np.logspace(-4, 0, 9)},
            "clf__C",
        ),
        "numeric_emb_hgb": (
            make_pipe(False, True, True,
                      HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"clf__max_iter": [10, 25, 50, 100, 200, 400, 800],
             "clf__learning_rate": [0.05, 0.1],
             "clf__max_depth": [3]},
            "clf__max_iter",
        ),
        # Reduce embedding dimension (768 -> k) before LR: fewer params, less overfitting
        # "emb_pca_lr": (
        #     Pipeline([("pre", ColumnTransformer([("emb", StandardScaler(), EMB_COLS)])),
        #               ("pca", PCA(random_state=RANDOM_STATE)),
        #               ("clf", LogisticRegression(max_iter=5000))]),
        #     {"pca__n_components": [8, 16, 32, 64, 128, 256, 512],
        #      "clf__C": [0.001, 0.01, 0.1, 1]},
        #     "pca__n_components",
        # ),
        # Embeddings + numeric features, linear model
        "emb_num_lr": (
            make_pipe(False, True, True, LogisticRegression(max_iter=5000)),
            {"clf__C": np.logspace(-4, 0, 9)},
            "clf__C",
        ),
        # # Small neural net on embeddings; alpha = L2 penalty (bigger = simpler model)
        # "emb_mlp": (
        #     make_pipe(False, False, True,
        #               MLPClassifier(early_stopping=True, max_iter=500,
        #                             random_state=RANDOM_STATE)),
        #     {"clf__alpha": np.logspace(-3, 3, 7),
        #      "clf__hidden_layer_sizes": [(64,), (256,)]},
        #     "clf__alpha",
        # ),
        # Wider search for gradient boosting on numeric + embeddings
        "numeric_emb_hgb_tuned": (
            make_pipe(False, True, True,
                      HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
            {"clf__max_iter": [10, 25, 50, 100, 200, 400, 800],
             "clf__learning_rate": [0.05],
             "clf__max_depth": [2, 3, 5],
             "clf__min_samples_leaf": [20, 50]},
            "clf__max_iter",
        ),
        # # Average the probabilities of a linear model and a boosted model
        # "ensemble_lr_hgb": (
        #     VotingClassifier([
        #         ("lr", make_pipe(False, True, True, LogisticRegression(max_iter=5000))),
        #         ("hgb", make_pipe(False, True, True,
        #                           HistGradientBoostingClassifier(
        #                               learning_rate=0.05, max_depth=3,
        #                               random_state=RANDOM_STATE))),
        #     ], voting="soft"),
        #     {"lr__clf__C": [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        #      "hgb__clf__max_iter": [50, 100, 200]},
        #     "lr__clf__C",
        # ),
    })
else:
    print("NOTE: no embeddings configured (EMB_TRAIN/EMB_TEST); skipping embedding experiments.")


def report_search(search, plot_param, name):
    """Tables + rubric figure for one experiment (same style as problem1.py)."""
    res = pd.DataFrame(search.cv_results_)
    res.drop(columns=["params"]).to_csv(f"cv_results_{name}.csv", index=False)
    best = search.best_params_
    param_cols = [c for c in res.columns if c.startswith("param_")]

    # Top configurations
    print("\nTop 5 configurations by mean validation AUROC:")
    show_cols = param_cols + ["mean_train_score", "mean_test_score", "std_test_score"]
    print(res.sort_values("rank_test_score")[show_cols].head(5).round(4).to_string(index=False))

    # Best validation AUROC for each value of each hyperparameter:
    # shows which choices matter (big differences) and which don't (all about equal)
    print("\nBest mean validation AUROC for each value of each hyperparameter:")
    for col in param_cols:
        g = res.groupby(res[col].astype(str))["mean_test_score"].max().sort_values(ascending=False)
        print(f"  {col[len('param_'):]}:")
        for val, sc in g.items():
            print(f"      {val:>30s}  {sc:.4f}")

    # Figure: train AND validation AUROC vs ONE hyperparameter, others fixed at best values.
    # One dot per fold + the mean.
    mask = np.ones(len(res), dtype=bool)
    for k, v in best.items():
        if k != plot_param:
            mask &= (res[f"param_{k}"].astype(str) == str(v)).values
    sub = res[mask].copy()
    xs_num = pd.to_numeric(sub[f"param_{plot_param}"], errors="coerce")
    numeric_x = not xs_num.isna().any()
    if numeric_x:
        sub["x"] = xs_num.astype(float)
        sub = sub.sort_values("x")
        xs = sub["x"].values
    else:  # e.g. tuples or None: plot as categories
        sub = sub.reset_index(drop=True)
        xs = np.arange(len(sub))
        xlabels = sub[f"param_{plot_param}"].astype(str).tolist()
    if len(xs) < 5:
        print(f"WARNING: only {len(xs)} values of {plot_param}; the rubric wants at least 5.")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for kind, color, label in [("test", "tab:blue", "validation"),
                               ("train", "tab:orange", "train")]:
        per_fold = np.array([sub[f"split{i}_{kind}_score"].values for i in range(N_FOLDS)])
        ax.scatter(np.tile(xs, (N_FOLDS, 1)).ravel(), per_fold.ravel(),
                   color=color, alpha=0.35, s=14, label=f"{label}: one dot per fold")
        ax.plot(xs, per_fold.mean(axis=0), color=color, marker="o",
                label=f"{label}: mean over folds")
    best_x = best[plot_param]
    if numeric_x:
        ax.axvline(float(best_x), color="gray", linestyle="--", alpha=0.6, label="selected value")
        if xs.min() > 0 and xs.max() / xs.min() > 50:
            ax.set_xscale("log")
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_xlabel(plot_param)
    ax.set_ylabel("AUROC")
    ax.set_title(f"{name}: {N_FOLDS}-fold CV AUROC vs {plot_param}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"cv_{name}.png", dpi=200)
    plt.close(fig)
    print(f"Saved cv_{name}.png")

    # Numbers behind the figure + under/overfitting diagnostics for the caption
    tbl = pd.DataFrame({
        plot_param: sub[f"param_{plot_param}"].astype(str).values,
        "mean_train": sub["mean_train_score"].values,
        "mean_valid": sub["mean_test_score"].values,
        "std_valid": sub["std_test_score"].values,
    })
    tbl["train-valid gap"] = tbl["mean_train"] - tbl["mean_valid"]
    print("\nValues shown in the figure:")
    print(tbl.round(4).to_string(index=False))
    if numeric_x and float(best_x) in (xs.min(), xs.max()):
        print(f"\nWARNING: selected {plot_param}={best_x} is at the EDGE of the grid. "
              "Extend the grid in that direction and rerun.")
    print("Look for: low train AND low validation at one end (underfitting); "
          "high train with validation dropping at the other end (overfitting).")


def top_confident(y_true, proba, meta, k=3):
    """Most confidently correct / incorrect heldout predictions (confidence in the TRUE class)."""
    p_true = np.where(y_true == 1, proba, 1 - proba)
    order = np.argsort(p_true)
    pick = lambda idx: meta.iloc[idx].assign(
        p_true_class=p_true[idx], p_upper=proba[idx], true_label=y_true[idx])
    correct = pick(order[::-1][:k])          # highest probability on the true class
    wrong = pick(order[:k])                  # lowest probability on the true class
    return correct, wrong


# Experiments
meta = x_tr[["author", "title", "passage_id"]]
summary = []
searches = {}
fold_table = {}

for name, (pipe, grid, plot_param) in EXPERIMENTS.items():
    print(f"\n=== {name} ===")
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=splits, refit=True,
                          n_jobs=-1, return_train_score=True, verbose=1)
    search.fit(X_tr, y)
    searches[name] = search
    best_i = search.best_index_
    fold_scores = [search.cv_results_[f"split{i}_test_score"][best_i] for i in range(N_FOLDS)]
    print(f"best CV AUROC {search.best_score_:.4f}  (std over folds {np.std(fold_scores):.4f})")
    print("best params:", search.best_params_)
    print("per-fold validation AUROC:", np.round(fold_scores, 4))
    summary.append((name, search.best_score_, np.std(fold_scores)))
    fold_table[name] = fold_scores

    report_search(search, plot_param, name)

    # Heldout predictions from CV, for the 3 most confidently correct/incorrect
    heldout = cross_val_predict(search.best_estimator_, X_tr, y, cv=splits,
                                method="predict_proba", n_jobs=-1)[:, 1]
    correct, wrong = top_confident(y, heldout, meta)
    print("\nMost confidently CORRECT (heldout):\n", correct.to_string())
    print("\nMost confidently INCORRECT (heldout):\n", wrong.to_string())

    # Test-set predictions from the model refit on ALL training data
    np.savetxt(f"yproba_test_{name}.txt",
               search.best_estimator_.predict_proba(X_te)[:, 1], fmt="%.8f")

print(f"\n=== SUMMARY ({'author-grouped' if GROUP_BY_AUTHOR else 'RANDOM-fold'} CV AUROC) ===")
for name, score, sd in sorted(summary, key=lambda t: -t[1]):
    print(f"{name:22s} {score:.4f} +/- {sd:.4f}")
best_name = max(summary, key=lambda t: t[1])[0]
print(f"\nBest: {best_name}. Copy yproba_test_{best_name}.txt to yproba1_test.txt to submit.")

# Per-fold comparison on identical folds: is the winner better on most folds, or is it noise?
print("\n=== Per-fold AUROC of each experiment's best config ===")
print(pd.DataFrame(fold_table, index=[f"fold{i}" for i in range(N_FOLDS)]).T.round(3).to_string())
print(f"\nPaired difference vs {best_name} (same folds):")
for name, fs in fold_table.items():
    if name != best_name:
        d = np.array(fold_table[best_name]) - np.array(fs)
        print(f"  {best_name} - {name}: {d.mean():+.4f} +/- {d.std():.4f}"
              f"  (better on {(d > 0).sum()}/{N_FOLDS} folds)")