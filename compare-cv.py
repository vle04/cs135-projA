"""
Compare ordinary CV vs. author-grouped CV with the same model.
Run: python compare_cv.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

DATA_DIR = "data_readinglevel"

# Load data (the snippet needs these variables defined)
x_tr = pd.read_csv(f"{DATA_DIR}/x_train.csv")
y_tr = pd.read_csv(f"{DATA_DIR}/y_train.csv")
texts_tr = x_tr["text"].astype(str)
y = (y_tr["Coarse Label"] == "Key Stage 4-5").astype(int).values
groups = x_tr["author"].values  # only used to build folds, never as a feature

# Problem 1 rule: features must be unigrams, so use CountVectorizer, not TF-IDF
def make_pipe(C):
    return Pipeline([
        ("bow", CountVectorizer(lowercase=True, binary=True, min_df=2, ngram_range=(1, 1))),
        ("clf", LogisticRegression(C=C, solver="liblinear", max_iter=2000)),
    ])

cv_methods = {
    "StratifiedKFold (ordinary)": StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    "StratifiedGroupKFold (by author)": StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0),
}

# Try a few values of C, since the gap can depend on how flexible the model is
for C in [0.01, 0.1, 1.0]:
    print(f"\n===== C = {C} =====")
    for name, splitter in cv_methods.items():
        if "Group" in name:
            splits = list(splitter.split(texts_tr, y, groups))
        else:
            splits = list(splitter.split(texts_tr, y))

        results = cross_validate(
            make_pipe(C), texts_tr, y,
            cv=splits, scoring="roc_auc", return_train_score=True, n_jobs=-1,
        )
        print(f"\n{name}")
        print("  Per-fold validation AUROC:", np.round(results["test_score"], 4))
        print("  Validation AUROC (mean): %.4f" % results["test_score"].mean())
        print("  Fold standard deviation: %.4f" % results["test_score"].std())
        print("  Training AUROC (mean):   %.4f" % results["train_score"].mean())