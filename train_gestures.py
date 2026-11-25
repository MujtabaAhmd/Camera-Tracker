#!/usr/bin/env python3
"""
Train and compare three gesture classifiers (kNN, SVM, MLP) on the CSV made by
collect_gestures.py, and save them for use by gesture_runtime.py.

Usage:
    python train_gestures.py --data gesture_dataset.csv
    python train_gestures.py --data ali.csv sara.csv omar.csv --test-subjects omar
    python train_gestures.py --data gesture_dataset.csv --quick          # fast check
    python train_gestures.py --data gesture_dataset.csv --curves --loso  # full analysis

What it does
  1. Loads the CSV and recomputes the 60 features from the raw landmark columns
     (so the features always match gesture_runtime.py).
  2. Holds out a TEST set by person (not by frame). Frames from one recording are
     near-duplicates, so a random frame split would give misleadingly high scores.
  3. Tunes each model with grouped cross-validation on the training people only
     (hyperparameters are never chosen using the test set).
  4. Evaluates on the held-out person(s): accuracy, per-class precision / recall / F1,
     confusion matrices, prediction time, and a confidence-threshold sweep that shows
     how many "none" frames would wrongly trigger a command.
  5. Optional: --curves (learning curves, MLP loss curves), --loso (leave one person
     out evaluation, the most honest summary when you only have a few people).
  6. Saves everything to --out-dir (default: models/).

Needs: numpy pandas scikit-learn joblib matplotlib   (and torch for the MLP;
without torch it falls back to scikit-learn's MLPClassifier).
"""

import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from gesture_features import RAW_COLS, NumpyMLP, landmarks_to_features

NONE_LABEL = "none"
MODEL_NAMES = ["knn", "svm", "mlp"]


# --------------------------------------------------------------------------
# MLP trained with PyTorch, exposed with the scikit-learn interface so the same
# cross-validation code works for all three models. After training, the weights
# are copied into a numpy model, so predictions do not need PyTorch.
# --------------------------------------------------------------------------
class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden=(64, 32), dropout=0.2, weight_decay=1e-4, lr=1e-3,
                 epochs=40, batch_size=128, balanced=True, random_state=0):
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.balanced = balanced
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        import torch
        from torch import nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        k = len(self.classes_)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-6
        xs = (X - self.mean_) / self.std_

        torch.manual_seed(self.random_state)
        rng = np.random.RandomState(self.random_state)

        layers, d = [], xs.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = h
        layers.append(nn.Linear(d, k))
        net = nn.Sequential(*layers)

        weight = None
        if self.balanced:
            counts = np.bincount(yi, minlength=k).astype(np.float32)
            weight = torch.tensor(counts.sum() / (k * np.maximum(counts, 1)))
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        plain_loss = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        xt = torch.tensor(xs)
        yt = torch.tensor(yi, dtype=torch.long)
        if eval_set is not None:
            xv = torch.tensor((np.asarray(eval_set[0], dtype=np.float32) - self.mean_) / self.std_)
            yv = torch.tensor(np.searchsorted(self.classes_, np.asarray(eval_set[1])), dtype=torch.long)

        self.history_ = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        n = len(xt)
        for _ in range(self.epochs):
            net.train()
            perm = torch.tensor(rng.permutation(n))
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            if eval_set is not None:
                net.eval()
                with torch.no_grad():
                    out_t, out_v = net(xt), net(xv)
                    self.history_["train_loss"].append(plain_loss(out_t, yt).item())
                    self.history_["val_loss"].append(plain_loss(out_v, yv).item())
                    self.history_["train_acc"].append((out_t.argmax(1) == yt).float().mean().item())
                    self.history_["val_acc"].append((out_v.argmax(1) == yv).float().mean().item())

        linears = [m for m in net if isinstance(m, nn.Linear)]
        self.numpy_ = NumpyMLP(
            [m.weight.detach().numpy().T.copy() for m in linears],
            [m.bias.detach().numpy().copy() for m in linears],
            self.mean_, self.std_, self.classes_,
        )
        return self

    def predict_proba(self, X):
        return self.numpy_.predict_proba(X)

    def predict(self, X):
        return self.numpy_.predict(X)


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(paths, stride):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in RAW_COLS + ["subject_id", "session_id", "label", "take", "handedness"]
               if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing[:5]} ... Is this from collect_gestures.py?")
    df = df.dropna(subset=RAW_COLS + ["label", "handedness"]).reset_index(drop=True)
    df["subject_id"] = df["subject_id"].astype(str)
    df["session_id"] = df["session_id"].astype(str)
    if stride > 1:
        df = df[df["frame_idx"] % stride == 0].reset_index(drop=True)

    raw = df[RAW_COLS].to_numpy(np.float32).reshape(-1, 21, 3)
    handed = df["handedness"].to_numpy()
    feats, keep = [], []
    for i in range(len(df)):
        f = landmarks_to_features(raw[i], handed[i])
        if f is not None:
            feats.append(f)
            keep.append(i)
    df = df.iloc[keep].reset_index(drop=True)
    X = np.stack(feats).astype(np.float32)
    return df, X


def print_dataset_summary(df):
    print("\n=== Dataset summary ===")
    print(f"Rows: {len(df)}   People: {df.subject_id.nunique()}   "
          f"Sessions: {df.groupby(['subject_id', 'session_id']).ngroups}   "
          f"Takes: {df.groupby(['subject_id', 'session_id', 'label', 'take']).ngroups}")
    print("\nRows per label:")
    print(df.label.value_counts().to_string())
    print("\nRows per person:")
    print(df.subject_id.value_counts().to_string())
    print("\nRows per hand label:")
    print(df.handedness.value_counts().to_string())
    counts = df.label.value_counts()
    if counts.max() > 3 * counts.min():
        print("\nWARNING: classes are quite imbalanced (largest is more than 3x the smallest). "
              "SVM and MLP use class weights, but consider recording more of the small classes.")
    if NONE_LABEL not in counts.index:
        print(f"\nWARNING: no '{NONE_LABEL}' class found. Without it the system cannot reject "
              "random hand poses.")
    print()


def make_split(df, test_subjects, seed):
    """Returns (test_mask, level, description)."""
    subjects = sorted(df.subject_id.unique())
    if test_subjects:
        ts = [s.strip() for s in test_subjects.split(",") if s.strip()]
        unknown = [s for s in ts if s not in subjects]
        if unknown:
            raise SystemExit(f"Unknown test subject(s) {unknown}. Available: {subjects}")
        if len(ts) >= len(subjects):
            raise SystemExit("Cannot hold out every person; keep at least one for training.")
        return df.subject_id.isin(ts).to_numpy(), "person", f"held-out person(s): {ts}"
    if len(subjects) >= 2:
        pick = np.random.RandomState(seed).choice(subjects)
        return (df.subject_id == pick).to_numpy(), "person", \
            f"held-out person (chosen at random, seed {seed}): ['{pick}']"
    sessions = sorted(df.session_id.unique())
    if len(sessions) >= 2:
        return (df.session_id == sessions[-1]).to_numpy(), "session", \
            f"held-out session: ['{sessions[-1]}']"
    return (df["take"] % 5 == 0).to_numpy(), "take", "held-out takes (every 5th take)"


def cv_groups(df):
    """Groups used for cross-validation inside the training data."""
    if df.subject_id.nunique() >= 3:
        return df.subject_id.to_numpy(), "person"
    key = df.subject_id + "|" + df.session_id
    if key.nunique() >= 3:
        return key.to_numpy(), "session"
    key = key + "|" + df.label + "|" + df["take"].astype(str)
    return key.to_numpy(), "take"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def base_estimator(name, mlp_backend, epochs, seed, probability):
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier())])
    if name == "svm":
        return Pipeline([("scale", StandardScaler()),
                         ("clf", SVC(probability=probability, class_weight="balanced",
                                     random_state=seed))])
    if name == "mlp":
        if mlp_backend == "torch":
            return TorchMLP(epochs=epochs, random_state=seed)
        return Pipeline([("scale", StandardScaler()),
                         ("clf", MLPClassifier(max_iter=300, random_state=seed))])
    raise ValueError(name)


def param_grid(name, mlp_backend, quick):
    if name == "knn":
        return {"clf__n_neighbors": [3, 7, 15] if quick else [1, 3, 5, 7, 11, 15, 21],
                "clf__weights": ["uniform", "distance"]}
    if name == "svm":
        if quick:
            return [{"clf__kernel": ["rbf"], "clf__C": [1, 10], "clf__gamma": ["scale"]}]
        return [{"clf__kernel": ["rbf"], "clf__C": [0.1, 1, 10, 100],
                 "clf__gamma": ["scale", 0.01, 0.001]},
                {"clf__kernel": ["linear"], "clf__C": [0.1, 1, 10]}]
    if name == "mlp":
        if mlp_backend == "torch":
            if quick:
                return {"hidden": [(32,), (64, 32)], "dropout": [0.2], "weight_decay": [1e-4]}
            return {"hidden": [(32,), (64, 32), (128, 64)], "dropout": [0.0, 0.2],
                    "weight_decay": [1e-5, 1e-3]}
        if quick:
            return {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-4]}
        return {"clf__hidden_layer_sizes": [(32,), (64, 32), (128, 64)],
                "clf__alpha": [1e-5, 1e-3, 1e-1]}
    raise ValueError(name)


def build_with_params(name, best_params, mlp_backend, epochs, seed, probability):
    est = base_estimator(name, mlp_backend, epochs, seed, probability)
    est.set_params(**best_params)
    return est


def tune(name, X, y, groups, n_splits, mlp_backend, epochs, seed, quick):
    est = base_estimator(name, mlp_backend, epochs, seed, probability=False)
    grid = param_grid(name, mlp_backend, quick)
    n_jobs = 1 if name == "mlp" else -1
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=GroupKFold(n_splits=n_splits),
                      n_jobs=n_jobs, refit=False, error_score="raise")
    gs.fit(X, y, groups=groups)
    return gs.best_params_, float(gs.best_score_), pd.DataFrame(gs.cv_results_)


def full_proba(est, X, n_classes):
    """Probabilities with one column per class (even if a class was unseen)."""
    p = est.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float32)
    out[:, np.asarray(est.classes_, dtype=int)] = p
    return out


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def evaluate(proba, y_true, class_names):
    k = len(class_names)
    pred = proba.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=range(k), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=range(k), average="macro", zero_division=0)),
        "per_class": {class_names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])} for i in range(k)},
        "confusion": confusion_matrix(y_true, pred, labels=range(k)),
    }


def plot_confusion(cm, class_names, title, path):
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(class_names), 0.8 + 0.8 * len(class_names)))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def time_single_predictions(est, X, n=200):
    idx = np.random.RandomState(0).choice(len(X), size=min(n, len(X)), replace=False)
    est.predict_proba(X[idx[:1]])   # warm up
    t0 = time.perf_counter()
    for i in idx:
        est.predict_proba(X[i:i + 1])
    return 1000.0 * (time.perf_counter() - t0) / len(idx)


def threshold_sweep(proba, y_true, class_names):
    """How accuracy / false commands change if low-confidence predictions become 'none'."""
    if NONE_LABEL not in class_names:
        return None
    none_idx = class_names.index(NONE_LABEL)
    k = len(class_names)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    rows = []
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        eff = np.where(conf >= t, pred, none_idx)
        is_none = y_true == none_idx
        real = ~is_none
        false_cmd = float(np.mean(eff[is_none] != none_idx)) if is_none.any() else np.nan
        recall_real = float(np.mean(eff[real] == y_true[real])) if real.any() else np.nan
        rows.append({"min_conf": t,
                     "macro_f1": float(f1_score(y_true, eff, labels=range(k), average="macro",
                                                zero_division=0)),
                     "false_command_rate_on_none_frames": false_cmd,
                     "correct_rate_on_real_gesture_frames": recall_real})
    return pd.DataFrame(rows)


def pca_plot(X, y, class_names, path):
    xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, X.shape[1])).fit(xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    z = pca.transform(xs)[:, :2]
    idx = np.random.RandomState(0).choice(len(z), size=min(4000, len(z)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, name in enumerate(class_names):
        m = y[idx] == i
        ax.scatter(z[idx][m, 0], z[idx][m, 1], s=6, alpha=0.6, label=name)
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.set_title("PCA of training features")
    ax.legend(markerscale=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cum


def learning_curve_by_groups(factory, X_tr, y_tr, g_tr, X_te, y_te, class_names, seed, path, title):
    """Train on a growing number of groups (people/sessions/takes), test on held-out data."""
    k = len(class_names)
    ug = np.unique(g_tr)
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for frac in fractions:
        n_g = max(1, int(round(frac * len(ug))))
        for rep in range(3 if frac < 1.0 else 1):
            rng = np.random.RandomState(seed + rep)
            chosen = rng.choice(ug, size=n_g, replace=False)
            m = np.isin(g_tr, chosen)
            if len(np.unique(y_tr[m])) < 2:
                continue
            est = factory().fit(X_tr[m], y_tr[m])
            p_tr = est.predict(X_tr[m])
            p_te = est.predict(X_te)
            rows.append({"n_groups": n_g,
                         "train_f1": f1_score(y_tr[m], p_tr, labels=range(k), average="macro", zero_division=0),
                         "test_f1": f1_score(y_te, p_te, labels=range(k), average="macro", zero_division=0)})
    d = pd.DataFrame(rows).groupby("n_groups").mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(d.n_groups, d.train_f1, "o-", label="train")
    ax.plot(d.n_groups, d.test_f1, "o-", label="held-out test")
    ax.set_xlabel("Number of training groups")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return d


def mlp_loss_curves(best_params, X_tr, y_tr, g_tr, epochs, seed, path):
    ug = np.unique(g_tr)
    rng = np.random.RandomState(seed)
    n_val = max(1, int(round(0.2 * len(ug))))
    val_groups = rng.choice(ug, size=n_val, replace=False)
    vm = np.isin(g_tr, val_groups)
    est = TorchMLP(epochs=epochs, random_state=seed)
    est.set_params(**best_params)
    est.fit(X_tr[~vm], y_tr[~vm], eval_set=(X_tr[vm], y_tr[vm]))
    h = est.history_
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(h["train_loss"], label="train")
    axs[0].plot(h["val_loss"], label="validation (unseen groups)")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-entropy loss")
    axs[0].legend()
    axs[0].set_title("MLP loss")
    axs[1].plot(h["train_acc"], label="train")
    axs[1].plot(h["val_acc"], label="validation (unseen groups)")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].set_title("MLP accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="one or more CSV files")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--test-subjects", default=None,
                    help="comma separated subject IDs to hold out (default: one chosen at random)")
    ap.add_argument("--models", default="knn,svm,mlp", help="comma separated subset of knn,svm,mlp")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth frame of each take (3 makes training much faster)")
    ap.add_argument("--epochs", type=int, default=40, help="MLP epochs")
    ap.add_argument("--mlp-backend", choices=["torch", "sklearn"], default="torch")
    ap.add_argument("--quick", action="store_true", help="small grids, 3 folds, stride 3")
    ap.add_argument("--curves", action="store_true", help="learning curves and MLP loss curves")
    ap.add_argument("--loso", action="store_true", help="leave-one-person-out evaluation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    # Newer scikit-learn versions warn that SVC(probability=True) is deprecated; it still works.
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(args.out_dir, exist_ok=True)
    stride = max(args.stride, 3) if args.quick else args.stride

    mlp_backend = args.mlp_backend
    if mlp_backend == "torch" and not torch_available():
        print("PyTorch not found: using scikit-learn's MLPClassifier instead "
              "(pip install torch to use the PyTorch MLP).")
        mlp_backend = "sklearn"
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in model_names:
        if m not in MODEL_NAMES:
            raise SystemExit(f"Unknown model '{m}'. Options: {MODEL_NAMES}")

    # ---- load -----------------------------------------------------------
    df, X = load_data(args.data, stride)
    print_dataset_summary(df)
    le = LabelEncoder().fit(df.label.to_numpy())
    class_names = list(le.classes_)
    y = le.transform(df.label.to_numpy())
    k = len(class_names)

    # ---- split ----------------------------------------------------------
    test_mask, level, desc = make_split(df, args.test_subjects, args.seed)
    tr, te = ~test_mask, test_mask
    X_tr, y_tr, X_te, y_te = X[tr], y[tr], X[te], y[te]
    df_tr, df_te = df[tr].reset_index(drop=True), df[te].reset_index(drop=True)
    groups, cv_level = cv_groups(df_tr)
    n_groups = len(np.unique(groups))
    n_splits = min(3 if args.quick else 5, n_groups)

    print("=== Split ===")
    print(f"Test set: {desc}  ({len(X_te)} rows)   Training: {len(X_tr)} rows")
    print(f"Cross-validation groups: {cv_level} ({n_groups} groups, {n_splits} folds)")
    if level != "person":
        print("WARNING: only one person in the data. Results below do NOT show how well the "
              "model works for an unseen person and will be optimistic. Record more people.")
    if cv_level == "take":
        print("WARNING: too few people/sessions for person-level cross-validation; "
              "falling back to take-level groups (optimistic).")
    if n_splits < 2:
        raise SystemExit("Not enough groups for cross-validation. Record more data.")
    missing_te = [class_names[i] for i in range(k) if (y_te == i).sum() == 0]
    if missing_te:
        print(f"NOTE: test set has no examples of {missing_te}")
    print()

    cum = pca_plot(X_tr, y_tr, class_names, os.path.join(args.out_dir, "pca.png"))
    print(f"PCA: first 2 components explain {100 * cum[1]:.1f}% of variance; "
          f"{int(np.searchsorted(cum, 0.95) + 1) if cum[-1] >= 0.95 else '>10'} components for 95%.\n")

    # ---- tune, fit, evaluate -------------------------------------------
    results, all_metrics, best_all, proba_test = [], {}, {}, {}
    for name in model_names:
        print(f"=== {name.upper()} ===")
        t0 = time.time()
        best_params, cv_score, cv_table = tune(name, X_tr, y_tr, groups, n_splits, mlp_backend,
                                               args.epochs, args.seed, args.quick)
        cv_table.to_csv(os.path.join(args.out_dir, f"cv_results_{name}.csv"), index=False)
        print(f"Best params: {best_params}   CV macro-F1 ({cv_level}-grouped): {cv_score:.3f}")

        est = build_with_params(name, best_params, mlp_backend, args.epochs, args.seed, probability=True)
        est.fit(X_tr, y_tr)
        train_time = time.time() - t0

        proba = full_proba(est, X_te, k)
        proba_test[name] = proba
        m = evaluate(proba, y_te, class_names)
        ms = time_single_predictions(est, X_te)
        print(f"TEST accuracy {m['accuracy']:.3f}   macro-F1 {m['macro_f1']:.3f}   "
              f"{ms:.2f} ms per prediction   total time {train_time:.0f}s")
        for cname, v in m["per_class"].items():
            print(f"   {cname:<12} P {v['precision']:.2f}  R {v['recall']:.2f}  "
                  f"F1 {v['f1']:.2f}  (n={v['support']})")
        print()

        plot_confusion(m["confusion"], class_names, f"{name.upper()} confusion (test)",
                       os.path.join(args.out_dir, f"confusion_{name}.png"))
        sweep = threshold_sweep(proba, y_te, class_names)
        if sweep is not None:
            sweep.to_csv(os.path.join(args.out_dir, f"threshold_sweep_{name}.csv"), index=False)

        # per held-out person
        per_person = {}
        if df_te.subject_id.nunique() > 1:
            for s in sorted(df_te.subject_id.unique()):
                mm = (df_te.subject_id == s).to_numpy()
                per_person[s] = float(f1_score(y_te[mm], proba[mm].argmax(1), labels=range(k),
                                               average="macro", zero_division=0))

        # save bundle for gesture_runtime.py
        to_save = est.numpy_ if isinstance(est, TorchMLP) else est
        bundle = {"name": name, "estimator": to_save, "classes": class_names,
                  "meta": {"best_params": {str(a): str(b) for a, b in best_params.items()},
                           "test_macro_f1": m["macro_f1"], "test_accuracy": m["accuracy"],
                           "n_features": int(X.shape[1])}}
        joblib.dump(bundle, os.path.join(args.out_dir, f"{name}.joblib"))

        best_all[name] = best_params
        all_metrics[name] = {
            "cv_macro_f1": cv_score, "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
            "per_class": m["per_class"], "ms_per_prediction": ms,
            "best_params": {str(a): str(b) for a, b in best_params.items()},
            "per_held_out_person_macro_f1": per_person,
        }
        results.append({"model": name, "cv_macro_f1": cv_score, "test_accuracy": m["accuracy"],
                        "test_macro_f1": m["macro_f1"], "ms_per_prediction": ms,
                        "best_params": json.dumps(all_metrics[name]["best_params"])})

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    print("=== Comparison ===")
    print(res.drop(columns=["best_params"]).round(3).to_string(index=False))

    # ---- threshold sweep -------------------------------------------------
    if NONE_LABEL in class_names:
        print("\n=== Confidence threshold sweep (test set, frame level) ===")
        print("Low-confidence predictions are treated as 'none'. 'false command rate' is the share of\n"
              "'none' frames that would be classified as a real gesture. Frame-level numbers are\n"
              "worse than real use, because gesture_runtime.py also requires the gesture to be held.\n"
              "Choosing a threshold from this table is tuning on the test set, so treat it as a guide.")
        for name in model_names:
            sw = pd.read_csv(os.path.join(args.out_dir, f"threshold_sweep_{name}.csv"))
            print(f"\n{name.upper()}")
            print(sw.round(3).to_string(index=False))

    # ---- optional: learning curves ---------------------------------------
    if args.curves and not args.quick:
        print("\n=== Learning curves ===")
        for name in model_names:
            factory = lambda n=name: build_with_params(n, best_all[n], mlp_backend, args.epochs,
                                                      args.seed, probability=False)
            d = learning_curve_by_groups(factory, X_tr, y_tr, groups, X_te, y_te, class_names,
                                         args.seed, os.path.join(args.out_dir, f"learning_curve_{name}.png"),
                                         f"{name.upper()}: more {cv_level}s in training")
            print(name, d.round(3).to_dict("list"))
        if "mlp" in model_names and mlp_backend == "torch":
            mlp_loss_curves(best_all["mlp"], X_tr, y_tr, groups, args.epochs, args.seed,
                            os.path.join(args.out_dir, "mlp_loss_curves.png"))
            print("Saved mlp_loss_curves.png")

    # ---- optional: leave-one-person-out ----------------------------------
    if args.loso:
        subjects = sorted(df.subject_id.unique())
        if len(subjects) < 2:
            print("\n--loso needs at least 2 people in the data; skipped.")
        else:
            print("\n=== Leave-one-person-out (macro-F1 on the person left out) ===")
            rows = []
            for s in subjects:
                te_m = (df.subject_id == s).to_numpy()
                for name in model_names:
                    est = build_with_params(name, best_all[name], mlp_backend, args.epochs,
                                            args.seed, probability=False)
                    est.fit(X[~te_m], y[~te_m])
                    pred = est.predict(X[te_m])
                    rows.append({"left_out": s, "model": name,
                                 "macro_f1": f1_score(y[te_m], pred, labels=range(k), average="macro",
                                                      zero_division=0),
                                 "accuracy": accuracy_score(y[te_m], pred)})
            lo = pd.DataFrame(rows)
            lo.to_csv(os.path.join(args.out_dir, "loso.csv"), index=False)
            print(lo.pivot(index="left_out", columns="model", values="macro_f1").round(3).to_string())
            summary = lo.groupby("model").macro_f1.agg(["mean", "std"]).round(3)
            print("\nMean and standard deviation across people:")
            print(summary.to_string())

    # ---- save metadata ---------------------------------------------------
    meta = {"classes": class_names, "split": desc, "split_level": level, "cv_level": cv_level,
            "n_rows": int(len(df)), "n_people": int(df.subject_id.nunique()),
            "stride": stride, "mlp_backend": mlp_backend, "metrics": all_metrics}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved models, plots and metrics to: {os.path.abspath(args.out_dir)}")
    print("Use one of the .joblib files with gesture_runtime.py, e.g.  --model "
          f"{os.path.join(args.out_dir, 'svm.joblib')}")


if __name__ == "__main__":
    main()
