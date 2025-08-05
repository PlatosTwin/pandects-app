import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .classifier_utils import extract_features
from .constants import CLASSIFIER_XGB_PATH


df = pd.read_parquet("../data/page-data.parquet")
if not {"html", "text", "label"}.issubset(df.columns):
    raise ValueError("CSV must contain 'html', 'text', and 'label' columns")
print(f"[data] loaded {df.shape[0]} rows.")

# Map labels to integers
labels = sorted(df["label"].unique())
label2idx = {l: i for i, l in enumerate(labels)}
y = df["label"].map(label2idx).values

# Build feature matrix
X = np.vstack(
    [extract_features(r["text"], r["html"], r["order"]) for _, r in df.iterrows()]
)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)


def macro_f1_eval(preds: np.ndarray, dmatrix: xgb.DMatrix):
    # true labels
    labels = dmatrix.get_label().astype(int)
    # number of samples
    n = labels.shape[0]
    # total preds length = n * num_class
    num_class = int(preds.size / n)
    # reshape into (n_samples, n_classes)
    preds = preds.reshape(n, num_class)
    # pick the class with highest prob
    y_pred = preds.argmax(axis=1)
    return "f1_macro", f1_score(labels, y_pred, average="macro")


# Objective for Optuna
def objective(trial):
    param = {
        "objective": "multi:softprob",
        "num_class": len(labels),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "eta": trial.suggest_float("eta", 1e-3, 1e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    bst = xgb.train(
        param,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        custom_metric=macro_f1_eval,
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    y_prob = bst.predict(dval)
    y_pred = np.argmax(y_prob, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    return f1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=400)

print("Best params:", study.best_params)

# Train final model
best = study.best_params
best.update(
    {
        "objective": "multi:softprob",
        "num_class": len(labels),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
    }
)
final_bst = xgb.train(best, dtrain, num_boost_round=study.best_trial.number)

final_bst.save_model(CLASSIFIER_XGB_PATH)

# Evaluate
y_hat = np.argmax(final_bst.predict(dval), axis=1)
acc = accuracy_score(y_val, y_hat)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_hat, average="macro")
print(f"Val Accuracy: {acc:.4f}")
print(f"Val Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
