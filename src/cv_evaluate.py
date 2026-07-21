# src/cv_evaluate.py
import os
import sys
import time
import numpy as np

# Ensure local imports work when running from project root
sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from preprocess import (
    load_data,
    split_xy,
    build_preprocessor,
    build_preprocessor_for_mlp,
    build_preprocessor_for_linear,
    ID_COL
)

# If xgboost isn't installed, this import will fail
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGB_OK = False


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def evaluate_model_cv(name, preprocessor, model, X, y, cv):
    fold_scores = []
    start = time.time()

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        score = rmse(y_va, preds)
        fold_scores.append(score)
        print(f"{name:16s} | Fold {fold} RMSE: {score:.4f}")

    elapsed = time.time() - start
    fold_scores = np.array(fold_scores)

    mean_score = fold_scores.mean()
    std_score = fold_scores.std(ddof=1) if len(fold_scores) > 1 else 0.0
    print(f"{name:16s} | CV RMSE: {mean_score:.4f} ± {std_score:.4f}  (time: {elapsed:.1f}s)\n")
    return mean_score, std_score


def main():
    train_df, _ = load_data()
    X, y = split_xy(train_df)

    # Remove ID from features if present
    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    # Preprocessors
    prep_default = build_preprocessor(X)              # tree models + xgb
    prep_linear = build_preprocessor_for_linear(X)    # ridge/lasso
    prep_mlp = build_preprocessor_for_mlp(X)          # mlp (scaled + dense)

    # CV strategy
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = []

    # Linear baselines
    models.append(("Ridge", prep_linear, Ridge(alpha=1.0, random_state=42)))
    models.append(("LassoCV", prep_linear, LassoCV(
        alphas=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
        cv=5,
        max_iter=20000,
        random_state=42
    )))

    # Tree ensemble + boosting
    models.append(("RandomForest", prep_default, RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1
    )))
    models.append(("HistGBR", prep_default, HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.06,
        max_leaf_nodes=63,
        max_depth=None
    )))

    # Neural network baseline
    models.append(("MLP", prep_mlp, MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=42,
        early_stopping=True
    )))

    # XGBoost (if available)
    if XGB_OK:
        models.append(("XGBoost", prep_default, XGBRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )))
    else:
        print("⚠️ XGBoost not installed. Run: python -m pip install xgboost\n")

    results = []

    print("=== 5-Fold CV Evaluation (RMSE lower is better) ===\n")
    for name, prep, model in models:
        mean_score, std_score = evaluate_model_cv(name, prep, model, X, y, cv)
        results.append((name, mean_score, std_score))

    # Sort by mean RMSE
    results.sort(key=lambda x: x[1])

    print("=== CV Summary (sorted) ===")
    for name, mean_score, std_score in results:
        print(f"{name:16s} -> RMSE {mean_score:.4f} ± {std_score:.4f}")

    best = results[0]
    print(f"\nBest by CV mean RMSE: {best[0]} (RMSE {best[1]:.4f} ± {best[2]:.4f})")


if __name__ == "__main__":
    main()
