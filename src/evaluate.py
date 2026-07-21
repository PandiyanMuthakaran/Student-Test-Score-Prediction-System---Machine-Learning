import os
import sys

# This makes "import preprocess" work on Windows when running from project root
sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from preprocess import build_preprocessor_for_mlp

from sklearn.linear_model import LassoCV
from xgboost import XGBRegressor

from preprocess import build_preprocessor_for_linear


from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from preprocess import load_data, split_xy, build_preprocessor, ID_COL


def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5


def main():
    train_df, _ = load_data()
    X, y = split_xy(train_df)

    # Remove ID from features
    X_features = X.drop(columns=[ID_COL])

    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_features)

    preprocessor_mlp = build_preprocessor_for_mlp(X_features)

    preprocessor_linear = build_preprocessor_for_linear(X_features)


    models = {
    # 1) Linear baseline (different technique)
    "RidgeRegression": (preprocessor, Ridge(alpha=1.0, random_state=42)),

    # 2) Tree-based boosting (your best so far)
    "HistGBR_tuned": (preprocessor, HistGradientBoostingRegressor(
        random_state=42, learning_rate=0.06, max_leaf_nodes=63, max_depth=None
    )),

    # 3) Tree ensemble
    "RandomForest": (preprocessor, RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1
    )),

    # 4) Neural network (MLP)
    "MLPRegressor": (preprocessor_mlp, MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )),
    
    "LassoCV": (preprocessor_linear, LassoCV(
    alphas=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0],
    cv=5,
    max_iter=20000,
    random_state=42
)),

"XGBoost": (preprocessor, XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)),

}


    results = []

    for name, (prep, model) in models.items():
        pipe = Pipeline([("prep", prep), ("model", model)])

        print(f"\nTraining: {name} ...")
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_val)
        score = rmse(y_val, preds)
        print(f"{name} RMSE: {score:.4f}")
        results.append((name, score))

    results.sort(key=lambda x: x[1])
    print("\n=== Summary (lower RMSE is better) ===")
    for name, score in results:
        print(f"{name:15s} -> RMSE {score:.4f}")

    best_name, best_score = results[0]
    print(f"\nBest model: {best_name} (RMSE {best_score:.4f})")


if __name__ == "__main__":
    main()
