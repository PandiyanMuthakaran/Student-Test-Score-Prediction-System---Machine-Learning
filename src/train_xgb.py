
import os
import sys
import joblib
import pandas as pd

# Ensure local imports work when running from project root
sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from preprocess import load_data, split_xy, build_preprocessor, ID_COL


def main():
    # Paths (robust)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    train_df, test_df = load_data(
        train_path=os.path.join(root_dir, "data", "train.csv"),
        test_path=os.path.join(root_dir, "data", "test.csv"),
    )

    # Split X/y
    X_train_full, y_train_full = split_xy(train_df)

    # Keep test IDs for submission
    if ID_COL not in test_df.columns:
        raise ValueError(f"Expected '{ID_COL}' column in test.csv, but it was not found.")
    test_ids = test_df[ID_COL].copy()

    # Remove id from features (train + test)
    if ID_COL in X_train_full.columns:
        X_train_full = X_train_full.drop(columns=[ID_COL])

    X_test = test_df.drop(columns=[ID_COL])

    # Preprocessor (same as your best tabular pipeline)
    preprocessor = build_preprocessor(X_train_full)

    # XGBoost model (same hyperparams you used in evaluation)
    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", xgb),
    ])

    print("Training XGBoost on FULL training data...")
    pipe.fit(X_train_full, y_train_full)

    # Save model
    model_path = os.path.join(models_dir, "model_xgb.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model: {model_path}")

    # Predict on test
    test_pred = pipe.predict(X_test)

    # Build submission
    submission = pd.DataFrame({
        ID_COL: test_ids,
        "exam_score": test_pred
    })

    out_path = os.path.join(root_dir, "submission_xgb.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (upload this to Kaggle)")


if __name__ == "__main__":
    main()
