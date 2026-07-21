import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor


DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "exam_score"
ID_COL = "id"


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identify categorical + numerical columns
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    model = HistGradientBoostingRegressor(random_state=42)

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])
    return pipe


def main():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Split X/y
    y = train_df[TARGET_COL]
    X = train_df.drop(columns=[TARGET_COL])

    # Keep ID separately, drop from features
    X_id = X[ID_COL]
    X = X.drop(columns=[ID_COL])

    test_ids = test_df[ID_COL]
    X_test = test_df.drop(columns=[ID_COL])

    pipe = build_pipeline(X)

    # Quick validation (prints RMSE)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    mse = mean_squared_error(y_val, val_pred)
    rmse = mse ** 0.5
    print(f"Validation RMSE: {rmse:.4f}")

    # Train full model
    pipe.fit(X, y)

    # Save model for UI
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model to: {model_path}")

    # Predict test and create submission
    test_pred = pipe.predict(X_test)
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_pred})
    submission.to_csv("submission.csv", index=False)
    print("Saved: submission.csv (upload this to Kaggle)")


if __name__ == "__main__":
    main()
