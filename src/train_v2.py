import os
import sys
import joblib
import pandas as pd

# Make imports work when running from project root
sys.path.append(os.path.dirname(__file__))

from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from preprocess import load_data, split_xy, build_preprocessor, ID_COL, TARGET_COL

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    train_df, test_df = load_data()

    # Split into features/target
    X, y = split_xy(train_df)

    # Remove ID from features
    X_features = X.drop(columns=[ID_COL])
    X_test = test_df.drop(columns=[ID_COL])
    test_ids = test_df[ID_COL]

    # Preprocess (same as evaluate)
    preprocessor = build_preprocessor(X_features)

    # Best model from your evaluation (HistGBR)
    model = HistGradientBoostingRegressor(
        random_state=42,
        learning_rate=0.06,
        max_leaf_nodes=63,
        max_depth=None
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    print("Training V2 model on FULL training data...")
    pipe.fit(X_features, y)

    # Save model for UI later
    model_path = os.path.join(MODEL_DIR, "model_v2.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model: {model_path}")

    # Predict test
    test_pred = pipe.predict(X_test)

    # Create submission_v2.csv
    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: test_pred
    })

    out_path = "submission_v2.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path} (upload this to Kaggle)")

if __name__ == "__main__":
    main()
