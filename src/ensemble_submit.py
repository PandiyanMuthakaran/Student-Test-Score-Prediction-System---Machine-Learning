# src/ensemble_submit.py
import os
import joblib
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

XGB_PATH  = os.path.join(ROOT_DIR, "models", "model_xgb.joblib")
HGB_PATH  = os.path.join(ROOT_DIR, "models", "model_v2.joblib")  # your HistGBR pipeline
TEST_PATH = os.path.join(ROOT_DIR, "data", "test.csv")
OUT_PATH  = os.path.join(ROOT_DIR, "submission_ensemble.csv")

ID_COL = "id"
TARGET_COL = "exam_score"

def main():
    test_df = pd.read_csv(TEST_PATH)

    ids = test_df[ID_COL].copy()
    X_test = test_df.drop(columns=[ID_COL])

    xgb = joblib.load(XGB_PATH)
    hgb = joblib.load(HGB_PATH)

    pred_xgb = xgb.predict(X_test)
    pred_hgb = hgb.predict(X_test)

    # Simple average (start here)
    pred = 0.5 * pred_xgb + 0.5 * pred_hgb

    # Optional: clip to a realistic range (only if needed)
    # pred = np.clip(pred, 0, 100)

    sub = pd.DataFrame({ID_COL: ids, TARGET_COL: pred})
    sub.to_csv(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} (upload to Kaggle)")
    print(f"Mean preds -> XGB: {pred_xgb.mean():.4f} | HGB: {pred_hgb.mean():.4f} | ENS: {pred.mean():.4f}")

if __name__ == "__main__":
    main()
