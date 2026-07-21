from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Project root (student-score-prediction/)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Use your FINAL best model
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_xgb.joblib")

# --- Load dropdown options from train.csv (so UI matches dataset) ---
TRAIN_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
train_df = pd.read_csv(TRAIN_PATH)

def uniq(col):
    # dropna, convert to string, sort for clean UI
    return sorted(train_df[col].dropna().astype(str).unique().tolist())

CATEGORIES = {
    "gender": uniq("gender"),
    "course": uniq("course"),
    "internet_access": uniq("internet_access"),
    "sleep_quality": uniq("sleep_quality"),
    "study_method": uniq("study_method"),
    "facility_rating": uniq("facility_rating"),
    "exam_difficulty": uniq("exam_difficulty"),
}

# Load the saved pipeline (preprocessing + model)
model = joblib.load(MODEL_PATH)

FEATURES = [
    "age", "gender", "course", "study_hours", "class_attendance",
    "internet_access", "sleep_hours", "sleep_quality", "study_method",
    "facility_rating", "exam_difficulty"
]

@app.route("/")
def home():
    return render_template("index.html")
    

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Build a one-row dataframe in correct order
    row = {f: data.get(f) for f in FEATURES}
    df = pd.DataFrame([row])

    # Ensure correct numeric types
    df["age"] = int(df["age"])
    df["study_hours"] = float(df["study_hours"])
    df["class_attendance"] = float(df["class_attendance"])
    df["sleep_hours"] = float(df["sleep_hours"])

    pred = float(model.predict(df)[0])

    return jsonify({
        "predicted_exam_score": round(pred, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
