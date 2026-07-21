# src/app.py
import os
import joblib
import pandas as pd
import streamlit as st

# ----------------------------
# Paths (robust for Windows + VS Code)
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_TRAIN_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "model_xgb.joblib")

ID_COL = "id"
TARGET_COL = "exam_score"

FEATURE_COLS = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty",
]

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="Student Exam Score Predictor", layout="centered")
st.title("Student Exam Score Predictor")
st.write(
    "Enter student details and predict the exam score using the trained Kaggle model "
    " XGBoostRegressor pipeline (saved from training).."
)

# ----------------------------
# Load model + training metadata
# ----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_train_for_metadata():
    """
    Used only to:
    - get allowed category values
    - get realistic min/max for sliders
    """
    if not os.path.exists(DATA_TRAIN_PATH):
        return None
    df = pd.read_csv(DATA_TRAIN_PATH)
    return df

try:
    model = load_model()
except Exception as e:
    st.error(" Could not load the model.")
    st.code(str(e))
    st.info(
        "Check that 'models/model_v2.joblib' exists. You should have created it by running:\n"
        "python src/train_v2.py"
    )
    st.stop()

train_df = load_train_for_metadata()

# ----------------------------
# Helper functions
# ----------------------------
def safe_unique(df, col, fallback):
    if df is None or col not in df.columns:
        return fallback
    vals = sorted([v for v in df[col].dropna().unique().tolist()])
    return vals if vals else fallback

def safe_range(df, col, fallback_min, fallback_max, step=1):
    if df is None or col not in df.columns:
        return fallback_min, fallback_max, step
    mn = df[col].min()
    mx = df[col].max()
    # handle weird cases
    if pd.isna(mn) or pd.isna(mx) or mn == mx:
        return fallback_min, fallback_max, step
    return mn, mx, step

# Category options (pulled from your train.csv so they match training)
gender_opts = safe_unique(train_df, "gender", ["female", "male", "other"])
course_opts = safe_unique(train_df, "course", ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"])
internet_opts = safe_unique(train_df, "internet_access", ["yes", "no"])
sleep_quality_opts = safe_unique(train_df, "sleep_quality", ["poor", "average", "good"])
study_method_opts = safe_unique(train_df, "study_method", ["self-study", "group study", "online videos", "coaching", "mixed"])
facility_opts = safe_unique(train_df, "facility_rating", ["low", "medium", "high"])
difficulty_opts = safe_unique(train_df, "exam_difficulty", ["easy", "moderate", "hard"])

# Numeric ranges (pulled from train.csv for realistic defaults)
age_min, age_max, _ = safe_range(train_df, "age", 16, 30, 1)
study_min, study_max, _ = safe_range(train_df, "study_hours", 0.0, 12.0, 0.1)
att_min, att_max, _ = safe_range(train_df, "class_attendance", 0.0, 100.0, 0.1)
sleep_min, sleep_max, _ = safe_range(train_df, "sleep_hours", 0.0, 12.0, 0.1)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Input Features")

# Defaults: pick something sensible in-range
default_age = int(max(age_min, min(age_max, 21)))
default_study = float(max(study_min, min(study_max, 4.0)))
default_att = float(max(att_min, min(att_max, 75.0)))
default_sleep = float(max(sleep_min, min(sleep_max, 7.0)))

age = st.sidebar.number_input(
    "Age",
    min_value=int(age_min),
    max_value=int(age_max),
    value=int(default_age),
    step=1
)

gender = st.sidebar.selectbox("Gender", gender_opts, index=0 if "female" in gender_opts else 0)
course = st.sidebar.selectbox("Course", course_opts, index=0)

study_hours = st.sidebar.number_input(
    "Study Hours (per day)",
    min_value=float(study_min),
    max_value=float(study_max),
    value=float(default_study),
    step=0.01,
    format="%.2f"
)
class_attendance = st.sidebar.number_input(
    "Class Attendance (%)",
    min_value=float(att_min),
    max_value=float(att_max),
    value=float(default_att),
    step=0.1,
    format="%.1f"
)

internet_access = st.sidebar.selectbox("Internet Access", internet_opts, index=0 if "yes" in internet_opts else 0)

sleep_hours = st.sidebar.number_input(
    "Sleep Hours (per day)",
    min_value=float(sleep_min),
    max_value=float(sleep_max),
    value=float(default_sleep),
    step=0.01,
    format="%.2f"
)

sleep_quality = st.sidebar.selectbox("Sleep Quality", sleep_quality_opts, index=0 if "average" in sleep_quality_opts else 0)

study_method = st.sidebar.selectbox("Study Method", study_method_opts, index=0)
facility_rating = st.sidebar.selectbox("Facility Rating", facility_opts, index=0)
exam_difficulty = st.sidebar.selectbox("Exam Difficulty", difficulty_opts, index=0)

# ----------------------------
# Build input dataframe (must match training feature names)
# ----------------------------
input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "course": course,
    "study_hours": study_hours,
    "class_attendance": class_attendance,
    "internet_access": internet_access,
    "sleep_hours": sleep_hours,
    "sleep_quality": sleep_quality,
    "study_method": study_method,
    "facility_rating": facility_rating,
    "exam_difficulty": exam_difficulty,
}])

st.subheader("Preview: Input Sent to Model")
st.dataframe(input_df, width="stretch")

# ----------------------------
# Predict + Download
# ----------------------------
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("Predict Exam Score"):
        try:
            pred = float(model.predict(input_df)[0])
            st.success(f"Predicted Exam Score: {pred:.2f}")

            out = input_df.copy()
            out["predicted_exam_score"] = pred
            csv_bytes = out.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction as CSV",
                data=csv_bytes,
                file_name="prediction.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error("Prediction failed. This usually happens if the UI inputs don't match the model’s expected features.")
            st.code(str(e))

with col2:
    st.markdown("### How it works")
    st.write(
        "- Loads the trained model pipeline from `models/model_v2.joblib`.\n"
        "- Preprocesses your inputs (categorical one-hot encoding + numeric handling).\n"
        "- Predicts `exam_score` using a tree-based boosting model.\n"
        "- Lets you download the inputs + predicted score."
    )

st.caption(f"Model path: {MODEL_PATH}")
