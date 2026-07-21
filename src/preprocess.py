import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ID_COL = "id"
TARGET_COL = "exam_score"

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def split_xy(train_df: pd.DataFrame):
    y = train_df[TARGET_COL]
    X = train_df.drop(columns=[TARGET_COL])
    return X, y

def build_preprocessor(X_features: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_features.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_features.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )
    return preprocessor

from sklearn.preprocessing import StandardScaler

def build_preprocessor_for_mlp(X_features: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X_features.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_features.columns if c not in cat_cols]

    # For older sklearn, use sparse=False (not sparse_output)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

    ])

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor

def build_preprocessor_for_linear(X_features: pd.DataFrame) -> ColumnTransformer:
    """
    For linear models (Ridge/Lasso): scale numeric columns.
    One-hot for categorical columns.
    """
    cat_cols = X_features.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_features.columns if c not in cat_cols]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))  # keep default sparse_output
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return preprocessor
