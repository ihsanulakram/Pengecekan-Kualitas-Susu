from __future__ import annotations

import json
from importlib import metadata as importlib_metadata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import (
    FEATURE_COLUMNS,
    LABEL_ENCODER_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PIPELINE_PATH,
    SCALER_PATH,
    TARGET_COLUMN,
)
from .data import clean_training_data

DEFAULT_RANDOM_STATE = 42
DEFAULT_MODEL_NAME = "Gaussian Naive Bayes"

MODEL_FACTORIES: "OrderedDict[str, Callable[[], Any]]" = OrderedDict(
    [
        ("Gaussian Naive Bayes", lambda: GaussianNB()),
        ("Decision Tree", lambda: DecisionTreeClassifier(random_state=DEFAULT_RANDOM_STATE, max_depth=6)),
        (
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=200,
                random_state=DEFAULT_RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        ("K-Nearest Neighbors", lambda: KNeighborsClassifier(n_neighbors=5)),
        (
            "Logistic Regression",
            lambda: LogisticRegression(
                max_iter=1_000,
                random_state=DEFAULT_RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
    ]
)


def available_model_names() -> list[str]:
    return list(MODEL_FACTORIES.keys())


def encode_target(labels: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels.astype(str).str.lower())
    return encoded, encoder


def _get_cv(y: np.ndarray, random_state: int = DEFAULT_RANDOM_STATE) -> StratifiedKFold:
    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_pipeline(model_name: str = DEFAULT_MODEL_NAME) -> Pipeline:
    if model_name not in MODEL_FACTORIES:
        raise ValueError(
            f"Model '{model_name}' tidak tersedia. Pilihan: {', '.join(available_model_names())}"
        )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", MODEL_FACTORIES[model_name]()),
        ]
    )


def compare_models(data: pd.DataFrame, random_state: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    """Compare candidate algorithms using stratified cross-validation."""
    training_data = clean_training_data(data)
    x = training_data[FEATURE_COLUMNS]
    y, _ = encode_target(training_data[TARGET_COLUMN])
    cv = _get_cv(y, random_state=random_state)

    rows: list[dict[str, Any]] = []
    for order, model_name in enumerate(available_model_names(), start=1):
        scores = cross_val_score(build_pipeline(model_name), x, y, cv=cv, scoring="accuracy")
        rows.append(
            {
                "Model": model_name,
                "CV Mean": float(scores.mean()),
                "CV Std": float(scores.std()),
                "CV Mean (%)": float(scores.mean() * 100),
                "CV Std (%)": float(scores.std() * 100),
                "Fold Scores": [float(score) for score in scores],
                "Urutan": order,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(
        by=["CV Mean", "CV Std", "Urutan"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    comparison.insert(0, "Rank", range(1, len(comparison) + 1))
    return comparison.drop(columns=["Urutan"])


def select_best_model_name(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return DEFAULT_MODEL_NAME
    return str(comparison.iloc[0]["Model"])


def train_final_model(
    data: pd.DataFrame,
    model_name: str | None = None,
) -> Tuple[Pipeline, LabelEncoder, pd.DataFrame, str, pd.DataFrame]:
    training_data = clean_training_data(data)
    comparison = compare_models(training_data)
    selected_model_name = model_name or select_best_model_name(comparison)

    x = training_data[FEATURE_COLUMNS]
    y, encoder = encode_target(training_data[TARGET_COLUMN])
    pipeline = build_pipeline(selected_model_name)
    pipeline.fit(x, y)
    return pipeline, encoder, training_data, selected_model_name, comparison


def evaluate_model(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    training_data = clean_training_data(data)
    x = training_data[FEATURE_COLUMNS]
    y, encoder = encode_target(training_data[TARGET_COLUMN])
    model_comparison = compare_models(training_data, random_state=random_state)
    best_model_name = select_best_model_name(model_comparison)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(best_model_name)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    class_names = encoder.classes_.tolist()
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    matrix_df = pd.DataFrame(matrix, index=class_names, columns=class_names)

    best_scores = model_comparison.iloc[0]["Fold Scores"]
    cv_scores = np.array(best_scores, dtype=float)

    return {
        "best_model_name": best_model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": report_df,
        "confusion_matrix": matrix_df,
        "cv_scores": cv_scores,
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "model_comparison": model_comparison,
        "test_size": test_size,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "training_rows_after_dedup": int(len(training_data)),
        "class_names": class_names,
    }


def _json_safe_model_comparison(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in comparison.to_dict(orient="records"):
        safe_record: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, np.generic):
                safe_record[key] = value.item()
            elif isinstance(value, np.ndarray):
                safe_record[key] = value.tolist()
            else:
                safe_record[key] = value
        records.append(safe_record)
    return records




def _package_version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"

def save_artifacts(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    training_data: pd.DataFrame,
    model_name: str,
    model_comparison: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR,
) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(pipeline.named_steps["model"], MODEL_PATH)
    joblib.dump(pipeline.named_steps["scaler"], SCALER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    best_row = model_comparison.iloc[0].to_dict() if not model_comparison.empty else {}
    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classes": label_encoder.classes_.tolist(),
        "training_rows_after_dedup": int(len(training_data)),
        "selected_model": model_name,
        "model": f"{model_name} + StandardScaler",
        "selection_method": "Highest mean accuracy from stratified cross-validation after duplicate removal",
        "best_cv_mean": float(best_row.get("CV Mean", 0.0)),
        "best_cv_std": float(best_row.get("CV Std", 0.0)),
        "model_comparison": _json_safe_model_comparison(model_comparison),
        "dependency_versions": {
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit-learn": _package_version("scikit-learn"),
            "joblib": _package_version("joblib"),
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def read_metadata() -> Dict[str, Any]:
    if not METADATA_PATH.exists():
        return {}
    try:
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def artifacts_match_runtime(metadata: Dict[str, Any]) -> bool:
    """Return True when persisted model artifacts are safe to unpickle.

    scikit-learn explicitly warns that persisted estimators are not guaranteed
    to be compatible across versions. To avoid ``InconsistentVersionWarning``
    and reduce the risk of invalid results, the app retrains from the CSV
    before loading pickles when the recorded scikit-learn version differs from
    the installed runtime version.
    """
    saved_versions = metadata.get("dependency_versions", {})
    saved_sklearn = saved_versions.get("scikit-learn")
    runtime_sklearn = _package_version("scikit-learn")
    return bool(saved_sklearn) and saved_sklearn == runtime_sklearn


def load_artifacts() -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
    metadata = read_metadata()
    if not artifacts_match_runtime(metadata):
        saved_version = metadata.get("dependency_versions", {}).get("scikit-learn", "unknown")
        runtime_version = _package_version("scikit-learn")
        raise RuntimeError(
            "Model artifacts were created with scikit-learn "
            f"{saved_version}, but the current environment uses {runtime_version}. "
            "Retraining is required before loading persisted estimators."
        )

    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return pipeline, label_encoder, metadata


def load_or_train_artifacts(data: pd.DataFrame) -> Tuple[Pipeline, LabelEncoder, Dict[str, Any]]:
    try:
        return load_artifacts()
    except Exception:
        pipeline, label_encoder, training_data, model_name, comparison = train_final_model(data)
        metadata = save_artifacts(pipeline, label_encoder, training_data, model_name, comparison)
        return pipeline, label_encoder, metadata


def predict_grade(
    pipeline: Pipeline,
    label_encoder: LabelEncoder,
    input_row: pd.DataFrame,
) -> Dict[str, Any]:
    encoded_prediction = pipeline.predict(input_row)[0]
    predicted_grade = label_encoder.inverse_transform([encoded_prediction])[0]

    probabilities = pipeline.predict_proba(input_row)[0]
    probability_table = pd.DataFrame(
        {
            "Grade": label_encoder.classes_,
            "Probabilitas": probabilities,
            "Probabilitas (%)": probabilities * 100,
        }
    ).sort_values("Probabilitas", ascending=False)

    return {
        "grade": str(predicted_grade),
        "confidence": float(probabilities.max()),
        "probabilities": probability_table.reset_index(drop=True),
    }
