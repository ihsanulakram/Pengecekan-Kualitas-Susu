from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from milk_quality.config import DATASET_PATH, FEATURE_COLUMNS
from milk_quality.data import load_dataset
from milk_quality.model import compare_models, evaluate_model, predict_grade, train_final_model


def test_model_can_train_and_predict():
    data = load_dataset(DATASET_PATH)
    pipeline, encoder, _, model_name, comparison = train_final_model(data)
    sample = data[FEATURE_COLUMNS].head(1)
    result = predict_grade(pipeline, encoder, sample)
    assert model_name in set(comparison["Model"])
    assert result["grade"] in {"high", "medium", "low"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_model_evaluation_returns_valid_accuracy():
    data = load_dataset(DATASET_PATH)
    metrics = evaluate_model(data)
    assert metrics["best_model_name"] in set(metrics["model_comparison"]["Model"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["cv_mean"] <= 1.0
    assert metrics["training_rows_after_dedup"] > 0


def test_model_comparison_has_multiple_candidates():
    data = load_dataset(DATASET_PATH)
    comparison = compare_models(data)
    assert len(comparison) >= 5
    assert list(comparison.columns) == ["Rank", "Model", "CV Mean", "CV Std", "CV Mean (%)", "CV Std (%)", "Fold Scores"]
    assert comparison["CV Mean"].between(0, 1).all()
