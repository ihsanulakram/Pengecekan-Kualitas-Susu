from pathlib import Path

from milk_quality.config import LABEL_ENCODER_PATH, METADATA_PATH, MODEL_PATH, PIPELINE_PATH, SCALER_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_artifacts_exist():
    for path in [PIPELINE_PATH, MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH, METADATA_PATH]:
        assert path.exists(), f"Missing model artifact: {path}"
        assert path.stat().st_size > 0, f"Empty model artifact: {path}"


def test_primary_entry_point_is_app_py():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    devcontainer = (PROJECT_ROOT / ".devcontainer" / "devcontainer.json").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "app.py" in dockerfile and "kualitas_susu.py" not in dockerfile
    assert "streamlit run app.py" in devcontainer and "kualitas_susu.py" not in devcontainer
    assert "streamlit run app.py" in readme


def test_plotly_chart_calls_do_not_use_deprecated_width_kwargs():
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    assert "st.plotly_chart(fig, width=" not in app_source
    assert "use_container_width" not in app_source
    assert "config={" in app_source


def test_model_loader_checks_sklearn_version_before_unpickling():
    model_source = (PROJECT_ROOT / "src" / "milk_quality" / "model.py").read_text(encoding="utf-8")
    assert "artifacts_match_runtime" in model_source
    assert "scikit-learn" in model_source
    assert "Retraining is required before loading persisted estimators" in model_source
