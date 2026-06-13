from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PREDICTION_DIR = DATA_DIR / "prediksi"
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "milkdata.csv"

FEATURE_COLUMNS = [
    "pH",
    "Temperature",
    "Taste",
    "Odor",
    "Fat",
    "Turbidity",
    "Colour",
]
TARGET_COLUMN = "Grade"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]
BINARY_COLUMNS = ["Taste", "Odor", "Fat", "Turbidity"]
GRADE_ORDER = ["high", "medium", "low"]

PIPELINE_PATH = MODEL_DIR / "milk_grade_pipeline.pkl"
MODEL_PATH = MODEL_DIR / "milk_grade_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
