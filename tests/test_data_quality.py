from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from milk_quality.config import DATASET_PATH, REQUIRED_COLUMNS, TARGET_COLUMN
from milk_quality.data import clean_training_data, load_dataset


def test_dataset_has_required_columns():
    data = load_dataset(DATASET_PATH)
    assert list(data.columns) == REQUIRED_COLUMNS


def test_dataset_has_no_missing_values():
    data = load_dataset(DATASET_PATH)
    assert data.isna().sum().sum() == 0


def test_grade_values_are_valid():
    data = load_dataset(DATASET_PATH)
    assert set(data[TARGET_COLUMN].unique()).issubset({"high", "medium", "low"})


def test_clean_training_data_removes_duplicates():
    data = load_dataset(DATASET_PATH)
    clean_data = clean_training_data(data)
    assert clean_data.duplicated().sum() == 0
    assert len(clean_data) <= len(data)
