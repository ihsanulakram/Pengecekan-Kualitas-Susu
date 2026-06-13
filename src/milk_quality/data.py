from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import BINARY_COLUMNS, FEATURE_COLUMNS, REQUIRED_COLUMNS, TARGET_COLUMN


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the old typo column name normalized to Temperature."""
    normalized = data.copy()
    if "Temprature" in normalized.columns and "Temperature" not in normalized.columns:
        normalized = normalized.rename(columns={"Temprature": "Temperature"})
    return normalized


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the milk dataset and validate the required schema."""
    path = Path(path)
    data = pd.read_csv(path)
    data = standardize_columns(data)
    validate_dataset(data)
    return data[REQUIRED_COLUMNS].copy()


def validate_dataset(data: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing_columns)}")

    missing_values = int(data[REQUIRED_COLUMNS].isna().sum().sum())
    if missing_values > 0:
        raise ValueError(f"Dataset memiliki {missing_values} nilai kosong. Bersihkan data terlebih dahulu.")

    invalid_binary: List[str] = []
    for column in BINARY_COLUMNS:
        values = set(data[column].dropna().unique().tolist())
        if not values.issubset({0, 1}):
            invalid_binary.append(column)
    if invalid_binary:
        raise ValueError(
            "Kolom biner hanya boleh berisi 0 atau 1: " + ", ".join(invalid_binary)
        )

    allowed_grades = {"high", "medium", "low"}
    actual_grades = set(data[TARGET_COLUMN].astype(str).str.lower().unique().tolist())
    if not actual_grades.issubset(allowed_grades):
        raise ValueError(
            "Label Grade hanya boleh high, medium, atau low. Label ditemukan: "
            + ", ".join(sorted(actual_grades))
        )


def clean_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and normalize the Grade text before model training/evaluation."""
    cleaned = standardize_columns(data)
    cleaned = cleaned[REQUIRED_COLUMNS].copy()
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(str).str.strip().str.lower()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    validate_dataset(cleaned)
    return cleaned


def dataset_summary(data: pd.DataFrame) -> Dict[str, int]:
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicates": int(data.duplicated().sum()),
        "unique_rows": int(len(data.drop_duplicates())),
        "missing_values": int(data.isna().sum().sum()),
    }


def feature_ranges(data: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for column in FEATURE_COLUMNS:
        stats.append(
            {
                "Fitur": column,
                "Minimum": data[column].min(),
                "Rata-rata": data[column].mean(),
                "Maksimum": data[column].max(),
            }
        )
    return pd.DataFrame(stats)


def out_of_range_messages(input_values: Dict[str, float], data: pd.DataFrame) -> List[str]:
    messages: List[str] = []
    for column, value in input_values.items():
        min_value = data[column].min()
        max_value = data[column].max()
        if value < min_value or value > max_value:
            messages.append(
                f"{column} = {value} berada di luar rentang data training "
                f"({min_value} - {max_value}). Prediksi bisa kurang akurat."
            )
    return messages
