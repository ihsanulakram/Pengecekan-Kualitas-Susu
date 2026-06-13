from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import PREDICTION_DIR


def ensure_prediction_dir(directory: str | Path = PREDICTION_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prediction_file_for_today(directory: str | Path = PREDICTION_DIR) -> Path:
    directory = ensure_prediction_dir(directory)
    today = datetime.today().strftime("%Y-%m-%d")
    return directory / f"prediksi_susu_{today}.csv"


def append_prediction(record: Dict[str, object], directory: str | Path = PREDICTION_DIR) -> Path:
    path = prediction_file_for_today(directory)
    row = pd.DataFrame([record])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path


def list_prediction_files(directory: str | Path = PREDICTION_DIR) -> List[Path]:
    directory = ensure_prediction_dir(directory)
    return sorted(directory.glob("prediksi_susu_*.csv"), reverse=True)


def load_prediction_history(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
