"""Train, compare, and save the milk quality model artifacts.

Run from the project root:
    python scripts/train_model.py
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from milk_quality.config import DATASET_PATH
from milk_quality.data import load_dataset
from milk_quality.model import evaluate_model, save_artifacts, train_final_model


def main() -> None:
    data = load_dataset(DATASET_PATH)
    metrics = evaluate_model(data)
    pipeline, encoder, training_data, model_name, comparison = train_final_model(data)
    metadata = save_artifacts(pipeline, encoder, training_data, model_name, comparison)

    print("Model berhasil dibandingkan, dilatih, dan disimpan.")
    print(f"Model terpilih: {model_name}")
    print(f"Akurasi holdout model terpilih: {metrics['accuracy'] * 100:.2f}%")
    print(f"Cross-validation model terpilih: {metrics['cv_mean'] * 100:.2f}% ± {metrics['cv_std'] * 100:.2f}%")
    print("\nPerbandingan model:")
    print(comparison[["Rank", "Model", "CV Mean (%)", "CV Std (%)"]].to_string(index=False))
    print(f"\nMetadata: {metadata}")


if __name__ == "__main__":
    main()
