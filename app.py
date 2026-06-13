from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from streamlit_option_menu import option_menu
except Exception:  # pragma: no cover - fallback when optional menu package is unavailable
    option_menu = None

from milk_quality.config import DATASET_PATH, FEATURE_COLUMNS, PREDICTION_DIR, TARGET_COLUMN
from milk_quality.data import (
    clean_training_data,
    dataset_summary,
    feature_ranges,
    load_dataset,
    out_of_range_messages,
)
from milk_quality.model import evaluate_model, load_or_train_artifacts, predict_grade
from milk_quality.storage import append_prediction, list_prediction_files, load_prediction_history


st.set_page_config(
    page_title="Pengecekan Kualitas Susu",
    page_icon="🥛",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def get_clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    return clean_training_data(data)


@st.cache_data(show_spinner=False)
def get_model_metrics(data: pd.DataFrame) -> dict:
    return evaluate_model(data)


@st.cache_resource(show_spinner=False)
def get_model_artifacts(data: pd.DataFrame):
    return load_or_train_artifacts(data)


def render_metric_cards(summary: dict) -> None:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jumlah Data", f"{summary['rows']:,}")
    col2.metric("Jumlah Kolom", f"{summary['columns']:,}")
    col3.metric("Data Duplikat", f"{summary['duplicates']:,}")
    col4.metric("Data Unik", f"{summary['unique_rows']:,}")
    col5.metric("Missing Value", f"{summary['missing_values']:,}")


def render_grade_result(grade: str, confidence: float) -> None:
    grade_upper = grade.upper()
    message = f"Grade Susu: {grade_upper} | Keyakinan Model: {confidence * 100:.2f}%"
    if grade == "high":
        st.success(message)
        st.write("Saran: kualitas susu baik. Susu layak diproses lebih lanjut sesuai standar operasional.")
    elif grade == "medium":
        st.warning(message)
        st.write("Saran: kualitas susu sedang. Lakukan pemeriksaan tambahan sebelum digunakan untuk konsumsi/produksi.")
    else:
        st.error(message)
        st.write("Saran: kualitas susu rendah. Tidak disarankan untuk dikonsumsi langsung tanpa pemeriksaan lanjutan.")

    if confidence < 0.70:
        st.warning(
            "Model kurang yakin dengan prediksi ini. Disarankan melakukan pemeriksaan tambahan atau uji laboratorium sebelum mengambil keputusan."
        )


def render_sidebar_menu() -> str:
    menu_items = [
        "Pengecekan Grade Susu",
        "Data Susu",
        "Riwayat Prediksi",
        "Evaluasi Model",
        "Tentang Dataset",
    ]
    icons = ["check-circle", "database", "clock-history", "bar-chart", "info-circle"]

    with st.sidebar:
        st.markdown("# 🥛")
        st.markdown("## Kualitas Susu")
        st.caption("Prediksi grade susu dengan model ML terbaik hasil cross-validation")
        if option_menu is not None:
            return option_menu(
                "Menu",
                menu_items,
                icons=icons,
                menu_icon="list",
                default_index=0,
            )
        return st.radio("Menu", menu_items, index=0)


def render_plotly_chart(fig) -> None:
    """Render Plotly charts without deprecated Streamlit kwargs.

    Streamlit deprecated passing Plotly configuration through arbitrary
    keyword arguments. Keeping chart options inside ``config`` avoids the
    deprecation warning across supported Streamlit versions.
    """
    fig.update_layout(autosize=True)
    st.plotly_chart(
        fig,
        config={
            "responsive": True,
            "displaylogo": False,
        },
    )


def render_prediction_page(data: pd.DataFrame, clean_data: pd.DataFrame) -> None:
    pipeline, label_encoder, metadata = get_model_artifacts(data)

    st.title("🥛 Pengecekan Grade Susu")
    st.write(
        "Masukkan karakteristik susu untuk memprediksi grade kualitas: "
        "**high**, **medium**, atau **low**."
    )
    st.info(
        "Catatan: hasil prediksi hanya sebagai bantuan awal. Keputusan konsumsi atau produksi tetap perlu mengikuti uji laboratorium dan standar keamanan pangan."
    )

    with st.expander("Lihat rentang data training"):
        st.dataframe(feature_ranges(clean_data), width="stretch")

    with st.form(key="milk_quality_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            nilai_ph = st.number_input(
                "pH",
                min_value=0.0,
                max_value=14.0,
                value=float(round(clean_data["pH"].median(), 1)),
                step=0.1,
                format="%.1f",
                help="Rentang umum pH adalah 0-14. Aplikasi akan memberi peringatan jika nilai berada di luar data training.",
            )
            nilai_temperature = st.number_input(
                "Temperature (°C)",
                min_value=0,
                max_value=100,
                value=int(clean_data["Temperature"].median()),
                step=1,
            )
            nilai_colour = st.slider(
                "Colour",
                min_value=240,
                max_value=255,
                value=int(clean_data["Colour"].median()),
                step=1,
            )
        with col2:
            nilai_taste = st.selectbox("Taste", options=[0, 1], format_func=lambda x: "Bad (0)" if x == 0 else "Good (1)")
            nilai_odor = st.selectbox("Odor", options=[0, 1], format_func=lambda x: "Bad (0)" if x == 0 else "Good (1)")
        with col3:
            nilai_fat = st.selectbox("Fat", options=[0, 1], format_func=lambda x: "Bad (0)" if x == 0 else "Good (1)")
            nilai_turbidity = st.selectbox("Turbidity", options=[0, 1], format_func=lambda x: "Bad (0)" if x == 0 else "Good (1)")

        submitted = st.form_submit_button("CEK GRADE SUSU", width="stretch")

    if not submitted:
        return

    input_values = {
        "pH": float(nilai_ph),
        "Temperature": int(nilai_temperature),
        "Taste": int(nilai_taste),
        "Odor": int(nilai_odor),
        "Fat": int(nilai_fat),
        "Turbidity": int(nilai_turbidity),
        "Colour": int(nilai_colour),
    }

    for message in out_of_range_messages(input_values, clean_data):
        st.warning(message)

    input_df = pd.DataFrame([input_values], columns=FEATURE_COLUMNS)
    prediction = predict_grade(pipeline, label_encoder, input_df)
    grade = prediction["grade"].lower()
    confidence = prediction["confidence"]
    probability_df = prediction["probabilities"]

    st.divider()
    render_grade_result(grade, confidence)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Data Input")
        st.dataframe(input_df, width="stretch")
    with col2:
        st.subheader("Probabilitas Tiap Grade")
        display_prob = probability_df.copy()
        display_prob["Probabilitas (%)"] = display_prob["Probabilitas (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_prob[["Grade", "Probabilitas (%)"]], width="stretch")

    probability_columns = {
        f"Probability_{row['Grade']}": round(float(row["Probabilitas"]) * 100, 4)
        for _, row in probability_df.iterrows()
    }
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **input_values,
        "PredictedGrade": grade,
        "ConfidencePercent": round(confidence * 100, 4),
        **probability_columns,
        "Verified": False,
    }

    try:
        saved_path = append_prediction(record, PREDICTION_DIR)
        st.success(f"Hasil prediksi disimpan ke: {saved_path.relative_to(PROJECT_ROOT)}")
    except Exception as exc:
        st.error(f"Hasil prediksi tidak berhasil disimpan: {exc}")

    result_df = pd.DataFrame([record])
    st.download_button(
        label="Unduh Hasil Prediksi Ini",
        data=result_df.to_csv(index=False),
        file_name="hasil_prediksi_susu.csv",
        mime="text/csv",
        width="stretch",
    )

    if metadata:
        st.caption(
            f"Model: {metadata.get('selected_model', metadata.get('model', 'Model'))} | "
            f"Data training unik: {metadata.get('training_rows_after_dedup', '-')} baris"
        )


def render_data_page(data: pd.DataFrame, clean_data: pd.DataFrame) -> None:
    st.title("📊 Data Susu")
    st.write("Halaman ini menampilkan dataset asli, dataset bersih, statistik fitur, dan distribusi grade.")

    render_metric_cards(dataset_summary(data))
    if dataset_summary(data)["duplicates"] > 0:
        st.warning(
            "Dataset asli mengandung duplikat. Model dilatih dan dievaluasi menggunakan data yang sudah dibersihkan dari duplikat."
        )

    tab_original, tab_clean, tab_visual = st.tabs(["Dataset Asli", "Dataset Bersih", "Visualisasi"])

    with tab_original:
        st.dataframe(data, width="stretch")
        st.download_button(
            "Unduh Dataset Asli",
            data=data.to_csv(index=False),
            file_name="milkdata.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab_clean:
        st.dataframe(clean_data, width="stretch")
        st.download_button(
            "Unduh Dataset Bersih Tanpa Duplikat",
            data=clean_data.to_csv(index=False),
            file_name="milkdata_clean.csv",
            mime="text/csv",
            width="stretch",
        )

    with tab_visual:
        col1, col2 = st.columns(2)
        with col1:
            grade_counts = clean_data[TARGET_COLUMN].value_counts().reset_index()
            grade_counts.columns = ["Grade", "Jumlah"]
            fig = px.bar(grade_counts, x="Grade", y="Jumlah", text="Jumlah", title="Distribusi Grade")
            fig.update_traces(textposition="outside")
            render_plotly_chart(fig)
        with col2:
            fig = px.pie(grade_counts, values="Jumlah", names="Grade", title="Persentase Grade")
            render_plotly_chart(fig)

        st.subheader("Statistik Fitur")
        st.dataframe(feature_ranges(clean_data), width="stretch")

        numeric_df = clean_data[FEATURE_COLUMNS].melt(var_name="Fitur", value_name="Nilai")
        fig = px.box(numeric_df, x="Fitur", y="Nilai", title="Sebaran Nilai Fitur")
        render_plotly_chart(fig)


def render_history_page() -> None:
    st.title("🕘 Riwayat Prediksi")
    files = list_prediction_files(PREDICTION_DIR)
    if not files:
        st.info("Belum ada riwayat prediksi. Lakukan prediksi terlebih dahulu pada menu Pengecekan Grade Susu.")
        return

    selected_file = st.selectbox(
        "Pilih file riwayat",
        files,
        format_func=lambda path: path.name,
    )
    history = load_prediction_history(selected_file)
    st.dataframe(history, width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Prediksi", f"{len(history):,}")
    if "PredictedGrade" in history.columns:
        most_common = history["PredictedGrade"].mode().iloc[0]
        col2.metric("Grade Terbanyak", str(most_common).upper())
    if "ConfidencePercent" in history.columns:
        col3.metric("Rata-rata Keyakinan", f"{history['ConfidencePercent'].mean():.2f}%")

    st.download_button(
        "Unduh Riwayat Prediksi",
        data=history.to_csv(index=False),
        file_name=selected_file.name,
        mime="text/csv",
        width="stretch",
    )

    if "PredictedGrade" in history.columns:
        grade_counts = history["PredictedGrade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Jumlah"]
        fig = px.bar(grade_counts, x="Grade", y="Jumlah", text="Jumlah", title="Distribusi Hasil Prediksi")
        fig.update_traces(textposition="outside")
        render_plotly_chart(fig)


def render_evaluation_page(data: pd.DataFrame, clean_data: pd.DataFrame) -> None:
    st.title("📈 Evaluasi Model")
    st.write(
        "Evaluasi dilakukan pada dataset yang sudah dibersihkan dari duplikat agar skor tidak terlihat terlalu tinggi secara tidak adil."
    )

    metrics = get_model_metrics(data)
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model Terbaik", metrics["best_model_name"])
    col2.metric("Akurasi Holdout", f"{metrics['accuracy'] * 100:.2f}%")
    col3.metric("Cross-Validation", f"{metrics['cv_mean'] * 100:.2f}%")
    col4.metric("CV Std", f"{metrics['cv_std'] * 100:.2f}%")
    col5.metric("Data Unik", f"{metrics['training_rows_after_dedup']:,}")

    st.caption(
        f"Split evaluasi: {metrics['train_rows']} data training dan {metrics['test_rows']} data testing. "
        f"Model terbaik dipilih berdasarkan rata-rata akurasi stratified cross-validation."
    )

    tab_comparison, tab_report, tab_matrix, tab_quality = st.tabs([
        "Perbandingan Model",
        "Classification Report",
        "Confusion Matrix",
        "Kualitas Data",
    ])

    with tab_comparison:
        st.subheader("Perbandingan Algoritma")
        comparison = metrics["model_comparison"].copy()
        display_comparison = comparison[["Rank", "Model", "CV Mean (%)", "CV Std (%)"]].copy()
        display_comparison["CV Mean (%)"] = display_comparison["CV Mean (%)"].map(lambda x: f"{x:.2f}%")
        display_comparison["CV Std (%)"] = display_comparison["CV Std (%)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(display_comparison, width="stretch")

        fig = px.bar(
            comparison,
            x="Model",
            y="CV Mean (%)",
            text="CV Mean (%)",
            title="Rata-rata Akurasi Cross-Validation per Model",
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        render_plotly_chart(fig)

        st.info(
            f"Model yang dipakai untuk prediksi saat ini: **{metrics['best_model_name']}**. "
            "Jika dataset diperbarui, jalankan `python scripts/train_model.py` untuk memilih dan menyimpan model terbaik kembali."
        )

    with tab_report:
        st.dataframe(metrics["classification_report"], width="stretch")

    with tab_matrix:
        matrix_df = metrics["confusion_matrix"]
        st.dataframe(matrix_df, width="stretch")
        fig = px.imshow(
            matrix_df,
            text_auto=True,
            labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
            title="Confusion Matrix",
        )
        render_plotly_chart(fig)

    with tab_quality:
        st.subheader("Ringkasan Dataset")
        render_metric_cards(dataset_summary(data))
        st.subheader("Jumlah Data per Grade Setelah Duplikat Dihapus")
        st.dataframe(clean_data[TARGET_COLUMN].value_counts().rename_axis("Grade").reset_index(name="Jumlah"), width="stretch")
        st.subheader("Skor Cross-Validation per Fold untuk Model Terbaik")
        cv_df = pd.DataFrame(
            {
                "Fold": range(1, len(metrics["cv_scores"]) + 1),
                "Accuracy": metrics["cv_scores"],
                "Accuracy (%)": metrics["cv_scores"] * 100,
            }
        )
        st.dataframe(cv_df, width="stretch")


def render_about_page(data: pd.DataFrame, clean_data: pd.DataFrame) -> None:
    st.title("ℹ️ Tentang Dataset dan Program")
    st.markdown(
        """
        Aplikasi ini memprediksi kualitas susu menggunakan model machine learning terbaik berdasarkan perbandingan cross-validation.
        Data training dibersihkan dari duplikat sebelum digunakan untuk evaluasi dan pelatihan model final.

        ### Fitur Dataset
        | Fitur | Keterangan |
        |---|---|
        | pH | Tingkat keasaman susu |
        | Temperature | Suhu susu dalam derajat Celsius |
        | Taste | 0 = buruk, 1 = baik |
        | Odor | 0 = buruk, 1 = baik |
        | Fat | 0 = buruk, 1 = baik |
        | Turbidity | 0 = buruk, 1 = baik |
        | Colour | Nilai warna susu |
        | Grade | Label kualitas susu: high, medium, low |

        ### Batasan Aplikasi
        - Hasil prediksi tidak menggantikan uji laboratorium resmi.
        - Model bergantung pada kualitas dan cakupan dataset.
        - Jika input berada di luar rentang data training, prediksi bisa kurang akurat.
        - Riwayat prediksi disimpan terpisah dan **tidak otomatis** dimasukkan kembali ke dataset training.
        """
    )

    st.subheader("Ringkasan Data")
    render_metric_cards(dataset_summary(data))
    st.subheader("Rentang Fitur Data Bersih")
    st.dataframe(feature_ranges(clean_data), width="stretch")


def main() -> None:
    try:
        data = get_dataset()
        clean_data = get_clean_dataset(data)
    except Exception as exc:
        st.error(f"Aplikasi tidak dapat memuat dataset: {exc}")
        st.stop()

    selected = render_sidebar_menu()

    if selected == "Pengecekan Grade Susu":
        render_prediction_page(data, clean_data)
    elif selected == "Data Susu":
        render_data_page(data, clean_data)
    elif selected == "Riwayat Prediksi":
        render_history_page()
    elif selected == "Evaluasi Model":
        render_evaluation_page(data, clean_data)
    elif selected == "Tentang Dataset":
        render_about_page(data, clean_data)


if __name__ == "__main__":
    main()
