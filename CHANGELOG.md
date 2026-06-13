# Changelog

## v2.0.2 - Warning-free runtime polish
- Avoid loading persisted scikit-learn estimators when artifact/runtime versions differ; the app retrains from CSV first to prevent `InconsistentVersionWarning`.
- Replaced Plotly chart width kwargs with `config={...}` based rendering to avoid Streamlit Plotly kwargs deprecation warnings.

## v2.0.1 - Streamlit API Compatibility
- Replaced deprecated `use_container_width` parameters with `width="stretch"`.
- Updated minimum Streamlit version to 1.51.0 for flex-layout width compatibility.

## v2.1.0 - Final Polish

- Menjadikan `app.py` sebagai entry point utama aplikasi.
- Menjaga `kualitas_susu.py` sebagai wrapper kompatibilitas lama.
- Memperbarui Dockerfile dan devcontainer agar menjalankan `app.py`.
- Memperbarui `CODEOWNERS` menjadi `@ihsanulakram`.
- Menambahkan `data/README.md` untuk dokumentasi dataset.
- Menambahkan `pyproject.toml` untuk metadata proyek dan konfigurasi pytest.
- Menambahkan `NOTICE` dan memperbarui placeholder copyright di `LICENSE`.
- Menambahkan test untuk memastikan model artifacts tersedia.
- Memperbarui GitHub Actions agar melakukan training ulang sebelum test.
- Merapikan nama folder distribusi menjadi lowercase-kebab-case.

## v2.0.0 - 10/10 Final

- Menambahkan perbandingan 5 model machine learning.
- Menambahkan pemilihan model terbaik otomatis berdasarkan cross-validation.
- Menambahkan confidence warning jika keyakinan model rendah.
- Menambahkan Dockerfile, `.dockerignore`, dan GitHub Actions.
- Menambahkan `MODEL_CARD.md`.
- Menambahkan preview aplikasi di folder `assets/`.
- Memisahkan `requirements.txt` dan `requirements-dev.txt`.
- Memperbarui README secara lengkap.

## v1.0.0 - Initial Improved Version

- Memperbaiki evaluasi model agar memakai train/test split.
- Menambahkan classification report dan confusion matrix yang valid.
- Memisahkan riwayat prediksi dari dataset training.
- Menambahkan validasi input dan rekomendasi hasil prediksi.
- Membersihkan duplikat dataset untuk training dan evaluasi.
- Memperbaiki dependency dan struktur proyek.
