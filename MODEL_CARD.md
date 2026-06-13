# Model Card - Pengecekan Kualitas Susu

## Ringkasan Model

Model digunakan untuk mengklasifikasikan kualitas susu menjadi tiga kelas: `high`, `medium`, dan `low`.

Versi saat ini tidak hanya memakai satu algoritma secara statis. Aplikasi membandingkan beberapa kandidat model dengan stratified cross-validation, lalu menyimpan model dengan rata-rata akurasi terbaik.

## Model Terpilih Saat Ini

- **Model:** Gaussian Naive Bayes + StandardScaler
- **Metode pemilihan:** rata-rata akurasi tertinggi dari stratified cross-validation
- **Data training setelah duplikat dihapus:** 83 baris
- **Holdout accuracy:** 82.35%
- **Cross-validation mean:** 85.59%
- **Cross-validation std:** 11.47%

## Kandidat Model yang Dibandingkan

| Rank | Model | CV Mean | CV Std |
|---:|---|---:|---:|
| 1 | Gaussian Naive Bayes | 85.59% | 11.47% |
| 2 | Random Forest | 84.34% | 7.99% |
| 3 | Decision Tree | 80.74% | 8.20% |
| 4 | Logistic Regression | 74.49% | 7.80% |
| 5 | K-Nearest Neighbors | 68.60% | 11.59% |

## Input Model

| Fitur | Keterangan |
|---|---|
| `pH` | Tingkat keasaman susu |
| `Temperature` | Suhu susu dalam derajat Celsius |
| `Taste` | 0 = buruk, 1 = baik |
| `Odor` | 0 = buruk, 1 = baik |
| `Fat` | 0 = buruk, 1 = baik |
| `Turbidity` | 0 = buruk, 1 = baik |
| `Colour` | Nilai warna susu |

## Output Model

Model menghasilkan:

- prediksi kelas grade susu,
- tingkat keyakinan prediksi,
- probabilitas untuk setiap kelas.

Jika tingkat keyakinan model kurang dari 70%, aplikasi akan menampilkan peringatan agar pengguna melakukan pemeriksaan tambahan.

## Data dan Evaluasi

Dataset asli memiliki banyak duplikat. Karena itu, training dan evaluasi dilakukan pada data yang sudah dibersihkan dari duplikat agar skor tidak terlalu optimistis.

Evaluasi yang digunakan:

- stratified train/test split,
- stratified cross-validation,
- classification report,
- confusion matrix,
- perbandingan beberapa algoritma.

## Penggunaan yang Disarankan

Model ini cocok untuk:

- demo aplikasi klasifikasi kualitas susu,
- prototipe sistem pendukung keputusan,
- pembelajaran Streamlit dan machine learning klasifikasi.

## Batasan

- Dataset relatif kecil setelah duplikat dihapus.
- Model hanya belajar dari fitur yang tersedia di dataset.
- Prediksi bisa kurang akurat jika input berada di luar rentang data training.
- Hasil prediksi tidak menggantikan uji laboratorium resmi.
- Hasil prediksi tidak otomatis dimasukkan ke dataset training agar tidak mencemari data asli.

## Catatan Keamanan

Aplikasi ini tidak boleh menjadi satu-satunya dasar keputusan konsumsi, distribusi, atau produksi pangan. Keputusan akhir tetap harus mengikuti pemeriksaan laboratorium, standar keamanan pangan, dan prosedur operasional yang berlaku.
