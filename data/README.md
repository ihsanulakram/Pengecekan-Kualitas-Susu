# Dataset Kualitas Susu

File utama dataset berada di:

```text
data/milkdata.csv
```

Dataset ini digunakan untuk melatih dan mengevaluasi model klasifikasi kualitas susu.

## Ringkasan Dataset

| Item | Nilai |
|---|---:|
| Jumlah baris asli | 1.059 |
| Jumlah kolom | 8 |
| Jumlah baris duplikat | 976 |
| Jumlah baris unik setelah deduplikasi | 83 |
| Missing value | 0 |

## Distribusi Label

| Grade | Jumlah |
|---|---:|
| low | 429 |
| medium | 374 |
| high | 256 |

## Fitur Dataset

| Kolom | Keterangan |
|---|---|
| `pH` | Tingkat keasaman susu |
| `Temperature` | Suhu susu dalam derajat Celsius |
| `Taste` | 0 = buruk, 1 = baik |
| `Odor` | 0 = buruk, 1 = baik |
| `Fat` | 0 = buruk, 1 = baik |
| `Turbidity` | 0 = buruk, 1 = baik |
| `Colour` | Nilai warna susu |
| `Grade` | Label kualitas susu: `high`, `medium`, atau `low` |

## Kebijakan Data Prediksi

Hasil prediksi pengguna disimpan di folder:

```text
data/prediksi/
```

File prediksi sengaja **tidak** otomatis digabungkan ke `milkdata.csv`, karena label prediksi berasal dari model, bukan dari pemeriksaan manual atau uji laboratorium. Data prediksi hanya boleh dimasukkan ke dataset training jika sudah diverifikasi.

## Catatan Kualitas Data

Dataset asli memiliki duplikat yang sangat banyak. Karena itu, proses training dan evaluasi model memakai data yang sudah dibersihkan dari duplikat agar hasil evaluasi tidak terlalu optimistis.
