# Machine Learning Repository

Selamat datang di repositori pembelajaran **Machine Learning (ML)**. Repositori ini dirancang untuk mencakup seluruh dokumentasi mata kuliah Machine Learning dari praktikum pertemuan berjalan.

---

## Daftar Isi

1. [Siklus Data & Preparation](#1-siklus-data--preparation)
2. [Paradigma Pembelajaran](#2-paradigma-pembelajaran)
3. [Konsep Modeling & Evaluasi](#3-konsep-modeling--evaluasi)
4. [Bias, Variance, & Fitting](#4-bias-variance--fitting)
5. [Algoritma & Model](#5-algoritma--model)

---

## 1. Siklus Data & Preparation

Dalam Machine Learning, data melalui beberapa tahap sebelum siap digunakan oleh model:

### Data Lifecycle
- **Data Acquisition**: Proses pengumpulan data dari berbagai sumber (API, Database, CSV, Scrapping).
- **Data Understanding**: Memahami struktur, tipe data, dan karakteristik awal dari dataset.
- **Data Analysis (EDA)**: Menemukan pola, anomali, dan korelasi antar fitur menggunakan visualisasi.

### Data Preparation & Preprocessing
Tahap krusial untuk memastikan kualitas data:
- **Data Cleaning**: Menangani *missing values*, duplikasi, dan *outliers*.
- **Data Preprocessing**: Transformasi data mentah menjadi format yang lebih bersih.
- **Data Transformation**: Proses seperti *Scaling* dan *Normalization* agar fitur memiliki rentang nilai yang seragam.
- **Categorical Encoding**: Mengubah data teks/kategori menjadi angka.
    - **OneHotEncoding**: Membuat kolom baru untuk setiap kategori (representasi biner).
- **Feature Selection**: Memilih fitur yang paling relevan untuk meningkatkan efisiensi dan akurasi model.

---

## 2. Paradigma Pembelajaran

Repositori ini mencakup empat pilar utama metode pembelajaran:

| Paradigma                 | Penjelasan                                                                           |
| :------------------------ | :----------------------------------------------------------------------------------- |
| **Supervised Learning**   | Belajar dari data yang memiliki label (Target). Contoh: Klasifikasi & Regresi.       |
| **Unsupervised Learning** | Menemukan pola tersembunyi pada data tanpa label. Contoh: Clustering & PCA.          |
| **Semi-supervised**       | Kombinasi data berlabel dan tidak berlabel. Menggunakan teknik seperti **Pseudo Labelling**. |
| **Reinforcement**         | Pembelajaran berdasarkan sistem _reward_ dan _punishment_ dari interaksi lingkungan. |

---

## 3. Konsep Modeling & Evaluasi

### Prosedur Pengembangan Model

Untuk menghasilkan model yang handal, kita menggunakan pembagian data:

1. **Training Set**: Data yang digunakan untuk melatih algoritma.
2. **Validation Set**: Digunakan untuk tuning hyperparameter dan mencegah overfitting.
3. **Testing Set**: Data "buta" yang digunakan untuk mengevaluasi performa akhir model.

Selain pembagian statis, kita juga dapat menggunakan **K-Fold Cross Validation** untuk memastikan model stabil pada berbagai subset data.

### Konsep Modeling
Membangun model melibatkan pemilihan fitur, penentuan arsitektur algoritma, dan penentuan **Training Objectives** (apa yang ingin dioptimalkan oleh model).

### Performance Evaluation
Untuk mengukur keberhasilan model, kita menggunakan **Confusion Matrix** (TP, TN, FP, FN) dan metrik turunan seperti:
- **Accuracy**: Rasio prediksi benar dibandingkan total data.
- **Precision**: Kemampuan model untuk tidak memberikan label positif pada sampel yang sebenarnya negatif.
- **Recall**: Kemampuan model untuk menemukan semua sampel positif.
- **F1-Score**: Rata-rata harmonik antara Precision dan Recall (berguna jika data tidak seimbang).

---

## 4. Bias, Variance, & Fitting

Memahami trade-off dalam ML sangat krusial:

- **Bias**: Error karena asumsi yang terlalu sederhana (menyebabkan Underfitting).
- **Variance**: Sensitivitas berlebih terhadap fluktuasi kecil pada data training (menyebabkan Overfitting).

| Kondisi          | Penjelasan                                                       | Solusi                                      |
| :--------------- | :--------------------------------------------------------------- | :------------------------------------------ |
| **Underfitting** | Model terlalu sederhana, performa buruk di training & testing.   | Tambah fitur, gunakan model lebih kompleks. |
| **Overfitting**  | Model terlalu hafal data training, gagal melakukan generalisasi. | Regularisasi, kurangi fitur, tambah data.   |

---

## 5. Algoritma & Model

### Regresi (Linear & Logistik)
- **Linear Regression**: Memprediksi nilai kontinu berdasarkan hubungan linear antar variabel.
- **Logistic Regression**: Digunakan untuk klasifikasi biner dengan memetakan output ke probabilitas antara 0 dan 1.

### K-Nearest Neighbors (K-NN)
Algoritma berbasis instansi yang melakukan klasifikasi berdasarkan mayoritas label dari tetangga terdekat.

### Naïve Bayes Model
Model probabilitas berdasarkan Teorema Bayes dengan asumsi "independensi" antar fitur yang kuat.

### Decision Tree
Model prediksi berupa struktur pohon keputusan yang membagi data berdasarkan aturan logika tertentu.

### Ensemble Learning
Teknik menggabungkan beberapa model (seperti Random Forest atau Gradient Boosting) untuk meningkatkan akurasi dan stabilitas prediksi dibandingkan model tunggal.

---

## Persyaratan Sistem

- Python 3.8+
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn

---

## Informasi Tambahan

| Informasi          | Keterangan                    |
| ------------------ | ----------------------------- |
| **Dosen Pengampu** | ABU SALAM, M.Kom              |
| **Mata Kuliah**    | Pembelajaran Mesin            |
| **Nama**           | Alfaturachman Maulana Pahlevi |
| **NIM**            | A11.2025.16609                |
| **Kelas**          | A11.4405                      |

---
_Dibuat untuk keperluan akademik - Machine Learning_
