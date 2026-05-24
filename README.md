# Machine Learning Repository

Selamat datang di repositori pembelajaran **Machine Learning (Pembelajaran Mesin)**. Repositori ini dirancang untuk mencakup seluruh dokumentasi mata kuliah Machine Learning dari praktikum pertemuan berjalan.

---

## Daftar Isi

- [Struktur Repositori](#struktur-repositori)
- [Cara Penggunaan](#cara-penggunaan)
- [Notebooks](#notebooks)
- [Paradigma Pembelajaran](#paradigma-pembelajaran)
- [Siklus Data & Preparation](#siklus-data--preparation)
- [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
- [Algoritma & Model](#algoritma--model)
- [Prosedur Pengembangan Model](#prosedur-pengembangan-model)
- [Bias, Variance & Fitting](#bias-variance--fitting)
- [Unsupervised Learning](#unsupervised-learning)
- [Feature Engineering](#feature-engineering)
- [Error Analysis](#error-analysis)
- [Ringkasan Performa](#ringkasan-performa)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Informasi Tambahan](#informasi-tambahan)

---

## Struktur Repositori

```
REPO/
├── data/                           # Dataset terpusat
│   ├── Social_Network_Ads.csv
│   ├── Dataset Iris.csv
│   ├── SMSSpam.csv
│   ├── heart_disease_data.csv
│   └── dataset_kelulusan_mahasiswa.csv
├── notebooks/                      # Notebook pembelajaran
│   ├── 01-Linear-Regression/       # Linear Regression - SMS Spam
│   ├── 02-Logistic-Regression/     # Logistic Regression - Heart Disease
│   ├── 03-K-Nearest-Neighbors/     # K-NN - Social Network Ads
│   ├── 04-Naive-Bayes/             # Naive Bayes - Social & SMS
│   ├── 05-Decision-Tree/           # Decision Tree - Iris Dataset
│   ├── 06-Random-Forest/           # Random Forest - Kelulusan
│   ├── 07-Unsupervised-Learning/   # K-Means + PCA - Iris
│   ├── 08-Feature-Engineering/     # Polynomial, Interaction, Binning
│   ├── 09-Cross-Validation/        # K-Fold CV - Heart Disease
│   └── 10-Error-Analysis/          # FP/FN Case Study - Heart
├── utils/                          # Modul Python reusable
│   ├── __init__.py
│   ├── evaluation.py               # Confusion matrix, ROC, dll
│   └── preprocessing.py            # Split, encoding, scaling
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Cara Penggunaan

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Jalankan Jupyter:

    ```bash
    jupyter notebook
    ```

3. Buka notebook di `notebooks/` dan jalankan sel-selnya.

---

## Notebooks

| #   | Algoritma                 | Dataset             | Topik                                               |
| --- | ------------------------- | ------------------- | --------------------------------------------------- |
| 1   | Linear Regression         | SMS Spam            | Klasifikasi teks dengan regresi linear              |
| 2   | Logistic Regression       | Heart Disease       | Prediksi penyakit jantung                           |
| 3   | K-Nearest Neighbors       | Social Network Ads  | Klasifikasi pembelian produk                        |
| 4   | Naive Bayes (Gaussian)    | Social Network Ads  | Klasifikasi pembelian produk                        |
| 5   | Naive Bayes (Multinomial) | SMS Spam            | Deteksi spam dengan TF-IDF                          |
| 6   | Decision Tree             | Iris (built-in)     | Klasifikasi 3 spesies Iris                          |
| 7   | Decision Tree             | Iris (CSV)          | Klasifikasi biner Iris                              |
| 8   | Random Forest             | Kelulusan Mahasiswa | Prediksi kelulusan                                  |
| 9   | K-Means + PCA             | Iris (built-in)     | Unsupervised Learning: clustering & reduksi dimensi |
| 10  | Feature Engineering       | Social Network Ads  | Polynomial features, interaction, binning           |
| 11  | K-Fold Cross Validation   | Heart Disease       | Perbandingan stabilitas antar model                 |
| 12  | Error Analysis            | Heart Disease       | False Positive & False Negative case study          |

---

## Paradigma Pembelajaran

Repositori ini mencakup empat pilar utama metode pembelajaran:

| Paradigma                 | Penjelasan                                                                                   |
| :------------------------ | :------------------------------------------------------------------------------------------- |
| **Supervised Learning**   | Belajar dari data yang memiliki label (Target). Contoh: Klasifikasi & Regresi.               |
| **Unsupervised Learning** | Menemukan pola tersembunyi pada data tanpa label. Contoh: Clustering & PCA.                  |
| **Semi-supervised**       | Kombinasi data berlabel dan tidak berlabel. Menggunakan teknik seperti **Pseudo Labelling**. |
| **Reinforcement**         | Pembelajaran berdasarkan sistem _reward_ dan _punishment_ dari interaksi lingkungan.         |

---

## Siklus Data & Preparation

### Data Lifecycle

- **Data Acquisition**: Proses pengumpulan data dari berbagai sumber (API, Database, CSV, Scrapping).
- **Data Understanding**: Memahami struktur, tipe data, dan karakteristik awal dari dataset.
- **Data Analysis (EDA)**: Menemukan pola, anomali, dan korelasi antar fitur menggunakan visualisasi.

### Data Preparation & Preprocessing

- **Data Cleaning**: Menangani _missing values_, duplikasi, dan _outliers_.
- **Data Preprocessing**: Transformasi data mentah menjadi format yang lebih bersih.
- **Data Transformation**: Proses seperti _Scaling_ dan _Normalization_ agar fitur memiliki rentang nilai yang seragam.
- **Categorical Encoding**: Mengubah data teks/kategori menjadi angka.
    - **OneHotEncoding**: Membuat kolom baru untuk setiap kategori (representasi biner).
- **Feature Selection**: Memilih fitur yang paling relevan untuk meningkatkan efisiensi dan akurasi model.

---

## EDA (Exploratory Data Analysis)

Setiap notebook melakukan eksplorasi data sebelum modeling dengan pola umum berikut:

| Analisis               | Tujuan                                                     | Visualisasi                    |
| ---------------------- | ---------------------------------------------------------- | ------------------------------ |
| **Distribusi Target**  | Cek keseimbangan kelas (imbalance)                         | Countplot, Pie chart           |
| **Distribusi Fitur**   | Pahami rentang, sebaran, dan outlier tiap fitur            | Histogram + KDE, Boxplot       |
| **Korelasi Fitur**     | Deteksi multikolinearitas dan hubungan fitur dengan target | Heatmap korelasi               |
| **Analisis per Kelas** | Bandingkan profil fitur antar kelas target                 | Histplot dengan hue, Pairplot  |
| **Feature Importance** | Urutkan fitur berdasarkan pengaruh terhadap prediksi       | Bar chart koefisien / impurity |

### Temuan EDA

| Dataset                | Temuan                                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Social Network Ads** | Fitur `Age` dan `EstimatedSalary` memiliki skala sangat berbeda => perlu **StandardScaler** sebelum K-NN/Naive Bayes |
| **SMS Spam**           | Distribusi panjang pesan spam vs ham berbeda signifikan (spam cenderung lebih panjang)                               |
| **Heart Disease**      | Tidak ada missing values; fitur `cp` (chest pain type) dan `thalach` berkorelasi kuat dengan target                  |
| **Iris**               | Fitur `PetalLengthCm` dan `PetalWidthCm` paling diskriminatif antar spesies => cocok untuk PCA & clustering          |
| **Kelulusan**          | Dataset seimbang (50:50); `IPS Rata-rata` dan `IPS Tren` paling berpengaruh                                          |
| **Cross-Validation**   | Random Forest paling stabil (variance rendah) dibanding Logistic Regression & K-NN                                   |
| **Error Analysis**     | FN lebih berbahaya daripada FP dalam konteks medis — model cenderung _under-confidence_ pada kasus borderline        |

---

## Algoritma & Model

### Regresi (Linear & Logistik)

- **Linear Regression**: Memprediksi nilai kontinu berdasarkan hubungan linear antar variabel.
- **Logistic Regression**: Digunakan untuk klasifikasi biner dengan memetakan output ke probabilitas antara 0 dan 1.

### K-Nearest Neighbors (K-NN)

Algoritma berbasis instansi yang melakukan klasifikasi berdasarkan mayoritas label dari tetangga terdekat.

### Naive Bayes Model

Model probabilitas berdasarkan Teorema Bayes dengan asumsi "independensi" antar fitur yang kuat.

### Decision Tree

Model prediksi berupa struktur pohon keputusan yang membagi data berdasarkan aturan logika tertentu.

### Ensemble Learning

Teknik menggabungkan beberapa model (seperti Random Forest atau Gradient Boosting) untuk meningkatkan akurasi dan stabilitas prediksi dibandingkan model tunggal.

---

## Prosedur Pengembangan Model

### Pembagian Data

1. **Training Set**: Data yang digunakan untuk melatih algoritma.
2. **Validation Set**: Digunakan untuk tuning hyperparameter dan mencegah overfitting.
3. **Testing Set**: Data "buta" yang digunakan untuk mengevaluasi performa akhir model.

Selain pembagian statis, **K-Fold Cross Validation** digunakan untuk memastikan model stabil pada berbagai subset data.

### Pipeline Modeling

Setiap notebook mengikuti pipeline yang konsisten:

#### 1. Data Preparation

- **Handling Missing Values**: Diisi dengan median untuk fitur numerik
- **Categorical Encoding**: LabelEncoder (ordinal) atau OneHotEncoding
- **Feature Scaling**: `StandardScaler` untuk algoritma berbasis jarak (K-NN, Logistic Regression)
- **Text Feature Extraction**: `TfidfVectorizer` dengan `stop_words='english'` dan `max_features=3000` untuk SMS

#### 2. Train/Test Split

- **Split ratio**: 80/20 (Iris, K-NN, Heart) atau 60/20/20 (SMS — train/val/test)
- **Stratifikasi**: `stratify=y` untuk menjaga proporsi kelas di setiap split

#### 3. Model Training

| Notebook | Algoritma                | Parameter Kunci                                         |
| -------- | ------------------------ | ------------------------------------------------------- |
| 01       | `LinearRegression`       | Threshold optimal via ROC                               |
| 02       | `LogisticRegression`     | `C=0.01`, `solver='liblinear'`, tuning via GridSearchCV |
| 03       | `KNeighborsClassifier`   | `n_neighbors=3`, `metric='manhattan'`                   |
| 04       | `GaussianNB`             | Default                                                 |
| 05       | `MultinomialNB`          | Default                                                 |
| 06       | `DecisionTreeClassifier` | `random_state=0`                                        |
| 07       | `DecisionTreeClassifier` | `random_state=0`                                        |
| 08       | `RandomForestClassifier` | `n_estimators=100`, `random_state=42`                   |
| 09       | `KMeans` + `PCA`         | `n_clusters=3`, `n_components=2`                        |
| 10       | `LogisticRegression`     | Feature engineering + scaling                           |
| 11       | Multiple models          | K-Fold & StratifiedKFold (k=5,10)                       |
| 12       | `LogisticRegression`     | Error analysis & confidence check                       |

#### 4. Cross Validation

- **K-Fold CV** (k=5, 10) dan **StratifiedKFold** diterapkan pada Logistic Regression, K-NN, Decision Tree, dan Random Forest
- Evaluasi stabilitas model: rata-rata accuracy ± std dev
- Boxplot perbandingan skor CV antar model untuk memilih model paling stabil

#### 5. Hyperparameter Tuning

- **Logistic Regression**: GridSearchCV pada `C` (0.01–100) dengan `StratifiedKFold` (5-fold)
- **Random Forest**: `n_estimators=100` (default yang cukup stabil)

#### 6. Evaluation Metrics

| Metrik        | Rumus                 | Kegunaan                              |
| ------------- | --------------------- | ------------------------------------- |
| **Accuracy**  | (TP+TN)/(TP+TN+FP+FN) | Akurasi keseluruhan                   |
| **Precision** | TP/(TP+FP)            | Ketepatan prediksi positif            |
| **Recall**    | TP/(TP+FN)            | Kemampuan menangkap semua positif     |
| **F1-Score**  | 2 x P x R / (P+R)     | Rata-rata harmonik precision & recall |
| **AUC-ROC**   | Area under curve      | Kemampuan diskriminasi kelas          |

Visualisasi evaluasi: **Confusion Matrix** (heatmap), **ROC Curve**, dan **Feature Importance** (bar chart).

### Performance Evaluation

Untuk mengukur keberhasilan model, digunakan **Confusion Matrix** (TP, TN, FP, FN) dan metrik turunan:

- **Accuracy**: Rasio prediksi benar dibandingkan total data.
- **Precision**: Kemampuan model untuk tidak memberikan label positif pada sampel yang sebenarnya negatif.
- **Recall**: Kemampuan model untuk menemukan semua sampel positif.
- **F1-Score**: Rata-rata harmonik antara Precision dan Recall (berguna jika data tidak seimbang).

---

## Bias, Variance & Fitting

Memahami trade-off dalam ML sangat krusial:

- **Bias**: Error karena asumsi yang terlalu sederhana (menyebabkan Underfitting).
- **Variance**: Sensitivitas berlebih terhadap fluktuasi kecil pada data training (menyebabkan Overfitting).

| Kondisi          | Penjelasan                                                      | Solusi                                     |
| :--------------- | :-------------------------------------------------------------- | :----------------------------------------- |
| **Underfitting** | Model terlalu sederhana, performa buruk di training & testing   | Tambah fitur, gunakan model lebih kompleks |
| **Overfitting**  | Model terlalu hafal data training, gagal melakukan generalisasi | Regularisasi, kurangi fitur, tambah data   |

---

## Unsupervised Learning

### K-Means Clustering

- Algoritma clustering yang mempartisi data menjadi **K cluster** berdasarkan jarak Euclidean ke centroid
- **Elbow Method** untuk menentukan K optimal (Sum of Squared Distances)
- Evaluasi dengan **Silhouette Score** (seberapa mirip suatu titik dengan clusternya vs cluster lain)
- **Adjusted Rand Index (ARI)** untuk membandingkan label cluster dengan ground truth

### PCA (Principal Component Analysis)

- Teknik reduksi dimensi **linear** yang memproyeksikan data ke komponen utama (variance terbesar)
- Digunakan untuk visualisasi data dimensi tinggi ke 2D/3D
- Menunjukkan bahwa Iris dataset terpisah secara alami bahkan tanpa label

---

## Feature Engineering

Teknik yang dieksplorasi:

| Teknik                  | Deskripsi                                                           | Dampak                                    |
| ----------------------- | ------------------------------------------------------------------- | ----------------------------------------- |
| **Polynomial Features** | Kombinasi fitur eksisting pangkat 2 (Age^2, Salary^2, Age x Salary) | Menangkap hubungan non-linear             |
| **Interaction Terms**   | Produk antar fitur (Age x Salary)                                   | Model memahami ketergantungan antar fitur |
| **Binning**             | Ubah Age kontinu => kategori (Young/Middle/Old)                     | Robust terhadap outlier                   |
| **Log Transform**       | log(EstimatedSalary)                                                | Normalisasi distribusi right-skewed       |

Hasil: Accuracy meningkat ~1-2% setelah feature engineering.

---

## Error Analysis

Analisis mendalam terhadap **kesalahan prediksi** model Logistic Regression:

1. **False Positives (FP)**: Pasien sehat tapi diprediksi sakit
    - Ciri: nilai `thalach` rendah, `oldpeak` tinggi — mirip pola sakit
    - Probabilitas model: cukup yakin (0.6-0.8)

2. **False Negatives (FN)**: Pasien sakit tapi diprediksi sehat
    - Ciri: semua fitur di ambang batas normal, sulit dibedakan
    - Probabilitas model: borderline (0.4-0.6) — model kurang yakin

3. **Rekomendasi**:
    - Tambah data pada subgroup dengan error tinggi
    - Threshold tuning untuk mengurangi FN (prioritas diagnosis medis)
    - Feature engineering pada interaksi fitur kritis

---

## Ringkasan Performa

| Notebook                   | Model                    | Akurasi | AUC-ROC | Catatan                                          |
| -------------------------- | ------------------------ | ------- | ------- | ------------------------------------------------ |
| Linear Regression SMS      | `LinearRegression`       | 0.88    | —       | Threshold optimal via validation set             |
| Logistic Regression Heart  | `LogisticRegression`     | 0.79    | 0.88    | Hyperparameter tuning via GridSearchCV           |
| K-NN Social                | `KNeighborsClassifier`   | 0.93    | —       | k=3, Manhattan distance                          |
| Naive Bayes Social         | `GaussianNB`             | 0.90    | —       | Decision boundary linear                         |
| Naive Bayes SMS            | `MultinomialNB`          | —       | —       | Perbandingan dengan Linear Regression            |
| Decision Tree Iris         | `DecisionTreeClassifier` | 0.97    | —       | Visualisasi pohon keputusan                      |
| Decision Tree Iris CSV     | `DecisionTreeClassifier` | 1.00    | —       | Binary classification (2 kelas)                  |
| Random Forest Kelulusan    | `RandomForestClassifier` | 0.50    | 0.50    | Dataset perlu investigasi lebih lanjut           |
| K-Means + PCA Iris         | `KMeans` + `PCA`         | —       | —       | ARI & Silhouette Score untuk evaluasi clustering |
| Feature Engineering Social | `LogisticRegression`     | 0.91    | —       | PolynomialFeatures + Interaction + Binning       |
| K-Fold CV Heart            | Multiple models          | —       | —       | Boxplot stabilitas model, RF paling stabil       |
| Error Analysis Heart       | `LogisticRegression`     | 0.79    | 0.88    | Deep dive FP/FN dengan confidence analysis       |

---

## Persyaratan Sistem

- Python 3.8+
- Library: lihat `requirements.txt`

---

## Informasi Tambahan

| Informasi          | Keterangan                    |
| ------------------ | ----------------------------- |
| **Dosen Pengampu** | ABU SALAM, M.Kom              |
| **Mata Kuliah**    | Pembelajaran Mesin            |
| **Nama**           | Alfaturachman Maulana Pahlevi |
| **NIM**            | A11.2025.16609                |
| **Kelas**          | A11.4405                      |
