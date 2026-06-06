# Machine Learning Repository

Selamat datang di repositori pembelajaran **Machine Learning (Pembelajaran Mesin)**. Repositori ini dirancang untuk mencakup seluruh dokumentasi mata kuliah Machine Learning dari praktikum pertemuan berjalan.

---

## Daftar Isi

- [Notebooks](#notebooks)
- [Paradigma Pembelajaran](#paradigma-pembelajaran)
- [Siklus Data & Preparation](#siklus-data--preparation)
- [EDA (Exploratory Data Analysis)](#eda-exploratory-data-analysis)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Artificial Neural Network](#artificial-neural-network)
- [Prosedur Pengembangan Model](#prosedur-pengembangan-model)
- [Bias, Variance & Fitting](#bias-variance--fitting)
- [Feature Engineering](#feature-engineering)
- [Error Analysis](#error-analysis)
- [Model Deployment](#model-deployment)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Informasi Tambahan](#informasi-tambahan)

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
| 13  | SVM                       | Iris (built-in)     | Support Vector Machine dengan 3 kernel              |
| 14  | Gradient Boosting         | Heart Disease       | AdaBoost & GradientBoosting                         |
| 15  | MLP Neural Network        | Medical Conditions  | Multi-Layer Perceptron dengan tuning                |
| 16  | Regularization            | Social Network Ads  | Lasso, Ridge, ElasticNet                            |
| 17  | Anomaly Detection         | Transaction Fraud   | Isolation Forest & Local Outlier Factor             |
| 18  | Model Deployment          | Iris (built-in)     | Save/load model & FastAPI REST API                  |
| 19  | Recommendation System     | E-commerce Behavior | Content-Based Filtering & Cosine Similarity         |

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

- **Data Acquisition**: Proses pengumpulan data dari berbagai sumber (API, Database, CSV, Scraping).
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

## Supervised Learning

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

### SVM (Support Vector Machine)

SVM mencari hyperplane optimal yang memisahkan kelas dengan margin maksimal.

**Dataset**: Iris (built-in) | **Notebook**: `11-SVM/SVM-Iris.ipynb`

| Kernel     | Kegunaan                          | Kelebihan                       |
| :--------- | :-------------------------------- | :------------------------------ |
| Linear     | Data terpisah secara linear       | Cepat, interpretable            |
| RBF        | Data non-linear, universal kernel | Fleksibel, cocok berbagai kasus |
| Polynomial | Data dengan pola polynomial       | Menangkap interaksi kompleks    |

- Hyperparameter `C` mengontrol trade-off margin vs misclassification
- `gamma` mengontrol radius pengaruh support vector
- GridSearchCV digunakan untuk menemukan kombinasi optimal

### Gradient Boosting

Teknik ensemble sekuensial yang memperbaiki kesalahan model sebelumnya.

**Dataset**: Heart Disease | **Notebook**: `12-Gradient-Boosting/Gradient-Boosting-Heart.ipynb`

| Algoritma             | Konsep Dasar                                         | Parameter Kunci                   |
| :-------------------- | :--------------------------------------------------- | :-------------------------------- |
| **AdaBoost**          | Memberi bobot lebih pada data yang salah klasifikasi | `n_estimators`, `learning_rate`   |
| **Gradient Boosting** | Meminimalkan loss function dengan gradient descent   | `n_estimators`, `max_depth`, `lr` |

- Lebih akurat dibanding model tunggal (decision tree)
- Rentan overfitting jika `n_estimators` terlalu tinggi

### Regularization

Teknik mencegah overfitting dengan menambahkan penalty pada fungsi loss.

**Dataset**: Social Network Ads | **Notebook**: `14-Regularization/Regularization-Social.ipynb`

| Teknik         | Penalty | Efek                                            |
| :------------- | :------ | :---------------------------------------------- |
| **Ridge (L2)** | Σ(w^2)  | Menurunkan koefisien secara proporsional        |
| **Lasso (L1)** | Σ\|w\|  | Membuat sebagian koefisien menjadi nol (sparse) |
| **ElasticNet** | L1 + L2 | Kombinasi sparsity dan distribusi bobot         |

- Parameter `C` (inverse regularization strength): semakin kecil → regularisasi semakin kuat
- Lasso berguna untuk feature selection otomatis

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

| Notebook | Algoritma                      | Parameter Kunci                                         |
| -------- | ------------------------------ | ------------------------------------------------------- |
| 01       | `LinearRegression`             | Threshold optimal via ROC                               |
| 02       | `LogisticRegression`           | `C=0.01`, `solver='liblinear'`, tuning via GridSearchCV |
| 03       | `KNeighborsClassifier`         | `n_neighbors=3`, `metric='manhattan'`                   |
| 04       | `GaussianNB`                   | Default                                                 |
| 05       | `MultinomialNB`                | Default                                                 |
| 06       | `DecisionTreeClassifier`       | `random_state=0`                                        |
| 07       | `DecisionTreeClassifier`       | `random_state=0`                                        |
| 08       | `RandomForestClassifier`       | `n_estimators=100`, `random_state=42`                   |
| 09       | `KMeans` + `PCA`               | `n_clusters=3`, `n_components=2`                        |
| 10       | `LogisticRegression`           | Feature engineering + scaling                           |
| 11       | Multiple models                | K-Fold & StratifiedKFold (k=5,10)                       |
| 12       | `LogisticRegression`           | Error analysis & confidence check                       |
| 13       | `SVC`                          | Linear, RBF, Polynomial kernels + GridSearchCV          |
| 14       | `AdaBoost`, `GradientBoosting` | `n_estimators=100`, `learning_rate=0.1`                 |
| 15       | `MLPClassifier`                | `(50,)`, `(100,)`, `(50,25)` + tuning activation/lr     |
| 16       | `LogisticRegression`           | `penalty='l1'/'l2'/'elasticnet'`, tuning C              |
| 17       | `IsolationForest`, `LOF`       | `contamination=0.1`, `n_neighbors=20`                   |
| 18       | `RandomForestClassifier`       | Save with joblib & pickle, load & predict               |
| 19       | Cosine Similarity, K-NN        | Content-Based Filtering, NearestNeighbors               |

#### 4. Cross Validation

- **K-Fold CV** (k=5, 10) dan **StratifiedKFold** diterapkan pada Logistic Regression, K-NN, Decision Tree, dan Random Forest
- Evaluasi stabilitas model: rata-rata accuracy ± std dev
- Boxplot perbandingan skor CV antar model untuk memilih model paling stabil

#### 5. Hyperparameter Tuning

- **Logistic Regression**: GridSearchCV pada `C` (0.01–100) dengan `StratifiedKFold` (5-fold)
- **Random Forest**: `n_estimators=100` (default yang cukup stabil)
- **Grid Search vs Random Search**: `GridSearchCV` melakukan pencarian komprehensif pada kombinasi parameter tertentu, sedangkan `RandomizedSearchCV` memilih kombinasi acak untuk efisiensi komputasi pada ruang hyperparameter yang luas.

#### 6. Evaluation Metrics

##### Metrik Klasifikasi

| Metrik        | Rumus                 | Kegunaan                              |
| ------------- | --------------------- | ------------------------------------- |
| **Accuracy**  | (TP+TN)/(TP+TN+FP+FN) | Akurasi keseluruhan                   |
| **Precision** | TP/(TP+FP)            | Ketepatan prediksi positif            |
| **Recall**    | TP/(TP+FN)            | Kemampuan menangkap semua positif     |
| **F1-Score**  | 2 x P x R / (P+R)     | Rata-rata harmonik precision & recall |
| **AUC-ROC**   | Area under curve      | Kemampuan diskriminasi kelas          |

##### Metrik Regresi

| Metrik                        | Rumus / Penjelasan                                            | Kegunaan                                                               |
| ----------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **MSE** (Mean Squared Error)  | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$                | Mengukur rata-rata kuadrat kesalahan (sensitif terhadap outlier)       |
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum_{i=1}^{n}\vert y_i - \hat{y}_i\vert$         | Mengukur rata-rata absolut kesalahan (lebih robust terhadap outlier)   |
| **R-squared** ($R^2$)         | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y}_i)^2}$ | Mengukur persentase varians target yang berhasil dijelaskan oleh model |

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

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

- Algoritma clustering berbasis kepadatan untuk mengidentifikasi cluster dengan bentuk arbitrer (tidak harus bulat)
- Menangani noise/outliers secara alami (titik di area dengan densitas rendah dianggap sebagai noise)
- Parameter kunci: `eps` (radius tetangga maksimal) dan `min_samples` (jumlah minimal titik tetangga dalam radius eps)

### Hierarchical Clustering (Agglomerative)

- Membangun klaster bertingkat secara bertahap menggunakan pendekatan _bottom-up_ (menggabungkan titik terdekat)
- Hubungan antar cluster ditentukan oleh kriteria _linkage_ (Single, Complete, Average, atau Ward)
- Visualisasi hubungan klaster menggunakan diagram pohon bernama **Dendrogram**

### Gaussian Mixture Models (GMM)

- Pendekatan klasterisasi probabilistik (_soft clustering_) berasumsi data dibangkitkan dari campuran beberapa distribusi Gaussian
- Menggunakan algoritma **Expectation-Maximization (EM)** untuk mengestimasi parameter model
- Memberikan probabilitas keanggotaan suatu titik di setiap klaster, bukan sekadar label kaku (_hard assignment_)

### PCA (Principal Component Analysis)

- Teknik reduksi dimensi **linear** yang memproyeksikan data ke komponen utama (variance terbesar)
- Digunakan untuk visualisasi data dimensi tinggi ke 2D/3D
- Menunjukkan bahwa Iris dataset terpisah secara alami bahkan tanpa label

### t-SNE (t-Distributed Stochastic Neighbor Embedding)

- Teknik reduksi dimensi **non-linear** yang dikhususkan untuk visualisasi data dimensi tinggi ke ruang 2D/3D
- Menjaga kemiripan titik-titik bertetangga dekat (local structure) lebih baik dibandingkan PCA pada data kompleks

### Association Rule Learning (Aturan Asosiasi)

- Metode untuk menemukan hubungan asosiatif menarik antar item dalam transaksi besar (Market Basket Analysis)
- Algoritma populer: **Apriori** (berbasis threshold minimum support) dan **FP-Growth** (menggunakan struktur pohon untuk efisiensi)
- Metrik kunci: **Support** (popularitas item), **Confidence** (kepastian aturan), dan **Lift** (kekuatan aturan asosiasi)

### Anomaly Detection

Mengidentifikasi pola langka/outlier yang menyimpang dari mayoritas data.

**Dataset**: Transaction Fraud | **Notebook**: `15-Anomaly-Detection/Anomaly-Detection-Transactions.ipynb`

| Algoritma                      | Konsep                                             | Parameter Kunci                |
| :----------------------------- | :------------------------------------------------- | :----------------------------- |
| **Isolation Forest**           | Mengisolasi outlier dengan random forest partition | `contamination`                |
| **Local Outlier Factor (LOF)** | Membandingkan density lokal antar titik            | `n_neighbors`, `contamination` |

- Unsupervised — tidak memerlukan label untuk training
- Parameter `contamination` harus disesuaikan dengan ekspektasi proporsi anomali

### Recommendation System

Sistem rekomendasi yang menyarankan item berdasarkan kemiripan profil.

**Dataset**: E-commerce Customer Behavior | **Notebook**: `17-Recommendation-System/Recommendation-Ecommerce.ipynb`

| Pendekatan            | Konsep                                             |
| :-------------------- | :------------------------------------------------- |
| **Content-Based**     | Merekomendasikan item serupa berdasarkan fitur     |
| **Cosine Similarity** | Mengukur kemiripan vektor antar profil             |
| **K-NN**              | Mencari K tetangga terdekat dalam ruang fitur      |
| **Segmentasi**        | Rekomendasi berdasarkan kelompok spending/behavior |

- Content-based filtering tidak memerlukan data interaksi pengguna lain
- Segmentasi membantu personalisasi rekomendasi

---

## Artificial Neural Network

**Artificial Neural Network (ANN)** atau Jaringan Saraf Tiruan adalah model komputasi yang meniru struktur dan fungsi jaringan saraf biologis manusia untuk memproses informasi dan mengenali pola yang kompleks.

### Varian Arsitektur ANN

| Arsitektur      | Karakteristik Utama                                                                   | Contoh Penggunaan                                          |
| :-------------- | :------------------------------------------------------------------------------------ | :--------------------------------------------------------- |
| **FNN / MLP**   | Aliran informasi searah (_feedforward_), terdiri dari input, hidden, dan output layer | Data tabular, regresi, klasifikasi dasar                   |
| **CNN**         | Menggunakan operasi konvolusi untuk mengenali pola spasial                            | Klasifikasi gambar, deteksi objek (_Computer Vision_)      |
| **RNN / LSTM**  | Memiliki memori/umpan balik untuk memproses data berurutan                            | Analisis deret waktu (_time-series_), NLP, translasi       |
| **Transformer** | Menggunakan mekanisme _Self-Attention_ untuk pemrosesan paralel                       | _Large Language Models_ (LLM) seperti GPT/Gemini           |
| **Autoencoder** | Arsitektur kompresi-rekonstruksi (_Encoder-Decoder_)                                  | Reduksi dimensi, deteksi anomali, _denoising_              |
| **GAN**         | Jaringan saraf yang berkompetisi (_Generator_ vs _Discriminator_)                     | Generasi gambar sintetis, transfer gaya (_style transfer_) |

### Multi-Layer Perceptron (MLP)

Multi-Layer Perceptron adalah arsitektur neural network dasar dengan _feedforward_ dan _backpropagation_.

**Dataset**: Medical Conditions | **Notebook**: `13-MLP-Neural-Network/MLP-Medical-Conditions.ipynb`

| Konsep                  | Penjelasan                                                                                  |
| :---------------------- | :------------------------------------------------------------------------------------------ |
| **Hidden Layer**        | Lapisan tersembunyi yang menangkap pola non-linear                                          |
| **Activation Function** | Fungsi non-linear (ReLU, Tanh, atau Sigmoid) untuk memproses output neuron                  |
| **Backpropagation**     | Algoritma pembaruan bobot (_weights_) berdasarkan gradien error dari belakang ke depan      |
| **Loss Function**       | Mengukur tingkat kesalahan prediksi (Cross-entropy untuk klasifikasi, MSE untuk regresi)    |
| **Learning Rate**       | Kecepatan model dalam memperbarui bobot — nilai terlalu tinggi dapat menyebabkan divergensi |

- Arsitektur sederhana (1-2 hidden layer) cukup untuk dataset kecil/sedang
- Loss curve digunakan untuk memonitor konvergensi selama proses training

### Convolutional Neural Network (CNN)

Convolutional Neural Network adalah arsitektur neural network yang dirancang khusus untuk memproses data spasial seperti gambar atau grid 2D/3D.

| Komponen CNN              | Fungsi & Deskripsi                                                                                                        |
| :------------------------ | :------------------------------------------------------------------------------------------------------------------------ |
| **Convolutional Layer**   | Menggunakan filter/kernel yang bergeser di atas gambar untuk mengekstrak fitur spasial (seperti garis, bentuk, atau pola) |
| **Activation (ReLU)**     | Memperkenalkan sifat non-linear setelah proses konvolusi                                                                  |
| **Pooling Layer**         | Mereduksi dimensi spasial (_downsampling_) menggunakan Max Pooling atau Average Pooling untuk menghemat komputasi         |
| **Fully Connected Layer** | Meratakan (_flatten_) peta fitur menjadi satu vektor panjang dan menghubungkannya ke output layer klasifikasi             |

- Sangat efisien karena menggunakan konsep _Weight Sharing_ (filter yang sama digunakan di seluruh bagian gambar)
- Memiliki sifat _Translation Invariance_ (mampu mendeteksi objek di mana pun posisinya di dalam gambar)

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

## Model Deployment

Proses menyimpan model terlatih dan mengintegrasikannya ke aplikasi.

**Dataset**: Iris (built-in) | **Notebook**: `16-Model-Deployment/Model-Deployment-Iris.ipynb`

| Tahap               | Tools                            | Deskripsi                                   |
| :------------------ | :------------------------------- | :------------------------------------------ |
| **Serialization**   | `joblib`, `pickle`               | Menyimpan model ke file disk                |
| **Deserialization** | `joblib.load()`, `pickle.load()` | Memuat model dari file disk                 |
| **REST API**        | FastAPI + Uvicorn                | Mendeploy model sebagai web service         |
| **Prediction**      | `model.predict()`                | Menggunakan model untuk inferensi data baru |

- Joblib lebih direkomendasikan untuk scikit-learn dibanding pickle
- REST API memungkinkan integrasi dengan aplikasi web/mobile

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
