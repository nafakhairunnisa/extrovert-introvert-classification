# Extrovert vs Introvert Classification with MLOps

## Overview
Proyek ini bertujuan untuk mengklasifikasikan kepribadian seseorang sebagai **Extrovert** atau **Introvert** berdasarkan perilaku sehari-hari. Model machine learning dikembangkan dan dikelola menggunakan praktik **MLOps** untuk memastikan pipeline otomatis, reproducible, dan dapat dimonitor secara real-time.

## Tujuan Proyek
- Membangun model klasifikasi kepribadian berbasis data perilaku.
- Mengimplementasikan workflow MLOps end-to-end mulai dari preprocessing otomatis, training, tuning, hingga monitoring dan alerting.

## Metodologi
1. **Dataset**: Extrovert vs Introvert Behavior (2900 sampel, 8 fitur) yang diambil dari [Kaggle](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data).
2. **Preprocessing**:
   - Encoding (LabelEncoder)
   - Standarisasi (StandardScaler)
3. **Modeling**:
   - RandomForestClassifier tanpa tuning (tracking dengan MLflow lokal)
   - RandomForestClassifier dengan tuning GridSearchCV (tracking & registry ke DagsHub)
4. **Monitoring & Logging**:
   - Ekspos metrik via Prometheus
   - Alerting dan visualisasi performa model via Grafana

## Evaluasi
Model dievaluasi dengan metrik:
- Basic Model: accuracy, f1_score (weighted), precision (macro), recall (macro), confusion matrix
- Tuned Model: accuracy, f1_score (weighted), roc_auc, precision class 0, recall class 1, log_loss, classification report, confusion matrix

Akurasi model sebelum tuning 91%.

Akurasi model setelah tuning 92.84%.

## Insight Utama
- Model Random Forest memberikan performa stabil baik tanpa tuning maupun setelah hyperparameter tuning.
- Proses tuning meningkatkan akurasi dan menyeimbangkan metrik antar kelas.

## Tools dan Teknologi
- **Python**, **Scikit-learn**, **Pandas**, **Matplotlib**, **Seaborn**
- **MLflow** (Tracking & Model Registry)
- **DagsHub** (Model Tracking & Storage)
- **Prometheus** (Monitoring Metrik)
- **Grafana** (Visualisasi & Alerting)

## Cara Menjalankan
1. Clone repositori:
   ```bash
   git clone https://github.com/nafakhairunnisa/extrovert-introvert-classification.git
   cd extrovert-introvert-classification
   ```
2. Buat virtual environment & install dependensi:
```
python -m venv venv
source venv/bin/activate  # atau .\venv\Scripts\activate di Windows
pip install -r requirements.txt
```
3. Jalankan preprocessing otomatis:
```
python preprocessing.py
```
4. Jalankan training model (tanpa tuning):
```
python modelling.py
```
5. Jalankan training model (dengan tuning + registry ke DagsHub):
```
python modelling_tuning.py
```
6. Monitoring (Opsional):
   - Jalankan Prometheus untuk mengumpulkan metrik dari model
   - Jalankan Grafana untuk memantau performa model secara visual
   - Alert akan dikirimkan jika performa model turun di bawah threshold
> Catatan: Tahap ini memerlukan konfigurasi manual berupa input metrik satu per satu ke Prometheus, serta pembuatan dashboard dan alerting di Grafana secara mandiri.

## Future Works
- Integrasi deployment ke layanan cloud (misalnya: Streamlit + Docker).
- Implementasi retraining pipeline otomatis berbasis data baru.
