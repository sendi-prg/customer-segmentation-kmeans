# Customer Segmentation menggunakan K-Means Clustering

Project tugas mandiri Machine Learning. Aplikasi ini memprediksi cluster pelanggan berdasarkan model K-Means (K=8) yang dilatih pada dataset `Customers.csv` dengan alur CRISP-DM. Aplikasi dibuat menggunakan Streamlit.

Link aplikasi online: *http://192.168.110.222:8501*

## Struktur Folder

```text
tugas_mandiri/
├── notebook/
│   └── clustering_final_crispdm.ipynb   # proses CRISP-DM dan training model
├── dataset/
│   └── Customers.csv                    # dataset asli
└── streamlit/
    ├── app.py                           # aplikasi Streamlit
    ├── requirements.txt                 # dependencies aplikasi
    ├── scaler_customer.joblib           # StandardScaler hasil training
    ├── kmeans_customer.joblib           # K-Means hasil training (K=8)
    ├── Customers.csv                    # salinan dataset untuk aplikasi
    └── README.md
```

| Folder       | Fungsi                                                                                                                                                                           |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebook/`  | Berisi notebook untuk Data Understanding sampai Evaluation, serta bagian Deployment yang menyimpan `scaler_customer.joblib` dan `kmeans_customer.joblib` ke folder `streamlit/`. |
| `dataset/`   | Dataset asli yang dibaca oleh notebook (`../dataset/Customers.csv`).                                                                                                             |
| `streamlit/` | Folder yang berdiri sendiri untuk deployment. Semua file yang dibutuhkan aplikasi berada di folder ini.                                                                          |

`Customers.csv` di folder `streamlit/` adalah salinan dataset. Salinan ini diperlukan karena folder `streamlit/` harus lengkap saat di-deploy, dan aplikasi memakainya untuk pilihan `Profession`, rentang input, serta profil cluster.

## Cara Kerja

```text
Notebook: preprocessing → encoding → StandardScaler → K-Means (K=8) → evaluation
    ↓
Simpan scaler_customer.joblib dan kmeans_customer.joblib
    ↓
Streamlit: load kedua file joblib
    ↓
Input customer → DataFrame → preprocessing → scaler.transform() → kmeans.predict()
    ↓
Prediksi Cluster: Cluster X
```

Aplikasi tidak melatih ulang K-Means. Konfigurasi model: `n_clusters=8`, `random_state=42`, `n_init=20`. Fitur yang digunakan: `Gender`, `Age`, `Annual Income ($)`, `Spending Score (1-100)`, `Profession`, `Work Experience`, dan `Family Size` (`CustomerID` tidak digunakan).

Urutan dan jumlah kolom input pada aplikasi mengikuti `scaler.feature_names_in_`, yaitu daftar kolom yang dipakai saat training, sehingga hasil preprocessing aplikasi sama dengan notebook.
