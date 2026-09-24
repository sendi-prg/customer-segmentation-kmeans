"""Aplikasi Streamlit: Customer Segmentation menggunakan K-Means Clustering.

Aplikasi ini hanya memuat scaler dan model hasil training dari notebook
(scaler_customer.joblib dan kmeans_customer.joblib). Tidak ada proses
training ulang pada file ini.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent
SCALER_PATH = BASE_DIR / "scaler_customer.joblib"
MODEL_PATH = BASE_DIR / "kmeans_customer.joblib"
DATA_PATH = BASE_DIR / "Customers.csv"

# Mapping Gender sama dengan tahap Encoding pada notebook
GENDER_MAP = {"Male": 0, "Female": 1}
NUMERIC_COLS = [
    "Age",
    "Annual Income ($)",
    "Spending Score (1-100)",
    "Work Experience",
    "Family Size",
]

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️")


# ---------------------------------------------------------------------------
# 1. Loading scaler, model, dan data
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Memuat StandardScaler dan K-Means hasil training (tanpa fit ulang)."""
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(MODEL_PATH)
    return scaler, kmeans


@st.cache_data
def load_data():
    """Membaca dataset. Baris dengan Profession kosong dihapus seperti pada notebook."""
    df = pd.read_csv(DATA_PATH)
    return df.dropna(subset=["Profession"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Preprocessing input (identik dengan fungsi pada bagian Deployment notebook)
# ---------------------------------------------------------------------------
def preprocess_input(df_input, feature_columns):
    """Mengubah input pelanggan menjadi fitur dengan kolom dan urutan yang sama dengan training."""
    df_proc = df_input.copy()
    df_proc["Gender"] = df_proc["Gender"].map(GENDER_MAP)
    df_proc = pd.get_dummies(df_proc, columns=["Profession"], prefix="Profession")
    # Penyesuaian feature columns: urutan dan jumlah kolom mengikuti scaler dari training.
    # Kolom yang tidak ada pada training (baseline Profession_Artist) otomatis dibuang.
    df_proc = df_proc.reindex(columns=feature_columns, fill_value=0)
    return df_proc.astype(float)


# ---------------------------------------------------------------------------
# 3. Profil cluster (dihitung dari data training melalui jalur prediksi yang sama)
# ---------------------------------------------------------------------------
@st.cache_data
def build_cluster_profile():
    scaler, kmeans = load_artifacts()
    df = load_data()
    feature_columns = list(scaler.feature_names_in_)
    features = df[["Gender", "Age", "Annual Income ($)", "Spending Score (1-100)",
                   "Profession", "Work Experience", "Family Size"]]
    labels = kmeans.predict(scaler.transform(preprocess_input(features, feature_columns)))
    df = df.assign(Cluster=labels)

    profile = df.groupby("Cluster")[NUMERIC_COLS].mean()
    profile["Jumlah Pelanggan"] = df.groupby("Cluster").size()
    profile["Persentase (%)"] = profile["Jumlah Pelanggan"] / len(df) * 100

    top = df.groupby("Cluster")["Profession"].agg(lambda s: s.value_counts().index[0])
    top_share = df.groupby("Cluster")["Profession"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0] * 100
    )
    profile["Profesi Dominan"] = top
    profile["Porsi Profesi Dominan (%)"] = top_share
    return profile


# ---------------------------------------------------------------------------
# Tampilan aplikasi
# ---------------------------------------------------------------------------
st.title("Customer Segmentation")
st.write(
    "Aplikasi ini memprediksi cluster pelanggan menggunakan model K-Means "
    "yang telah dilatih pada notebook. Masukkan data pelanggan, lalu klik tombol prediksi."
)

if not (SCALER_PATH.exists() and MODEL_PATH.exists() and DATA_PATH.exists()):
    st.error(
        "File scaler_customer.joblib, kmeans_customer.joblib, atau Customers.csv "
        "tidak ditemukan di folder aplikasi. Jalankan bagian Deployment pada notebook "
        "terlebih dahulu."
    )
    st.stop()

scaler, kmeans = load_artifacts()
data = load_data()
feature_columns = list(scaler.feature_names_in_)
st.caption(f"Model: K-Means dengan {kmeans.n_clusters} cluster dan {len(feature_columns)} fitur.")

st.subheader("Input Data Pelanggan")
col_left, col_right = st.columns(2)


def number_field(container, label, column, step=1):
    """Input numerik dengan rentang dan nilai awal yang mengikuti data training."""
    return container.number_input(
        label,
        min_value=int(data[column].min()),
        max_value=int(data[column].max()),
        value=int(data[column].median()),
        step=step,
    )


with col_left:
    gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
    age = number_field(st, "Age", "Age")
    income = number_field(st, "Annual Income ($)", "Annual Income ($)", step=1000)
    spending = number_field(st, "Spending Score (1-100)", "Spending Score (1-100)")

with col_right:
    profession = st.selectbox("Profession", sorted(data["Profession"].unique()))
    experience = number_field(st, "Work Experience (tahun)", "Work Experience")
    family = number_field(st, "Family Size", "Family Size")

st.caption("Rentang nilai input mengikuti rentang data yang digunakan saat training.")

if st.button("Prediksi Cluster"):
    # Input Pengguna -> DataFrame
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Annual Income ($)": income,
        "Spending Score (1-100)": spending,
        "Profession": profession,
        "Work Experience": experience,
        "Family Size": family,
    }])

    # Preprocessing -> Scaler.transform() -> kmeans.predict()
    input_processed = preprocess_input(input_df, feature_columns)
    input_scaled = scaler.transform(input_processed)
    cluster = int(kmeans.predict(input_scaled)[0])

    st.success(f"Prediksi Cluster: Cluster {cluster}")

    # Profil cluster hasil prediksi
    profile = build_cluster_profile()
    row = profile.loc[cluster]

    st.subheader(f"Profil Cluster {cluster}")
    st.write(
        f"Cluster {cluster} berisi {int(row['Jumlah Pelanggan'])} pelanggan "
        f"({row['Persentase (%)']:.2f}% dari data training). "
        f"Profesi yang paling banyak pada cluster ini adalah {row['Profesi Dominan']} "
        f"({row['Porsi Profesi Dominan (%)']:.1f}% anggota cluster)."
    )

    comparison = pd.DataFrame({
        "Input Anda": [age, income, spending, experience, family],
        f"Rata-rata Cluster {cluster}": [row[c] for c in NUMERIC_COLS],
    }, index=NUMERIC_COLS).astype(float).round(2)
    st.dataframe(comparison)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#d62728" if i == cluster else "#9ecae1" for i in profile.index]
    ax.bar(profile.index.astype(str), profile["Jumlah Pelanggan"], color=colors)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.set_title("Ukuran Cluster (cluster hasil prediksi berwarna merah)")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Hasil ini merupakan segmentasi eksploratif. Pada notebook, fitur Profession "
        "terbukti berpengaruh besar terhadap pembentukan cluster, dan hasil K-Means "
        "masih sensitif terhadap random_state."
    )
