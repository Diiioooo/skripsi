import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Evaluasi LSTM + Word2Vec",
    layout="wide"
)

st.title("📊 Evaluasi Model LSTM + Word2Vec")
st.caption("Analisis performa model dengan dan tanpa SMOTE")

# PATH
PATH = {
    "Non-SMOTE": {
        "default": "models/non_smote/hasil_lstm_default.pkl",
        "skenario": "models/non_smote/lstm_skenario_results.pkl",
        "cm_dir": "CM",
        "cm_default": "cm_s2w2default.png",
        "cm_prefix": "cm_s2w2"
    },
    "SMOTE": {
        "default": "models/smote/hasil_lstm_defaultS2W1_smote.pkl",
        "skenario": "models/smote/lstm_skenarioS2W1_results_smote.pkl",
        "cm_dir": "CMSM",
        "cm_default": "cm_smoteS2W1_default.png",
        "cm_prefix": "cm_smoteS2W1_"
    }
}

# HELPER
def load_pkl(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data if isinstance(data, list) else [data]

def show_metrics(row):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{row['accuracy']:.4f}")
    c2.metric("Precision", f"{row['precision']:.4f}")
    c3.metric("Recall", f"{row['recall']:.4f}")
    c4.metric("F1-Score", f"{row['f1_score']:.4f}")

def show_hyperparam(row):
    hp = {
        "Units LSTM": row.get("units", "-"),
        "Dropout": row.get("dropout", "-"),
        "Optimizer": row.get("optimizer", "-"),
        "Learning Rate": row.get("learning_rate", "-"),
        "Durasi (detik)": f"{row['duration']:.2f}"
    }
    st.table(pd.DataFrame(hp.items(), columns=["Parameter", "Nilai"]))

def show_cm(path):
    if os.path.exists(path):
        st.image(path, width=380)
    else:
        st.info("Confusion Matrix tidak ditemukan.")

# DROPDOWN
mode = st.selectbox(
    "🔀 Pilih Metode Penyeimbangan Data",
    ["Non-SMOTE", "SMOTE"]
)

cfg = PATH[mode]

data = load_pkl(cfg["default"]) + load_pkl(cfg["skenario"])

# EVALUASI 
st.header(f"📌 Hasil Evaluasi {mode}")

for row in data:
    nama = row.get("skenario", "Default")

    st.subheader(f"🔹 {nama}")
    show_metrics(row)

    with st.expander("⚙️ Detail Hyperparameter"):
        show_hyperparam(row)

    # Confusion Matrix
    if nama.lower() == "default":
        cm_file = cfg["cm_default"]
    else:
        cm_file = f"{cfg['cm_prefix']}{nama.replace(' ', '_')}.png"

    show_cm(os.path.join(cfg["cm_dir"], cm_file))
    st.divider()


# LOAD DATA
non_smote_data = (
    load_pkl(PATH["Non-SMOTE"]["default"]) +
    load_pkl(PATH["Non-SMOTE"]["skenario"])
)

smote_data = (
    load_pkl(PATH["SMOTE"]["default"]) +
    load_pkl(PATH["SMOTE"]["skenario"])
)


# DIAGRAM PERBANDINGAN
st.header("📈 Diagram Perbandingan Model")

compare_mode = st.selectbox(
    "Pilih jenis perbandingan:",
    ["Non-SMOTE", "SMOTE", "Non-SMOTE vs SMOTE"]
)

# Siapkan Data
df_non = pd.DataFrame(non_smote_data)
df_non["Tipe"] = "Non-SMOTE"
df_non["skenario"] = df_non["skenario"].fillna("Default")

df_sm = pd.DataFrame(smote_data)
df_sm["Tipe"] = "SMOTE"
df_sm["skenario"] = df_sm["skenario"].fillna("Default")

# Pilih Data Sesuai Dropdown
if compare_mode == "Non-SMOTE":
    df_plot = df_non
    title = "Perbandingan Akurasi Non-SMOTE"

elif compare_mode == "SMOTE":
    df_plot = df_sm
    title = "Perbandingan Akurasi SMOTE"

else:
    df_plot = pd.concat([df_non, df_sm], ignore_index=True)
    title = "Perbandingan Akurasi Non-SMOTE vs SMOTE"

# PLOT
fig, ax = plt.subplots(figsize=(9, 4))

if compare_mode == "Non-SMOTE":
    df_plot = df_non
    title = "Akurasi Model Non-SMOTE"

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_plot["skenario"], df_plot["accuracy"])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=9)
elif compare_mode == "SMOTE":
    df_plot = df_sm
    title = "Akurasi Model SMOTE"

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df_plot["skenario"], df_plot["accuracy"])

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                ha="center", fontsize=9)
else:
    df_plot = pd.concat([df_non, df_sm])

    title = "Perbandingan Akurasi Non-SMOTE vs SMOTE per Skenario"

    fig, ax = plt.subplots(figsize=(9, 4))

    skenario = df_plot["skenario"].unique()
    x = np.arange(len(skenario))
    width = 0.35

    acc_non = df_non.set_index("skenario").loc[skenario]["accuracy"]
    acc_sm = df_sm.set_index("skenario").loc[skenario]["accuracy"]

    bars1 = ax.bar(x - width/2, acc_non, width, label="Non-SMOTE")
    bars2 = ax.bar(x + width/2, acc_sm, width, label="SMOTE")

    ax.set_xticks(x)
    ax.set_xticklabels(skenario)
    ax.legend()

    # angka di atas bar
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.3f}", ha="center", fontsize=9)

ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_xlabel("Skenario")
ax.set_title(title)

st.pyplot(fig)
