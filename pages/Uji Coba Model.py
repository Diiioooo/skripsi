import re
import os
import pickle
import numpy as np
import streamlit as st

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# LABEL
LABELS = ["Negatif 😡", "Netral 😐", "Positif 😊"]
SCENARIO_NON_SMOTE = ["Default", "A","C", "D", "E"]
SCENARIO_SMOTE = ["A", "B", "D", "E"]

st.set_page_config(page_title="Analisis Sentimen LSTM", layout="centered")


# LOAD PIPELINE
@st.cache_resource(show_spinner=False)
def load_pipeline(mode, scenario):
    """
    Load model LSTM, Word2Vec, dan max_len sesuai mode & skenario
    """
    base_path = f"models/{mode}"

    # Pilih file model
    if scenario == "Default":
        model_file = "lstm_default.keras" if mode == "non_smote" else "lstm_defaultS2W1_smote.keras"
    else:
        model_file = (
            f"Skenario_{scenario}.keras" if mode == "non_smote" else f"model_S2W1Skenario_{scenario}_smote.keras"
        )

    model = load_model(os.path.join(base_path, model_file), compile=False)

    # WORD2VEC
    w2v_path = "modelS2W2_w2v.model" if mode == "non_smote" else "model_w2vsmoteS2.model"
    w2v = Word2Vec.load(os.path.join(base_path, w2v_path))

    # MAX LEN
    maxlen_path = "max_len_s2w2.pkl" if mode == "non_smote" else "max_len_w2vsmoteS2.pkl"
    with open(os.path.join(base_path, maxlen_path), "rb") as f:
        max_len = pickle.load(f)

    # WORD INDEX
    word_index_path = "word_index_s2w2.pkl" if mode == "non_smote" else "word_index_w2vsmoteS2.pkl"
    with open(os.path.join(base_path, word_index_path), "rb") as f:
        word_index = pickle.load(f)

    return model, w2v, word_index, max_len

# PREPROCESS
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)   # baguuuus -> bagus
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()

    tokens = [t for t in tokens if  len(t) > 2]
    return tokens

# UI
st.title("🔍 Analisis Sentimen Ulasan Jamu Madura")
st.caption("LSTM + Word2Vec | SMOTE vs Non-SMOTE")

# Pilih dataset
mode_ui = st.selectbox("Dataset", ["Non-SMOTE", "SMOTE"])
mode = "non_smote" if mode_ui == "Non-SMOTE" else "smote"

# Pilih skenario
if mode == "non_smote":
    scenario = st.selectbox(
        "Skenario Model (Non-SMOTE)",
        SCENARIO_NON_SMOTE
    )
else:
    scenario = st.selectbox(
        "Skenario Model (SMOTE)",
        SCENARIO_SMOTE
    )


# Input teks
text = st.text_area("Masukkan ulasan:")

# Tombol prediksi
if st.button("Prediksi"):
    if not text.strip():
        st.warning("Teks kosong")
    else: 
        model, w2v, word_index, max_len = load_pipeline(mode, scenario)

        tokens = preprocess_text(text)
        valid_tokens = [t for t in tokens if t in word_index]

        if len(valid_tokens) == 0:
            st.warning("Teks terlalu jauh dari data latih")
        else:
            seq = [word_index[t] for t in valid_tokens]
            X = pad_sequences([seq], maxlen=max_len, padding="post")

            pred = model.predict(X, verbose=0)[0]
            label = np.argmax(pred)

            st.success(f"Hasil Prediksi: **{LABELS[label]}**")
            st.json({
                "Negatif": float(pred[0]),
                "Netral": float(pred[1]),
                "Positif": float(pred[2])
            })






