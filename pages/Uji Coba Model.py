import re
import os
import pickle
import numpy as np
import streamlit as st

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences



# LABEL
LABELS = ["Negatif", "Netral", "Positif"]
SCENARIOS = ["best model"]

st.set_page_config(page_title="Analisis Sentimen LSTM", layout="centered")


# LOAD PIPELINE
@st.cache_resource(show_spinner=False)
def load_pipeline(scenario):
    base_path = "models"

    # Pilih model sesuai skenario
    if scenario == "Default":
        model_file = "non_smote/lstm_default.keras"
    else:
        model_file = f"non_smote/{scenario}.keras"

    model = load_model(os.path.join(base_path, model_file), compile=False)

    # Word2Vec
    w2v = Word2Vec.load(os.path.join(base_path, "non_smote/modelS2W2_w2v.model"))

    # Max length
    with open(os.path.join(base_path, "non_smote/max_len_s2w2.pkl"), "rb") as f:
        max_len = pickle.load(f)

    # Word index
    with open(os.path.join(base_path, "non_smote/word_index_s2w2.pkl"), "rb") as f:
        word_index = pickle.load(f)

    return model, w2v, word_index, max_len



# PREPROCESS

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]

    return tokens

# UI
st.title("Analisis Sentimen Ulasan Jamu Madura")
st.caption("LSTM + Word2Vec | Skenario Model")

scenario = st.selectbox("Skenario Model", SCENARIOS)
text = st.text_area("Masukkan ulasan:")


# PREDIKSI
if st.button("Prediksi"):
    if not text.strip():
        st.warning("Teks kosong")
    else:
        model, w2v, word_index, max_len = load_pipeline(scenario)

        tokens = preprocess_text(text)
        valid_tokens = [t for t in tokens if t in word_index]

        if len(valid_tokens) == 0:
            st.warning("Kalimat terlalu jauh dari data latih")
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
