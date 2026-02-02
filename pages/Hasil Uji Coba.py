import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image

# PAGE CONFIG
st.set_page_config(
    page_title="Evaluasi Model LSTM + Word2Vec",
    layout="wide"
)

st.title("Evaluasi Model Terbaik LSTM + Word2Vec")
st.caption("Hasil pengujian model terbaik pada dataset ulasan Jamu Madura")

# PATH
EVAL_PATH = "models/BEST_PIPELINE_80_20.pkl"
CM_PATH = "CM/cm_best_grid_w2v_vs60_w10.png"

@st.cache_resource
def load_eval():
    with open(EVAL_PATH, "rb") as f:
        return pickle.load(f)

data = load_eval()


st.header("Hasil Evaluasi Model Terbaik")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{data['accuracy']:.4f}")
c2.metric("Precision", f"{data['precision']:.4f}")
c3.metric("Recall", f"{data['recall']:.4f}")
c4.metric("F1-Score", f"{data['f1_score']:.4f}")


# DETAIL MODEL

with st.expander("Detail Model Terbaik"):

    detail = {
        "Model": "LSTM + Word2Vec",
        "Word2Vec": "Vector Size 60, Window 10",
        "Split Data": "80 : 20",
        "Units LSTM": data.get("units", "-"),
        "Dropout": data.get("dropout", "-"),
        "Optimizer": data.get("optimizer", "-"),
        "Learning Rate": data.get("learning_rate", "-")
    }

    st.table(pd.DataFrame(detail.items(), columns=["Parameter", "Nilai"]))


# CONFUSION MATRIX

st.subheader("Confusion Matrix")

if os.path.exists(CM_PATH):
    st.image(CM_PATH, width=420)
else:
    st.warning("Confusion matrix tidak ditemukan.")
