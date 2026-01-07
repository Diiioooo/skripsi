import streamlit as st

st.set_page_config(page_title="Skripsi DIO – Dashboard", layout="wide")

st.title(" Dashboard Penelitian Skripsi")
st.markdown("---")

st.header(" Gambaran Umum Penelitian")
st.write("""
Aplikasi ini merupakan dashboard hasil penelitian tugas akhir mengenai  
**Analisis Sentimen Ulasan Produk Jamu Madura** menggunakan pendekatan 
**Deep Learning Long Short-Term Memory (LSTM)** dengan representasi kata 
**Word2Vec**.

Penelitian ini membandingkan performa model berdasarkan:
- **Word2Vec Non-SMOTE** dan **Word2Vec SMOTE**
- **LSTM Default** dan **LSTM dengan variasi hyperparameter**
- Evaluasi menggunakan **Accuracy, Precision, Recall, dan F1-Score**
""")

st.subheader(" Struktur Pengujian Model")
st.write("""
Pengujian dilakukan melalui beberapa skenario konfigurasi model LSTM, yaitu:

- **Default** → Konfigurasi dasar LSTM  
- **Skenario A – E** → Variasi jumlah neuron, dropout, optimizer, dan learning rate  

Setiap skenario diuji pada dua kondisi data:
- **Non-SMOTE**
- **SMOTE**
""")

st.subheader(" Halaman yang Tersedia")
st.write("""
1. **Hasil Uji Coba**  
   Menampilkan hasil evaluasi seluruh skenario model berupa metrik performa 
   dan confusion matrix.

2. **Prediksi Sentimen**  
   Pengguna dapat memasukkan teks ulasan dan mencoba prediksi sentimen 
   menggunakan model LSTM terlatih (SMOTE maupun Non-SMOTE).

3. **Perbandingan Model**  
   Visualisasi perbandingan akurasi antar skenario dan metode penyeimbangan data.
""")

st.success("Sistem siap digunakan. Silakan pilih halaman dari sidebar.")

