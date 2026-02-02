import streamlit as st

st.set_page_config(page_title="Skripsi DIO – Dashboard", layout="wide")

st.title(" Dashboard Penelitian Skripsi")
st.markdown("---")

st.markdown("""
**Nama** : Septian Dio Dwinata Hendratno  
**NIM**  : 210411100026
""")

st.header(" Gambaran Umum Penelitian")
st.write("""
Aplikasi ini merupakan dashboard hasil penelitian tugas akhir mengenai  
**Analisis Sentimen Ulasan Produk Jamu Madura** menggunakan pendekatan 
**Deep Learning Long Short-Term Memory (LSTM)** dengan representasi kata 
**Word2Vec**.

Model digunakan untuk mengklasifikasikan ulasan ke dalam tiga kelas sentimen:
**Negatif, Netral, dan Positif**.
""")

st.subheader(" Struktur Pengujian Model")
st.write("""
Berdasarkan seluruh eksperimen yang dilakukan pada tahap penelitian,  
satu **model terbaik** telah dipilih berdasarkan nilai **Accuracy, Precision, Recall, dan F1-Score** tertinggi.

Model terbaik inilah yang digunakan pada sistem ini untuk melakukan
prediksi sentimen ulasan secara real-time.
""")

st.subheader("Konfigurasi Model")
st.write("""
Model yang digunakan adalah **LSTM + Word2Vec** dengan konfigurasi terbaik:

- **Word2Vec**: Vector Size 60, Window 10  
- **LSTM**: hasil tuning hyperparameter terbaik  
- **Split data**: 80% data latih dan 20% data uji  
""")


st.subheader(" Halaman yang Tersedia")
st.write("""
1. **Hasil Uji Coba**  
   Menampilkan performa model terbaik berupa Accuracy, Precision, Recall, F1-Score, 
   dan Confusion Matrix.

2. **Uji Coba Model**  
   Pengguna dapat memasukkan teks ulasan dan mencoba prediksi sentimen 
   menggunakan model terbaik.

3. **Tentang**  
   Berisi informasi singkat mengenai penelitian dan sistem.
""")

st.success("Sistem siap digunakan. Silakan pilih halaman dari sidebar.")

