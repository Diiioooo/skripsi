import streamlit as st

st.title("Penutup & Kesimpulan")

st.write("""
### Kesimpulan Penelitian

1. Penelitian ini berhasil menerapkan metode **Long Short-Term Memory (LSTM)** 
   dengan representasi kata **Word2Vec** untuk melakukan analisis sentimen 
   terhadap ulasan produk Jamu Madura.

2. Penerapan **SMOTE** sebagai metode penyeimbangan data terbukti dapat 
   meningkatkan stabilitas performa model, terutama pada kelas sentimen 
   minoritas.

3. Variasi hyperparameter LSTM seperti **jumlah neuron, dropout, optimizer, 
   dan learning rate** memberikan pengaruh yang signifikan terhadap performa 
   klasifikasi sentimen pada setiap skenario pengujian.

4. Sistem yang dibangun mampu menampilkan hasil evaluasi model serta melakukan 
   **prediksi sentimen ulasan secara interaktif** melalui dashboard Streamlit.

---

### Ucapan
Terima kasih atas perhatian dan penggunaan sistem ini sebagai media 
visualisasi dan analisis hasil penelitian skripsi.

Sebagai pengembangan di masa mendatang, penelitian ini dapat diperluas dengan:
- Penambahan metode lain seperti **CNN atau Transformer-based model**
- Penggunaan **pre-trained embedding** (FastText atau IndoBERT)
- Analisis lanjutan terhadap kesalahan klasifikasi dan distribusi sentimen
""")

st.info("Gunakan sidebar untuk kembali ke halaman lain pada dashboard.")

