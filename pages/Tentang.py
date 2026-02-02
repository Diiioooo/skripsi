import streamlit as st

st.title("Penutup & Kesimpulan")

st.write("""
### Kesimpulan Penelitian

1. Penelitian ini berhasil menerapkan metode **Long Short-Term Memory (LSTM)** 
   dengan representasi kata **Word2Vec** untuk melakukan analisis sentimen 
   terhadap ulasan produk Jamu Madura menjadi tiga kelas, yaitu **Negatif, Netral, dan Positif**.

2. Melalui serangkaian eksperimen dan pengujian konfigurasi model, 
   satu **model terbaik** berhasil diperoleh berdasarkan nilai 
   **Accuracy, Precision, Recall, dan F1-Score** yang paling optimal.

3. Variasi hyperparameter LSTM seperti **jumlah neuron, dropout, optimizer, 
   dan learning rate** terbukti memberikan pengaruh signifikan terhadap 
   performa klasifikasi sentimen, sehingga proses tuning menjadi tahap penting 
   dalam memperoleh model yang optimal.

4. Sistem yang dibangun mampu menampilkan hasil evaluasi model terbaik 
   serta melakukan **prediksi sentimen ulasan secara interaktif** melalui 
   dashboard berbasis Streamlit.

---

### Pengembangan Selanjutnya

Sebagai pengembangan di masa mendatang, penelitian ini dapat diperluas dengan:
- Penambahan metode lain seperti **CNN atau Transformer-based model**
- Penggunaan **pre-trained embedding** seperti **FastText atau IndoBERT**
- Analisis lanjutan terhadap kesalahan klasifikasi dan distribusi sentimen
""")

st.info("Gunakan sidebar untuk kembali ke halaman lain pada dashboard.")
