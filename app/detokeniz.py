import pandas as pd
import ast

def detokenize(text):
    try:
        # Ubah string list menjadi list python beneran
        tokens = ast.literal_eval(text)
        # Gabungkan kembali menjadi kalimat
        if isinstance(tokens, list):
            return ' '.join(tokens)
        return text
    except (ValueError, SyntaxError):
        # Bersihkan manual jika formatnya error
        return text.replace("['", "").replace("']", "").replace("', '", " ")

# 1. Baca file
df = pd.read_csv('databaru2.csv', delimiter=';')

# 2. Update kolom 'Ulasan' langsung dengan versi yang sudah dibersihkan
df['Ulasan'] = df['Ulasan'].apply(detokenize)

# 3. Simpan hanya data bersihnya (menimpa struktur lama)
df.to_csv('databaru2_bersih.csv', index=False, sep=';')

# Cek hasil
print(df.head())