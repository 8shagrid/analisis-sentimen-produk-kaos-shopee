import pandas as pd
import tensorflow as tf
# Menonaktifkan pesan warning
tf.get_logger().setLevel('ERROR')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# Load model yang telah disimpan
model = load_model("model/saved_model/")  # Ganti dengan path tempat model disimpan

# Input kalimat yang ingin diprediksi sentimennya
text_to_predict = input("Masukkan kalimat yang ingin diprediksi sentimennya: ")

# Buat DataFrame untuk kalimat yang ingin diprediksi
new_df = pd.DataFrame({'txt': [text_to_predict]})

# Preprocessing teks pada data pelatihan
data = pd.read_csv("dataset/clean_rks.csv")
max_features = 1000  # Jumlah kata unik yang digunakan saat tokenisasi
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data["txt"].values)

# Transformasi teks baru menjadi sekuen angka
X_new = tokenizer.texts_to_sequences(new_df['txt'].values)
maxlen = max(len(seq) for seq in tokenizer.texts_to_sequences(data['txt'].values))
X_new = pad_sequences(X_new, maxlen=maxlen)

# Lakukan prediksi pada data baru
prediction = model.predict(X_new)

# Tampilkan hasil prediksi
if prediction > 0.5:
    print("Sentimen: Positif")
else:
    print("Sentimen: Negatif")
