import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import gensim
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from keras.utils import to_categorical
# from gensim.models import FastText
from keras.callbacks import EarlyStopping
# import fasttext.util

# Set Value untuk text size dan epochs
test_size_param = 0.3
epochs_param = 30
# ----------- dataset dari csv label.py
# Membaca file CSV data.csv yang berisi teks dan label sentimen.
# Menyimpan teks dan label dalam variabel texts dan labels.
data = pd.read_csv('data.csv')
texts = data['text'].values
labels = data['sentiment'].values

# Baca data normalisasi dari file Excel
normalization_df = pd.read_excel('normalization.xlsx')

# Buat dictionary untuk mapping sebelum dan sesudah normalisasi
normalization_mapping = dict(zip(normalization_df['Before'], normalization_df['After']))

# Fungsi untuk melakukan normalisasi teks berdasarkan mapping yang diberikan
def normalize_text(text, normalization_mapping):
    normalized_text = []
    words = text.split()
    for word in words:
        normalized_word = normalization_mapping.get(word.lower(), word)  # Ambil kata yang telah dinormalisasi, jika tidak ada gunakan kata aslinya
        normalized_text.append(normalized_word)
    return ' '.join(normalized_text)

# Normalisasi teks sebelum proses stopword removal
texts = [normalize_text(text, normalization_mapping) for text in texts]

# Membaca stopwords dari file stopwords.txt
with open('stopwords.txt', 'r') as file:
    stopwords = set(file.read().split())

# Fungsi untuk menghapus stopword dari teks
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stopwords])

# Menghapus stopword dari teks
texts = [remove_stopwords(text) for text in texts]

# Mengganti teks yang telah dihapus stopword-nya ke dalam dataframe
data['text'] = texts

#----------- Encode sentiment labels global
# Menggunakan LabelEncoder untuk mengubah label sentimen dari format teks menjadi format numerik.
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

#----------- Split the data
# Membagi data menjadi set pelatihan (80%) dan pengujian (20%).
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=test_size_param, random_state=42)

#----------- Tokenize text
# Membuat tokenizer dan menyesuaikan pada teks pelatihan untuk membuat indeks kata.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

#----------- Encode label worcloud
# Meng-encode label menjadi format numerik dan kemudian mengubahnya menjadi format kategorikal untuk visualisasi word cloud.
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# ----------- Convert texts to sequences
# Mengubah teks menjadi urutan indeks kata dan melakukan padding agar semua urutan memiliki panjang yang sama
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
data_seq = tokenizer.texts_to_sequences(data['text'])

# -----------Pad sequences to the same length
max_length = max([len(x) for x in X_train_seq])
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
data_pad = pad_sequences(data_seq, maxlen=max_length, padding='post')

# ----------- Load FastText model cc.id.300.vec, download dari https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.id.300.vec.gz, ekstrak ke folder projectnya
# Memuat model FastText yang telah diunduh dan diekstrak sebelumnya.
w2vec_model = gensim.models.KeyedVectors.load_word2vec_format('cc.id.300.vec', binary=False)

# -----------membuat embedding matrix dari model fasttext
# Membuat matriks embedding yang akan digunakan dalam model berdasarkan kata-kata dalam data pelatihan dan embedding dari model FastText.
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in w2vec_model:
        embedding_matrix[i] = w2vec_model[word]

# Membagi kembali data yang sudah diproses menjadi set pelatihan dan pengujian.
X_train, X_test, y_train, y_test = train_test_split(data_pad, data['sentiment'], test_size=test_size_param, random_state=42)

# -----------Build the model-----------
# Membangun Model CNN untuk Klasifikasi
# Membangun model Sequential dengan lapisan embedding, convolutional, global max pooling, dan dense untuk klasifikasi sentimen.
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))

# -----------Compile the model-----------
# Mengompilasi model dengan optimizer adam dan loss sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------Train the model-----------
# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train_pad, y_train, epochs=epochs_param, validation_data=(X_test_pad, y_test), batch_size=32, callbacks=[early_stopping])

# Memprediksi label pada data pengujian dan menghitung metrik evaluasi seperti confusion matrix dan classification report.
# Menyimpan sejarah pelatihan dalam DataFrame dan menampilkan akurasi pelatihan dan validasi.
# -----------Evaluate the model-----------
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)

#----------- Create a DataFrame from the history
history_df = pd.DataFrame(history.history)

#----------- Display the accuracy and validation accuracy
accuracy_table = history_df[['accuracy', 'val_accuracy']].copy()
accuracy_table.columns = ['Train Accuracy', 'Validation Accuracy']
print(accuracy_table)

#----------- Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

#----------- Classification report
class_report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_)

print(conf_matrix)
print(class_report)

#----------- Plot confusion matrix
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

sentiments = ['positive', 'negative']

#----------- Generate word clouds
for sentiment in sentiments:
    # Filter texts by sentiment
    sentiment_indices = np.where(labels == label_encoder.transform([sentiment])[0])[0]
    sentiment_texts = [texts[idx] for idx in sentiment_indices]
    
    # Convert sentiment texts to strings
    sentiment_texts = [str(text) for text in sentiment_texts]
    
    # Check if there are words for the sentiment
    if sentiment_texts:
        # Join all texts into a single string
        sentiment_text = ' '.join(sentiment_texts)
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
        
        # Plot word cloud
        plt.figure(figsize=(5, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
    else:
        print(f"No texts available for {sentiment.capitalize()} sentiment.")

#----------- Visualize training history
plt.figure(figsize=(12, 5))

#----------- Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

#----------- Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

#----------- Plot bar chart of sentiment counts
# keseluruhan data
# Count the number of positive and negative labels
# Sentiment distribution in dataset
sentiment_counts = data['sentiment'].value_counts()
colors = ['blue' if sentiment == 'positive' else 'red' for sentiment in label_encoder.classes_]
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=label_encoder.classes_, y=sentiment_counts, hue=label_encoder.classes_, palette=colors, dodge=False)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Untuk Keseluruhan Data')
plt.xticks(ticks=[0, 1], labels=label_encoder.classes_)
# Add count labels to the bars
for i, count in enumerate(sentiment_counts):
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom')

# Sentiment distribution in training data
train_sentiment_counts = pd.Series(y_train).value_counts()
plt.figure(figsize=(6, 4))
ax1 = sns.barplot(x=label_encoder.classes_, y=train_sentiment_counts, hue=label_encoder.classes_, palette=colors, dodge=False)
plt.xlabel('Sentiment')
plt.ylabel('Total')
plt.title('Sentiment Untuk Data Training')
plt.xticks(ticks=[0, 1], labels=label_encoder.classes_)
# Add count labels to the bars
for i, count in enumerate(train_sentiment_counts):
    ax1.text(i, count + 0.5, str(count), ha='center', va='bottom')

# Sentiment distribution in test data
testing_sentiment_counts = pd.Series(y_test).value_counts()
plt.figure(figsize=(6, 4))
ax2 = sns.barplot(x=label_encoder.classes_, y=testing_sentiment_counts, hue=label_encoder.classes_, palette=colors, dodge=False)
plt.xlabel('Sentiment')
plt.ylabel('Total')
plt.title('Sentiment Untuk Data Testing')
plt.xticks(ticks=[0, 1], labels=label_encoder.classes_)
# Add count labels to the bars
for i, count in enumerate(testing_sentiment_counts):
    ax2.text(i, count + 0.5, str(count), ha='center', va='bottom')

plt.show()