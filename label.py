import csv
import json
from textblob import TextBlob
from deep_translator import GoogleTranslator

# Membuka file komentar.csv.
file = open("komentar.csv", mode='r', encoding='utf-8')  # Menggunakan mode 'r' untuk membaca dan encoding 'utf-8'

# Membaca file CSV menggunakan csv.reader.
csvreader = csv.reader(file)

# Membaca baris pertama sebagai header.
header = next(csvreader)

# Inisialisasi Variabel
data = []  # List untuk menyimpan data hasil analisis sentimen.

# Membaca Baris CSV dan Menganalisis Sentimen
for row in csvreader:  # Loop melalui setiap baris dalam csvreader
    for i in range(len(row)): 
        translated = GoogleTranslator(source='id', target='en').translate(row[i])  # Menerjemahkan teks dari Bahasa Indonesia ke Bahasa Inggris
        analysis = TextBlob(translated)  # Melakukan analisis sentimen pada teks yang sudah diterjemahkan
        # Menentukan sentimen (positive, neutral, negative) berdasarkan nilai polaritas dari analisis sentimen
        if analysis.sentiment.polarity > 0:
            value = 'positive'  # Positive
        else:
            value = 'negative'  # Negative

        dictionary = {'text': row[i], 'sentiment': value}  # Menyimpan hasil analisis dalam dictionary dengan kunci text dan sentiment
        data.append(dictionary)  # Menambahkan dictionary ke dalam list data

# Menyimpan Data dalam Format JSON
json_object = json.dumps(data, indent=2)  # Mengubah list data menjadi string JSON dengan indentasi 2 spasi.
with open("data.json", "w", encoding='utf-8') as outfile:
    outfile.write(json_object)

# Menyimpan Data dalam Format CSV
with open("data.csv", mode='w', newline='', encoding='utf-8') as file:  # Membuka file data.csv untuk menulis data.
    writer = csv.DictWriter(file, fieldnames=['text', 'sentiment'])  # Membuat writer untuk menulis dictionary ke CSV
    writer.writeheader()  # Menulis header ke file CSV.
    writer.writerows(data)  # Menulis semua baris dari list data ke file CSV

file.close()