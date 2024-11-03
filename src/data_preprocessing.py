import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from spellchecker import SpellChecker

# NLTK durak kelimeleri indir
nltk.download('stopwords')

# Yazım düzeltme fonksiyonu
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []

    for word in words:
        # SpellChecker'dan önerilen kelimeleri al
        candidates = spell.candidates(word)
        if candidates:
            # İlk önerilen kelimeyi al (ya da orijinal kelimeyi döndür)
            corrected_words.append(next(iter(candidates)))
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Tek harfli ve iki harfli kelimeleri çıkarıp durak kelimeleri temizleyen fonksiyon
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

def load_and_preprocess_data(file_path):
    # Veriyi oku
    df = pd.read_csv(file_path)

    # Eksik değerlerin kontrolü
    print("Eksik değerler:\n", df.isnull().sum())

    # Temizleme işlemleri
    df_cleaned = df[['content']].dropna()  # Sadece içerik sütunu ve eksik verileri kaldır

    # Metin temizleme
    df_cleaned['content'] = df_cleaned['content'].str.lower()  # Küçük harfe çevir
    df_cleaned['content'] = df_cleaned['content'].str.replace(r'[^\w\s]', '', regex=True)  # Özel karakterleri kaldır
    df_cleaned['content'] = df_cleaned['content'].str.strip()  # Boşlukları kaldır

    # Yazım hatalarını düzelt ve metinleri ön işlemeden geçir
    df_cleaned['content'] = df_cleaned['content'].apply(correct_spelling)
    df_cleaned['content'] = df_cleaned['content'].apply(preprocess_text)

    print("Temizlenmiş veri:\n", df_cleaned.head())  # İlk 5 veriyi yazdır
    print("Temizlenmiş veri sayısı:", len(df_cleaned))  # Temizlenmiş veri sayısını yazdır

    # Temizlenmiş veriyi yeni bir CSV dosyasına kaydet
    df_cleaned.to_csv('data/cleaned_whatsapp_reviews.csv', index=False)

    return df_cleaned

# Eğer bu dosya doğrudan çalıştırılırsa, örnek dosya yolu ile fonksiyonu çağır
if __name__ == "__main__":
    cleaned_data = load_and_preprocess_data('data/com.whatsapp_reviews.csv')
