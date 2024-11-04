import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from spellchecker import SpellChecker
from langdetect import detect

# NLTK durak kelimelerini indir
nltk.download('stopwords')

# Dil filtresi
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

# Fiil içerikli veya tek kelimelik yorumları temizleyen fonksiyon
def remove_unwanted_verbs(text):
    verbs_to_exclude = {"must", "need", "try", "want"}
    words = text.split()
    return len(words) > 1 and not all(word in verbs_to_exclude for word in words)

# Yazım düzeltme fonksiyonu
def correct_spelling(text):
    spell = SpellChecker()
    words = text.split()
    corrected_words = []

    for word in words:
        candidates = spell.candidates(word)
        if candidates:
            corrected_words.append(next(iter(candidates)))
        else:
            corrected_words.append(word)

    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Veri yükleme ve ön işleme
def load_and_preprocess_data(file_path):
    # Veriyi oku
    df = pd.read_csv(file_path)
    
    # İngilizce dışındaki yorumları kaldır
    df = df[df['content'].apply(is_english)]
    
    # Tek kelimelik veya yalnızca fiil içeren yorumları çıkar
    df = df[df['content'].apply(remove_unwanted_verbs)]
    
    # Yazım hatalarını düzelt ve metinleri temizle
    df['content'] = df['content'].apply(correct_spelling)
    df['content'] = df['content'].apply(preprocess_text)
    
    print("Temizlenmiş veri sayısı:", len(df))  # Temizlenmiş veri sayısını yazdır
    
    # Temizlenmiş veriyi yeni bir CSV dosyasına kaydet
    df.to_csv('data/cleaned_whatsapp_reviews.csv', index=False)

    return df
