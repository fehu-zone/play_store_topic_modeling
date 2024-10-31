import pandas as pd
from spellchecker import SpellChecker

def create_word_pool(file_path):
    # Veriyi oku
    df_cleaned = pd.read_csv(file_path)

    # Tüm içerikleri birleştir ve kelimelere ayır
    all_words = ' '.join(df_cleaned['content']).split()
    word_pool = set(all_words)  # Tekil kelimeleri al

    return word_pool

if __name__ == "__main__":
    # Temizlenmiş veri dosyasının yolunu belirtin
    file_path = 'data/cleaned_whatsapp_reviews.csv'  # Bu dosya adını kendi dosyanıza göre güncelleyin
    word_pool = create_word_pool(file_path)

    # Kelime havuzunu SpellChecker'a yükle
    spell = SpellChecker()
    spell.word_frequency.load_words(list(word_pool))

    # Kelime havuzunu bir dosyaya kaydet
    with open('data/word_pool.txt', 'w') as f:
        for word in word_pool:
            f.write(word + '\n')

    print("Kelime havuzu oluşturuldu ve 'word_pool.txt' dosyasına kaydedildi.")
