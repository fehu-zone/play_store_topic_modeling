import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Temizlenmiş veriyi yükle
df_cleaned = pd.read_csv('data/cleaned_whatsapp_reviews.csv')

# Eksik değerleri kontrol et ve temizle
print("Eksik değerler:")
print(df_cleaned.isnull().sum())

# 'content' sütunundaki NaN değerlerini kaldır
df_cleaned = df_cleaned.dropna(subset=['content'])

# Veri kontrolü
print("Yüklenen veri sayısı:", len(df_cleaned))
print("İlk 5 veri:\n", df_cleaned.head())

# TF-IDF vektörizasyonu
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # En fazla 1000 kelime
tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['content'])

# LDA modeli
lda = LatentDirichletAllocation(n_components=5, random_state=42)  # 5 konu oluştur
lda.fit(tfidf_matrix)

# Sonuçları inceleme
for index, topic in enumerate(lda.components_):
    print(f"Topic {index + 1}:")
    print([tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
