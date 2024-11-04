import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def perform_topic_modeling(df_cleaned):
    # TF-IDF vektörizasyonu
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_cleaned['content'])

    # LDA modeli
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)

    # Sonuçları inceleme
    for index, topic in enumerate(lda.components_):
        print(f"Topic {index + 1}:")
        print([tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
