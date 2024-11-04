import pandas as pd
import gensim
from gensim import corpora
import matplotlib.pyplot as plt

def visualize_topics(lda_model, df_cleaned):
    # Metinleri token'lara ayır
    texts = df_cleaned['content'].apply(lambda x: x.split()).tolist()

    # Kelime dağarcığı oluştur
    dictionary = corpora.Dictionary(texts)

    # Belge terim matrisini oluştur
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Temalar ve kelimeleri göster
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    # Temaları görselleştirme
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Y ekseni için konum belirleme
    y_positions = range(len(topics))
    
    # Her tema için en üstteki terimlerin frekanslarını ve etiketlerini çek
    topic_terms = [lda_model.get_topic_terms(i, topn=5) for i in range(lda_model.num_topics)]
    term_labels = [[dictionary[id] for id, _ in terms] for terms in topic_terms]
    frequencies = [[frequency for _, frequency in terms] for terms in topic_terms]
    
    # Çubuk grafik çizimi
    for i, (freq, labels) in enumerate(zip(frequencies, term_labels)):
        ax.barh([pos + i * 0.15 for pos in y_positions], freq, height=0.1, label=f'Topic {i+1}', alpha=0.7)
        for j, label in enumerate(labels):
            ax.text(freq[j] + 0.02, y_positions[j] + i * 0.15, label, va='center')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f'Topic {i+1}' for i in range(len(topics))])
    plt.xlabel('Term Frequency')
    plt.title('Topical Distribution')
    plt.legend()
    plt.show()
