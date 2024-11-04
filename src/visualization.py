import pandas as pd
import gensim
from gensim import corpora
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def visualize_topics(lda_model, df_cleaned):
    # Temalar ve kelimeleri göster
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    # Temaları görselleştirme
    fig, ax = plt.subplots(figsize=(10, 6))
    term_labels = [lda_model.show_topic(i, topn=5) for i in range(lda_model.num_topics)]
    ax.barh(range(len(term_labels)), [sum([term[1] for term in label]) for label in term_labels])
    ax.set_yticks(range(len(term_labels)))
    ax.set_yticklabels([f'Topic {i+1}' for i in range(len(term_labels))])
    plt.xlabel('Term Frequency')
    plt.title('Topical Distribution')
    plt.show()

def visualize_wordcloud(df_cleaned):
    # Metinleri birleştir
    text = ' '.join(df_cleaned['content'])

    # WordCloud oluştur
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)

    # Görselleştirme
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Eksenleri kapat

    # Başlık ve açıklama ekle
    plt.title('En Sık Kullanılan Kelimeler', fontsize=10)
    plt.suptitle('Kelime büyüklüğü kullanım sıklığına göre belirlenmiştir', fontsize=5, y=0.93)

    plt.show()

    # En sık geçen kelimeleri sayısal olarak göster
    freq_dist = pd.Series(wordcloud.words_).reset_index()
    freq_dist.columns = ['Kelimeler', 'Frekans']
    print(freq_dist.sort_values(by='Frekans', ascending=False).head(10))
