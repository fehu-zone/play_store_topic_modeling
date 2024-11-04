from src.data_fetcher import fetch_and_save_reviews
from src.data_preprocessing import load_and_preprocess_data
from src.visualization import visualize_topics, visualize_wordcloud
from src.ai_analysis import generate_insight  # AI tabanlı yorum fonksiyonunu ekledik
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os

# Örnek uygulama ID'si
app_id = "com.whatsapp"  # Kendi uygulama ID'ni buraya ekleyebilirsin

# Veri yolu ayarı
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data/com.whatsapp_reviews.csv')

# Veri dosyasının varlığını kontrol et
if not os.path.exists(data_path):
    fetch_and_save_reviews(app_id=app_id, lang='en', country='us', count=300)
    print("Veri dosyası oluşturuldu.")
else:
    print("Veri dosyası zaten mevcut, yeniden oluşturulmadı.")

# Verileri işle
cleaned_data = load_and_preprocess_data(data_path)

# Kelime dağarcığını oluştur
dictionary = Dictionary([doc.split() for doc in cleaned_data['content']])  # 'content' sütunu metinleri içermelidir
corpus = [dictionary.doc2bow(doc.split()) for doc in cleaned_data['content']]

# LDA modelini oluştur
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Temaları görselleştirmek için çağırın
visualize_topics(lda_model, cleaned_data)  # LDA modelini ve veriyi burada geçiriyoruz

# Kelime bulutunu görselleştirmek için çağırın
visualize_wordcloud(cleaned_data)  # Temizlenmiş veriyi burada geçiriyoruz

# Her bir tema için yapay zeka tabanlı yorumları oluştur ve göster
for idx, topic in lda_model.show_topics(formatted=False):
    topic_terms = topic  # Kelimeler ve önem dereceleri
    insight = generate_insight(topic_terms)  # Her tema için AI tabanlı yorum oluştur
    print(f"Tema {idx+1} Yorumu: {insight}")  # Yorumları yazdır
