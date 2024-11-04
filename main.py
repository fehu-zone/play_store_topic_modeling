from src.data_fetcher import fetch_and_save_reviews
from src.data_preprocessing import load_and_preprocess_data
from src.visualization import visualize_topics
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import os

# Örnek uygulama ID'si
app_id = "com.whatsapp"  # Kendi uygulama ID'ni buraya ekleyebilirsin

# Yorumları çek ve kaydet
fetch_and_save_reviews(app_id=app_id, lang='en', country='us', count=300)

# Veri yolu ayarı
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data/com.whatsapp_reviews.csv')

# Verileri işle
cleaned_data = load_and_preprocess_data(data_path)

# Kelime dağarcığını oluştur
dictionary = Dictionary([doc.split() for doc in cleaned_data['content']])  # 'content' sütunu metinleri içermelidir
corpus = [dictionary.doc2bow(doc.split()) for doc in cleaned_data['content']]

# LDA modelini oluştur
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)

# Temaları görselleştirmek için çağırın
visualize_topics(lda_model, cleaned_data)  # LDA modelini ve veriyi burada geçiriyoruz
