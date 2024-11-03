from src.data_fetcher import fetch_and_save_reviews
from src.data_preprocessing import load_and_preprocess_data
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
