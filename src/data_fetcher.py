from google_play_scraper import reviews
import pandas as pd
import os

def fetch_and_save_reviews(app_id, lang='en', country='us', count=200, save_path="data/"):
    """
    Google Play Store'dan belirtilen uygulama ID'sine göre yorumları çeker ve CSV olarak kaydeder.
    """
    # Yorumları çek
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        count=count
    )
    # DataFrame'e dönüştür
    df = pd.DataFrame(result)
    
    # Kaydetme klasörünü kontrol et
    os.makedirs(save_path, exist_ok=True)
    
    # Dosya yolu oluştur
    file_path = os.path.join(save_path, f"{app_id}_reviews.csv")
    df.to_csv(file_path, index=False)
    
    print(f"Yorumlar başarıyla kaydedildi: {file_path}")
