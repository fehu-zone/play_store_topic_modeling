# src/ai_analysis.py

from transformers import pipeline

# Yorum üretme fonksiyonu
def generate_insight(topic_terms):
    summarizer = pipeline("text2text-generation", model="t5-small")  # Örnek model
    text = f"Tespit edilen konular: {', '.join([term for term, _ in topic_terms])}"
    summary = summarizer(text, max_length=50, min_length=20, do_sample=False)
    return summary[0]['generated_text']
