# RAG Chatbot with OpenRouter

OpenRouter API kullanarak akıllı dokümantasyon arama sistemi.

## Özellikler

- **Akıllı Tool Kullanımı**: Model, sadece gerektiğinde doküman araması yapar
- **Genel Sohbet**: "Merhaba" gibi basit mesajlara direkt cevap verir
- **Streaming Yanıtlar**: Gerçek zamanlı yanıt akışı
- **Vector DB**: FAISS ile yerel dokümantasyon arama

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install flask openai langchain-ollama langchain-community faiss-cpu python-dotenv pypdf
```

2. `.env` dosyasına OpenRouter API anahtarınızı ekleyin:
```
OPENROUTER_API_KEY=your_actual_api_key
```

3. Vector veritabanını oluşturun (ilk çalıştırmada):
```bash
python RAG/services/rag_service.py
```

4. Uygulamayı başlatın:
```bash
python RAG/app.py
```

## Kullanım

- Tarayıcıdan `http://localhost:5000` adresine gidin
- "Merhaba" yazın → Doküman araması yapmadan cevap verir
- Teknik soru sorun → Otomatik olarak doküman araması yapar ve cevap verir
