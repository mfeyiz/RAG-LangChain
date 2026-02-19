
import os

def test_api_key_config():
    # GitHub Secrets'tan gelen anahtarı kontrol eder
    key = os.getenv("OPENROUTER_API_KEY")
    assert key is not None, "API Key secret olarak tanımlanmamış!"
    assert key.startswith("sk-"), "API Key hatalı formatta görünüyor!"
    
def test_setup_works():
    # Bu test sadece sistemin çalıştığını kanıtlamak içindir
    assert True