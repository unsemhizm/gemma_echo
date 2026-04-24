import requests
import time
import os

def clear_ollama_vram():
    print("[TEST] Ollama VRAM tahliye isteği gönderiliyor...")
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma4-echo", # Model adının tam doğru olduğundan emin ol
        "keep_alive": 0
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"[BAŞARILI] İstek gönderildi. Süre: {time.time() - start_time:.2f}sn")
            print("[BİLGİ] Şimdi nvidia-smi ile VRAM'i kontrol et. Gemma silinmiş olmalı.")
        else:
            print(f"[HATA] Ollama hata döndürdü: {response.status_code}")
    except Exception as e:
        print(f"[HATA] Bağlantı kurulamadı: {e}")

if __name__ == "__main__":
    clear_ollama_vram()