# 🗣️ Gemma Echo 🌍

**Akıllı model orkestrasyonu ve ses klonlama ile gerçek zamanlı Türkçe–İngilizce sesli çeviri.**

Gemma Echo, konuşulan Türkçeyi yazıya döken, kendi kendini onaran bir model kademesi (cascade) ile İngilizceye çeviren ve sonucu konuşmacının klonlanmış sesiyle geri seslendiren bir masaüstü yapay zekâ asistanıdır — tüm bunları tüketici donanımı üzerinde, gerektiğinde tamamen çevrimdışı olarak yapar.

> 🇹🇷 Bu dosya, [README.md](README.md) dosyasının Türkçe çevirisidir. İngilizce sürüm ana referanstır; bir uyumsuzluk olursa İngilizce sürüm geçerlidir.

---

## 🎬 Tanıtım Videosu

[![Gemma Echo Tanıtım](https://img.youtube.com/vi/1FU11A3G6ig/maxresdefault.jpg)](https://www.youtube.com/watch?v=1FU11A3G6ig)

▶️ **[YouTube'da İzle](https://www.youtube.com/watch?v=1FU11A3G6ig)**

> 🧠 ** not:** Tanıtım videosu Türkçe çekildi. **Videoda duyduğunuz İngilizce dublaj ve gördüğünüz sinematik altyazılar tamamen Gemma Echo'nun kendisi tarafından**, tüketici donanımı üzerinde yerel olarak üretildi. Proje kelimenin tam anlamıyla kendi kendini sunuyor.

---

## 💻 Platform Desteği

> ⚠️ **Test edilen platform: Windows 11 + NVIDIA GPU (CUDA 13).**
>
> Şu an için **resmî olarak desteklenen tek konfigürasyon** budur. Proje, uçtan uca yalnızca bu yığın üzerinde geliştirildi ve doğrulandı.
>
> Linux ve macOS **şu anda test edilmemiştir** ve manuel uyarlama gerektirebilir:
>
> - **Linux + NVIDIA:** `libportaudio2` / `libasound2-dev` kurulduktan ve uygun `torch` CUDA wheel'i seçildikten sonra büyük olasılıkla çalışır; WASAPI loopback kaydedici (sistem sesi yakalama) yalnızca Windows'a özgüdür.
> - **macOS (Apple Silicon):** çıkarım cihazını `cuda`'dan `mps`'e geçirmeyi, `llama-cpp-python`'u `CMAKE_ARGS="-DLLAMA_METAL=on"` ile derlemeyi ve Tcl/Tk kurmayı (`brew install python-tk`) gerektirir; yazar tarafından test edilmemiştir.
> - **macOS (Intel) / NVIDIA'sız Linux:** Yalnızca CPU modu mümkündür ama yavaştır; `requirements.txt` içindeki `torch==2.11.0+cu130` uygun bir CUDA-dışı wheel ile değiştirilmelidir.
>
> Test edilmiş çapraz platform desteği ekleyen pull request'ler memnuniyetle karşılanır.

---

## ✨ Özellikler ve İş Akışları

Gemma Echo, çok modlu bir çeviri paketidir. Aşağıda açıklanan kademeli yapı, ana arayüzden erişilebilen **beş** farklı iş akışını besler:

| Mod                    | Girdi                                                          | Çıktı                                                     | Kullanım Alanı                                 |
| ---------------------- | -------------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------- |
| 🎙️ **Canlı**           | Mikrofon (push-to-talk veya VAD) veya sistem loopback (WASAPI) | Akış halinde metin + klonlanmış ses                       | Gerçek zamanlı sohbet, toplantı, canlı görüşme |
| 🎬 **Medya — Dublaj**  | Video dosyası (MP4, MKV, MOV, AVI, WebM)                       | Konuşmacının klonlanmış sesiyle dublajlı video            | Türkçe videoları İngilizce yeniden seslendirme |
| 📝 **Medya — Altyazı** | Video dosyası                                                  | Yumuşak `.srt` izi veya sert (yakılmış) sinematik altyazı | YouTube yükleme, erişilebilirlik, teslimat     |
| 📄 **Kitap / Belge**   | PDF, DOCX, TXT                                                 | Çevrilmiş `.txt` (opsiyonel olarak biçim koruyan `.docx`) | Akademik makaleler, kitaplar, uzun belgeler    |
| 📁 **Dosya / Metin**   | Ses/video dosyası veya yapıştırılan metin                      | Çevrilmiş transkript                                      | Toplu transkripsiyon, hızlı metin çevirisi     |

Beş mod da aynı kendi kendini onaran çeviri kademesini (Kültürel Harita → Gemma 4 Bulut → Gemini 2.5 Flash → Gemma 4 Q4 yerel) paylaşır ve bulut ile çevrimdışı çalışma arasında saydam biçimde geçiş yapar.

---

## 🏗️ Sistem Mimarisi

```
Mikrofon / Video Dosyası
        |
        v
[ faster-whisper STT ]   <-- VAD kontrollü, 16kHz mono, language=tr
        |
        v
[ Kültürel Harita ]       <-- 7 dilde 130 deyim (TR/AR/DE/ES/FR/IT/JA), gecikmesiz tam eşleşme
        |
        | (eşleşme yok)
        v
[ Gemma 4 26B — Gemini API ]   <-- Birincil: kalite öncelikli
        |
        | dinamik zaman aşımı (40-120sn) veya API hatası
        v
[ Gemini 2.5 Flash — Gemini API ] <-- Hız yedeği
        |
        | tüm bulut yolları başarısız / çevrimdışı mod
        v
[ Gemma 4 Q4 GGUF (Yerel Çıkarım Motoru) ]  <-- Bağımlılıksız yerel temel
        |
        v
[ XTTS-v2 Ses Klonlama ]   <-- Konuşmacıyı referans wav dosyasından klonlar
        |
        v
Ses Çıkışı / Dublajlı Video
```

---

## 🧠 Kendi Kendini Onaran Model Kademesi

| Katman | Model                               | Sağlayıcı            | Tetikleyici                                     |
| ------ | ----------------------------------- | -------------------- | ----------------------------------------------- |
| 0      | Kültürel Harita (130 girdi × 7 dil) | Yerel                | Kaynak dilde deyim tespit edildiğinde           |
| 1      | Gemma 4 26B (`gemma-4-26b-a4b-it`)  | Gemini API           | Varsayılan çevrimiçi yol                        |
| 2      | Gemini 2.5 Flash                    | Gemini API           | Katman 1 zaman aşımı / hatası                   |
| 3      | Gemma 4 Q4 GGUF                     | Yerel Çıkarım Motoru | Çevrimdışı mod / tüm bulut katmanları başarısız |

Kademe **kendi kendini onarır**: herhangi bir katman sessizce başarısız olabilir. Bir sonraki katman milisaniyeler içinde otomatik devreye girer. Pratikte sistem neredeyse her zaman Katman 1 veya 2'de sonuçlanır; Katman 3, sistemin **hiç internet olmasa bile** asla çökmemesi için vardır.

### 🤝 Neden Gemma 4 İki Uçta Birden?

Gemma 4, tasarım gereği hem Katman 1'de (bulut, 26B tam hassasiyet) hem de Katman 3'te (yerel, Q4 kuantize) yer alır:

- **Katman 1 (bulut):** Gerçek zamanlı konuşma için maksimum çeviri kalitesi. 26B parametreli model; karmaşık Türkçe gramerini, zamir çözümünü ve alana özgü kelime dağarcığını işler.
- **Katman 3 (yerel):** Gizliliği koruyan, ücretsiz, ağdan bağımsız yedek. Kotalar bittiğinde, internet kesildiğinde veya veri cihazda kalması gerektiğinde `gemma-4-q4.gguf` sıfır konfigürasyon değişikliğiyle devreye girer — yeniden başlatma yok, kullanıcı müdahalesi yok.

Bu, **kalite açısından simetrik, tamamen Gemma-yerli** bir kademe yaratır: sistemdeki her çeviri katmanı bir Google Gemma modelidir.

---

## ⚡ VRAM Optimizasyonu ve Performans

Tüketici GPU'ları (8–12 GB VRAM) tüm modelleri aynı anda tutamaz. Gemma Echo bunu mümkün kılmak için üç strateji kullanır:

### 🐢 1. Tembel Yükleme (Lazy Loading)

Modeller başlangıçta yüklenmez. Yerel Gemma 4 Q4 yalnızca ilk ihtiyaç duyulduğunda (çevrimdışı mod veya video dublaj) yüklenir. XTTS-v2 yalnızca TTS modu çevrimdışı/GPU'ya alındığında yüklenir.

```python
# translator.py — ilk çevrimdışı istek geldiğinde load_local_model() çağrılır
# (Llama, llama-cpp-python'un giriş sınıfıdır; GGUF modelleri için
#  jenerik bir C++ çıkarım motorudur. Burada Google Gemma 4 ağırlık
#  dosyasını çalıştırmak için kullanılır.)
self.local_llm = Llama(model_path="./models/gemma-4-q4.gguf", n_gpu_layers=-1)
```

### 🥷 2. Arka Plan Ön Yükleme (Ambush Mode)

Kullanıcı çevrimiçi moddayken XTTS-v2, sessizce bir daemon thread üzerinde sistem RAM'ine ön yüklenir. Kullanıcı çevrimdışı moda geçtiğinde model zaten sıcaktır — algılanan gecikme sıfırdır.

```python
# synthesizer.py — çevrimiçi modda etkinken başlar
synthesizer.preload_xtts_background(use_gpu=False)
```

### 🔄 3. Önbellek Tahliyeli Sıcak Değişim (Hot-Swap)

GPU ve CPU modları arasında geçiş, yeni konfigürasyonu yüklemeden önce kontrollü bir VRAM tahliyesi tetikler ve CUDA OOM hatalarını önler.

```python
# synthesizer.py — offload_xtts()
del self.xtts_model
gc.collect()
torch.cuda.empty_cache()   # bir sonraki yüklemeden önce VRAM tamamen serbest
```

Üç stratejinin birleşimi: sistem tek bir RTX 3060 Ti üzerinde (8 GB VRAM) çevrimiçi modda gerçek zamanlı çeviriyi ve çevrimdışı modda tam video dublajı, hangi modelin VRAM'i işgal ettiğini sıcak değiştirerek yürütür.

---

## ⚙️ Arka Uç (Backend) Konfigürasyonu

Ayarlar sayfası STT × LLM × TTS matrisini sergiler. Her eksen bağımsız seçilebilir ve 36+ geçerli kombinasyon üretir. Aşağıdaki ön ayarlar en yaygın olanlardır; **`Özel (Custom)`** her STT motorunu, her çeviri arka ucunu ve her TTS çıkışını karıştırmanıza izin verir.

| Ön Ayar                       | STT                                                         | Çeviri                                                       | TTS                           | İnternet     |
| ----------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ----------------------------- | ------------ |
| 🟢 **Çevrimiçi (varsayılan)** | faster-whisper yerel-GPU                                    | Gemma 4 26B → Gemini 2.5 Flash                               | ElevenLabs Turbo              | Gerekli      |
| ☁️ **Bulut STT hızlandırıcı** | Groq Whisper-large-v3 _veya_ Deepgram Nova                  | Gemma 4 26B → Gemini 2.5 Flash                               | ElevenLabs Turbo              | Gerekli      |
| 🛡️ **Çevrimdışı (CPU)**       | faster-whisper CPU                                          | Gemma 4 Q4 GGUF (CPU)                                        | XTTS-v2 CPU                   | Gerekmez     |
| 🚀 **Çevrimdışı (GPU)**       | faster-whisper yerel-GPU                                    | Gemma 4 Q4 GGUF (GPU)                                        | XTTS-v2 GPU                   | Gerekmez     |
| ⚖️ **Hibrit (önerilen)**      | faster-whisper yerel-GPU                                    | Gemma 4 26B → Gemini 2.5 Flash → Gemma 4 Q4 (otomatik yedek) | XTTS-v2 GPU                   | İsteğe bağlı |
| 🎥 **Video Dublaj**           | faster-whisper medium (zaman damgalı) + Demucs vokal ayrımı | Gemma 4 26B → Gemini 2.5 Flash → Gemma 4 Q4                  | XTTS-v2 (ses klonu)           | İsteğe bağlı |
| 🧩 **Özel (Custom)**          | yukarıdakilerin herhangi biri                               | yukarıdakilerin herhangi biri                                | yukarıdakilerin herhangi biri | duruma bağlı |

---

## 🎬 Video Dublaj İş Hattı

Gemma Echo, Türkçe bir videoyu konuşmacının orijinal sesini koruyarak İngilizceye dublajlayabilir:

```
1. ffmpeg          Videodan 16kHz mono WAV çıkar
2. faster-whisper  Zaman damgalı transkripsiyon (segment başına başlangıç/bitiş/metin)
3. Gemma 4 Q4      Her segmenti yerel olarak çevir (API maliyeti yok, kota limiti yok)
4. XTTS-v2         En uzun temiz segmentten (≤8sn) konuşmacı referansı çıkar
5. XTTS-v2         get_conditioning_latents() bir kez hesapla, her segmentte tekrar kullan
6. XTTS-v2         segment başına inference() → klonlanmış sesle İngilizce konuşma
7. ffmpeg atempo   İngilizce sesi orijinal segment süresine zaman-uzat
8. ffmpeg          Yeni ses izini orijinal videoya muxla (stream copy, yeniden kodlama yok)
```

Çıktı: `<kaynak_video>_dubbed.mp4`

Dublaj iş hattı **yalnızca yerel modelleri** kullanır (Adım 3–6); bu da onu hassas içerikler ve uzun videolar için API maliyeti kaygısı olmadan uygun kılar.

### 📝 Altyazı (dublaja alternatif)

Aynı 1–3. adımlar (çıkar → transkribe et → çevir), ardından farklı bir bitiş:

```
4. SRT yazıcı       Cümle bilinçli bölme, cue başına maks. 42 karakter × 2 satır
5a. yumuşak mux     ffmpeg .srt'yi seçilebilir altyazı izi olarak kopyalar (yeniden kodlama yok)
5b. sert yakma      ffmpeg subtitles filtresi sinematik altyazıları video karelerine işler
                    (beyaz Arial bold, siyah outline, yumuşak gölge — Netflix tarzı;
                     opak arka plan kutusu yok)
```

Çıktı: `<kaynak_video>_subtitled.mp4` (yumuşak) veya `<kaynak_video>_burned.mp4` (sert).

---

## 📄 Belge Çeviri İş Hattı

Uzun biçimli çeviri (PDF / DOCX / TXT) canlı sohbet yolundan ayrı mühendislenmiştir. Naif yaklaşım — tüm belgeyi tek LLM çağrısına vermek — terminoloji tutarlılığında başarısız olur, bağlam pencerelerini aşar ve bölümler arası kayma üretir. Gemma Echo 8 aşamalı bir iş hattı kullanır:

```
1. pdfplumber/python-docx   Paragraf sınırlarını koruyarak metin çıkar
2. Kayan Pencere            Paragrafları ~400 sözcüklük öbeklere %15 örtüşmeyle grupla
3. Terim Çıkarma            İlk tarama özel isim, atıf ve alan terimlerini çıkarır
                            → belgeye özel bir sözlük oluşturur
4. Kültürel Harita          Deyimleri ve kalıp ifadeleri ön-çevir (sıfır LLM maliyeti)
5. Çeviri Kademesi          Her öbek: Gemma 4 Bulut → Gemini 2.5 Flash → Gemma 4 Q4
                            (öbek sınır kontrolünden geçemezse paragraf-bazlı yedek)
6. Yuvarlanan Özet          Her 5 öbekte bir, o ana kadar belgenin 2 cümlelik özetini
                            yeniden üret → sonraki öbeklere bağlam olarak besle
                            (uzun belge tutarlılığı, zamir çözümü)
7. Yeniden Birleştirme      Çevrilen öbekleri birleştir; örtüşmeyi tekilleştir
8. Çıkış Yazıcı             Düz `.txt` (her zaman) veya biçim koruyan `.docx`
                            (isteğe bağlı, paragraf yapısını korur)
```

Çıktı: `<kaynak>_<dil>.txt` ve/veya `<kaynak>_<dil>.docx`.

Sözlük (Aşama 3) ve yuvarlanan özet (Aşama 6), makine çevirisi külüstürü ile yayımlanabilir bir taslak arasındaki farktır. Bu iş hattı ile çevrilen 50 sayfalık bir makale, hiçbir insan ön işlemi olmadan uçtan uca tutarlı terminoloji korur.

---

## 🚀 Kurulum

### 📋 Gereksinimler

- Python 3.11 (3.11.x önerilir — Coqui XTTS bu hatta doğrulandı)
- CUDA 13.0 destekleyen NVIDIA GPU sürücüsü (Windows'ta sürücü 580+; PyTorch kendi CUDA çalışma zamanını getirir)
- PATH üzerinde ffmpeg ≥ 6.0 (`ffmpeg -version` çözülmeli)
- ~12 GB boş disk alanı (≈4 GB Gemma 4 GGUF, ≈2 GB XTTS-v2, geri kalanı venv için)

### 📦 Kur

```bash
git clone https://github.com/unsemhizm/gemma_echo.git
cd gemma_echo
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 🧠 Gemma 4 Yerel Modelini İndir

Gemma Echo, `models/gemma-4-q4.gguf` yolunda **GGUF-kuantize edilmiş bir Gemma 4** ağırlık dosyası bekler. Bu dosya **repo ile birlikte gelmez** (~4 GB ve Git LFS sınırlarının dışındadır) ve manuel olarak indirilmelidir.

#### 📥 Önerilen — Doğrudan Tarayıcıdan İndirme

Bu, Windows'ta en güvenli yoldur çünkü Python ortamınıza **dokunmaz** (yeni Hugging Face CLI'si `huggingface_hub>=1.0` gerektirir, bu da bu projenin çeviri stabilitesi için sabitlediği `transformers==4.38.2` / `tokenizers==0.15.2` sürümleriyle uyumsuzdur — hub kitaplığını yükseltmek yerel çıkarımı bozar).

1. [Hugging Face'te Gemma 4 26B GGUF](https://huggingface.co/google/gemma-4-26b-a4b-it-qat-q4_0-gguf) sayfasını aç.
2. Ücretsiz bir Hugging Face hesabıyla giriş yap ve bir kereliğine **Gemma Kullanım Şartlarını (Gemma Terms of Use)** kabul et.
3. `gemma-4-26b-a4b-it-q4_0.gguf` dosyasını (~4 GB) indir.
4. Projenin `models/` klasörüne yerleştir ve adını `gemma-4-q4.gguf` olarak değiştir.

#### 💻 İsteğe Bağlı — `hf` CLI (yalnızca ileri düzey kullanıcılar)

Eski `huggingface-cli download …` komutu `huggingface_hub` 1.x itibarıyla **kaldırılmıştır** (deprecated) ve yerini `hf` aldı. Eğer `hf` zaten `PATH`'inizdeyse ve geçerli bir token'ınız varsa (`hf auth login`), şunu yapabilirsiniz:

```bash
hf download google/gemma-4-26b-a4b-it-qat-q4_0-gguf gemma-4-26b-a4b-it-q4_0.gguf --local-dir ./models
# Sonra projenin beklediği yola yeniden adlandır:
ren .\models\gemma-4-26b-a4b-it-q4_0.gguf gemma-4-q4.gguf     # PowerShell / Windows
# mv  ./models/gemma-4-26b-a4b-it-q4_0.gguf ./models/gemma-4-q4.gguf   # macOS / Linux
```

> Bu projenin `venv`'i içinde **kesinlikle** `pip install -U "huggingface_hub[cli]"` çalıştırmayın. Bu komut `huggingface_hub`'ı sessizce 1.0'ın üzerine yükseltir ve `transformers 4.38.2` + `tokenizers 0.15.2`'yi bozar. Yanlışlıkla yaptıysanız sabitlenmiş sürümü şu komutla geri yükleyin:
>
> ```bash
> pip install huggingface_hub==0.36.2
> ```

#### 📂 Beklenen Son Yerleşim

```
models/
  gemma-4-q4.gguf      # ~4 GB, Q4_K_M kuantizasyonu (Google Gemma 4 26B)
```

Final dosya adı `gemma-4-q4.gguf` olduğu sürece Gemma 4'ün başka herhangi bir Q4 GGUF derlemesi de çalışır (bu, `llm/translator.py` içinde sabit kodlanmış yoldur).

**XTTS-v2**, Coqui `TTS` kitaplığı tarafından ilk çevrimdışı-TTS kullanımında otomatik olarak indirilir (~2 GB; `~/.local/share/tts/` veya Windows eşdeğerine). Manuel adım gerekmez.

> **Jüri / ilk kez kullanıcılar için not:** Bu adımı atlarsanız proje yine de açılır ve çevrimiçi çeviri (Gemini API) çalışır, ancak **Çevrimdışı mod, Hibrit otomatik yedek ve Video Dublaj**, loglarda `models/gemma-4-q4.gguf not found` hatasıyla başarısız olur.

### 🔑 Ortam Değişkenleri

Proje kök dizininde bir `.env` dosyası oluşturun:

```
GEMINI_API_KEY=your_gemini_api_key            # Çeviri (Gemma 4 + Gemini Flash)
ELEVENLABS_API_KEY=your_elevenlabs_api_key    # Yüksek kaliteli çevrimiçi TTS (isteğe bağlı)
GROQ_API_KEY=your_groq_api_key                # Bulut Whisper-large-v3 STT hızlandırma (isteğe bağlı)
DEEPGRAM_API_KEY=your_deepgram_api_key        # Bulut STT yedeği (isteğe bağlı)
```

Uygulama, bu anahtarların hiçbirine ihtiyaç duymadan tamamen çevrimdışı çalışır (yalnızca yerel Gemma 4 Q4 + faster-whisper + XTTS-v2). Groq ve Deepgram anahtarları yalnızca düşük VRAM'li makinelerde bulut konuşma-metin dönüşümünü hızlandırır; **çeviri için kullanılmazlar**.

---

## 🎮 Kullanım

```bash
# Arayüzü başlat
python -m gui.app

# Veya doğrudan
python gui/app.py
```

---

## 🛠️ Teknoloji Yığını

| Bileşen                                | Kitaplık                                            |
| -------------------------------------- | --------------------------------------------------- |
| Arayüz                                 | CustomTkinter                                       |
| STT (yerel)                            | faster-whisper (CTranslate2 arka ucu)               |
| STT (bulut hızlandırıcı, isteğe bağlı) | Groq Whisper-large-v3, Deepgram Nova                |
| VAD (ses etkinliği algılama)           | webrtcvad                                           |
| Çeviri (bulut)                         | Google Gemini API — Gemma 4 26B → Gemini 2.5 Flash  |
| Çeviri (yerel)                         | `llama-cpp-python` üzerinden Gemma 4 Q4 GGUF        |
| TTS (çevrimiçi)                        | ElevenLabs                                          |
| TTS (çevrimdışı / ses klonlama)        | Coqui XTTS-v2 †                                     |
| Vokal/enstrüman ayrımı (dublaj)        | Demucs htdemucs (Meta, MIT)                         |
| Belge ayrıştırma (kitap çevirisi)      | pdfplumber, python-docx                             |
| Ses G/Ç                                | sounddevice, soundfile, soundcard (WASAPI loopback) |
| Video işleme                           | ffmpeg (CLI alt süreç)                              |

> **† TTS Motoru Lisans Uyarısı.** Gemma Echo'nun çekirdek orkestrasyon çerçevesi Apache 2.0 altında lisanslanmıştır. Ancak **varsayılan** çevrimdışı TTS motoru (Coqui XTTS-v2) **Coqui Public Model License (Ticari Olmayan)** altında lisanslanan model ağırlıklarını kullanır. Gemma Echo, herhangi bir TTS motorunu entegre edebilecek mimariyi sağlar. Ticari dağıtım için kullanıcılar XTTS-v2 ağırlıklarını ticari olarak izin verilebilir bir alternatifle (örn. VITS, Piper) değiştirmeli ya da Coqui GmbH'den ticari lisans almalıdır. Gemma Echo'nun kendi Apache 2.0 lisansı bundan etkilenmez.

---

## 🗺️ Yol Haritası / Deneysel Şerit

[`experimental/`](experimental/) dizini, **mevcut yarışma teslimine dahil olmayan** ama projenin lansman sonrası kişiselleştirme şeridini belgeleyen araştırma-geliştirme çalışmalarını içerir:

- **Mod-bilinçli Japonca ince ayar (fine-tune)** — Gemini API ile üretilen 500 örneklik sentetik veri seti (5 stilistik mod: acil durum, resmî, yayıncı, gündelik, edebî) ve Kaggle T4 not defterinde çalıştırılmaya hazır eksiksiz bir Unsloth + LoRA eğitim betiği. Eğitim henüz çalıştırılmadı; veri seti ve iş hattı çoğaltılabilir bir taslak olarak sunulmuştur. Bkz. [`experimental/kaggle_finetune/README.md`](experimental/kaggle_finetune/README.md).

Bu malzemeler, gelecek sürümlerde Japonca desteği ve stil-bilinçli çeviri ekleme yönündeki mühendislik yönelimini gösterir ve projenin geri kalanı ile aynı Apache 2.0 lisansı altında yayımlanır.

---

## 👨‍💻 Geliştirici ve Lisans

**Yusuf Semih Öksüzoğlu** tarafından **Google Gemma AI Hackathon 2026** için geliştirildi.

📝 **Lisans:** Apache License 2.0 — bkz. [LICENSE](LICENSE).

Telif Hakkı 2026 Yusuf Semih Öksüzoğlu
