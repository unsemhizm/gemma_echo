import os
import sys
import wave
import queue
import threading
import time
import collections
import sounddevice as sd
import webrtcvad


class Recorder:
    """
    VAD tabanli gercek zamanli mikrofon kaydedici.
    webrtcvad ile konusma algilama, sounddevice ile ses yakalama.

    Producer-Consumer Mimarisi:
      Producer (mikrofon): 16kHz mono ses akisini 30ms cerceveler halinde okur,
        VAD ile cumle bitisini algilayinca benzersiz isimli WAV yazar ve
        audio_queue'ya atar. Mikrofon hic kapanmaz.
      Consumer (islem thread): Kuyruktan WAV alir, orchestrator.process()
        cagirip dosyayi temizler. Producer ile paralel calisir.
    """

    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30  # webrtcvad: 10, 20 veya 30ms destekler

    def __init__(self, orchestrator, aggressiveness=2):
        """
        Args:
            orchestrator: Orchestrator ornegi — process() metodu cagrilir.
            aggressiveness: webrtcvad gurultu direnci (0=dusuk, 3=yuksek).
                            2: dengeli; gurultulu ortam icin 3 tercih edilir.
        """
        self.orchestrator = orchestrator
        self.aggressiveness = aggressiveness

        # Kare boyutu: 30ms * 16000Hz / 1000 = 480 ornek (int16 -> 960 byte)
        self.frame_size = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000)

        # 900ms sessizlik = kac kare?
        self._silence_threshold = 900 // self.FRAME_DURATION_MS  # 30 kare

        # On-tetik tamponu: 300ms (10 kare) — yalanci tetiklemeleri onler
        self._pre_trigger_size = 10

        self.vad = webrtcvad.Vad(aggressiveness)

        # GUI "Durdur" butonu bu event'i set eder
        self._stop_event = threading.Event()

        # Asenkron islem kuyrugu (maks 5 eleman — RAM tasmasini onler)
        self.audio_queue = queue.Queue(maxsize=5)

        # Gecici WAV dizini (her kayit benzersiz isim alir)
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmp_dir = os.path.join(_project_dir, ".tmp_audio")
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # ANA DINLEME DONGUSU
    # ═══════════════════════════════════════════════════════════

    def run(self):
        """
        Producer-Consumer dinleme dongusu. Ctrl+C ile durdurulur.

        Producer (bu metod): Mikrofonu hic kapamadan dinler, cumle bitince
          WAV'i kuyruga atar.
        Consumer (_consumer thread): Kuyruktan WAV ceker, STT->LLM->TTS
          hattini calistirip dosyayi temizler. Paralel calisir.
        """
        print("\n" + "=" * 50)
        print("[RECORDER] CANLI MIKROFON MODU BASLATILDI")
        print(f"[RECORDER] VAD aggressiveness={self.aggressiveness} | Sessizlik esigi=900ms")
        print("[RECORDER] Konusmaya baslayin. Cikis: Ctrl+C")
        print("=" * 50 + "\n")

        # Consumer thread'i baslat (daemon: Ctrl+C'de ana program kapaninca o da kapanir)
        threading.Thread(target=self._consumer, daemon=True).start()

        pre_trigger_buf = collections.deque(maxlen=self._pre_trigger_size)
        voiced_frames = []   # Kaydedilen ses kareleri
        triggered = False    # Kayit aktif mi?
        silent_count = 0     # Arka arkaya sessiz kare sayisi

        try:
            with sd.RawInputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.frame_size
            ) as stream:

                while not self._stop_event.is_set():
                    raw, overflowed = stream.read(self.frame_size)
                    if overflowed:
                        print("[UYARI] Ses tamponu tasti.")

                    frame_bytes = bytes(raw)
                    is_speech = self.vad.is_speech(frame_bytes, self.SAMPLE_RATE)

                    if not triggered:
                        # ── ON-TAMPON: tetiklenme bekleniyor ──────────
                        pre_trigger_buf.append((frame_bytes, is_speech))
                        voiced_in_buf = sum(1 for _, s in pre_trigger_buf if s)

                        # On-tamponda %80'den fazla konusma varsa tetikle
                        if voiced_in_buf > 0.8 * pre_trigger_buf.maxlen:
                            triggered = True
                            silent_count = 0
                            # On-tampondaki kareleri kayda dahil et (cumle baslangicini kesmemek icin)
                            voiced_frames = [f for f, _ in pre_trigger_buf]
                            pre_trigger_buf.clear()
                            print("[RECORDER] Konusma algilandi, kaydediliyor...")

                    else:
                        # ── KAYIT: sessizlik sayaci ───────────────────
                        voiced_frames.append(frame_bytes)

                        if is_speech:
                            silent_count = 0
                        else:
                            silent_count += 1

                        # 900ms kesintisiz sessizlik → cumle bitti
                        if silent_count >= self._silence_threshold:
                            triggered = False
                            silent_count = 0

                            wav_path = self._write_wav(voiced_frames)
                            try:
                                self.audio_queue.put_nowait(wav_path)
                                print(f"[RECORDER] Cumle kuyruga eklendi (bekleyen: {self.audio_queue.qsize()})")
                            except queue.Full:
                                print("[UYARI] Kuyruk dolu (Limit: 5)! Yeni ses atlandi.")
                                try:
                                    os.remove(wav_path)
                                except OSError:
                                    pass

                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[RECORDER] Dinleme durduruldu.")

    # ═══════════════════════════════════════════════════════════
    # CONSUMER: ASENKRON ISLEM THREAD'I
    # ═══════════════════════════════════════════════════════════

    def _consumer(self):
        """Kuyruktan WAV alir, orchestrator.process() cagirip dosyayi siler.
        Producer ile tam paralel calisir — mikrofon hic kapanmaz.
        Hata olsa bile thread olmez; dongu devam eder."""
        while True:
            wav_path = self.audio_queue.get()
            try:
                self.orchestrator.process(wav_path)
            except Exception as e:
                print(f"[HATA] Consumer thread islem hatasi (Atlatildi): {e}")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()

    # ═══════════════════════════════════════════════════════════
    # YARDIMCI: WAV YAZICI
    # ═══════════════════════════════════════════════════════════

    def _write_wav(self, frames: list) -> str:
        """Ses karelerini 16kHz mono WAV olarak benzersiz isimle yazar.
        Millisaniye timestamp ile isim uretir — dosya üzerine yazma riski yok."""
        filename = f"rec_{int(time.time() * 1000)}.wav"
        wav_path = os.path.join(self._tmp_dir, filename)
        audio_bytes = b"".join(frames)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16 = 2 byte
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return wav_path
