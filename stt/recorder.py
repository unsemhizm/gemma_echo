import os
import sys
import wave
import queue
import threading
import time
import collections
import sounddevice as sd
import webrtcvad
import numpy as np


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

    def __init__(self, orchestrator, aggressiveness=None, transcriber=None, config=None, device=None):
        """
        Args:
            orchestrator: Orchestrator ornegi — process() metodu cagrilir.
            aggressiveness: webrtcvad gurultu direnci (0=dusuk, 3=yuksek).
                            2: dengeli; gurultulu ortam icin 3 tercih edilir.
        """
        self.orchestrator = orchestrator
        self.transcriber = transcriber
        self.config = config
        self.device = device  # None = varsayilan mikrofon, int = loopback device index
        
        # Ayarlari config'den veya parametreden oku
        vad_aggr = aggressiveness
        if vad_aggr is None and config:
            vad_aggr = config.get("recording", "vad_aggressiveness", default=2)
        self.aggressiveness = vad_aggr or 2

        self.streaming_enabled = True
        if config:
            self.streaming_enabled = config.get("recording", "streaming_enabled", default=True)

        # Kare boyutu: 30ms * 16000Hz / 1000 = 480 ornek (int16 -> 960 byte)
        self.frame_size = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000)

        # Sessizlik esikleri (kare cinsinden)
        silence_ms = 900
        if config:
            silence_ms = config.get("recording", "silence_ms", default=900)
        self._silence_threshold = silence_ms // self.FRAME_DURATION_MS

        para_ms = 2000
        if config:
            para_ms = config.get("recording", "paragraph_silence_ms", default=2000)
        self._para_threshold = para_ms // self.FRAME_DURATION_MS

        self.vad_enabled = True
        self.hotkey_outbound = "space"
        if config:
            self.vad_enabled = config.get("recording", "vad_enabled", default=True)
            self.hotkey_outbound = config.get("inbound", "hotkey_outbound", default="space")

        # On-tetik tamponu: 300ms (10 kare) — yalanci tetiklemeleri onler
        self._pre_trigger_size = 10

        self.vad = webrtcvad.Vad(self.aggressiveness)

        # GUI "Durdur" butonu bu event'i set eder
        self._stop_event = threading.Event()

        # Asenkron islem kuyrugu (maks 10 eleman — RAM tasmasini onler)
        self.audio_queue = queue.Queue(maxsize=10)

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
            stream_kwargs = dict(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=self.frame_size,
            )
            if self.device is not None:
                stream_kwargs["device"] = self.device

            with sd.RawInputStream(**stream_kwargs) as stream:

                while not self._stop_event.is_set():
                    raw, overflowed = stream.read(self.frame_size)
                    if overflowed:
                        print("[UYARI] Ses tamponu tasti.")

                    frame_bytes = bytes(raw)
                    if self.vad_enabled:
                        is_speech = self.vad.is_speech(frame_bytes, self.SAMPLE_RATE)
                    else:
                        try:
                            import keyboard
                            is_speech = keyboard.is_pressed(self.hotkey_outbound)
                        except:
                            is_speech = False

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
                            
                            # ── 0-A: STREAMING CHUNKER (Noktalama Bazli) ──
                            # Her 15 karede bir (450ms) hizli kontrol et
                            if self.streaming_enabled and self.transcriber and len(voiced_frames) % 15 == 0:
                                # Mevcut birikimi gecici WAV'a yaz
                                tmp_wav = self._write_wav(voiced_frames, suffix="_partial")
                                partial_text = self._quick_transcribe(tmp_wav)
                                
                                if self._ends_with_punctuation(partial_text):
                                    print(f"[STREAMING] Noktalama algilandi: '{partial_text}'")
                                    try:
                                        self.audio_queue.put_nowait(tmp_wav)
                                        voiced_frames = []
                                        # triggered = True kalmaya devam eder, yeni cumle baslar
                                    except queue.Full:
                                        print("[UYARI] Streaming: Kuyruk dolu, noktalama chunk atlandi.")
                                        try:
                                            os.remove(tmp_wav)
                                        except OSError:
                                            pass
                                else:
                                    # Noktalama yoksa gecici dosyayi sil
                                    try:
                                        os.remove(tmp_wav)
                                    except OSError:
                                        pass
                        else:
                            silent_count += 1

                        # ── 0-B: DINAMIK ESIK (Cumle vs Paragraf) ─────
                        # Sessizlik esigi asildiginda kaydi sonlandir
                        threshold = self._silence_threshold
                        if self.streaming_enabled:
                            # Streaming aktifse sessizlik esigi daha agresif (kisa) olabilir
                            # Cunku noktalama gelmezse bile 900ms beklemek cok uzun.
                            # Ama burada paragraf modunu desteklemek icin para_threshold kullanalim.
                            threshold = self._para_threshold
                        
                        if silent_count >= threshold:
                            triggered = False
                            silent_count = 0

                            if voiced_frames:
                                wav_path = self._write_wav(voiced_frames)
                                try:
                                    self.audio_queue.put_nowait(wav_path)
                                    print(f"[RECORDER] Final flush: {self.audio_queue.qsize()} bekleyen.")
                                except queue.Full:
                                    print("[UYARI] Kuyruk dolu! Atlandi.")
                                    try:
                                        os.remove(wav_path)
                                    except OSError:
                                        pass

                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[RECORDER] Dinleme durduruldu.")

    def stop(self):
        """Kaydı durdurur ve bekleyen işleri temizler."""
        self._stop_event.set()
        while not self.audio_queue.empty():
            try:
                wav_path = self.audio_queue.get_nowait()
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()
            except:
                pass


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

    def _write_wav(self, frames: list, suffix: str = "") -> str:
        """Ses karelerini 16kHz mono WAV olarak benzersiz isimle yazar.
        RMS normalizasyon ile ses seviyesini dengeler (fisiltili/yuksek ses ortami)."""
        ts = int(time.time() * 1000); filename = f"rec_{ts}{suffix}.wav"
        wav_path = os.path.join(self._tmp_dir, filename)
        audio_bytes = b"".join(frames)

        # ── RMS Normalizasyon ─────────────────────────────────────
        # Ham sesi float'a cevir, RMS ol, hedef seviyeye cek, kırp
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        if rms > 50:   # gercek ses var (saf gurultu/sessizlik degil)
            gain = min(3000.0 / rms, 10.0)   # hedef RMS=3000, max 10x kazanim
            samples = np.clip(samples * gain, -32767, 32767)
        audio_bytes = samples.astype(np.int16).tobytes()
        # ─────────────────────────────────────────────────────────

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16 = 2 byte
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return wav_path

    # ===========================================================
    # YARDIMCI: STREAMING & DOSYA ISLEMLERI
    # ===========================================================

    def _quick_transcribe(self, wav_path: str) -> str:
        """Streaming icin hizli, dusuk kaliteli transkripsiyon."""
        if not self.transcriber:
            return ""
        
        try:
            # Beam size 1: En hizli sonuc (kalite ikincil)
            src_lang = "tr"
            if self.config:
                src_lang = self.config.get("language", "source", default="tr")
                
            res = self.transcriber._transcribe_local(wav_path, source_lang=src_lang)
            return res.get("text", "").strip()
        except Exception as e:
            print(f"[UYARI] Quick transcribe hatasi: {e}")
            return ""

    def _ends_with_punctuation(self, text: str) -> bool:
        """Metin nokta, soru isareti veya unlem ile bitiyor mu?"""
        if not text:
            return False
        clean = text.strip()
        if not clean:
            return False
        return clean[-1] in ".!?"


# ══════════════════════════════════════════════════════════════════════════════
# LOOPBACK RECORDER — soundcard ile WASAPI Loopback
# ══════════════════════════════════════════════════════════════════════════════

class LoopbackRecorder:
    """
    WASAPI Loopback kaydedici — karsı tarafin sesini (Zoom/Meet cikisi) yakalar.
    soundcard kutuphanesini kullanir; sounddevice ile ayni VAD + kuyruk mimarisi.

    Kullanim:
        recorder = LoopbackRecorder(orchestrator, config=cfg)
        threading.Thread(target=recorder.run, daemon=True).start()
        # durdurmak icin:
        recorder._stop_event.set()
    """

    SAMPLE_RATE       = 16000
    FRAME_DURATION_MS = 30
    BLOCKSIZE         = 4800   # ~300ms

    def __init__(self, orchestrator, config=None, aggressiveness=None, device_name: str = None):
        """
        Args:
            orchestrator  : process() metodu cagrilacak nesne.
            config        : ConfigManager — ayarlar buradan okunur.
            aggressiveness: webrtcvad gurultu direnci (0-3).
            device_name   : Loopback cihaz adi (None = ilk bulunan loopback).
        """
        self.orchestrator = orchestrator
        self.config       = config
        self.device_name  = device_name

        vad_aggr = aggressiveness
        if vad_aggr is None and config:
            vad_aggr = config.get("recording", "vad_aggressiveness", default=2)
        self.aggressiveness = vad_aggr or 2

        silence_ms = 900
        if config:
            silence_ms = config.get("recording", "silence_ms", default=900)
        self._silence_threshold = silence_ms // self.FRAME_DURATION_MS

        self._pre_trigger_size  = 10
        self._frame_size        = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000)

        self.vad_enabled = True
        self.hotkey_inbound = "alt"
        if config:
            self.vad_enabled = config.get("recording", "vad_enabled", default=True)
            self.hotkey_inbound = config.get("inbound", "hotkey_inbound", default="alt")

        self.vad              = webrtcvad.Vad(self.aggressiveness)
        self._stop_event      = threading.Event()
        self.audio_queue      = queue.Queue(maxsize=10)

        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmp_dir = os.path.join(_project_dir, ".tmp_audio")
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ── Ana Dongu ─────────────────────────────────────────────────────────────

    def run(self):
        """WASAPI Loopback'ten ses yakala, VAD ile cumle algilayinca kuyruga at."""
        try:
            import soundcard as sc
        except ImportError:
            print("[LOOPBACK] HATA: 'soundcard' kutuphanesi bulunamadi. 'pip install soundcard' calistir.")
            return

        # Loopback cihazi bul
        loopback_mics = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
        if not loopback_mics:
            print("[LOOPBACK] HATA: Hicbir WASAPI Loopback cihazi bulunamadi.")
            return

        device = loopback_mics[0]
        if self.device_name:
            matches = [m for m in loopback_mics if self.device_name.lower() in m.name.lower()]
            if matches:
                device = matches[0]

        print(f"\n[LOOPBACK] Cihaz: {device.name}")
        print("[LOOPBACK] Karsi taraf dinleniyor. Ctrl+C ile dur.")

        threading.Thread(target=self._consumer, daemon=True).start()

        pre_trigger_buf = collections.deque(maxlen=self._pre_trigger_size)
        voiced_frames   = []
        triggered       = False
        silent_count    = 0

        try:
            with device.recorder(samplerate=self.SAMPLE_RATE, channels=1,
                                  blocksize=self.BLOCKSIZE) as rec:
                while not self._stop_event.is_set():
                    # soundcard float32 dondurur → int16'ya cevir
                    data       = rec.record(numframes=self._frame_size)
                    samples_f  = data[:, 0] if data.ndim > 1 else data
                    samples_i  = np.clip(samples_f * 32767, -32768, 32767).astype(np.int16)
                    frame_bytes = samples_i.tobytes()

                    if self.vad_enabled:
                        is_speech = self.vad.is_speech(frame_bytes, self.SAMPLE_RATE)
                    else:
                        try:
                            import keyboard
                            is_speech = keyboard.is_pressed(self.hotkey_inbound)
                        except:
                            is_speech = False

                    if not triggered:
                        pre_trigger_buf.append((frame_bytes, is_speech))
                        voiced_in_buf = sum(1 for _, s in pre_trigger_buf if s)
                        if voiced_in_buf > 0.8 * pre_trigger_buf.maxlen:
                            triggered    = True
                            silent_count = 0
                            voiced_frames = [f for f, _ in pre_trigger_buf]
                            pre_trigger_buf.clear()
                            print("[LOOPBACK] Konusma algilandi...")
                    else:
                        voiced_frames.append(frame_bytes)
                        if is_speech:
                            silent_count = 0
                        else:
                            silent_count += 1

                        if silent_count >= self._silence_threshold:
                            triggered    = False
                            silent_count = 0
                            if voiced_frames:
                                wav_path = self._write_wav(voiced_frames)
                                try:
                                    self.audio_queue.put_nowait(wav_path)
                                except queue.Full:
                                    print("[LOOPBACK] Kuyruk dolu, atlandi.")
                                    try:
                                        os.remove(wav_path)
                                    except OSError:
                                        pass
                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[LOOPBACK] Durduruldu.")
        except Exception as e:
            print(f"[LOOPBACK] Hata: {e}")

    def stop(self):
        """Kaydı durdurur ve bekleyen işleri temizler."""
        self._stop_event.set()
        while not self.audio_queue.empty():
            try:
                wav_path = self.audio_queue.get_nowait()
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()
            except:
                pass

    # ── Consumer ─────────────────────────────────────────────────────────────

    def _consumer(self):
        while True:
            wav_path = self.audio_queue.get()
            try:
                self.orchestrator.process(wav_path)
            except Exception as e:
                print(f"[LOOPBACK] Consumer hatasi: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()

    # ── WAV Yazici ────────────────────────────────────────────────────────────

    def _write_wav(self, frames: list) -> str:
        ts        = int(time.time() * 1000)
        wav_path  = os.path.join(self._tmp_dir, f"loopback_{ts}.wav")
        audio_bytes = b"".join(frames)

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms     = np.sqrt(np.mean(samples ** 2))
        if rms > 50:
            gain    = min(3000.0 / rms, 10.0)
            samples = np.clip(samples * gain, -32767, 32767)
        audio_bytes = samples.astype(np.int16).tobytes()

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return wav_path
