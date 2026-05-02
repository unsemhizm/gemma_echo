import os
import sys
import wave
import collections
import sounddevice as sd
import webrtcvad


class Recorder:
    """
    VAD tabanli gercek zamanli mikrofon kaydedici.
    webrtcvad ile konusma algilama, sounddevice ile ses yakalama.

    Akis:
      1. 16kHz mono ses akisini 30ms'lik cerceveler halinde okur.
      2. Her cerceveyi webrtcvad'a gonderir (is_speech?).
      3. On-tampon (300ms) konusmalarla dolu olunca kaydi tetikler.
      4. Kayit sirasinda 900ms kesintisiz sessizlik algilayinca
         cumleyi tamamlanmis sayar, .wav'a yazar, orchestrator'a iletir.
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

        # Gecici WAV dosyasi
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _tmp_dir = os.path.join(_project_dir, ".tmp_audio")
        os.makedirs(_tmp_dir, exist_ok=True)
        self._tmp_wav = os.path.join(_tmp_dir, "live_recording.wav")

    # ═══════════════════════════════════════════════════════════
    # ANA DINLEME DONGUSU
    # ═══════════════════════════════════════════════════════════

    def run(self):
        """
        Engelleme (blocking) dinleme dongusu.
        Ctrl+C ile durdurulur.
        """
        print("\n" + "=" * 50)
        print("[RECORDER] CANLI MIKROFON MODU BASLATILDI")
        print(f"[RECORDER] VAD aggressiveness={self.aggressiveness} | Sessizlik esigi=900ms")
        print("[RECORDER] Konusmaya baslayin. Cikis: Ctrl+C")
        print("=" * 50 + "\n")

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

                while True:
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
                            print("[RECORDER] Cumle tamamlandi, isleniyor...")

                            wav_path = self._write_wav(voiced_frames)
                            self.orchestrator.process(wav_path)

                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[RECORDER] Dinleme durduruldu.")

    # ═══════════════════════════════════════════════════════════
    # YARDIMCI: WAV YAZICI
    # ═══════════════════════════════════════════════════════════

    def _write_wav(self, frames: list) -> str:
        """Ses karelerini (bytes listesi) 16kHz mono WAV olarak yazar.
        Yolu dondurur."""
        audio_bytes = b"".join(frames)
        with wave.open(self._tmp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16 = 2 byte
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return self._tmp_wav
