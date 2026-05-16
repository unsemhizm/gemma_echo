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
    VAD-driven real-time microphone capture.

    Uses webrtcvad for voice activity detection and sounddevice for audio I/O.

    Producer-Consumer architecture:
      Producer (microphone): continuously reads the 16 kHz mono input stream in
        30 ms frames, applies VAD to detect end-of-utterance, then writes a
        uniquely named WAV file and enqueues it. The microphone is never closed.
      Consumer (processing thread): pulls a WAV off the queue, runs the
        orchestrator.process() pipeline, and deletes the file. Runs in parallel
        with the producer.
    """

    SAMPLE_RATE = 16000
    FRAME_DURATION_MS = 30  # webrtcvad supports 10, 20 or 30 ms frames.

    def __init__(self, orchestrator, aggressiveness=None, transcriber=None, config=None, device=None):
        """
        Args:
            orchestrator: Orchestrator instance whose process() method is invoked.
            aggressiveness: webrtcvad noise tolerance (0 = low, 3 = high).
                            2 is balanced; noisy environments may prefer 3.
        """
        self.orchestrator = orchestrator
        self.transcriber = transcriber
        self.config = config
        self.device = device  # None = default microphone; int = loopback device index.

        # Read settings from config when not supplied via constructor arguments.
        vad_aggr = aggressiveness
        if vad_aggr is None and config:
            vad_aggr = config.get("recording", "vad_aggressiveness", default=2)
        self.aggressiveness = vad_aggr or 2

        self.streaming_enabled = True
        if config:
            self.streaming_enabled = config.get("recording", "streaming_enabled", default=True)

        # Frame size: 30 ms * 16000 Hz / 1000 = 480 samples (int16 → 960 bytes).
        self.frame_size = int(self.SAMPLE_RATE * self.FRAME_DURATION_MS / 1000)

        # Silence thresholds (expressed in frames).
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

        # Pre-trigger buffer: 300 ms (10 frames) — suppresses false-positive triggers.
        self._pre_trigger_size = 10

        self.vad = webrtcvad.Vad(self.aggressiveness)

        # The GUI "Stop" button raises this event.
        self._stop_event = threading.Event()

        # Async work queue (capped at 10 to avoid unbounded RAM growth).
        self.audio_queue = queue.Queue(maxsize=10)

        # Temp directory for the recorded WAV files (each gets a unique name).
        _project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._tmp_dir = os.path.join(_project_dir, ".tmp_audio")
        os.makedirs(self._tmp_dir, exist_ok=True)

    # ═══════════════════════════════════════════════════════════
    # PRIMARY LISTENING LOOP
    # ═══════════════════════════════════════════════════════════

    def run(self):
        """
        Producer-Consumer listening loop. Stopped with Ctrl+C.

        Producer (this method): never closes the microphone; enqueues a WAV
          file every time an utterance terminates.
        Consumer (_consumer thread): dequeues WAVs, runs the STT → LLM → TTS
          pipeline and deletes the file. Operates in parallel with the producer.
        """
        print("\n" + "=" * 50)
        print("[RECORDER] LIVE MICROPHONE MODE ACTIVE")
        print(f"[RECORDER] VAD aggressiveness={self.aggressiveness} | silence threshold=900 ms")
        print("[RECORDER] Start speaking. Exit: Ctrl+C")
        print("=" * 50 + "\n")

        # Start the consumer thread (daemon: terminates with the main program on Ctrl+C).
        threading.Thread(target=self._consumer, daemon=True).start()

        pre_trigger_buf = collections.deque(maxlen=self._pre_trigger_size)
        voiced_frames = []   # Frames being accumulated into the current utterance.
        triggered = False    # Is recording currently active?
        silent_count = 0     # Consecutive silent frames.

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
                        print("[WARN] Audio buffer overflow.")

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
                        # ── PRE-BUFFER: awaiting trigger ──────────────
                        pre_trigger_buf.append((frame_bytes, is_speech))
                        voiced_in_buf = sum(1 for _, s in pre_trigger_buf if s)

                        # Trigger once >80% of the pre-buffer is voiced.
                        if voiced_in_buf > 0.8 * pre_trigger_buf.maxlen:
                            triggered = True
                            silent_count = 0
                            # Carry the pre-buffer into the recording so we do not clip the onset.
                            voiced_frames = [f for f, _ in pre_trigger_buf]
                            pre_trigger_buf.clear()
                            print("[RECORDER] Speech detected — recording...")

                    else:
                        # ── RECORDING: silence counter ────────────────
                        voiced_frames.append(frame_bytes)

                        if is_speech:
                            silent_count = 0

                            # ── 0-A: STREAMING CHUNKER (punctuation-aware) ──
                            # Every 15 frames (~450 ms) run a quick check.
                            if self.streaming_enabled and self.transcriber and len(voiced_frames) % 15 == 0:
                                # Materialize the running buffer to a temporary WAV.
                                tmp_wav = self._write_wav(voiced_frames, suffix="_partial")
                                partial_text = self._quick_transcribe(tmp_wav)

                                if self._ends_with_punctuation(partial_text):
                                    print(f"[STREAMING] Punctuation detected: '{partial_text}'")
                                    try:
                                        self.audio_queue.put_nowait(tmp_wav)
                                        voiced_frames = []
                                        # Keep ``triggered`` True — the next sentence starts immediately.
                                    except queue.Full:
                                        print("[WARN] Streaming: queue full — punctuation chunk skipped.")
                                        try:
                                            os.remove(tmp_wav)
                                        except OSError:
                                            pass
                                else:
                                    # No punctuation yet — discard the temp file.
                                    try:
                                        os.remove(tmp_wav)
                                    except OSError:
                                        pass
                        else:
                            silent_count += 1

                        # ── 0-B: DYNAMIC THRESHOLD (sentence vs paragraph) ─────
                        # Terminate the recording once the silence threshold is crossed.
                        threshold = self._silence_threshold
                        if self.streaming_enabled:
                            # While streaming is enabled, a slightly more aggressive
                            # (shorter) silence threshold could apply, but here we
                            # use the paragraph threshold so multi-sentence flows
                            # remain supported.
                            threshold = self._para_threshold

                        if silent_count >= threshold:
                            triggered = False
                            silent_count = 0

                            if voiced_frames:
                                wav_path = self._write_wav(voiced_frames)
                                try:
                                    self.audio_queue.put_nowait(wav_path)
                                    print(f"[RECORDER] Final flush: {self.audio_queue.qsize()} pending.")
                                except queue.Full:
                                    print("[WARN] Queue full — skipped.")
                                    try:
                                        os.remove(wav_path)
                                    except OSError:
                                        pass

                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[RECORDER] Listening stopped.")

    def stop(self):
        """Stop the recorder and drain any pending work items."""
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
    # CONSUMER: ASYNC PROCESSING THREAD
    # ═══════════════════════════════════════════════════════════

    def _consumer(self):
        """Pull a WAV off the queue, run orchestrator.process(), delete the file.

        Runs fully in parallel with the producer — the microphone never closes.
        The loop is exception-tolerant: a thrown error never kills the thread.
        """
        while True:
            wav_path = self.audio_queue.get()
            try:
                self.orchestrator.process(wav_path)
            except Exception as e:
                print(f"[ERROR] Consumer thread processing error (recovered): {e}")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()

    # ═══════════════════════════════════════════════════════════
    # UTILITY: WAV WRITER
    # ═══════════════════════════════════════════════════════════

    def _write_wav(self, frames: list, suffix: str = "") -> str:
        """Write the supplied frames to a 16 kHz mono WAV under a unique name.

        Applies RMS-based gain normalization so quiet whispers and shouted speech
        end up at comparable levels for the STT model.
        """
        ts = int(time.time() * 1000); filename = f"rec_{ts}{suffix}.wav"
        wav_path = os.path.join(self._tmp_dir, filename)
        audio_bytes = b"".join(frames)

        # ── RMS normalization ─────────────────────────────────────
        # Convert raw int16 to float, compute RMS, scale to the target level, clip.
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        if rms > 50:   # Real speech (not pure silence / noise).
            gain = min(3000.0 / rms, 10.0)   # Target RMS=3000, max 10x gain.
            samples = np.clip(samples * gain, -32767, 32767)
        audio_bytes = samples.astype(np.int16).tobytes()
        # ─────────────────────────────────────────────────────────

        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # int16 = 2 bytes per sample.
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        return wav_path

    # ===========================================================
    # UTILITY: STREAMING & FILE HELPERS
    # ===========================================================

    def _quick_transcribe(self, wav_path: str) -> str:
        """Fast / low-quality transcription used during the streaming chunker."""
        if not self.transcriber:
            return ""

        try:
            # beam_size=1: fastest result (quality is secondary here).
            src_lang = "tr"
            if self.config:
                src_lang = self.config.get("language", "source", default="tr")

            res = self.transcriber._transcribe_local(wav_path, source_lang=src_lang)
            return res.get("text", "").strip()
        except Exception as e:
            print(f"[WARN] Quick transcribe error: {e}")
            return ""

    def _ends_with_punctuation(self, text: str) -> bool:
        """Return True when the text ends with '.', '?' or '!'."""
        if not text:
            return False
        clean = text.strip()
        if not clean:
            return False
        return clean[-1] in ".!?"


# ══════════════════════════════════════════════════════════════════════════════
# LOOPBACK RECORDER — soundcard-based WASAPI loopback
# ══════════════════════════════════════════════════════════════════════════════

class LoopbackRecorder:
    """
    WASAPI-loopback recorder — captures the counterpart's audio (e.g., the
    Zoom / Meet output stream).

    Uses the soundcard library; shares the VAD + queue architecture with the
    sounddevice-based Recorder.

    Usage:
        recorder = LoopbackRecorder(orchestrator, config=cfg)
        threading.Thread(target=recorder.run, daemon=True).start()
        # to stop:
        recorder._stop_event.set()
    """

    SAMPLE_RATE       = 16000
    FRAME_DURATION_MS = 30
    BLOCKSIZE         = 4800   # ~300 ms

    def __init__(self, orchestrator, config=None, aggressiveness=None, device_name: str = None):
        """
        Args:
            orchestrator  : Object whose process() method will be invoked.
            config        : ConfigManager — supplies runtime settings.
            aggressiveness: webrtcvad noise tolerance (0-3).
            device_name   : Loopback device name (None = first available loopback device).
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

    # ── Primary loop ─────────────────────────────────────────────────────────

    def run(self):
        """Capture audio from the WASAPI loopback and enqueue each detected utterance."""
        try:
            import soundcard as sc
        except ImportError:
            print("[LOOPBACK] ERROR: 'soundcard' library missing. Run 'pip install soundcard'.")
            return

        # Locate a loopback device.
        loopback_mics = [m for m in sc.all_microphones(include_loopback=True) if m.isloopback]
        if not loopback_mics:
            print("[LOOPBACK] ERROR: no WASAPI loopback device available.")
            return

        device = loopback_mics[0]
        if self.device_name:
            matches = [m for m in loopback_mics if self.device_name.lower() in m.name.lower()]
            if matches:
                device = matches[0]

        print(f"\n[LOOPBACK] Device: {device.name}")
        print("[LOOPBACK] Listening to counterpart audio. Stop with Ctrl+C.")

        threading.Thread(target=self._consumer, daemon=True).start()

        pre_trigger_buf = collections.deque(maxlen=self._pre_trigger_size)
        voiced_frames   = []
        triggered       = False
        silent_count    = 0

        try:
            with device.recorder(samplerate=self.SAMPLE_RATE, channels=1,
                                  blocksize=self.BLOCKSIZE) as rec:
                while not self._stop_event.is_set():
                    # soundcard returns float32 → convert to int16.
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
                            print("[LOOPBACK] Speech detected...")
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
                                    print("[LOOPBACK] Queue full — skipped.")
                                    try:
                                        os.remove(wav_path)
                                    except OSError:
                                        pass
                            voiced_frames = []
                            pre_trigger_buf.clear()

        except KeyboardInterrupt:
            print("\n[LOOPBACK] Stopped.")
        except Exception as e:
            print(f"[LOOPBACK] Error: {e}")

    def stop(self):
        """Stop the recorder and drain any pending work items."""
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

    # ── Consumer ────────────────────────────────────────────────────────────

    def _consumer(self):
        while True:
            wav_path = self.audio_queue.get()
            try:
                self.orchestrator.process(wav_path)
            except Exception as e:
                print(f"[LOOPBACK] Consumer error: {e}")
            finally:
                try:
                    os.remove(wav_path)
                except OSError:
                    pass
                self.audio_queue.task_done()

    # ── WAV writer ──────────────────────────────────────────────────────────

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
