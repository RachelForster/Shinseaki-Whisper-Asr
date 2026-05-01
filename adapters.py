"""Whisper-family ASR backends (plugin): faster-whisper + RealtimeSTT."""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Optional

from sdk.adapters.asr import ASRAdapter, TranscriptionCallback

from asr.asr_adapter import get_asr_log

_log = get_asr_log()

def _realtimestt_compute_sanitize(device: str, user_pref: str) -> str:
    """RealtimeSTT 在子进程内加载模型且无回退；CUDA 上 int8_float16 常在用户环境下报错。"""
    d = (device or "cpu").strip().lower()
    raw = (user_pref or "").strip()
    if not raw:
        return "float16" if d == "cuda" else "int8"
    c = raw.lower()
    if d == "cuda" and c == "int8_float16":
        _log.info(
            "RealtimeSTT: CUDA 下 int8_float16 已降级为 float16（可改 API 计算精度）"
        )
        return "float16"
    if d == "cpu" and c == "int8_float16":
        return "int8"
    return raw


class RealtimeSTTAdapter(ASRAdapter):
    """RealtimeSTT（realtimepy 包名 RealtimeSTT）：VAD + faster-whisper，实时字幕 + 每句结束 final。"""

    def __init__(
        self,
        language: str,
        callback: TranscriptionCallback,
        *,
        model_name: str = "small",
        device: str = "auto",
        compute_type: str = "",
    ):
        super().__init__(language, callback)
        self._model_name = (model_name or "small").strip()
        self._device_pref = device or "auto"
        self._compute_pref = compute_type or ""
        self._recorder: Any = None
        self._loop_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._paused = False
        if os.name == "nt":
            venv_path = sys.prefix
            nvidia_base = os.path.join(venv_path, r"Lib\site-packages\nvidia")
            for sub in (r"cudnn\bin", r"cublas\bin", r"curand\bin"):
                full_path = os.path.join(nvidia_base, sub)
                if os.path.exists(full_path):
                    try:
                        os.add_dll_directory(full_path)
                    except (OSError, AttributeError):
                        pass

    @staticmethod
    def _device_resolved(pref: str) -> str:
        p = (pref or "auto").strip().lower()
        if p == "cpu":
            return "cpu"
        if p == "cuda":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _compute_resolved(self, device: str) -> str:
        c = (self._compute_pref or "").strip()
        if c:
            return c
        return "float16" if device == "cuda" else "int8"

    def _compute_for_recorder(self, device: str) -> str:
        base = self._compute_resolved(device)
        return _realtimestt_compute_sanitize(device, base)

    def _initial_prompt_optional(self) -> Optional[str]:
        lang = (self.language or "").strip().lower()
        if lang.startswith("en"):
            return "English speech."
        if lang in ("ja", "jp"):
            return "日本語の会話です。"
        return None

    def _setup_recorder(self) -> None:
        from RealtimeSTT import AudioToTextRecorder

        dev = self._device_resolved(self._device_pref)
        ct = self._compute_for_recorder(dev)
        _log.info(
            "RealtimeSTT setup_recorder: model=%r device=%r compute_type=%r lang=%r",
            self._model_name,
            dev,
            ct,
            (self.language or "").strip(),
        )

        def on_rt_update(text: str) -> None:
            t = (text or "").strip()
            if t:
                _log.debug(
                    "RealtimeSTT realtime partial: %s",
                    t[:200] + ("…" if len(t) > 200 else ""),
                )
                self.callback(t, True)

        self._recorder = AudioToTextRecorder(
            model=self._model_name,
            language=(self.language or "").strip(),
            compute_type=ct,
            device=dev,
            enable_realtime_transcription=True,
            use_main_model_for_realtime=True,
            on_realtime_transcription_update=on_rt_update,
            spinner=False,
            initial_prompt=self._initial_prompt_optional(),
        )

    def _text_loop(self) -> None:
        cycle = 0
        while self._is_running:
            if self._paused:
                time.sleep(0.08)
                continue
            rec = self._recorder
            if rec is None:
                _log.warning("RealtimeSTT _text_loop: recorder is None, exit loop")
                break
            try:
                cycle += 1
                _log.debug("RealtimeSTT text() cycle #%s start", cycle)
                final = rec.text()
                if not self._is_running:
                    _log.info("RealtimeSTT text() returned but _is_running=False, stop")
                    break
                if self._paused:
                    _log.debug(
                        "RealtimeSTT text() returned while paused (len=%s), skip final",
                        len(final or ""),
                    )
                    continue
                ft = (final or "").strip()
                if ft:
                    _log.info(
                        "RealtimeSTT text() cycle #%s final: %s",
                        cycle,
                        ft[:300] + ("…" if len(ft) > 300 else ""),
                    )
                    self.callback(ft, False)
                else:
                    _log.debug(
                        "RealtimeSTT text() cycle #%s empty final (interrupted?)",
                        cycle,
                    )
            except Exception:
                if self._is_running:
                    _log.exception("RealtimeSTT _text_loop exception (cycle #%s)", cycle)

    def start(self) -> None:
        if self._is_running:
            _log.warning("RealtimeSTT start: already running")
            return
        try:
            if self._recorder is None:
                self._setup_recorder()
        except ImportError as e:
            _log.error("RealtimeSTT import failed: %s (pip install realtimestt)", e)
            return
        except Exception:
            _log.exception("RealtimeSTT setup_recorder failed")
            self._recorder = None
            return
        _log.info("RealtimeSTT starting loop thread")
        self._is_running = True
        self._paused = False
        self._loop_thread = threading.Thread(
            target=self._text_loop, name="realtimestt_loop", daemon=True
        )
        self._loop_thread.start()
        _log.info("RealtimeSTT started (thread=%s)", self._loop_thread.name)

    def stop(self) -> None:
        rec = self._recorder
        loop_thread = self._loop_thread
        if not self._is_running and rec is None:
            _log.debug("RealtimeSTT stop: idle, skip")
            return
        _log.info("RealtimeSTT stopping…")
        self._is_running = False
        self._paused = False
        self._recorder = None
        self._loop_thread = None
        if rec is not None:
            try:
                # 勿先 abort：Windows 下转写为子进程，管道已断时 abort 可能阻塞
                # was_interrupted.wait() 且加剧 poll_connection 的 BrokenPipe 刷屏。
                rec.shutdown()
            except Exception:
                _log.warning("RealtimeSTT stop: shutdown()", exc_info=True)
        if loop_thread is not None and loop_thread.is_alive():
            loop_thread.join(timeout=15.0)
            if loop_thread.is_alive():
                _log.warning("RealtimeSTT stop: loop thread still alive after join")
        _log.info("RealtimeSTT stopped")

    def get_status(self) -> str:
        return "Running" if self._is_running else "Stopped"

    def pause(self) -> None:
        # sendMessage 与 TTS 都会 pause；勿在主线程（Qt 槽）里同步 abort()——库内 was_interrupted.wait 易死锁
        if self._paused:
            _log.debug("RealtimeSTT pause: already paused, skip")
            return
        self._paused = True
        _log.info("RealtimeSTT pause: scheduling abort on helper thread")
        rec = self._recorder
        if rec is not None:

            def _abort_safe() -> None:
                try:
                    _log.debug(
                        "RealtimeSTT abort() on %s",
                        threading.current_thread().name,
                    )
                    rec.abort()
                    _log.debug("RealtimeSTT abort() finished")
                except Exception:
                    _log.exception("RealtimeSTT abort() failed")

            threading.Thread(
                target=_abort_safe, daemon=True, name="realtimestt_abort"
            ).start()

    def resume(self) -> None:
        _log.info("RealtimeSTT resume: clear pause + events + listen()")
        self._paused = False
        rec = self._recorder
        if rec is None:
            _log.warning("RealtimeSTT resume: recorder is None")
            return
        for attr in ("interrupt_stop_event", "was_interrupted"):
            ev = getattr(rec, attr, None)
            if ev is not None:
                try:
                    ev.clear()
                    _log.debug("RealtimeSTT resume: cleared %s", attr)
                except Exception:
                    _log.warning(
                        "RealtimeSTT resume: clear %s failed", attr, exc_info=True
                    )
        try:
            rec.listen()
            _log.info("RealtimeSTT resume: listen() ok")
        except Exception:
            _log.exception("RealtimeSTT resume: listen() failed")


def _resolve_whisper_device_compute(device_pref: str, compute_pref: str) -> tuple[str, str]:
    dp = (device_pref or "auto").strip().lower()
    cp = (compute_pref or "").strip()
    if dp == "auto":
        cuda_ok = False
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        if cuda_ok:
            return "cuda", cp or "float16"
        return "cpu", cp or "int8"
    if dp == "cuda":
        return "cuda", cp or "float16"
    return "cpu", cp or "int8"


def _whisper_compute_fallback_chain(device: str, preferred: str) -> list[str]:
    """在首选 compute_type 加载失败时依次尝试（如 int8 在当前后端不可用）。"""
    d = (device or "cpu").strip().lower()
    p = (preferred or "").strip().lower()
    if d == "cuda":
        order = ("float16", "int8_float16", "int8", "float32")
    else:
        order = ("int8", "int8_float32", "float32")
    out: list[str] = []
    if p:
        out.append(p)
    for x in order:
        if x not in out:
            out.append(x)
    return out


def _whisper_load_recoverable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "int8" in msg
        or "compute type" in msg
        or "compute_type" in msg
        or ("backend" in msg and "support" in msg)
        or "efficient" in msg
    )


class FasterWhisperAdapter(ASRAdapter):
    """faster-whisper 适配器：PyAudio 采集 + 端点检测 + WhisperModel.transcribe。"""

    SAMPLERATE = 16000
    CHUNK = 1024
    MIN_UTTER_SAMPLES = int(16000 * 0.9)
    PARTIAL_EVERY_SAMPLES = int(16000 * 1.4)
    SILENCE_SAMPLES_END = int(0.42 * 16000)

    @classmethod
    def get_config_schema(cls) -> dict[str, dict]:
        return {
            "rms_threshold": {
                "type": "float",
                "label": "RMS threshold",
                "default": 38.0,
                "min": 1.0,
                "max": 500.0,
                "step": 0.5,
            }
        }

    def __init__(
        self,
        language: str,
        callback: TranscriptionCallback,
        *,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "",
        rms_threshold: float = 38.0,
    ):
        super().__init__(language, callback)
        self._model_size = (model_size or "small").strip()
        self._device_pref = device or "auto"
        self._compute_pref = compute_type or ""
        self._rms_threshold = float(rms_threshold)
        self._model: Any = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._pause_event.set()

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            _log.error("faster-whisper not installed: %s", e)
            return
        dev, ct = _resolve_whisper_device_compute(self._device_pref, self._compute_pref)
        chain = _whisper_compute_fallback_chain(dev, ct)
        last_err: Optional[BaseException] = None
        for i, ctry in enumerate(chain):
            _log.info(
                "faster-whisper load try model=%r device=%s compute_type=%s",
                self._model_size,
                dev,
                ctry,
            )
            try:
                self._model = WhisperModel(self._model_size, device=dev, compute_type=ctry)
            except Exception as e:
                last_err = e
                if _whisper_load_recoverable_error(e) and i + 1 < len(chain):
                    continue
                _log.warning("faster-whisper load error: %s", e)
                self._model = None
                return
            if ctry != ct:
                _log.info(
                    "faster-whisper compute_type adjusted %r -> %r",
                    ct,
                    ctry,
                )
            return
        _log.error("faster-whisper load failed after retries: %s", last_err)
        self._model = None

    def _transcribe_numpy(self, audio_i16: Any) -> str:
        import numpy as np

        if self._model is None:
            return ""
        audio = np.asarray(audio_i16, dtype=np.float32) / 32768.0
        if audio.size < 256:
            return ""
        lang = (self.language or "").strip() or None
        segments, _ = self._model.transcribe(
            audio,
            language=lang,
            beam_size=5,
            vad_filter=True,
            without_timestamps=True,
        )
        return "".join(seg.text for seg in segments).strip()

    def _recognition_loop(self) -> None:
        import numpy as np
        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.SAMPLERATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )
        stream.start_stream()
        chunks: list[Any] = []
        silent_acc = 0
        last_partial_total = 0

        while self._is_running:
            if not self._pause_event.is_set():
                time.sleep(0.08)
                continue
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
            except Exception:
                _log.warning("faster-whisper stream.read failed, exit loop", exc_info=True)
                break
            np16 = np.frombuffer(data, dtype=np.int16)
            rms = float(
                np.sqrt(np.mean(np.square(np16.astype(np.float64))))
            )
            voice = rms >= self._rms_threshold

            if voice:
                silent_acc = 0
                chunks.append(np16)
                total = int(sum(len(c) for c in chunks))
                if (
                    total >= self.MIN_UTTER_SAMPLES
                    and total - last_partial_total >= self.PARTIAL_EVERY_SAMPLES
                ):
                    text = self._transcribe_numpy(np.concatenate(chunks))
                    if text:
                        self.callback(text, True)
                    last_partial_total = total
            else:
                if chunks:
                    silent_acc += len(np16)
                    if silent_acc >= self.SILENCE_SAMPLES_END:
                        audio = np.concatenate(chunks)
                        chunks = []
                        last_partial_total = 0
                        silent_acc = 0
                        text = self._transcribe_numpy(audio)
                        if text:
                            self.callback(text, False)
                else:
                    silent_acc = 0

        stream.stop_stream()
        stream.close()
        p.terminate()
        if chunks:
            text = self._transcribe_numpy(np.concatenate(chunks))
            if text:
                self.callback(text, False)
        _log.info("faster-whisper recognition loop ended")

    def start(self) -> None:
        if self._is_running:
            _log.warning("faster-whisper start: already running")
            return
        self._load_model()
        if self._model is None:
            _log.error("faster-whisper start: model not loaded")
            return
        _log.info("faster-whisper starting thread…")
        self._is_running = True
        self._thread = threading.Thread(
            target=self._recognition_loop, name="faster_whisper_asr", daemon=True
        )
        self._thread.start()
        _log.info("faster-whisper started")

    def stop(self) -> None:
        if not self._is_running:
            _log.debug("faster-whisper stop: not running")
            return
        _log.info("faster-whisper stopping…")
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8.0)
        self._thread = None
        _log.info("faster-whisper stopped")

    def get_status(self) -> str:
        return "Running" if self._is_running else "Stopped"

    def pause(self) -> None:
        if self._is_running:
            _log.info("faster-whisper pause")
            self._pause_event.clear()

    def resume(self) -> None:
        if self._is_running:
            _log.info("faster-whisper resume")
            self._pause_event.set()
