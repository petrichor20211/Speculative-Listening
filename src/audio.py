"""Audio conventions shared by the session builder, the replay and the targets."""
from __future__ import annotations

import io
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16000  # every target's input sample rate; one chunk is one second
SPEECH_RMS = 0.05  # a 20 ms frame above this RMS marks the chunk's audio as user speech
FRAME = RATE // 50


def read_mono_16k(source: str | Path | io.BytesIO) -> np.ndarray:
    wav, rate = sf.read(source, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if rate != RATE:
        g = gcd(rate, RATE)
        wav = resample_poly(wav, RATE // g, rate // g)
    return np.ascontiguousarray(wav, dtype=np.float32)


def has_speech(chunk: np.ndarray) -> bool:
    frames = len(chunk) // FRAME
    return frames > 0 and bool(np.sqrt((chunk[: frames * FRAME].reshape(frames, FRAME) ** 2).mean(1)).max() > SPEECH_RMS)
