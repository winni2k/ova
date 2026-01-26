import io
import wave

import numpy as np
import soxr


def numpy_to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
        arr = np.clip(arr, -1.0, 1.0)
    
    # RMS normalize
    arr = rms_normalize(arr)

    arr = np.clip(arr, -1.0, 1.0)
    arr_i16 = (arr * 32767.0).astype(np.int16)

    if arr_i16.ndim == 1:
        arr_i16 = arr_i16[:, None]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(arr_i16.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr_i16.tobytes())

    return buf.getvalue()


def resample(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)

    if src_sr == dst_sr or arr.size == 0:
        return arr
    
    return soxr.resample(arr, src_sr, dst_sr, quality="HQ")


def rms_normalize(arr: np.ndarray, target_rms=0.15, peak_limit=0.90, eps=1e-12) -> np.ndarray:
    x = arr.astype(np.float32)

    rms = np.sqrt(np.mean(x * x) + eps)
    if rms < eps:
        return x  # silence

    x = x * (target_rms / rms)

    # prevent clipping
    peak = np.max(np.abs(x)) + eps
    if peak > peak_limit:
        x = x * (peak_limit / peak)

    return x