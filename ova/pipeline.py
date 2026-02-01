import io
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Self

import nemo.collections.asr as nemo_asr
import numpy as np
import torch
from kokoro import KPipeline
from nemo.collections.asr.models import ASRModel
from ollama import chat
from qwen_tts import Qwen3TTSModel

from .audio import numpy_to_wav_bytes, resample
from .utils import get_device, logger

DEFAULT_SR = 24000  # default sample rate
DEFAULT_TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_TTS_VOICE = "af_heart"
VOICE_CLONE_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_CHAT_MODEL = "ministral-3:3b-instruct-2512-q4_K_M"
DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


class OVAProfile(Enum):
    DEFAULT = "default"
    DUA = "dua"

    @classmethod
    def from_str(cls, profile_str: str) -> Self:
        profile_enum = cls.DEFAULT
        if profile_str in [p.value for p in OVAProfile]:
            profile_enum = cls(profile_str)
        else:
            logger.warning(
                f"Unknown OVA profile '{profile_str}', defaulting to DEFAULT"
            )
        return profile_enum


class TTSProtocol(Protocol):
    def generate(self, text: str) -> bytes: ...


@dataclass
class TTSDefault:
    tts_model: KPipeline
    _is_warm: bool = False

    @classmethod
    def create(cls) -> Self:
        tts_model = KPipeline(
            lang_code="a", repo_id=DEFAULT_TTS_MODEL
        )  # 'a' => US/American English
        return cls(tts_model=tts_model)

    def generate(self, text: str) -> bytes:
        if not self._is_warm:
            # warm up
            self.tts_model("Just testing!", voice=DEFAULT_TTS_VOICE)
            self._is_warm = True

        generator = self.tts_model(text, voice=DEFAULT_TTS_VOICE)

        chunks = []
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.size > 0:
                chunks.append(audio)

        arr = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)

        wav_bytes = numpy_to_wav_bytes(arr, sr=DEFAULT_SR)

        return wav_bytes


@dataclass
class TTSVoiceClone:
    tts_model: Qwen3TTSModel
    voice_clone_prompt_items: Any

    @classmethod
    def create(cls, profile_dir: str, device: Any) -> Self:
        tts_model = Qwen3TTSModel.from_pretrained(
            VOICE_CLONE_TTS_MODEL,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        with open(f"{profile_dir}/ref_text.txt", "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

        voice_clone_prompt_items = tts_model.create_voice_clone_prompt(
            ref_audio=f"{profile_dir}/ref_audio.wav",
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        return cls(
            tts_model=tts_model, voice_clone_prompt_items=voice_clone_prompt_items
        )

    def generate(self, text: str) -> bytes:
        wavs, sr = self.tts_model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=self.voice_clone_prompt_items,
        )

        wav_bytes = numpy_to_wav_bytes(wavs[0], sr)

        return wav_bytes


@dataclass
class OVAPipeline:
    profile: OVAProfile
    device: Any
    system_prompt: str
    context: list[dict]
    tts_handler: TTSProtocol
    chat_model: str = DEFAULT_CHAT_MODEL
    asr_model_name: str = DEFAULT_ASR_MODEL
    _asr_model: ASRModel | None = None

    @classmethod
    def from_profile(cls, profile_enum: OVAProfile) -> Self:
        device = get_device()

        # prep for loading assistant profile / prompt
        profile_dir = f"profiles/{profile_enum.value}"

        with open(f"{profile_dir}/prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        context = [{"role": "system", "content": system_prompt}]

        # initialize TTS
        if profile_enum == OVAProfile.DUA:
            tts_handler = TTSVoiceClone.create(profile_dir, device)
        else:
            tts_handler = TTSDefault.create()

        return cls(
            profile=profile_enum,
            device=device,
            system_prompt=system_prompt,
            context=context,
            tts_handler=tts_handler,
        )

    @property
    def asr_model(self) -> ASRModel:
        if self._asr_model is None:
            self._asr_model = ASRModel.from_pretrained(model_name=self.asr_model_name)
        return self._asr_model

    def tts(self, text: str) -> bytes:
        return self.tts_handler.generate(text)

    def transcribe(self, wav_bytes: bytes) -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            num_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            src_sr = wf.getframerate()
            num_frames = wf.getnframes()
            pcm = wf.readframes(num_frames)

        # PCM -> float32 in [-1, 1]
        if sampwidth == 1:
            audio = np.frombuffer(pcm, dtype=np.uint8).astype(np.int16) - 128
            audio = audio.astype(np.float32) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            a_f32 = np.frombuffer(pcm, dtype=np.float32)
            if (
                np.isfinite(a_f32).all()
                and (np.abs(a_f32).max() <= 10.0)
                and (np.abs(a_f32).mean() < 0.5)
            ):
                audio = a_f32.astype(np.float32)
            else:
                audio = (
                    np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
                )
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

        # Downmix to mono if needed
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1).astype(np.float32)

        # Resample to model SR
        model_sr = int(
            getattr(getattr(self.asr_model, "cfg", None), "sample_rate", DEFAULT_SR)
        )
        audio = resample(audio, src_sr, model_sr)

        # Torch tensors on model device
        device = next(self.asr_model.parameters()).device
        audio_tensor = (
            torch.from_numpy(audio).unsqueeze(0).to(device=device, dtype=torch.float32)
        )  # [1, T]
        length_tensor = torch.tensor([audio.shape[0]], device=device, dtype=torch.long)

        self.asr_model.eval()
        with torch.inference_mode():
            out = self.asr_model(
                input_signal=audio_tensor, input_signal_length=length_tensor
            )

            if isinstance(out, (tuple, list)) and len(out) >= 2:
                logits, logit_lengths = out[0], out[1]
            elif isinstance(out, dict):
                logits = out.get("logits", out.get("encoded"))
                logit_lengths = out.get("logit_lengths", out.get("encoded_len"))
                if logits is None or logit_lengths is None:
                    raise RuntimeError(
                        f"Unexpected model output keys: {list(out.keys())}"
                    )
            else:
                raise RuntimeError(f"Unexpected model output type: {type(out)}")

            decoding = getattr(self.asr_model, "decoding", None)
            if decoding is None:
                raise RuntimeError("Model has no `decoding`; cannot decode.")

            if hasattr(decoding, "ctc_decoder_predictions_tensor"):
                texts = decoding.ctc_decoder_predictions_tensor(logits, logit_lengths)
            elif hasattr(decoding, "rnnt_decoder_predictions_tensor"):
                texts = decoding.rnnt_decoder_predictions_tensor(logits, logit_lengths)
            else:
                raise RuntimeError(
                    "No supported decoder method found on `asr_model.decoding`."
                )

        # Extract text from Hypothesis object if needed
        if texts and len(texts) > 0:
            text = texts[0]
            if hasattr(text, "text"):
                return text.text.strip()
            elif isinstance(text, str):
                return text.strip()
            else:
                return str(text).strip()

        return ""

    def chat(self, text: str) -> str:
        self.context.append({"role": "user", "content": text})

        response = chat(
            model=self.chat_model,
            messages=self.context,
            think=False,
            stream=False,
        )

        response = (
            response.message.content.replace("**", "")
            .replace("_", "")
            .replace("__", "")
            .replace("#", "")
            .strip()
        )

        self.context.append({"role": "assistant", "content": response})

        return response
