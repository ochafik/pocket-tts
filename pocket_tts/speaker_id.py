"""Speaker fingerprinting and identification module.

Extracts fixed-length speaker embeddings from audio and compares them
using cosine similarity to determine whether two audio samples come
from the same speaker.

Supports three backends:
  - "ecapa" : SpeechBrain ECAPA-TDNN (best accuracy, ~192-dim, requires speechbrain)
  - "resemblyzer" : Resemblyzer d-vectors (lightweight, ~256-dim, requires resemblyzer)
  - "mimi" : pocket-tts internal Mimi encoder (no extra deps, experimental)

Typical usage::

    from pocket_tts.speaker_id import SpeakerID

    sid = SpeakerID.load(backend="ecapa")
    fp1 = sid.embed_file("speaker_a.wav")
    fp2 = sid.embed_file("speaker_b.wav")
    score = sid.similarity(fp1, fp2)
    print(f"Same speaker: {sid.is_same_speaker(fp1, fp2)}")
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public dataclass-like container for a speaker fingerprint
# ---------------------------------------------------------------------------


class SpeakerFingerprint:
    """A fixed-length speaker embedding with metadata."""

    def __init__(self, embedding: np.ndarray, backend: str, source: str | None = None):
        if embedding.ndim != 1:
            raise ValueError(f"Embedding must be 1-D, got shape {embedding.shape}")
        self.embedding = embedding.astype(np.float32)
        self.backend = backend
        self.source = source

    def save(self, path: str | Path) -> None:
        """Persist fingerprint to a JSON file."""
        path = Path(path)
        data = {
            "embedding": self.embedding.tolist(),
            "backend": self.backend,
            "source": self.source,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "SpeakerFingerprint":
        """Load fingerprint from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls(
            embedding=np.array(data["embedding"], dtype=np.float32),
            backend=data["backend"],
            source=data.get("source"),
        )


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------


class SpeakerIDBackend(ABC):
    """Base class for speaker embedding extraction backends."""

    name: str

    @abstractmethod
    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 1-D float32 embedding from raw audio."""
        ...

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Load an audio file and return its embedding."""
        waveform, sr = self._load_audio(path)
        return self.embed_audio(waveform, sr)

    @staticmethod
    def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
        """Load audio file as mono float32 numpy array + sample rate."""
        import scipy.io.wavfile as wavfile

        sr, data = wavfile.read(str(path))
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        # Convert to mono if stereo
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data, sr


# ---------------------------------------------------------------------------
# ECAPA-TDNN backend (SpeechBrain)
# ---------------------------------------------------------------------------


class EcapaBackend(SpeakerIDBackend):
    """SpeechBrain ECAPA-TDNN speaker encoder.

    Produces 192-dim embeddings. Achieves ~0.87% EER on VoxCeleb1.
    """

    name = "ecapa"

    def __init__(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            raise ImportError(
                "SpeechBrain is required for the 'ecapa' backend. "
                "Install it with: pip install speechbrain"
            )
        logger.info("Loading SpeechBrain ECAPA-TDNN model...")
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": "cpu"},
        )

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = torch.from_numpy(waveform).unsqueeze(0).float()

        # Resample to 16kHz if needed (ECAPA-TDNN expects 16kHz)
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(sample_rate, 16000)
            waveform_16k = resample_poly(waveform, 16000 // g, sample_rate // g)
            signal = torch.from_numpy(waveform_16k).unsqueeze(0).float()

        embedding = self._model.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Resemblyzer backend
# ---------------------------------------------------------------------------


class ResemblyzerBackend(SpeakerIDBackend):
    """Resemblyzer d-vector speaker encoder.

    Produces 256-dim embeddings. Simpler model, lower accuracy (~5-7% EER).
    """

    name = "resemblyzer"

    def __init__(self):
        try:
            from resemblyzer import VoiceEncoder
        except ImportError:
            raise ImportError(
                "Resemblyzer is required for the 'resemblyzer' backend. "
                "Install it with: pip install resemblyzer"
            )
        logger.info("Loading Resemblyzer voice encoder...")
        self._encoder = VoiceEncoder()

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(waveform, source_sr=sample_rate)
        embedding = self._encoder.embed_utterance(wav)
        return embedding


# ---------------------------------------------------------------------------
# Mimi backend (uses pocket-tts internals, no extra deps)
# ---------------------------------------------------------------------------


class MimiBackend(SpeakerIDBackend):
    """Pocket-TTS Mimi encoder with speaker projection.

    Uses the same encoder that powers voice cloning to extract a speaker
    conditioning vector, then mean-pools across time to get a fixed-length
    embedding. This is EXPERIMENTAL -- the embeddings are optimised for
    TTS conditioning, not for speaker discrimination.
    """

    name = "mimi"

    def __init__(self, variant: str = "b6369a24"):
        from pocket_tts.models.tts_model import TTSModel

        logger.info("Loading pocket-tts model for Mimi speaker embeddings...")
        self._model = TTSModel.load_model(variant)

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        from pocket_tts.data.audio import convert_audio

        audio = torch.from_numpy(waveform).unsqueeze(0).float()
        target_sr = self._model.config.mimi.sample_rate
        if sample_rate != target_sr:
            audio = convert_audio(audio, sample_rate, target_sr, 1)

        with torch.no_grad():
            # _encode_audio returns [1, T, 1024] conditioning tensor
            conditioning = self._model._encode_audio(audio.unsqueeze(0).to(self._model.device))

        # Mean-pool across time dimension to get fixed-length embedding
        embedding = conditioning.squeeze(0).mean(dim=0).cpu().numpy()
        return embedding


# ---------------------------------------------------------------------------
# Main SpeakerID facade
# ---------------------------------------------------------------------------

_BACKENDS = {
    "ecapa": EcapaBackend,
    "resemblyzer": ResemblyzerBackend,
    "mimi": MimiBackend,
}


class SpeakerID:
    """Speaker fingerprinting and comparison.

    Args:
        backend: Which embedding backend to use ("ecapa", "resemblyzer", "mimi").
        threshold: Cosine similarity threshold for same-speaker decisions.
            Defaults depend on backend:
            - ecapa: 0.25  (calibrated on VoxCeleb)
            - resemblyzer: 0.75
            - mimi: 0.85 (experimental, tune on your data)
    """

    _DEFAULT_THRESHOLDS = {
        "ecapa": 0.25,
        "resemblyzer": 0.75,
        "mimi": 0.85,
    }

    def __init__(self, backend_instance: SpeakerIDBackend, threshold: float | None = None):
        self._backend = backend_instance
        self.threshold = (
            threshold
            if threshold is not None
            else self._DEFAULT_THRESHOLDS.get(backend_instance.name, 0.5)
        )

    @classmethod
    def load(cls, backend: str = "ecapa", threshold: float | None = None) -> "SpeakerID":
        """Create a SpeakerID instance with the given backend.

        Args:
            backend: One of "ecapa", "resemblyzer", "mimi".
            threshold: Optional cosine similarity threshold override.
        """
        if backend not in _BACKENDS:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(_BACKENDS.keys())}")
        return cls(_BACKENDS[backend](), threshold=threshold)

    def embed_file(self, path: str | Path) -> SpeakerFingerprint:
        """Extract a speaker fingerprint from an audio file."""
        embedding = self._backend.embed_file(path)
        return SpeakerFingerprint(
            embedding=embedding,
            backend=self._backend.name,
            source=str(path),
        )

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> SpeakerFingerprint:
        """Extract a speaker fingerprint from a raw audio waveform."""
        embedding = self._backend.embed_audio(waveform, sample_rate)
        return SpeakerFingerprint(
            embedding=embedding,
            backend=self._backend.name,
        )

    def similarity(self, a: SpeakerFingerprint, b: SpeakerFingerprint) -> float:
        """Cosine similarity between two fingerprints (higher = more similar)."""
        if a.backend != b.backend:
            raise ValueError(
                f"Cannot compare fingerprints from different backends: "
                f"'{a.backend}' vs '{b.backend}'"
            )
        return cosine_similarity(a.embedding, b.embedding)

    def is_same_speaker(self, a: SpeakerFingerprint, b: SpeakerFingerprint) -> bool:
        """Decide whether two fingerprints belong to the same speaker."""
        return self.similarity(a, b) >= self.threshold
