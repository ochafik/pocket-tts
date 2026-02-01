"""Tests for the speaker fingerprinting module.

These tests validate the core logic (fingerprint save/load, cosine similarity,
backend dispatch, input validation) without requiring heavy model downloads.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pocket_tts.speaker_id import (
    SpeakerFingerprint,
    SpeakerID,
    SpeakerIDBackend,
    _BACKENDS,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-7)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_high_dimensional(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal(192).astype(np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# SpeakerFingerprint
# ---------------------------------------------------------------------------


class TestSpeakerFingerprint:
    def test_create_valid(self):
        emb = np.random.randn(192).astype(np.float32)
        fp = SpeakerFingerprint(embedding=emb, backend="ecapa", source="test.wav")
        assert fp.embedding.shape == (192,)
        assert fp.backend == "ecapa"
        assert fp.source == "test.wav"

    def test_rejects_2d(self):
        emb = np.random.randn(1, 192).astype(np.float32)
        with pytest.raises(ValueError, match="1-D"):
            SpeakerFingerprint(embedding=emb, backend="ecapa")

    def test_save_and_load(self, tmp_path: Path):
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        fp = SpeakerFingerprint(embedding=emb, backend="test_backend", source="a.wav")
        out = tmp_path / "fp.json"
        fp.save(out)

        loaded = SpeakerFingerprint.load(out)
        np.testing.assert_allclose(loaded.embedding, emb, atol=1e-6)
        assert loaded.backend == "test_backend"
        assert loaded.source == "a.wav"

    def test_save_format_is_json(self, tmp_path: Path):
        emb = np.array([1.0, 2.0], dtype=np.float32)
        fp = SpeakerFingerprint(embedding=emb, backend="x")
        out = tmp_path / "fp.json"
        fp.save(out)
        data = json.loads(out.read_text())
        assert "embedding" in data
        assert "backend" in data


# ---------------------------------------------------------------------------
# Stub backend for testing SpeakerID without real models
# ---------------------------------------------------------------------------


class StubBackend(SpeakerIDBackend):
    """Returns a deterministic embedding based on a hash of the waveform."""

    name = "stub"

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        rng = np.random.default_rng(int(abs(waveform.sum() * 1000)))
        return rng.standard_normal(192).astype(np.float32)


class TestSpeakerID:
    def test_similarity_same_fingerprint(self):
        emb = np.random.randn(192).astype(np.float32)
        fp = SpeakerFingerprint(embedding=emb, backend="stub")
        sid = SpeakerID(StubBackend(), threshold=0.5)
        assert sid.similarity(fp, fp) == pytest.approx(1.0)

    def test_is_same_speaker_above_threshold(self):
        emb = np.random.randn(192).astype(np.float32)
        fp = SpeakerFingerprint(embedding=emb, backend="stub")
        sid = SpeakerID(StubBackend(), threshold=0.5)
        assert sid.is_same_speaker(fp, fp) is True

    def test_is_same_speaker_below_threshold(self):
        fp1 = SpeakerFingerprint(embedding=np.array([1.0, 0.0, 0.0]), backend="stub")
        fp2 = SpeakerFingerprint(embedding=np.array([0.0, 1.0, 0.0]), backend="stub")
        sid = SpeakerID(StubBackend(), threshold=0.5)
        assert sid.is_same_speaker(fp1, fp2) is False

    def test_rejects_mismatched_backends(self):
        fp1 = SpeakerFingerprint(embedding=np.array([1.0]), backend="a")
        fp2 = SpeakerFingerprint(embedding=np.array([1.0]), backend="b")
        sid = SpeakerID(StubBackend(), threshold=0.5)
        with pytest.raises(ValueError, match="different backends"):
            sid.similarity(fp1, fp2)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            SpeakerID.load(backend="nonexistent")

    def test_known_backends_registered(self):
        assert "ecapa" in _BACKENDS
        assert "resemblyzer" in _BACKENDS
        assert "mimi" in _BACKENDS

    def test_default_thresholds(self):
        sid = SpeakerID(StubBackend())
        # StubBackend falls back to 0.5
        assert sid.threshold == 0.5

    def test_custom_threshold(self):
        sid = SpeakerID(StubBackend(), threshold=0.99)
        assert sid.threshold == 0.99
