"""Embedding adapters (G07): OOD and duplicate signals for the escalation lane.

Embeddings here are *signals*, never the security boundary. They feed the
deterministic trigger evaluation in core/escalate.py (near-OOD inputs and
duplicate floods escalate); they never gate acceptance by themselves.

Per EVIDENCE_FIREWALL_V2.md section 6: gemini-embedding-001 + NumPy cosine.
No vector database at demo scale -- ClaimIndex and EvidenceIndex are plain
NumPy matrices.

Provider absence must not break imports: GeminiEmbedder imports google-genai
lazily inside embed(), exactly like GeminiCompiler (G06).
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Sequence

import numpy as np

__all__ = [
    "EMBED_MODEL_ENV",
    "DEFAULT_EMBED_MODEL",
    "cosine",
    "FakeEmbedder",
    "GeminiEmbedder",
    "ClaimIndex",
    "EvidenceIndex",
]

#: Env var naming the embedding model; see EVIDENCE_FIREWALL_V2.md section 6.
EMBED_MODEL_ENV = "EPISTEMICOS_EMBED_MODEL"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"


def _unit(vec: np.ndarray) -> np.ndarray | None:
    """Normalized copy of ``vec``, or None for the zero vector."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vec / norm


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; defined as 0.0 when either vector is zero."""
    ua = _unit(np.asarray(a, dtype=np.float64))
    ub = _unit(np.asarray(b, dtype=np.float64))
    if ua is None or ub is None:
        return 0.0
    return float(np.dot(ua, ub))


class FakeEmbedder:
    """Deterministic, provider-free embedder for tests and demos.

    ``embed(text)`` returns a unit vector derived from the SHA-256 of the
    text (used to seed a NumPy generator): identical text always yields an
    identical vector, distinct texts yield (with overwhelming probability)
    distinct near-orthogonal vectors.

    ``canned`` optionally overrides specific texts with fixed vectors so
    tests can force duplicates or OOD geometry exactly.
    """

    def __init__(self, dim: int = 32, canned: dict[str, list[float]] | None = None) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.canned = dict(canned) if canned else {}

    async def embed(self, text: str) -> list[float]:
        if text in self.canned:
            return list(self.canned[text])
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim)
        unit = _unit(vec)
        if unit is None:  # astronomically unlikely; keep the contract total
            vec = np.zeros(self.dim)
            vec[0] = 1.0
            unit = vec
        return [float(x) for x in unit]


class GeminiEmbedder:
    """Gemini-backed embedder.

    The google-genai client is imported and constructed lazily inside
    ``embed()`` (same pattern as G06's GeminiCompiler), so importing this
    module and constructing this class work without google-genai installed
    and without GEMINI_API_KEY set.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get(EMBED_MODEL_ENV, DEFAULT_EMBED_MODEL)
        self._client: Any = None
        self.last_usage: dict[str, int] | None = None

    def _get_client(self) -> Any:
        """Lazily import google-genai and build the client (cached)."""
        if self._client is None:
            from google import genai  # deferred: no import cost / key needed at module load

            self._client = genai.Client()
        return self._client

    async def embed(self, text: str) -> list[float]:
        client = self._get_client()
        self.last_usage = None
        response = await client.aio.models.embed_content(model=self.model, contents=text)
        metadata = getattr(response, "usage_metadata", None)
        if metadata is not None:
            prompt = getattr(metadata, "prompt_token_count", 0) or 0
            total = getattr(metadata, "total_token_count", prompt) or prompt
            self.last_usage = {
                "input_tokens": int(prompt),
                "output_tokens": 0,
                "thinking_tokens": 0,
                "cached_tokens": int(
                    getattr(metadata, "cached_content_token_count", 0) or 0
                ),
                "total_tokens": int(total),
            }
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            raise RuntimeError("provider returned no embeddings")
        return [float(x) for x in embeddings[0].values]


class ClaimIndex:
    """Precomputed claim-text embedding matrix (pure NumPy, no vector DB).

    Built once from ``{claim_id: claim_text}``; used on the hot path for the
    OOD signal (distance from the modeled claim space) and nearest-claim
    lookup. Rows are unit-normalized so similarity is a single matmul.
    """

    def __init__(self, claim_ids: list[str], matrix: np.ndarray) -> None:
        self.claim_ids = claim_ids
        self.matrix = matrix  # shape (n_claims, dim), rows unit-normalized

    @classmethod
    async def build(cls, claims: dict[str, str], embedder: Any) -> "ClaimIndex":
        claim_ids: list[str] = []
        rows: list[np.ndarray] = []
        for claim_id in claims:  # insertion order; deterministic
            vec = np.asarray(await embedder.embed(claims[claim_id]), dtype=np.float64)
            unit = _unit(vec)
            if unit is None:
                unit = np.zeros_like(vec)
            claim_ids.append(claim_id)
            rows.append(unit)
        matrix = np.stack(rows) if rows else np.zeros((0, 0))
        return cls(claim_ids, matrix)

    def _similarities(self, vec: Sequence[float]) -> np.ndarray | None:
        if len(self.claim_ids) == 0:
            return None
        unit = _unit(np.asarray(vec, dtype=np.float64))
        if unit is None:
            return np.zeros(len(self.claim_ids))
        return self.matrix @ unit

    def ood_distance(self, vec: Sequence[float]) -> float:
        """1 - max cosine similarity against all claims; 1.0 for an empty index."""
        sims = self._similarities(vec)
        if sims is None:
            return 1.0
        return float(1.0 - np.max(sims))

    def nearest(self, vec: Sequence[float]) -> tuple[str | None, float]:
        """(claim_id, cosine) of the nearest claim; (None, 0.0) when empty."""
        sims = self._similarities(vec)
        if sims is None:
            return None, 0.0
        idx = int(np.argmax(sims))
        return self.claim_ids[idx], float(sims[idx])


class EvidenceIndex:
    """Running store of prior evidence vectors (duplicate-flood signal).

    ``max_similarity`` against everything previously ``add``-ed is the
    duplicate signal consumed by evaluate_triggers (>= DUP_THRESHOLD
    escalates). Pure NumPy list-of-rows; demo scale needs nothing more.
    """

    def __init__(self) -> None:
        self._rows: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(self, vec: Sequence[float]) -> None:
        unit = _unit(np.asarray(vec, dtype=np.float64))
        if unit is None:
            unit = np.zeros(np.asarray(vec, dtype=np.float64).shape)
        self._rows.append(unit)

    def max_similarity(self, vec: Sequence[float]) -> float:
        """Max cosine vs all stored vectors; 0.0 when the store is empty."""
        if not self._rows:
            return 0.0
        unit = _unit(np.asarray(vec, dtype=np.float64))
        if unit is None:
            return 0.0
        return float(np.max(np.stack(self._rows) @ unit))
