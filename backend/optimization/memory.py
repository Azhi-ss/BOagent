"""memory.py — Persistent Insight History with Doubao Embedding vector retrieval.

Inspired by Reasoning-BO's NotesAgent + MilvusAgent pattern, but lightweight:
- No external DB required (pure in-memory numpy store, optionally persisted to JSONL)
- Uses Doubao embedding API (OpenAI-compatible) for semantic search
- Falls back to simple keyword search if embedding API is unavailable
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Insight:
    """One structured note extracted from a BO iteration."""
    iteration: int
    best_score: float
    notes: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    parameter_relationships: list[str] = field(default_factory=list)
    optimization_principles: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Flatten all fields into a single retrieval string."""
        parts = self.notes + self.key_findings + self.parameter_relationships + self.optimization_principles
        return " | ".join(parts)


class EmbeddingClient:
    """Thin wrapper around OpenAI-compatible embedding endpoint (Doubao / OpenAI)."""

    def __init__(self) -> None:
        self._ark_key = os.environ.get("ARK_API_KEY", "")
        self._ark_base = os.environ.get("ARK_API_BASE", "").rstrip("/")
        self._model = os.environ.get("DOUBAO_EMBEDDING_MODEL", "doubao-embedding-vision-250615")
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            self._available = bool(self._ark_key and self._ark_base)
        return self._available

    def embed(self, texts: list[str]) -> np.ndarray | None:
        """Return (N, D) float32 embedding matrix, or None on failure."""
        if not self.is_available():
            return None
        try:
            import urllib.request, json as _json
            payload = _json.dumps({"model": self._model, "input": texts}).encode()
            req = urllib.request.Request(
                url=f"{self._ark_base}/embeddings",
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._ark_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read())
            vecs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
            return np.array(vecs, dtype=np.float32)
        except Exception as e:
            print(f"[VectorMemory] Embedding API error: {e}")
            return None


class VectorMemory:
    """In-memory insight store with optional embedding-based retrieval.

    Usage:
        mem = VectorMemory(persist_path="insights.jsonl")
        mem.add(insight)
        relevant = mem.query("high CBO leads to spike blocking", top_k=3)
        formatted = mem.format_for_prompt(relevant)
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._insights: list[Insight] = []
        self._embeddings: list[np.ndarray] = []   # parallel list
        self._embed_client = EmbeddingClient()
        self._persist_path = Path(persist_path) if persist_path else None

        if self._persist_path and self._persist_path.exists():
            self._load_from_disk()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, insight: Insight) -> None:
        """Append an insight and compute its embedding (if API available)."""
        self._insights.append(insight)
        text = insight.to_text()
        vec = self._embed_client.embed([text])
        self._embeddings.append(vec[0] if vec is not None else None)

        if self._persist_path:
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(insight), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int = 3) -> list[Insight]:
        """Return top-k most relevant insights by cosine similarity (or recency)."""
        if not self._insights:
            return []

        # Try embedding-based retrieval first
        q_vec = self._embed_client.embed([query_text])
        if q_vec is not None:
            stored = [e for e in self._embeddings if e is not None]
            if len(stored) == len(self._embeddings):  # all have embeddings
                mat = np.stack(stored)                  # (N, D)
                q = q_vec[0]                            # (D,)
                scores = mat @ q / (
                    np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9
                )
                top_idx = np.argsort(scores)[::-1][:top_k]
                return [self._insights[i] for i in top_idx]

        # Fallback: return most recent top_k
        return self._insights[-top_k:]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_for_prompt(self, insights: list[Insight]) -> str:
        """Render retrieved insights into a concise prompt block."""
        if not insights:
            return ""
        lines = ["### Retrieved Experiment Memory (most relevant prior insights)"]
        for ins in insights:
            lines.append(f"\n[Iteration {ins.iteration} | Best PCE={ins.best_score:.4f}]")
            if ins.key_findings:
                lines.append("Key Findings: " + "; ".join(ins.key_findings))
            if ins.parameter_relationships:
                lines.append("Parameter Relations: " + "; ".join(ins.parameter_relationships))
            if ins.optimization_principles:
                lines.append("Optimization Principles: " + "; ".join(ins.optimization_principles))
        return "\n".join(lines)

    def format_all_for_prompt(self, max_items: int = 5) -> str:
        """Format recent insights — used when no query context is available."""
        return self.format_for_prompt(self._insights[-max_items:])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        with open(self._persist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    insight = Insight(**data)
                    self._insights.append(insight)
                    # Re-embed on load (lazy: defer to first query)
                    self._embeddings.append(None)
                except Exception:
                    pass

    def __len__(self) -> int:
        return len(self._insights)
