from __future__ import annotations

import os


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from typing import Optional  

DEFAULT_RERANKER = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
FALLBACK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "auto").lower()


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        batch_size: int = 8,
        normalize: bool = True,
    ):
        import torch

        if use_fp16 is None:
            use_fp16 = torch.cuda.is_available()
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self._backend: Optional[str] = None

        backend = RERANKER_BACKEND
        if backend == "auto":
            backend = "bge" if torch.cuda.is_available() else "fast"

        if backend == "fast":

            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(FALLBACK_RERANKER, max_length=512)
            self._backend = "CrossEncoder"
            return

        try:
            from FlagEmbedding import FlagReranker

            kwargs = {"use_fp16": use_fp16, "batch_size": batch_size}
            if device is not None:
                kwargs["devices"] = device
            self.model = FlagReranker(model_name, **kwargs)
            self._backend = "FlagReranker"
        except Exception as exc:  
            print(
                f"[reranker] FlagReranker init failed ({exc}); "
                f"falling back to {FALLBACK_RERANKER}"
            )
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(FALLBACK_RERANKER, max_length=512)
            self._backend = "CrossEncoder"


    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each (query, text) pair."""
        if not texts:
            return []
        pairs = [[query, t] for t in texts]

        if self._backend == "FlagReranker":
            scores = self.model.compute_score(  # type: ignore[union-attr]
                pairs,
                normalize=self.normalize,
                batch_size=self.batch_size,
            )
            if isinstance(scores, float):
                scores = [scores]
            return [float(s) for s in scores]

        # sentence-transformers CrossEncoder path
        import numpy as np

        if hasattr(self.model, "score"):
            out = self.model.score(pairs, batch_size=self.batch_size)  # type: ignore[union-attr]
        else:  # older API
            out = self.model.predict(pairs, batch_size=self.batch_size)  # type: ignore[union-attr]
        return [float(x) for x in np.asarray(out).reshape(-1)]

    def rerank(
        self,
        query: str,
        evidence: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:

        if not evidence:
            return []
        texts = [e.get("text") or "" for e in evidence]
        scores = self.score(query, texts)

        ranked = []
        for e, s in zip(evidence, scores):
            r = dict(e)
            r["rerank_score"] = float(s)
            ranked.append(r)
        ranked.sort(key=lambda r: r["rerank_score"], reverse=True)

        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked

    def __call__(
        self,
        query: str,
        evidence: list[dict],
        top_k: Optional[int] = None,
    ) -> list[dict]:
        return self.rerank(query, evidence, top_k)
