"""
R2 — BGE-M3 query encoder.

Wraps `BGEM3FlagModel("BAAI/bge-m3")` so a single call emits BOTH vectors the
`legal_hybrid` collection stores:
  - dense  : normalized 1024-d vector (the unnamed default vector)
  - sparse : `lexical_weights` (BGE-M3 learned/SPLADE-style) -> SparseVector

Because the stored embeddings were produced by the SAME model, this guarantees
the query lives in the exact same vector space as the index (no vocabulary to
persist — the token space is BGE-M3's own tokenizer).

The encoder is a callable:  encoder("some query") -> {"dense": [...],
"sparse": SparseVector}, so it can be injected directly into
`QdrantHybridStore(query_encoder=encoder)`.

First use downloads ~2.3 GB of model weights into the HuggingFace cache.

Windows note: the HF Hub cache defaults to symlinks, which fail with
WinError 1314 unless Developer Mode/admin is enabled. We force the copy
fallback via HF_HUB_DISABLE_SYMLINKS=1 BEFORE huggingface_hub is imported.
"""
from __future__ import annotations

import os

# Must be set before `FlagEmbedding`/`huggingface_hub` are imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from typing import Any, Optional  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from FlagEmbedding import BGEM3FlagModel  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402

DEFAULT_MODEL = os.getenv("BGE_MODEL", "BAAI/bge-m3")
DENSE_SIZE = 1024
SPARSE_VECTOR_NAME = "sparse"


class BgeM3QueryEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        devices: Optional[str] = None,
        use_fp16: Optional[bool] = None,
        query_max_length: int = 512,
        passage_max_length: int = 512,
    ):
        # CPU has no fp16 kernels; only enable fp16 when CUDA is present.
        if use_fp16 is None:
            use_fp16 = torch.cuda.is_available()
        if devices is None:
            devices = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.devices = devices
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            devices=devices,
            normalize_embeddings=True,     # match stored dense (cosine)
            query_max_length=query_max_length,
            passage_max_length=passage_max_length,
        )
        # Sanity-check the model dimensionality against the collection.
        dim = self.model.model.config.hidden_size
        if dim != DENSE_SIZE:
            raise RuntimeError(
                f"Model {model_name} emits {dim}-d dense vectors, "
                f"but legal_hybrid expects {DENSE_SIZE}. "
                "The stored collection does not match this model."
            )

    def encode(self, text: str) -> dict:
        """Encode a single query -> {"dense": [...1024 floats], "sparse": SparseVector}."""
        out = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        lexical = out["lexical_weights"][0]  # {token_id(str/int): weight}
        sparse = qm.SparseVector(
            indices=[int(k) for k in lexical],
            values=[float(v) for v in lexical.values()],
        )
        return {"dense": dense.tolist(), "sparse": sparse}

    def encode_batch(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        """Encode many passages (returns a dict per text)."""
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        results = []
        for i in range(len(texts)):
            dense = np.asarray(out["dense_vecs"][i], dtype=np.float32)
            lexical = out["lexical_weights"][i]
            results.append(
                {
                    "dense": dense.tolist(),
                    "sparse": qm.SparseVector(
                        indices=[int(k) for k in lexical],
                        values=[float(v) for v in lexical.values()],
                    ),
                }
            )
        return results

    # Allow injection into QdrantHybridStore(query_encoder=encoder).
    def __call__(self, text: str) -> dict:
        return self.encode(text)

    def close(self) -> None:
        try:
            self.model.model.cpu()
        except Exception:
            pass
