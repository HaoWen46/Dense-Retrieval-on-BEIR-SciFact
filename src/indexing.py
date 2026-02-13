from dataclasses import dataclass
from typing import Optional
import faiss
import numpy as np

@dataclass
class IndexConfig:
    kind: str                      # "flat" | "ivf" | "hnsw"
    nlist: int = 256               # IVF
    nprobe: int = 16               # IVF
    M: int = 32                    # HNSW
    efConstruction: int = 200      # HNSW
    efSearch: int = 64             # HNSW

def build_faiss_index(xb: np.ndarray, cfg: IndexConfig):
    """
    Builds a FAISS index for normalized embeddings.
    Uses inner product (dot product) which equals cosine similarity for normalized vectors.
    """
    d = xb.shape[1]

    if cfg.kind == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        return index

    if cfg.kind == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, cfg.nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(xb)
        index.add(xb)
        index.nprobe = cfg.nprobe
        return index

    if cfg.kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, cfg.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = cfg.efConstruction
        index.add(xb)
        index.hnsw.efSearch = cfg.efSearch
        return index

    raise ValueError(f"Unknown index kind: {cfg.kind}")
