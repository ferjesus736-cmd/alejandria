import os
import hashlib
import math
from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
RAG_OFFLINE = os.getenv("RAG_OFFLINE", "0") == "1"

@lru_cache(maxsize=None)
def _get_dense_model():
    if RAG_OFFLINE:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME)
    except Exception:
        return None

@lru_cache(maxsize=None)
def _get_sparse_model():
    if RAG_OFFLINE:
        return None
    try:
        from fastembed import SparseTextEmbedding # BM25 Sparse Embeddings
        return SparseTextEmbedding("Qdrant/bm25")
    except Exception:
        return None

def _fallback_dense(text: str) -> list[float]:
    vector = [0.0] * VECTOR_SIZE
    for index, token in enumerate(text.lower().split()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = digest[0] % VECTOR_SIZE
        value = ((digest[1] / 255.0) * 2.0) - 1.0
        vector[slot] += value

    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]

def _fallback_sparse(text: str):
    class SparseVectorLike:
        def __init__(self, indices, values):
            self.indices = indices
            self.values = values

    weights = {}
    for token in text.lower().split():
        if not token:
            continue
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % 100_000
        weights[index] = weights.get(index, 0.0) + 1.0

    indices = list(weights.keys())
    values = list(weights.values())
    return SparseVectorLike(indices, values)

def create_embeddings(chunks: list[str]) -> tuple[list[list[float]], list]:
    # Dense (Semantic)
    model = _get_dense_model()
    if model is None:
        dense_vectors = [_fallback_dense(chunk) for chunk in chunks]
    else:
        try:
            dense_vectors = model.encode(
                chunks,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True
            ).tolist()
        except Exception:
            dense_vectors = [_fallback_dense(chunk) for chunk in chunks]
    
    # Sparse (BM25 Keyword)
    sparse_model = _get_sparse_model()
    if sparse_model is None:
        sparse_vectors = [_fallback_sparse(chunk) for chunk in chunks]
    else:
        try:
            sparse_vectors = list(sparse_model.embed(chunks))
        except Exception:
            sparse_vectors = [_fallback_sparse(chunk) for chunk in chunks]
    
    return dense_vectors, sparse_vectors

def embed_query(query: str) -> tuple[list[float], any]:
    # Dense
    model = _get_dense_model()
    if model is None:
        dense_vector = _fallback_dense(query)
    else:
        try:
            dense_vector = model.encode([query], normalize_embeddings=True)[0].tolist()
        except Exception:
            dense_vector = _fallback_dense(query)
    
    # Sparse
    sparse_model = _get_sparse_model()
    if sparse_model is None:
        sparse_vector = _fallback_sparse(query)
    else:
        try:
            sparse_vector = list(sparse_model.query_embed(query))[0]
        except Exception:
            sparse_vector = _fallback_sparse(query)
    
    return dense_vector, sparse_vector
