import re
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

VECTOR_SIZE = 384
DENSE_VEC_NAME = "text-dense"
SPARSE_VEC_NAME = "text-sparse"

# Persistencia REAL local
client = QdrantClient(path="./qdrant_data")

def normalize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id.strip())
    return cleaned or "anonymous"

def get_collection_name(user_id: str) -> str:
    return f"user_{normalize_user_id(user_id)}"

def init_collection(user_id: str):
    collection_name = get_collection_name(user_id)
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VEC_NAME: VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VEC_NAME: SparseVectorParams()
            }
        )

def _to_sparse_vector(vector) -> SparseVector:
    if isinstance(vector, SparseVector):
        return vector

    indices = getattr(vector, "indices", None)
    values = getattr(vector, "values", None)

    if indices is None or values is None:
        raise TypeError("Unsupported sparse vector format")

    if hasattr(indices, "tolist"):
        indices = indices.tolist()
    else:
        indices = list(indices)

    if hasattr(values, "tolist"):
        values = values.tolist()
    else:
        values = list(values)

    return SparseVector(indices=indices, values=values)

def insert_vectors(chunks: list[str], vectors: tuple[list[list[float]], list], doc_id: str, user_id: str):
    collection_name = get_collection_name(user_id)
    dense_vecs, sparse_vecs = vectors
    points = []
    
    for i, (chunk, d_vec, s_vec) in enumerate(zip(chunks, dense_vecs, sparse_vecs)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    DENSE_VEC_NAME: d_vec,
                    SPARSE_VEC_NAME: _to_sparse_vector(s_vec)
                },
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "source": doc_id
                }
            )
        )
    client.upsert(collection_name=collection_name, points=points)
    return len(points)

def search_vectors_hybrid(query_vectors: tuple[list[float], any], user_id: str, top_k: int = 20) -> tuple[list, list]:
    collection_name = get_collection_name(user_id)
    
    try:
        client.get_collection(collection_name)
    except Exception:
        return [], []
        
    dense_query, sparse_query = query_vectors
    
    # Dense Semantic Search
    dense_results = client.query_points(
        collection_name=collection_name,
        query=dense_query,
        using=DENSE_VEC_NAME,
        limit=top_k
    ).points
    
    # Sparse BM25 Search
    sparse_results = client.query_points(
        collection_name=collection_name,
        query=_to_sparse_vector(sparse_query),
        using=SPARSE_VEC_NAME,
        limit=top_k
    ).points
    
    return dense_results, sparse_results
