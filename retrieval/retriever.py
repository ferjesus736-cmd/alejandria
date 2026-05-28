from functools import lru_cache

from embeddings.embedder import embed_query
from vectorstore.qdrant_store import search_vectors_hybrid

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RAG_OFFLINE = __import__("os").getenv("RAG_OFFLINE", "0") == "1"


@lru_cache(maxsize=1)
def _get_reranker():
    if RAG_OFFLINE:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        return None


def _payload_from_hit(hit) -> dict:
    payload = dict(hit.payload or {})
    payload["point_id"] = str(hit.id)
    return payload


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    rrf_scores = {}
    docs = {}

    for rank, hit in enumerate(dense_results, 1):
        doc_id = str(hit.id)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = hit

    for rank, hit in enumerate(sparse_results, 1):
        doc_id = str(hit.id)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        docs[doc_id] = hit

    sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    fused = []

    for doc_id, score in sorted_docs:
        payload = _payload_from_hit(docs[doc_id])
        payload["rrf_score"] = score
        fused.append(payload)

    return fused


def rerank_candidates(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    if not candidates:
        return []

    reranker = _get_reranker()
    if reranker is None:
        query_terms = set(query.lower().split())
        scores = []
        for candidate in candidates:
            candidate_terms = set(candidate.get("text", "").lower().split())
            overlap = len(query_terms & candidate_terms)
            scores.append(float(overlap))
    else:
        pairs = [(query, candidate.get("text", "")) for candidate in candidates]
        scores = reranker.predict(pairs)

    reranked = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["score"] = float(score)
        reranked.append(enriched)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def retrieve(query: str, top_k: int = 20, min_score: float = 0.0, user_id: str = "") -> list[dict]:
    """
    query -> hybrid embedding -> dense & sparse search -> RRF fusion -> rerank
    """
    query_vectors = embed_query(query)

    dense_results, sparse_results = search_vectors_hybrid(
        query_vectors, user_id=user_id, top_k=20
    )

    if not dense_results and not sparse_results:
        return []

    fused_results = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    candidate_pool = fused_results[:20]
    reranked_results = rerank_candidates(query, candidate_pool, top_k=top_k)

    if min_score > 0.0:
        reranked_results = [result for result in reranked_results if result["score"] >= min_score]

    return reranked_results


def build_context(results: list[dict], max_chars: int = 4000) -> str:
    context_parts = []
    current_length = 0

    for i, result in enumerate(results, 1):
        part = f"[{i}] (score={result['score']:.4f}, rrf={result.get('rrf_score', 0.0):.4f}) {result['text']}"

        if current_length + len(part) > max_chars:
            break

        context_parts.append(part)
        current_length += len(part) + 2

    return "\n\n".join(context_parts)


def retrieve_context(query: str, top_k: int = 5, user_id: str = "") -> str:
    results = retrieve(query, top_k, user_id=user_id)
    return build_context(results)
