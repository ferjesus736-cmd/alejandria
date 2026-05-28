import hashlib
import logging
import os
import time
import tempfile

from fastapi import FastAPI, File, HTTPException, Header, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.generator import generate_answer, stream_answer
from providers.huggingface_provider import HuggingFaceProvider
from providers.nvidia_provider import NvidiaProvider
from providers.ollama_provider import OllamaProvider
from retrieval.retriever import build_context, retrieve
from worker import process_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Alejandria")

app = FastAPI(title="Alejandria")

def _build_provider():
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider_name in {"hf", "huggingface", "huggingface-local"}:
        return HuggingFaceProvider(model=os.getenv("HF_MODEL"))
    if provider_name == "nvidia" or os.getenv("NVIDIA_API_KEY"):
        return NvidiaProvider(
            model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
            api_key=os.getenv("NVIDIA_API_KEY"),
        )
    return OllamaProvider(model=os.getenv("OLLAMA_MODEL", "llama3"))


provider = _build_provider()

retrieval_cache = {}


@app.on_event("startup")
async def startup():
    logger.info("Iniciando Sistema RAG Alejandria (Fase Avanzada)...")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), x_user_id: str = Header(...)):
    with tempfile.NamedTemporaryFile(
        suffix=os.path.splitext(file.filename)[1],
        delete=False
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    task = process_document.delay(tmp_path, file.filename, x_user_id)
    return {
        "status": "queued",
        "task_id": task.id,
        "doc": file.filename,
        "user_id": x_user_id,
    }


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


def _get_context_for_query(question: str, top_k: int, user_id: str):
    safe_top_k = min(top_k, 8)

    cache_key = hashlib.md5(f"{user_id}_{question}_{safe_top_k}".encode()).hexdigest()
    if cache_key in retrieval_cache:
        logger.info("Cache HIT para query: '%s...'", question[:30])
        return retrieval_cache[cache_key]

    start_time = time.time()
    results = retrieve(question, safe_top_k, user_id=user_id)

    if results:
        logger.info(
            "Retrieval: %s chunks encontrados en %.2fs. Top score: %.2f",
            len(results),
            time.time() - start_time,
            results[0]["score"],
        )
    else:
        logger.warning("Retrieval vacio (sin contexto) para: '%s...'", question[:30])

    context = build_context(results)
    retrieval_cache[cache_key] = (context, results)
    return context, results


@app.post("/chat")
async def chat(req: ChatRequest, x_user_id: str = Header(...)):
    if not req.question.strip():
        raise HTTPException(400, "Pregunta vacia")

    context, results = await run_in_threadpool(
        _get_context_for_query, req.question, req.top_k, x_user_id
    )

    gen_start = time.time()
    answer = await run_in_threadpool(generate_answer, req.question, context, provider)
    logger.info("Generacion LLM completada en %.2fs", time.time() - gen_start)

    return {
        "answer": answer,
        "sources": [
            {
                "doc_id": result.get("doc_id"),
                "chunk": result.get("chunk_index"),
                "score": round(result["score"], 3),
                "rrf_score": round(result.get("rrf_score", 0.0), 4),
            }
            for result in results
        ],
    }


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, x_user_id: str = Header(...)):
    if not req.question.strip():
        raise HTTPException(400, "Pregunta vacia")

    context, _results = await run_in_threadpool(
        _get_context_for_query, req.question, req.top_k, x_user_id
    )

    def token_generator():
        logger.info("Iniciando stream de respuesta...")
        for token in stream_answer(req.question, context, provider):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")
