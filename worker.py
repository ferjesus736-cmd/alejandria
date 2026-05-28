import logging
import os
import threading
import time
import uuid

from chunking.chunker import chunk_text
from embeddings.embedder import create_embeddings
from ingestion.reader import read_document
from vectorstore.qdrant_store import init_collection, insert_vectors

try:
    from celery import Celery
except ImportError:
    Celery = None

logger = logging.getLogger("Alejandria.Worker")

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_EXECUTION_MODE = os.getenv("CELERY_EXECUTION_MODE", "celery").lower()


def _process_document_impl(tmp_path: str, filename: str, user_id: str):
    start_time = time.time()

    try:
        logger.info("Procesando documento %s para usuario %s", filename, user_id)
        init_collection(user_id)

        text = read_document(tmp_path)
        chunks = chunk_text(text)

        if len(chunks) > 5000:
            raise ValueError(f"Upload descartado: {filename} excede 5000 chunks")

        embed_start = time.time()
        vectors = create_embeddings(chunks)
        embed_latency = time.time() - embed_start

        store_start = time.time()
        count = insert_vectors(chunks, vectors, doc_id=filename, user_id=user_id)
        store_latency = time.time() - store_start

        total_time = time.time() - start_time
        logger.info(
            "%s indexado con exito (user: %s): %s vectores. Embed: %.2fs, Store: %.2fs, Total: %.2fs",
            filename,
            user_id,
            count,
            embed_latency,
            store_latency,
            total_time,
        )

        return {
            "status": "completed",
            "doc": filename,
            "user_id": user_id,
            "chunks": count,
            "embed_seconds": round(embed_latency, 2),
            "store_seconds": round(store_latency, 2),
            "total_seconds": round(total_time, 2),
        }
    except Exception:
        logger.exception("Error procesando %s para %s", filename, user_id)
        raise
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            logger.warning("No se pudo eliminar el archivo temporal %s", tmp_path)


if Celery is not None and CELERY_EXECUTION_MODE != "local":
    celery_app = Celery(
        "alejandria",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND,
    )

    celery_app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        broker_connection_retry_on_startup=True,
        timezone="UTC",
        enable_utc=True,
    )

    process_document = celery_app.task(name="process_document")(_process_document_impl)
else:
    celery_app = None

    class _LocalAsyncResult:
        def __init__(self, task_id: str):
            self.id = task_id

    class _LocalTask:
        def delay(self, tmp_path: str, filename: str, user_id: str):
            task_id = str(uuid.uuid4())

            def runner():
                try:
                    _process_document_impl(tmp_path, filename, user_id)
                except Exception:
                    logger.exception("Error en la tarea local %s", task_id)

            thread = threading.Thread(target=runner, daemon=True)
            thread.start()
            return _LocalAsyncResult(task_id)

    process_document = _LocalTask()
