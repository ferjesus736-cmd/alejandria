# Alejandria

RAG distribuido con soporte multi-tenant, búsqueda híbrida, reranking y cola de trabajo para ingesta.

## Features

- Multi-tenant por `x-user-id`
- Almacenamiento en Qdrant con colecciones aisladas por usuario
- Búsqueda híbrida con dense + sparse vectors y RRF
- Reranking con `CrossEncoder`
- Ingesta asíncrona con Celery
- Providers de generación para Ollama, NVIDIA y Hugging Face local

## Requisitos

- Python 3.11+
- Redis si usas Celery real
- Opcional: Ollama, NVIDIA API o modelo local de Hugging Face

## Instalación local

```bash
pip install -r requirements.txt
```

## Arranque rápido

```bash
uvicorn app:app --reload
```

## Variables de entorno

- `LLM_PROVIDER`: `huggingface`, `nvidia` u `ollama`
- `HF_MODEL`: modelo local de Hugging Face
- `NVIDIA_API_KEY`: clave de NVIDIA API
- `NVIDIA_MODEL`: modelo NVIDIA a usar
- `OLLAMA_MODEL`: modelo de Ollama
- `CELERY_EXECUTION_MODE`: `celery` o `local`
- `CELERY_BROKER_URL`: broker de Celery
- `CELERY_RESULT_BACKEND`: backend de resultados

## Docker Compose

```bash
docker compose up --build
```

Levanta:

- `api` en `http://localhost:8000`
- `worker` para procesar documentos
- `redis` como broker de Celery

## Uso

Subida de documento:

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "x-user-id: fernando" \
  -F "file=@documento.txt"
```

Chat:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "x-user-id: fernando" \
  -d '{"question":"¿Qué dice mi documento?","top_k":5}'
```

## Notas

- El provider de Hugging Face local por defecto usa `HuggingFaceTB/SmolLM2-135M-Instruct`.
- Si prefieres NVIDIA, configura `LLM_PROVIDER=nvidia` y añade `NVIDIA_API_KEY`.
