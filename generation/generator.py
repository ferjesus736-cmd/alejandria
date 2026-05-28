from providers.base import BaseGenerator
from typing import Iterator

SYSTEM_PROMPT = """
Eres un asistente RAG.

REGLAS:
- Usa únicamente el contexto proporcionado.
- No infieras ni completes información externa.
- Si no hay evidencia directa en el contexto, responde "No encontrado".
- Sé preciso y directo.

FORMATO:
Responde en español claro.
"""

def build_prompt(query: str, context: str) -> str:
    # Estructura limpia y estricta para asegurar que el modelo diferencie el contexto de las instrucciones
    return f"""{SYSTEM_PROMPT}

[INICIO DEL CONTEXTO EXTRAÍDO]
{context}
[FIN DEL CONTEXTO EXTRAÍDO]

PREGUNTA DEL USUARIO:
{query}

RESPUESTA:"""

def generate_answer(query: str, context: str, provider: BaseGenerator) -> str:
    prompt = build_prompt(query, context)
    return provider.generate(prompt)

def stream_answer(query: str, context: str, provider: BaseGenerator) -> Iterator[str]:
    prompt = build_prompt(query, context)
    return provider.stream(prompt)
