import json
import os
from typing import Iterator

import requests

from providers.base import BaseGenerator


class NvidiaProvider(BaseGenerator):
    def __init__(
        self,
        model: str = "meta/llama-3.1-8b-instruct",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        api_key: str | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            response = requests.post(
                self._chat_completions_url(),
                headers=self._headers(),
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "")
        except Exception as exc:
            return f"Error conectando con NVIDIA: {exc}"

    def stream(self, prompt: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        try:
            with requests.post(
                self._chat_completions_url(),
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    chunk = line.removeprefix("data:").strip()
                    if chunk == "[DONE]":
                        break
                    data = json.loads(chunk)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    yield delta.get("content", "")
        except Exception as exc:
            yield f"\n[Error de Streaming NVIDIA: {exc}]"

    def is_available(self) -> tuple[bool, str]:
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code == 200:
                return (True, f"NVIDIA online and responding on {self.base_url}")
            return (False, f"NVIDIA returned status code: {response.status_code}")
        except requests.exceptions.RequestException as exc:
            return (False, f"Timeout or connection error connecting to NVIDIA: {exc}")
