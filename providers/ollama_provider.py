import requests
import json
from typing import Iterator
from providers.base import BaseGenerator

class OllamaProvider(BaseGenerator):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as e:
            return f"Error conectando con Ollama: {str(e)}"
            
    def stream(self, prompt: str) -> Iterator[str]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))
                        yield data.get("response", "")
        except Exception as e:
            yield f"\n[Error de Streaming: {str(e)}]"
            
    def is_available(self) -> tuple[bool, str]:
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return (True, f"Ollama online and responding on {self.base_url}")
            return (False, f"Ollama returned status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            return (False, f"Timeout or connection error connecting to model server: {str(e)}")
