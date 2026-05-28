from abc import ABC, abstractmethod
from typing import Iterator

class BaseGenerator(ABC):
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Genera respuesta dado un prompt."""
        pass
        
    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Genera respuesta mediante tokens en streaming (como ChatGPT)."""
        pass
    
    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """
        Verifica si el provider está disponible.
        Retorna (True, "Mensaje de éxito") o (False, "Motivo del error").
        """
        pass
