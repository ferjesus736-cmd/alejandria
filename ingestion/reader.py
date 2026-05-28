from pathlib import Path
from pdfminer.high_level import extract_text
import re

def read_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No encontrado: {path}")
    
    try:
        if p.suffix.lower() == ".pdf":
            text = extract_text(path)
        elif p.suffix.lower() == ".txt":
            text = p.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Formato no soportado: {p.suffix}")
            
        # Normalización de texto (limpiar espacios y saltos extra, clave para PDFs)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    except Exception as e:
        raise RuntimeError(f"Error al leer el documento {path}: {str(e)}")
