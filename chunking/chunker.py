def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: list = None
) -> list[str]:
    """Recursive chunking manual verdadero — sin LangChain."""
    if separators is None:
        # Párrafos -> Líneas -> Frases -> Palabras -> Caracteres
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    chunks = []
    
    def split(text: str, seps: list):
        # 1. Si el texto entra perfecto, lo guardamos y salimos
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        
        sep = seps[0] if seps else ""
        parts = text.split(sep) if sep else list(text)
        
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            
            # 2. Si concatenarlo no se pasa del límite, seguimos acumulando
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                # 3. Nos pasamos. Guardamos lo que llevamos acumulado
                if current.strip():
                    chunks.append(current.strip())
                
                # Guardamos el overlap para conectar con lo que sigue
                tail = current[-overlap:] if overlap and current else ""
                
                # 4. LA VERDADERA RECURSIÓN
                # Si la 'part' individual es gigante por sí sola y aún tenemos separadores
                if len(part) > chunk_size and len(seps) > 1:
                    # Aplicamos recursión con el siguiente separador más fino (ej: de \n a " ")
                    split(part, seps[1:])
                    # current se vacía porque 'part' ya fue picada y añadida en la recursión
                    current = ""
                else:
                    # Empezamos el nuevo chunk con el overlap y la nueva parte
                    current = tail + sep + part if tail else part
        
        # 5. Guardar lo que sobre al final
        if current.strip():
            if len(current) > chunk_size and len(seps) > 1:
                split(current, seps[1:])
            else:
                chunks.append(current.strip())
    
    split(text, separators)
    return chunks
