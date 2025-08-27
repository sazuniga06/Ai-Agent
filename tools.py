from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import os
import tempfile
import shutil

def save_to_txt(data: str, filename: str = "outputs/research_output.txt"):
    """
    Guarda contenido de investigación en un archivo .txt dentro de la carpeta 'outputs/'.

    - Crea el directorio si no existe.
    - Rechaza guardados demasiado cortos (menos de 200 caracteres) devolviendo
      'REFUSE_SAVE: content_too_short' para evitar que el modelo intente
      guardar textos triviales o incompletos.
    - Incluye un timestamp al inicio de cada bloque guardado.
    - Añade contenido en modo append (no sobrescribe lo existente).
    
    Args:
        data (str): Texto a guardar en el archivo.
        filename (str): Ruta del archivo donde se guardará (por defecto 'outputs/research_output.txt').

    Returns:
        str: Mensaje de confirmación indicando la ruta del archivo,
             o una cadena de rechazo si el contenido es muy corto.
    """
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    # Bloquea guardados triviales (protección contra llamadas tempranas del agente)
    if len(data.strip()) < 200:
        return "REFUSE_SAVE: content_too_short"

    # Genera timestamp para el bloque
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = (
        f"--- Research Output ---\n"
        f"Timestamp: {timestamp}\n\n"
        f"{data.rstrip()}\n\n"
    )

    # Guardado en modo append
    with open(filename, "a", encoding="utf-8", newline="\n") as f:
        f.write(formatted)

    return f"Data successfully saved to {filename}"


# Tool para LangChain que permite al agente invocar la función de guardado.
save_tool = Tool(
    name="save_to_file",
    func=save_to_txt,
    description="Save research data to a text file"
)   


# Herramienta de búsqueda web usando DuckDuckGo
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the Web for information"
)


# Wrapper de Wikipedia que limita a 1 resultado y 100 caracteres de contenido por doc
# (ajústalo según la cantidad de contexto que quieras permitir).
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=100
)

# Herramienta de consulta a Wikipedia usando el wrapper configurado
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
