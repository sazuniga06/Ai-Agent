"""
Research Agent (LangChain 0.2+ / Pydantic v2)

Este script implementa un agente de investigación que:
1) Usa herramientas (search/wiki) para reunir información.
2) Agrega hallazgos "en bruto".
3) Llama una herramienta de guardado con las notas completas.
4) Devuelve SOLO un JSON con el esquema definido por Pydantic.

Requisitos:
- langchain, langchain-core 0.2+
- pydantic 2.x
- langchain-openai
- Un archivo .env con OPENAI_API_KEY (o variable de entorno configurada)
- tools.py con: search_tool, wiki_tool, save_tool (Tools válidas de LangChain)

Notas:
- El AgentExecutor devuelve la salida final como STRING en result["output"].
- Si el modelo no devuelve JSON perfecto, simplemente se corta en silencio
  (no se muestran errores en consola).
"""

from __future__ import annotations

from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool  # Deben ser Tool/StructuredTool válidos
import json
import sys


# Carga variables de entorno (por ejemplo, OPENAI_API_KEY desde .env)
load_dotenv()


class ResearchResponse(BaseModel):
    """
    Esquema de salida estructurada del agente.

    Atributos:
        topic: Tema central de la investigación.
        summary: Resumen final (conciso) de los hallazgos.
        sources: Lista de fuentes consultadas (URLs o títulos).
        tools_used: Nombres de herramientas utilizadas por el agente.
    """
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]


def build_parser() -> PydanticOutputParser:
    """
    Crea el parser de salida basado en Pydantic v2.

    En LangChain moderno, el argumento correcto es `pydantic_object`
    (no `pydantic_model`, que era de Pydantic v1).
    """
    return PydanticOutputParser(pydantic_object=ResearchResponse)


def build_llm() -> ChatOpenAI:
    """
    Instancia el modelo de lenguaje a usar.

    Ajusta temperature/max_tokens según lo necesites.
    - gpt-4o-mini es económico y suficiente para prototipos.
    """
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def build_prompt(parser: PydanticOutputParser) -> ChatPromptTemplate:
    """
    Define el prompt del agente.

    Uso de MessagesPlaceholder:
    - chat_history: historial opcional (lista de mensajes).
    - agent_scratchpad: espacio que el agente usa para registrar llamadas a tools.
    """
    system_instructions = (
        "You are a research assistant.\n"
        "Follow this exact order:\n"
        "1) Use search/wiki tools to gather information. Do NOT call save yet.\n"
        "2) Aggregate FULL raw findings into comprehensive notes (no truncation).\n"
        "3) Call the `save_to_file` tool ONCE with the FULL notes (plain text).\n"
        "4) Finally, return ONLY the JSON per schema:\n{format_instructions}\n"
        "Never call `save_to_file` before you have gathered and aggregated the notes."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instructions),
            MessagesPlaceholder("chat_history"),   # Puedes omitirlo si no usarás historial
            ("human", "{query}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt


def build_agent(parser: PydanticOutputParser) -> AgentExecutor:
    """
    Crea el agente con sus herramientas y ejecutor.

    - create_tool_calling_agent: compone LLM + prompt + tools.
    - AgentExecutor: orquesta la ejecución del agente.
      Por defecto, retorna `{"output": "<string JSON final>"}`.
    """
    llm = build_llm()
    prompt = build_prompt(parser)

    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)

    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def run(query: str) -> None:
    """
    Ejecuta el agente para una consulta dada y muestra la salida estructurada.

    Flujo:
        - Invoca el agente con la query.
        - Parsea el string JSON resultante a ResearchResponse.
        - Si la salida no es JSON válido, simplemente se corta en silencio.
    """
    parser = build_parser()
    agent_executor = build_agent(parser)

    raw_response = agent_executor.invoke({
        "query": query,
        "chat_history": []  # Si no usas historial, deja la lista vacía o elimina el placeholder.
    })

    output_str = raw_response.get("output", "")

    try:
        structured: ResearchResponse = parser.parse(output_str)
        print(structured.model_dump_json(indent=2, ensure_ascii=False))
    except Exception:
        # Si no se puede parsear (ej. 'Agent stopped due to max iterations.'),
        # simplemente se corta en silencio sin mostrar error.
        pass


def main() -> None:
    """
    Punto de entrada de la aplicación.
    Pide una consulta por stdin si no se pasó como argumento CLI.
    """
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("What can I help you research? ").strip()

    if not query:
        print("Empty query. Aborting.")
        return

    run(query)


if __name__ == "__main__":
    main()
