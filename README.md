# 🧠 Research Agent con LangChain

Este proyecto implementa un **agente de investigación** utilizando [LangChain](https://www.langchain.com/), modelos de OpenAI/Anthropic y herramientas externas (búsqueda en DuckDuckGo, consultas a Wikipedia y guardado en archivos locales).

El flujo básico es:

1. El usuario introduce una consulta de investigación.
2. El agente usa herramientas (`search`, `wiki`) para obtener información.
3. Agrega los hallazgos y llama a la herramienta `save_to_file` para persistir los resultados.
4. Devuelve un objeto estructurado en formato **JSON**, validado con **Pydantic**.

---

## ⚙️ Requisitos

- Python 3.9+
- Paquetes principales:
  - `langchain`
  - `langchain-core`
  - `langchain-openai`
  - `langchain-community`
  - `pydantic>=2`
  - `python-dotenv`

Instala dependencias con:

```bash
pip install -r requirements.txt
```

## 🔑 Configuración de variables de entorno

Debes crear un archivo llamado **`.env`** en la raíz del proyecto.
Dentro coloca tus claves de API en este formato:

```env
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```

## ▶️ Ejecución

Para ejecutar el agente de investigación:

```bash
python main.py
```

El script pedirá por consola una consulta, por ejemplo:

```
What can I help you research? White Sharks
```

El agente buscará información, guardará los resultados en `outputs/research_output.txt` y devolverá una respuesta estructurada en JSON.

## 📂 Estructura del proyecto

```
├── __pycache__/             # Archivos cacheados de Python
├── venv/                    # Entorno virtual de Python
├── .env                     # Variables de entorno (con las API keys)
├── .gitignore               # Archivo que define qué no subir a git
├── main.py                  # Script principal del agente
├── README.md                # Documentación del proyecto
├── requirements.txt         # Dependencias del proyecto
└── tools.py                 # Definición de herramientas (search, wiki, save)
```

## 📝 Notas

* Si quieres cambiar el modelo, ajusta `ChatOpenAI(model="gpt-4o-mini")` en `main.py`.
* El guardado en TXT incluye un timestamp y evita guardar textos triviales de menos de 200 caracteres.
* Puedes ampliar el wrapper de Wikipedia (`doc_content_chars_max`) para traer más contenido.
