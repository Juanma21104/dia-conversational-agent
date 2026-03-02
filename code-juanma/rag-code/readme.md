# RAG (FastAPI + Docling + ChromaDB)

Sistema RAG local. El sistema lee, procesa y fragmenta documentos de forma inteligente para ofrecer respuestas precisas basadas estrictamente en el contexto proporcionado.

## Características

* **Parsing:** usa `Docling` (Hybrid Chunker) para extraer texto entendiendo la estructura visual del documento (tablas, títulos, párrafos) sin romper el contexto.
* **Embeddings:** utiliza el modelo `BAAI/bge-m3` para búsquedas semánticas avanzadas en español e inglés.
* **Búsqueda (RAG-Fusion):** implementa *Multi-Query* y *Reciprocal Rank Fusion (RRF)* para recuperar los fragmentos más relevantes.
* **Local:** El LLM funciona de manera local a través de LM Studio.
* **Interfaz web (gradio):** incluye un chat con memoria y un *Inspector* para auditar qué fragmentos exactos del documento usó el modelo para responder.

---

## Requisitos

Antes de ejecutar el proyecto, asegúrate de tener instalado:
1. **Python 3.10+**
2. **Docker** (para levantar la base de datos ChromaDB).
3. **LM Studio** corriendo un modelo local (ej. `llama-3.2-3b-instruct`) con el servidor local activado en `http://127.0.0.1:1234/v1`.

---

## Instalación y uso

```bash
pip install -r requirements.txt

```

Para ChromaDB, hay un docker-compose para levantar el contenedor:
```bash
docker-compose up
```
Para eliminar la base de datos:
```bash
docker-compose down -v
```

Para el frontend:
```bash
python frontend.py
```

Para el backend:
```bash
uvicorn backend:app --port 8001 --reload 
```

