# Local RAG Question–Answering System

This project implements a fully local **RAG (Retrieval-Augmented Generation)** system using Python.

## Technologies

- [Ollama](https://ollama.com) - Local LLM and embeddings  
- [ChromaDB](https://www.trychroma.com) - Vector database  
- [LlamaIndex](https://www.llamaindex.ai) - RAG pipeline orchestrator  
- [Streamlit](https://streamlit.io) - Web interface  

## How to Use

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
```bash
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

3. Installl dependencies:
```bash
pip install -r requirements.txt
```

4. Place .txt files inside data/

5. Index the documents:
```bash
python src/ingest.py
```

6. Run the web app:
```bash
streamlit run src/app.py
```