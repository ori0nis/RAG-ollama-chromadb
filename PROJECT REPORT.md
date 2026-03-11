# 1. Project Objective
Build a fully local RAG system that can answer a user's questions, powered by Llama3 LLM and supported by proprietary documents.

# 2. Technology Stack and Justification
- **Ollama** → local LLM + embeddings  
- **ChromaDB** → vector database for similarity search  
- **LlamaIndex** → to orchestrate the RAG pipeline  
- **Streamlit** → simple web interface  

# 3. Development / Implementation
- **Environment preparation:** `venv` virtual environment, installation of dependencies, local download of Ollama, download of models  
- **Indexing:** preparation of a simple `.txt` document, document reading, chunking, storing embeddings in ChromaDB  
- **Query pipeline:** question embeddings → fragment search → prompt to the LLM → generated response  
- **Interface:** creation of a Streamlit interface to test the model and confirm that responses and the fragments used are provided  

# 4. Problems Encountered and Solutions Applied
The implementation of ChromaDB and the document indexing worked from the beginning. However, `query.py` did cause problems due to version issues.

When searching for information in the `LlamaIndex` documentation, the recommendation was to implement `SimpleDirectoryReader` to format the documents and generate embeddings. However, this implementation is deprecated and because of this I was not able to get the LLM to answer the questions.

To fix this, I replaced `SimpleDirectoryReader` with `Document(text=...)`.

## Minor Problems
- I had to use `#type: ignore` to remove a minor typing issue with **Pylance** in `app.py`.
- I encountered a CUDA problem during the first launch of the frontend, so I had to terminate the **Ollama** process in PowerShell and restart it in VSCode.

# 5. Results
- The system responds correctly
- The fragments used as context are displayed
- It runs entirely locally, without relying on external APIs