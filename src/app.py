import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

st.title("Local RAG QA System")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("rag_docs")

vector_store = ChromaVectorStore(chroma_collection=collection)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
llm = Ollama(model="llama3")
query_engine = index.as_query_engine(llm=llm)

# LLM feedback
user_question = st.text_input("Ask a question")

if user_question:
    response = query_engine.query(user_question)

    st.subheader("Response: ")
    st.write(response.response) # type: ignore

    st.subheader("Fragments used: ")
    for node in response.source_nodes:
        st.write(node.node.get_content())
