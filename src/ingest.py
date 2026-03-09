from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Embeddings model
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

# Create Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

collection = chroma_client.get_or_create_collection("rag_docs")

vector_store = ChromaVectorStore(chroma_collection=collection)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    vector_store=vector_store
)

print("Documents correctly indexed")