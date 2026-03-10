import os
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Folder where documents are located
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    raise ValueError(f"Folder {DATA_DIR} doesn't exist. Create your .txt files there.")

# Read .txt files and create docs
documents = []
for filename in os.listdir(DATA_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(Document(text=text))

print(f"Number of docs loaded: {len(documents)}")

# Embeddings model
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("rag_docs")

# Vector store
vector_store = ChromaVectorStore(chroma_collection=collection)

# Storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Save persistent index
index.storage_context.persist("./chroma_db")
print("✅ Indexed and saved docs")
