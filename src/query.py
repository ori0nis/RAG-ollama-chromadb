from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Load Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("rag_docs")

vector_store = ChromaVectorStore(chroma_collection=collection)

embed_model = OllamaEmbedding(model_name="nomic-embed-text")

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

# Language model
llm = Ollama(model="llama3")

query_engine = index.as_query_engine(llm=llm)
question = input("Ask a question: ")
response = query_engine.query(question)

print("\nResponse: ")
print(response)

print("\nUsed context:")
for node in response.source_nodes:
    print(node.text)

retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(question)

print("Nodes recovered:", len(nodes))