import chromadb # If using local chroma, or use qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

def ingest_data():
    # Setup Qdrant
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "project_docs"
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Create collection if it doesn't exist
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 384, "distance": "Cosine"}
        )

    # 2. Data to store
    docs = [
        {"text": "Alpha project is in the Design phase. Completion expected Q4 2026.", "id": 1},
        {"text": "Beta is a testing framework for infrastructure as code.", "id": 2}
    ]

    # 3. Embed and Upsert
    points = [
        PointStruct(
            id=doc["id"], 
            vector=embedder.encode(doc["text"]).tolist(), 
            payload={"text": doc["text"]}
        )
        for doc in docs
    ]
    
    client.upsert(collection_name=collection_name, points=points)
    print("Vector database populated!")

if __name__ == "__main__":
    ingest_data()