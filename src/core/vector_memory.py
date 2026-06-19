# src/core/vector_memory.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import uuid

# 1. Initialize Global Objects
encoder = SentenceTransformer('all-MiniLM-L6-v2')
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "chat_memory"

# 2. Initialization
def init_qdrant():
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
            )
        print("✅ Qdrant initialized")
    except Exception as e:
        # This prevents the app from crashing at startup
        print(f"⚠️ Qdrant could not be reached: {e}. Running in 'offline' mode.")

# 3. Storage (Upsert)
def upsert_to_vector_db(user_id: str, role: str, content: str):
    """Embeds and stores a single message turn."""
    point_id = str(uuid.uuid4())
    vector = encoder.encode(content).tolist()
    
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={"user_id": user_id, "role": role, "text": content}
            )
        ]
    )

# 4. Retrieval (Search)
def search_memory(user_id: str, query: str, limit: int = 3):
    """Finds top N most similar past interactions."""
    query_vector = encoder.encode(query).tolist()
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=limit
    )
    
    # Return just the text content of the matches
    return [hit.payload["text"] for hit in results]