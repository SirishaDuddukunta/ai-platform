from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

# 1. We must delete the old one because you can't change the 'Distance' of an existing collection
client.delete_collection(collection_name="day19_test")

# 2. Create the production-grade version
client.create_collection(
    collection_name="production_docs",
    vectors_config=VectorParams(
        size=4,                # Keep it at 4 for your test, but usually 1536 for OpenAI
        distance=Distance.COSINE # <--- This is the key change!
    ),
)

print("✅ Collection recreated with COSINE similarity.")