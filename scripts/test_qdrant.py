from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding

client = QdrantClient(host="localhost", port=6333)
collection_name = "production_docs"

# 1. RESET: Fresh start
if client.collection_exists(collection_name):
    print(f"🗑️ Deleting old collection '{collection_name}'...")
    client.delete_collection(collection_name)

# 2. CREATE: Explicitly define the named vector slot
print(f"🏗️ Creating fresh collection: {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "fast-bge-small-en": VectorParams(size=384, distance=Distance.COSINE)
    }
)

# 3. EMBED: Generate a real AI vector matching the retriever's model
print("🧠 Generating real embedding using FastEmbed...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
text_content = "To optimize Qdrant for high-speed writes, use the gRPC interface and increase the indexing threshold."
embedding = list(model.embed([text_content]))[0].tolist()

# 4. UPSERT: Push directly into the named vector slot
print("💉 Injecting point into 'fast-bge-small-en' slot...")
client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=1,
            vector={"fast-bge-small-en": embedding},
            payload={"text": text_content}
        )
    ]
)

print("✅ Setup complete! Database is synced and production-ready.")