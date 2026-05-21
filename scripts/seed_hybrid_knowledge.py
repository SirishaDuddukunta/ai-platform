from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from src.core.processor import DocumentProcessor

client = QdrantClient(host="localhost", port=6333)
collection_name = "hybrid_platform_docs"
processor = DocumentProcessor()

# 1. Reset Environment
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "fast-bge-small-en": VectorParams(size=384, distance=Distance.COSINE)
    }
)

# Raw unstructured document input
sample_document = (
    "Platform Architecture Guidelines. Section 1.0 System Architecture Overview: "
    "The system utilizes a dual-engine core separating vectors from structural indices. "
    "Section 2.0 Database Optimization Configurations: To optimize high-speed concurrent writes "
    "on Qdrant, activate the gRPC interface on port 6334 and raise the indexing threshold to 25000."
)

# 2. Extract Data for Both Worlds
page_index_tree = processor.parse_to_pageindex(sample_document, doc_title="Platform Architecture Guide")
linear_chunks = processor.chunk_for_vector_db(sample_document, chunk_size=30)

# 3. Embedding Generation for Vector DB
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

print("⚡ Injecting dual-layer knowledge base into Qdrant...")
for item in linear_chunks:
    vector = list(model.embed([item["text"]]))[0].tolist()
    
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=item["chunk_id"],
                vector={"fast-bge-small-en": vector},
                payload={
                    "text": item["text"],
                    "chunk_id": item["chunk_id"],
                    # Attaching the structural pageindex tree to the metadata payload
                    "page_index_root": page_index_tree.model_dump() 
                }
            )
        ]
    )

print("🚀 Successfully seeded both Vector Embeddings and PageIndex trees!")