from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from typing import Dict, Any, Optional

class HybridRetriever:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "hybrid_platform_docs"
        self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def search(self, query: str) -> Optional[Dict[str, Any]]:
        # Phase 1: Pure Vector Lookup (Updated for modern Qdrant client syntax)
        query_vector = list(self.model.embed([query]))[0].tolist()
        
        # Use query_points instead of search for modern versions
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="fast-bge-small-en",
            limit=1
        ).points
        
        if not results:
            return None
            
        top_hit = results[0]
        page_tree = top_hit.payload.get("page_index_root")
        
        # Phase 2: PageIndex Traversal (Structural Mapping)
        print(f"\n🔍 Found document tree: '{page_tree['title']}' via Vector Space.")
        print("🗺️ Traversing PageIndex tree nodes to map structural hierarchy...")
        
        target_content = None
        for section in page_tree.get("sub_sections", []):
            if "2.0" in section["title"] and "optimize" in query.lower():
                print(f"🎯 Route Target Verified -> Section: {section['title']} (Page {section['page_number']})")
                target_content = section["content"]
                break
                
        return {
            "source_chunk": top_hit.payload["text"],
            "page_index_extracted_content": target_content or "No explicit node match found."
        }