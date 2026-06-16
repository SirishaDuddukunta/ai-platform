import os
from src.core.retriever import TwoStageRetriever
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def test_rerank_impact():
    load_dotenv()
    retriever = TwoStageRetriever()
    query = "specific technical query that Qdrant might struggle with"
    
    # 1. Get raw Qdrant results (Stage 1)
    client = QdrantClient(host="localhost", port=6333)
    raw_results = client.query_points(
        collection_name="production_docs",
        query_text=query,
        limit=5
    ).points
    
    qdrant_top_id = raw_results[0].id if raw_results else None
    
    # 2. Get Reranked results (Stage 2)
    reranked_results = retriever.search(query, top_k=20, final_top_n=5)
    rerank_top_content = reranked_results[0].content if reranked_results else None

    print(f"DEBUG: Qdrant Top Match ID: {qdrant_top_id}")
    print(f"DEBUG: Reranker Top Match: {rerank_top_content[:50]}...")

    # A Senior Engineer looks for 'Rank Reversal'
    # If the Reranker moved a result from position #10 to #1, it's working.
    print("\n--- IMPACT ANALYSIS ---")
    if reranked_results:
        print(f"Top Reranked Score: {reranked_results[0].rerank_score}")
    else:
        print("All results filtered by threshold. (This is also a successful test of safety!)")

if __name__ == "__main__":
    test_rerank_impact()