from src.core.retriever import TwoStageRetriever

def run_evaluation():
    retriever = TwoStageRetriever()
    
    # Technical query testing for specific intent
    query = "How to optimize Qdrant collection for high-speed concurrent writes?"
    
    results = retriever.search(query)
    
    print(f"\n🚀 QUERY: {query}\n" + "="*50)
    
    if not results:
        print("❌ No relevant context found above the threshold.")
        return

    for idx, doc in enumerate(results):
        print(f"[{idx+1}] Rerank Score: {doc.rerank_score:.4f} | Retrieval Score: {doc.retrieval_score:.4f}")
        print(f"📄 Content: {doc.content[:120]}...\n")

if __name__ == "__main__":
    run_evaluation()