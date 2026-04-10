import os
from dotenv import load_dotenv
from src.core.factory import LLMFactory
from src.core.processor import DocumentProcessor
from src.schemas.invoice import Invoice

load_dotenv()

def run_week1_capstone():
    factory = LLMFactory()
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    
    print("=== 🚀 AI PLATFORM: WEEK 1 CAPSTONE ===")
    print("1. Structured Invoice Extraction (Resilience Test)")
    print("2. Document Chunking (RAG Foundation)")
    print("3. Streaming Chatbot (UX Test)")
    
    choice = input("\nSelect a mode (1-3): ")

    if choice == "1":
        raw_data = "Acme Corp sent a bill for 2 widgets at $50 each. Total is $100."
        data, stats = factory.get_structured(Invoice, raw_data)
        print(f"\n✅ Extracted: {data.vendor} | Total: ${data.total}")
        print(f"📊 Usage: {stats.total_tokens} tokens")

    elif choice == "2":
        text = "This is a long document about DevOps and AI... " * 10
        chunks = processor.split_text(text)
        print(f"\n📦 Split document into {len(chunks)} chunks for the vector DB.")

    elif choice == "3":
        history = [{"role": "system", "content": "You are a Senior AI Engineer."}]
        user_input = input("\nYou: ")
        history.append({"role": "user", "content": user_input})
        
        print("AI: ", end="", flush=True)
        stream = factory.stream_chat(history)
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    run_week1_capstone()