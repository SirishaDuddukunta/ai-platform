import os
from dotenv import load_dotenv
from src.core.factory import LLMFactory
from src.schemas.invoice import Invoice

load_dotenv()

def run_day4_chaos_test():
    factory = LLMFactory(provider="groq")
    raw_data = "I bought an AI chip from Silicon-Valley. Qty is 0, price is TBD."
    
    print("🚀 Running Resilience Test...")
    try:
        # These variables ONLY exist inside this 'try' block
        invoice, stats = factory.get_structured(Invoice, raw_data)
        
        print(f"✅ Success! Vendor: {invoice.vendor}")
        print(f"Total: ${invoice.total}")
        print(f"Items: {invoice.items}")
        print(f"Detailed Reasoning: {invoice.reasoning}")
        print(f"Tokens Used: {stats.total_tokens}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    run_day4_chaos_test()

from src.core.processor import DocumentProcessor

def run_day6_chunking_test():
    # 1. Simulate a long document
    mega_bill = "VENDOR: Silicon-Valley. " + ("ITEM: AI Chip, PRICE: 1200. " * 50)
    
    # 2. Process chunks
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.split_text(mega_bill)
    
    print(f"📦 Document split into {len(chunks)} chunks.")
    print(f"🔗 Sample Chunk 1: {chunks[0][:100]}...")

if __name__ == "__main__":
    run_day6_chunking_test()