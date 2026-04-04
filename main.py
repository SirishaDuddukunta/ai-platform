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