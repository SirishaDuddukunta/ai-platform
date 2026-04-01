import os
from dotenv import load_dotenv
from src.core.factory import LLMFactory
from src.schemas.invoice import Invoice

load_dotenv()

def run_day4_chaos_test():
    factory = LLMFactory(provider="groq")
    
    # CHAOS DATA: 0 is mathematically valid but logically invalid for an invoice
    # 'TBD' is a string, which violates the 'float' type for price.
    raw_data = """
    I bought a specialized AI chip from Silicon-Valley. 
    The quantity is 0 because it's a pre-order, and the price is 'TBD'. 
    """
    
    print("🚀 Day 4: Starting Self-Healing Test...")
    
    try:
        invoice, stats = factory.get_structured(Invoice, raw_data)
        
        print(f"\n--- DAY 4 RESILIENCE REPORT ---")
        print(f"Vendor: {invoice.vendor}")
        # Observe how the LLM handled the '0' and 'TBD'
        print(f"Parsed Quantity: {invoice.items[0].quantity}") 
        print(f"Reasoning: {invoice.reasoning}")
        print(f"Tokens Used: {stats.total_tokens}")
        print(f"----------------------------------")
        
    except Exception as e:
        print(f"❌ System Failed: {e}")

if __name__ == "__main__":
    run_day4_chaos_test()