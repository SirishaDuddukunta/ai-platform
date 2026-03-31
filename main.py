from src.core.factory import LLMFactory
from src.schemas.invoice import Invoice
from dotenv import load_dotenv
import sys

load_dotenv()

def run_invoice_task():
    factory = LLMFactory(provider="groq")
    raw_data = "Hey, I need to bill for 2 GPUs at 500 each. Wait, no—the price dropped to 450 today. Also, add a Monitor for 300. Actually, ignore the second GPU, just 1 GPU and 1 Monitor. Apply a 50 dollar loyalty discount at the end. Vendor is Tech-Hub (we used to use AI-Shop but not anymore)."
    invoice = factory.get_structured(Invoice, raw_data)
    print(f"\n[PARSED INVOICE]\nVendor: {invoice.vendor}\nTotal: ${invoice.total}")
def run_day3_task():
    factory = LLMFactory() # Defaults to Groq
    raw_data = "Bill 1 laptop for 1200. Apply 10% tax."
    
    # Receive the invoice AND the metrics
    invoice, stats = factory.get_structured(Invoice, raw_data)
    
    print(f"\n--- DAY 3 OBSERVABILITY REPORT ---")
    print(f"Provider: {stats.provider}")
    print(f"Tokens Used: {stats.total_tokens} (Prompt: {stats.prompt_tokens}, Completion: {stats.completion_tokens})")
    print(f"Estimated Cost: ${stats.total_tokens * 0.0000001:.6f}") # Example math for Llama-3
    print(f"----------------------------------")
    print(f"Final Invoice Total: ${invoice.total}")

if __name__ == "__main__":
    run_day3_task()
