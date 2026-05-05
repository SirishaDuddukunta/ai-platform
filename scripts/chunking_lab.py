import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

# 1. Load your raw technical documentation
raw_text = """
### API Authentication
To use the API, you must provide a valid API key in the `X-API-KEY` header.
Rate limits are strictly enforced at 100 requests per minute.

### Error Codes
- 401: Unauthorized access. Check your API key.
- 429: Rate limit exceeded. Implement exponential backoff.
- 500: Internal server error. Contact the DevOps team.
"""

# 2. Strategy A: Fixed-Size (Baseline)
# Fast, but often splits words or technical terms in half.
fixed_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0
)

# 3. Strategy B: Recursive (The 2026 Standard)
# Attempts to split by \n\n (paragraphs), then \n (lines), then spaces.
# Uses the roadmap's recommended 500/50 ratio.
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

def analyze_chunks(splitter, text, name):
    chunks = splitter.split_text(text)
    print(f"\n--- {name} Results ---")
    print(f"Total Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:2]):  # Peek at first 2
        print(f"Chunk {i+1} length: {len(chunk)}")
        print(f"Content: {repr(chunk[:50])}...")

# Execute the comparison
analyze_chunks(fixed_splitter, raw_text, "Fixed-Size")
analyze_chunks(recursive_splitter, raw_text, "Recursive")