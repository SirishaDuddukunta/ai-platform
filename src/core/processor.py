#  Chunking & Logic
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        # Setting the industry standard parameters for technical docs
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_text(self, text: str):
        """
        Splits raw text into chunks ready for Day 18's Vector DB.
        """
        chunks = self.splitter.split_text(text)
        return chunks