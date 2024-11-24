import re
from typing import List

class TextChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise ValueError("Input to chunk_text must be a string")
        if not text.strip():
            return [] 

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:] + sentence + " "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks
