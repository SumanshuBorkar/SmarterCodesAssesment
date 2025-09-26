from transformers import AutoTokenizer
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class ChunkTokenizer:
    def __init__(self, model_name: str = "bert-base-uncased", max_tokens: int = 500):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
    
    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into chunks of maximum tokens"""
        if not text.strip():
            return []
        
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            
            # Try to break at sentence boundaries (simple approach)
            if end < len(tokens):
                # Look for a period, question mark, or exclamation mark to break naturally
                for i in range(end, start, -1):
                    if i < len(tokens) and self._is_sentence_boundary(tokens, i):
                        end = i
                        break
            
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            chunks.append((chunk_text, start, end))
            start = end
            
        return chunks
    
    def _is_sentence_boundary(self, tokens: List[int], position: int) -> bool:
        """Check if position is a reasonable sentence boundary"""
        # Simple check for punctuation tokens
        punctuation = ['.', '!', '?', '。', '！', '？']
        token_text = self.tokenizer.decode([tokens[position-1]])
        return any(punct in token_text for punct in punctuation)