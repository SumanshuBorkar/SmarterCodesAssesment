from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class URLRequest(BaseModel):
    url: HttpUrl
    query: str

class Chunk(BaseModel):
    chunk_id: int
    content: str
    token_count: int
    start_position: int
    end_position: int

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    relevance_rank: int

class IndexResponse(BaseModel):
    message: str
    total_chunks: int
    total_tokens: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_matches: int