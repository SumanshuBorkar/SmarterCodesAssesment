from fastapi import APIRouter, HTTPException
from app.models import URLRequest, SearchResponse, IndexResponse, SearchResult, Chunk
from app.services.html_parser import HTMLParser
from app.services.tokenizer import ChunkTokenizer
from app.services.vector_db import VectorDatabase
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
html_parser = HTMLParser()
tokenizer = ChunkTokenizer()
vector_db = VectorDatabase()

@router.post("/index-url", response_model=IndexResponse)
async def index_url(request: URLRequest):
    """Index HTML content from URL"""
    try:
        # Fetch and parse HTML
        clean_text = html_parser.process_url(request.url)
        if not clean_text:
            raise HTTPException(status_code=400, detail="Failed to fetch or parse URL")
        
        # Chunk text
        chunks_data = tokenizer.chunk_text(clean_text)
        
        # Prepare chunks for indexing
        chunks = []
        for i, (chunk_text, start, end) in enumerate(chunks_data):
            chunks.append({
                "chunk_id": i,
                "content": chunk_text,
                "token_count": tokenizer.count_tokens(chunk_text),
                "start_position": start,
                "end_position": end
            })
        
        # Index in vector database
        total_chunks = vector_db.index_chunks(str(request.url), chunks)
        total_tokens = sum(chunk["token_count"] for chunk in chunks)
        
        return IndexResponse(
            message="URL indexed successfully",
            total_chunks=total_chunks,
            total_tokens=total_tokens
        )
        
    except Exception as e:
        logger.error(f"Error indexing URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search(request: URLRequest):
    """Search for query in indexed content"""
    try:
        # Search in vector database
        results = vector_db.search(request.query, str(request.url))
        
        # Format results
        search_results = []
        for i, result in enumerate(results):
            search_results.append(
                SearchResult(
                    chunk=Chunk(**result["chunk"]),
                    score=result["score"],
                    relevance_rank=i + 1
                )
            )
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_matches=len(search_results)
        )
        
    except Exception as e:
        logger.error(f"Error searching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = vector_db.get_collection_stats()
        return {
            "status": "healthy",
            "vector_db": "connected",
            "collection_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")