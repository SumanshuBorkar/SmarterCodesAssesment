from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, host: str = "localhost", port: str = "19530", 
                 model_name: str = "all-MiniLM-L6-v2"):
        self.host = host
        self.port = port
        self.model = SentenceTransformer(model_name)
        self.collection_name = "html_chunks"
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.collection = None
        
        self._connect()
        self._create_collection()
    
    def _connect(self):
        """Connect to Milvus"""
        try:
            connections.connect("default", host=self.host, port=self.port)
            logger.info("Connected to Milvus successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise
    
    def _create_collection(self):
        """Create collection if it doesn't exist"""
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4000),
                FieldSchema(name="token_count", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
            ]
            
            schema = CollectionSchema(fields, "HTML chunks with embeddings")
            self.collection = Collection(self.collection_name, schema)
            
            # Create index for faster search
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index("embedding", index_params)
            logger.info("Collection created successfully")
        else:
            self.collection = Collection(self.collection_name)
            logger.info("Collection loaded successfully")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        return self.model.encode(text).tolist()
    
    def index_chunks(self, url: str, chunks: List[Dict]) -> int:
        """Index chunks in vector database"""
        if not chunks:
            return 0
        
        # Prepare data for insertion
        data = {
            "url": [url] * len(chunks),
            "chunk_id": [chunk["chunk_id"] for chunk in chunks],
            "content": [chunk["content"] for chunk in chunks],
            "token_count": [chunk["token_count"] for chunk in chunks],
            "embedding": [self.generate_embedding(chunk["content"]) for chunk in chunks]
        }
        
        # Insert data
        insert_result = self.collection.insert(data)
        self.collection.flush()
        
        logger.info(f"Indexed {len(chunks)} chunks for URL: {url}")
        return len(chunks)
    
    def search(self, query: str, url: str = None, limit: int = 10) -> List[Dict]:
        """Search for similar chunks"""
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Build filter
        filter_expr = None
        if url:
            filter_expr = f'url == "{url}"'
        
        # Perform search
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=filter_expr,
            output_fields=["url", "chunk_id", "content", "token_count"]
        )
        
        # Format results
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append({
                    "chunk": {
                        "chunk_id": hit.entity.get("chunk_id"),
                        "content": hit.entity.get("content"),
                        "token_count": hit.entity.get("token_count"),
                        "start_position": 0,  # Would need to store this
                        "end_position": 0     # Would need to store this
                    },
                    "score": float(hit.score),
                    "id": hit.id
                })
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        if not self.collection:
            return {}
        
        return {
            "total_entities": self.collection.num_entities,
            "collection_name": self.collection_name
        }