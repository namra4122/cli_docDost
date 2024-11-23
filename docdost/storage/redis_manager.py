import logging
from typing import List,Dict,Any
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from sentence_transformers import SentenceTransformer
from ..config import Config

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self, config: Config):
        self.config = config,
        self.index = self._init_redis_storage()
    
    def _init_redis_storage(self) -> SearchIndex:
        storage_schema = {
            "index": { "name": "docDost_schema", "prefix": "docDost_schema_docs" },
            "fields": [
                {"name": "page_number","type":"numeric"},
                {"name": "sentence_chunks","type":"text"},
                {
                    "name": "byte_embeddings",
                    "type": "vector",
                    "attrs": {
                        "dims": self.config[0].embedding_size,
                        "distance_metric" :  "cosine",
                        "algorithm" : "flat",
                        "datatype" : "float32"
                    }
                }
            ],
        }

        try:
            index = SearchIndex.from_dict(storage_schema)
            redis_client = Redis.from_url(self.config[0].redis_url)
            index.set_client(redis_client)
            index.connect(self.config[0].redis_url)
            if(index.exists()):
                logger.info(f"Redis Index Exists")
            else:
                index.create(overwrite=True)
            return index
        
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
            raise
    
    def add_to_redis(self, data: List[Dict[str,Any]]):
        try:
            self.index.load(data)
        except Exception as e:
            logger.error(f"Error uploading data to Redis: {e}")
            raise
    
    def similarity_search(self, query: str, embedding_model: SentenceTransformer, topk: int = 5) -> List[Dict[str, Any]]:
        try:
            query_embedding = embedding_model.encode(query).tolist()
            vector_query = VectorQuery(
                vector=query_embedding,
                vector_field_name="byte_embeddings",
                return_fields=["sentence_chunks", "vector_distance"],
                num_results=topk
            )

            return self.index.query(vector_query)
        except Exception as e:
            logger.error(f"Error performing similarity search {e}")
            raise