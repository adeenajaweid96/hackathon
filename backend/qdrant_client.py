"""
Qdrant Client Wrapper for Vector Store Integration

This module provides a wrapper around the Qdrant client for vector storage
and similarity search operations in the RAG system.
"""

import asyncio
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """Wrapper class for Qdrant client operations"""

    def __init__(
        self,
        host: str = os.getenv("QDRANT_HOST", "localhost"),
        port: int = int(os.getenv("QDRANT_PORT", 6333)),
        api_key: Optional[str] = os.getenv("QDRANT_API_KEY"),
        collection_name: str = "book_content"
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = None

    async def init_client(self):
        """Initialize the Qdrant client"""
        try:
            # Initialize Qdrant client
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port
                )

            # Ensure collection exists
            await self._ensure_collection_exists()

            logger.info(f"Qdrant client initialized successfully for collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            raise

    async def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't"""
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                # Create collection
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),  # Default size for sentence transformers
                    # You can add other configurations as needed
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def upload_points(self, points: List[Dict], batch_size: int = 64) -> bool:
        """Upload points to the collection in batches"""
        try:
            # Prepare points for upload
            prepared_points = []
            for point in points:
                prepared_point = models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"]
                )
                prepared_points.append(prepared_point)

            # Upload in batches
            for i in range(0, len(prepared_points), batch_size):
                batch = prepared_points[i:i + batch_size]
                await self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1} of {len(prepared_points)//batch_size + 1}")

            logger.info(f"Successfully uploaded {len(prepared_points)} points to collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error uploading points: {e}")
            return False

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter: Optional[models.Filter] = None
    ) -> List[models.ScoredPoint]:
        """Search for similar vectors"""
        try:
            # Perform search
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filter
            )

            logger.info(f"Found {len(search_results)} results for search query")
            return search_results

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

    async def delete_collection(self) -> bool:
        """Delete the collection (use with caution!)"""
        try:
            await self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False

    async def get_collection_info(self):
        """Get information about the collection"""
        try:
            info = await self.client.get_collection(collection_name=self.collection_name)
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    async def count_points(self) -> int:
        """Count the number of points in the collection"""
        try:
            count = await self.client.count(
                collection_name=self.collection_name
            )
            return count.count
        except Exception as e:
            logger.error(f"Error counting points: {e}")
            return 0

    async def filter_search(
        self,
        query_vector: List[float],
        filters: Dict[str, Any],
        limit: int = 5
    ) -> List[models.ScoredPoint]:
        """Search with filters"""
        try:
            # Convert filters to Qdrant filter format
            filter_conditions = []
            for key, value in filters.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

            qdrant_filter = models.Filter(must=filter_conditions)

            # Perform filtered search
            search_results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=qdrant_filter
            )

            logger.info(f"Found {len(search_results)} results for filtered search")
            return search_results

        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []

    async def batch_search(
        self,
        query_vectors: List[List[float]],
        limit: int = 5
    ) -> List[List[models.ScoredPoint]]:
        """Perform multiple searches in batch"""
        try:
            search_requests = [
                models.SearchRequest(
                    vector=query_vector,
                    limit=limit
                )
                for query_vector in query_vectors
            ]

            results = await self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_requests
            )

            logger.info(f"Completed batch search with {len(query_vectors)} queries")
            return results

        except Exception as e:
            logger.error(f"Error in batch search: {e}")
            return []


# Example usage and testing function
async def test_qdrant_client():
    """Test function to verify Qdrant client functionality"""
    client_wrapper = QdrantClientWrapper()

    try:
        # Initialize client
        await client_wrapper.init_client()

        # Test collection info
        info = await client_wrapper.get_collection_info()
        print(f"Collection info: {info}")

        # Test point count
        count = await client_wrapper.count_points()
        print(f"Number of points in collection: {count}")

    except Exception as e:
        print(f"Error in test: {e}")


if __name__ == "__main__":
    # Run test if executed directly
    asyncio.run(test_qdrant_client())