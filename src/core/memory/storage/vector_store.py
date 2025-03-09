# src/core/memory/storage/vector_store.py
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import os
import chromadb
from chromadb.config import Settings
import time
import asyncio

from .base import StorageBase
from src.config.config import DatabaseConfig

logger = logging.getLogger(__name__)

class VectorStoreManager(StorageBase):
    """ChromaDB vector store implementation with improved error handling"""
    
    def __init__(self):
        self.client = None
        self.collections = {}
        self.retry_attempts = 3
        self.retry_delay = 1  # seconds
        self.operation_locks = {}
        self.collection_stats = {}

    async def initialize(self) -> bool:
        """Initialize ChromaDB with retry logic"""
        try:
            # Ensure persistence directory exists
            os.makedirs(DatabaseConfig.CHROMA_DIR, exist_ok=True)

            # Initialize ChromaDB client with retries
            for attempt in range(self.retry_attempts):
                try:
                    self.client = chromadb.PersistentClient(
                        path=DatabaseConfig.CHROMA_DIR,
                        settings=Settings(
                            anonymized_telemetry=False,
                            allow_reset=True
                        )
                    )
                    break
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    time.sleep(self.retry_delay)

            # Initialize collections with dimension specification
            collections = ["episodic", "semantic", "contextual", "general", "abstracted"]
            for layer in collections:
                collection_name = f"{layer}_memories"
                try:
                    # Create collection with specific embedding dimensions for nomic-embed-text
                    self.collections[layer] = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": f"Memory storage for {layer} layer"},
                        embedding_function=None  # We'll provide embeddings directly
                    )
                    # Initialize operation lock for this collection
                    self.operation_locks[layer] = asyncio.Lock()
                    # Initialize stats for this collection
                    self.collection_stats[layer] = {
                        "total_documents": 0,
                        "operations": {"add": 0, "update": 0, "delete": 0, "query": 0},
                        "errors": {"add": 0, "update": 0, "delete": 0, "query": 0}
                    }
                    logger.info(f"Retrieved existing collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error creating collection {collection_name}: {str(e)}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            return False

    async def validate_connection(self) -> bool:
        """Validate ChromaDB connection and collections"""
        try:
            if not self.client:
                logger.error("ChromaDB client not initialized")
                return False

            for layer in self.collections:
                collection = self.collections[layer]
                try:
                    # Test basic operations
                    test_id = f"test_{layer}_{datetime.now().timestamp()}"
                    test_embedding = [0.0] * 768  # Dimension for nomic-embed-text
                    
                    # Test add operation
                    collection.add(
                        ids=[test_id],
                        embeddings=[test_embedding],
                        documents=["test document"],
                        metadatas=[{"test": True}]
                    )
                    
                    # Test query operation
                    results = collection.get(ids=[test_id])
                    if not results['ids']:
                        raise Exception(f"Failed to retrieve test document in {layer}")
                    
                    # Test delete operation
                    collection.delete(ids=[test_id])
                    
                    # Verify deletion
                    results = collection.get(ids=[test_id])
                    if results['ids']:
                        raise Exception(f"Failed to delete test document in {layer}")
                        
                except Exception as e:
                    logger.error(f"Validation failed for collection {layer}: {str(e)}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating vector store: {str(e)}")
            return False

    async def store_memory(
        self,
        key: str,
        value: Dict[str, Any],
        layer: str,
        memory_hash: str,
        importance_score: float
    ) -> bool:
        """Store memory in vector store with improved error handling"""
        if layer not in self.collections or layer not in self.operation_locks:
            return False

        async with self.operation_locks[layer]:
            try:
                collection = self.collections[layer]
                
                # Prepare document and metadata
                document = f"{value.get('input', '')} {value.get('response', '')}"
                metadata = {
                    "timestamp": datetime.now().timestamp(),
                    "layer": layer,
                    "importance_score": importance_score,
                    "memory_hash": memory_hash,
                    "key": key
                }

                # Get embedding from value
                embedding = value.get('embedding')
                if not embedding:
                    return False

                # Check if document exists first
                try:
                    existing_docs = collection.get(
                        ids=[memory_hash],
                        include=['metadatas']
                    )
                    
                    if existing_docs and existing_docs['ids']:
                        # Update existing document
                        collection.update(
                            ids=[memory_hash],
                            embeddings=[embedding],
                            documents=[document],
                            metadatas=[metadata]
                        )
                        self.collection_stats[layer]["operations"]["update"] += 1
                    else:
                        # Add new document
                        collection.add(
                            ids=[memory_hash],
                            embeddings=[embedding],
                            documents=[document],
                            metadatas=[metadata]
                        )
                        self.collection_stats[layer]["operations"]["add"] += 1
                        self.collection_stats[layer]["total_documents"] += 1
                    
                    return True

                except Exception as e:
                    # If error occurs during update/add, try to recreate the document
                    logger.warning(f"Error during document operation, attempting recreation: {str(e)}")
                    try:
                        # Delete if exists
                        collection.delete(ids=[memory_hash])
                        # Add as new
                        collection.add(
                            ids=[memory_hash],
                            embeddings=[embedding],
                            documents=[document],
                            metadatas=[metadata]
                        )
                        self.collection_stats[layer]["operations"]["add"] += 1
                        self.collection_stats[layer]["total_documents"] += 1
                        return True
                    except Exception as e2:
                        logger.error(f"Failed to recreate document: {str(e2)}")
                        self.collection_stats[layer]["errors"]["add"] += 1
                        return False

            except Exception as e:
                logger.error(f"Error storing memory in vector store: {str(e)}")
                self.collection_stats[layer]["errors"]["add"] += 1
                return False

    async def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        layer: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from vector store with improved handling"""
        if layer not in self.collections or layer not in self.operation_locks:
            return []

        # Ensure limit is positive
        limit = max(1, limit)

        async with self.operation_locks[layer]:
            try:
                collection = self.collections[layer]
                
                # Get total count of documents in collection
                collection_size = collection.count()
                if collection_size == 0:
                    return []
                    
                # Adjust limit if it exceeds collection size
                limit = min(limit, collection_size)
                
                # Query similar documents with retry logic
                for attempt in range(self.retry_attempts):
                    try:
                        results = collection.query(
                            query_embeddings=[embedding],
                            n_results=limit
                        )
                        
                        self.collection_stats[layer]["operations"]["query"] += 1
                        
                        memories = []
                        if results and 'documents' in results:
                            for idx, (doc, metadata) in enumerate(zip(
                                results['documents'][0],
                                results['metadatas'][0]
                            )):
                                try:
                                    similarity = 1.0 - results['distances'][0][idx]
                                    memories.append({
                                        "content": doc,
                                        "metadata": metadata,
                                        "similarity": similarity
                                    })
                                except Exception as e:
                                    logger.error(f"Error processing result {idx}: {str(e)}")
                                    continue
                        
                        return memories

                    except Exception as e:
                        if attempt == self.retry_attempts - 1:
                            raise
                        await asyncio.sleep(self.retry_delay)

            except Exception as e:
                logger.error(f"Error retrieving memories from vector store: {str(e)}")
                self.collection_stats[layer]["errors"]["query"] += 1
                return []

    async def cleanup_old_memories(
        self,
        days_threshold: int,
        importance_threshold: float
    ) -> Dict[str, int]:
        """Clean up old memories from vector store"""
        cleanup_stats = {"removed": 0}
        
        try:
            cutoff_timestamp = (datetime.now() - timedelta(days=days_threshold)).timestamp()
            
            for layer, collection in self.collections.items():
                async with self.operation_locks[layer]:
                    try:
                        # Get all documents with metadata
                        results = collection.get(
                            where={
                                "$and": [
                                    {"timestamp": {"$lt": cutoff_timestamp}},
                                    {"importance_score": {"$lt": importance_threshold}}
                                ]
                            }
                        )
                        
                        if results and results['ids']:
                            collection.delete(ids=results['ids'])
                            deleted_count = len(results['ids'])
                            cleanup_stats["removed"] += deleted_count
                            
                            # Update stats
                            self.collection_stats[layer]["operations"]["delete"] += 1
                            self.collection_stats[layer]["total_documents"] -= deleted_count
                            
                    except Exception as e:
                        logger.error(f"Error cleaning up collection {layer}: {str(e)}")
                        self.collection_stats[layer]["errors"]["delete"] += 1
                        continue

            return cleanup_stats

        except Exception as e:
            logger.error(f"Error cleaning up vector store: {str(e)}")
            return cleanup_stats

    async def cleanup(self) -> None:
        """Cleanup vector store resources"""
        try:
            # Save collection statistics
            stats_file = os.path.join(
                DatabaseConfig.DATA_DIR,
                f"vector_store_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(stats_file, 'w') as f:
                json.dump(self.collection_stats, f, indent=2)
            
            # Clear collections and client
            self.collections.clear()
            self.operation_locks.clear()
            self.collection_stats.clear()
            self.client = None
            
        except Exception as e:
            logger.error(f"Error cleaning up vector store: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self.collection_stats.copy()

    def __str__(self) -> str:
        """String representation"""
        total_docs = sum(
            stats["total_documents"]
            for stats in self.collection_stats.values()
        )
        return f"VectorStoreManager(collections={len(self.collections)}, total_documents={total_docs})"

