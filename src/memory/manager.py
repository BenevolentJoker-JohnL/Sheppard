"""
Enhanced memory manager with proper embedding generation and storage.
File: src/memory/manager.py
"""

import logging
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime

from src.memory.stores.redis import RedisMemoryStore
from src.memory.stores.chroma import ChromaMemoryStore
from src.memory.models import Memory, MemorySearchResult
from src.memory.exceptions import (
    MemoryStoreConnectionError,
    StorageError,
    RetrievalError,
    MemoryValidationError
)
from src.llm.client import OllamaClient
from src.config.settings import settings

logger = logging.getLogger(__name__)

class MemoryManager:
    """Manages memory operations with enhanced embedding support."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.redis_store = RedisMemoryStore()
        self.chroma_store = ChromaMemoryStore()
        self.ollama_client = None
        self._initialized = False
        self._active_conversations: Dict[str, Dict[str, Any]] = {}
        self._stored_memories: Set[str] = set()
        
        # Configure embedding settings
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self.embedding_batch_size = settings.EMBEDDING_BATCH_SIZE
        
        # Configure retry settings
        self.max_retries = settings.MAX_RETRIES
        self.retry_delay = settings.RETRY_DELAY
        
        # Fallback mode settings
        self._fallback_mode = False
        self._failed_embeddings: Set[str] = set()

    def set_ollama_client(self, client: OllamaClient) -> None:
        """Set the Ollama client instance."""
        self.ollama_client = client
        logger.info("Ollama client set in memory manager")

    async def initialize(self) -> None:
        """Initialize memory stores and load existing data."""
        if self._initialized:
            return

        try:
            # Verify Ollama client is set
            if not self.ollama_client:
                raise MemoryStoreConnectionError(
                    "memory manager",
                    "Ollama client not set. Call set_ollama_client() before initialize()"
                )
            
            # Initialize Redis store first for fast retrieval
            await self.redis_store.initialize()
            
            # Initialize ChromaDB store with Ollama client for embeddings
            await self.chroma_store.initialize(self.ollama_client)
            
            self._initialized = True
            logger.info("Memory manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory manager: {str(e)}")
            await self.cleanup()
            raise MemoryStoreConnectionError("memory manager", str(e))

    async def store(
        self,
        memory: Union[Memory, Dict[str, Any]],
        generate_embedding: bool = True,
        retry_on_failure: bool = True
    ) -> str:
        """Store a memory with embedding generation."""
        if not self._initialized:
            raise StorageError("Memory manager not initialized")
            
        try:
            # Convert dict to Memory if needed
            if isinstance(memory, dict):
                memory = Memory(**memory)
            
            # Ensure memory content is a string
            if not isinstance(memory.content, str):
                if isinstance(memory.content, list):
                    memory.content = "\n".join(str(item) for item in memory.content)
                else:
                    memory.content = str(memory.content)
                    
            # Check if we're in fallback mode
            if self._fallback_mode and not generate_embedding:
                logger.warning("Operating in fallback mode - skipping embedding generation")
                memory_id = await self._store_without_embedding(memory)
                return memory_id
            
            # Generate embedding if requested and not present
            if generate_embedding and not memory.embedding and self.ollama_client:
                try:
                    memory.embedding = await self._generate_embedding_with_retries(
                        memory.content,
                        max_retries=self.max_retries if retry_on_failure else 1
                    )
                except Exception as e:
                    if retry_on_failure:
                        logger.error(f"Failed to generate embedding after retries: {str(e)}")
                        # Switch to fallback mode
                        self._fallback_mode = True
                        self._failed_embeddings.add(memory.embedding_id)
                        memory_id = await self._store_without_embedding(memory)
                        return memory_id
                    else:
                        raise
            
            # Validate embedding if present
            if memory.embedding:
                self._validate_embedding(memory.embedding)
            
            # Store in both backends
            memory_id = await self.chroma_store.store(memory)
            await self.redis_store.store(memory)
            
            # Track stored memory
            self._stored_memories.add(memory_id)
            
            # Update conversation tracking if applicable
            conversation_id = memory.metadata.get('conversation_id')
            if conversation_id and conversation_id in self._active_conversations:
                self._active_conversations[conversation_id]['memories'].append(memory_id)
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {str(e)}")
            raise StorageError(f"Failed to store memory: {str(e)}")

    async def _store_without_embedding(self, memory: Memory) -> str:
        """Store memory without embedding in fallback mode."""
        try:
            # Store in Redis only
            memory_id = await self.redis_store.store(memory)
            self._stored_memories.add(memory_id)
            logger.info(f"Stored memory {memory_id} without embedding in fallback mode")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to store memory in fallback mode: {str(e)}")
            raise StorageError(f"Failed to store memory in fallback mode: {str(e)}")

    async def _generate_embedding_with_retries(
        self,
        content: str,
        max_retries: int = 3
    ) -> Optional[List[float]]:
        """Generate embedding with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                embedding = await self.ollama_client.generate_embedding(content)
                
                # Validate embedding
                self._validate_embedding(embedding)
                return embedding
                
            except Exception as e:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Embedding generation failed (attempt {attempt + 1}): {str(e)}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                continue
        
        raise StorageError("Failed to generate valid embedding after retries")

    def _validate_embedding(self, embedding: List[float]) -> None:
        """Validate embedding dimension and values."""
        if not isinstance(embedding, list):
            raise MemoryValidationError("Embedding must be a list")
        
        if len(embedding) != self.embedding_dimension:
            raise MemoryValidationError(
                f"Embedding dimension mismatch: got {len(embedding)}, "
                f"expected {self.embedding_dimension}"
            )
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise MemoryValidationError("All embedding values must be numeric")
        
        # Check for NaN/Inf values and reasonable bounds
        if any(not -1e6 <= x <= 1e6 for x in embedding):
            raise MemoryValidationError("Invalid embedding values detected")

    async def search(
        self,
        query: str,
        limit: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        min_relevance: float = 0.0,
        include_embeddings: bool = False
    ) -> List[MemorySearchResult]:
        """Search for memories with semantic search."""
        if not self._initialized:
            raise RetrievalError("Memory manager not initialized")
        try:
            # Use text-based search if in fallback mode
            if self._fallback_mode:
                return await self._text_based_search(
                    query,
                    limit=limit,
                    metadata_filter=metadata_filter
                )
            # Generate query embedding if available
            query_embedding = None
            if self.ollama_client and len(query.strip()) > 0:
                try:
                    query_embedding = await self._generate_embedding_with_retries(
                        query,
                        max_retries=1  # Don't retry for search queries
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate query embedding: {str(e)}")
                    # Fallback to text-based search
                    return await self._text_based_search(
                        query,
                        limit=limit,
                        metadata_filter=metadata_filter
                    )
            
            # Search using ChromaDB for semantic search
            try:
                results = await self.chroma_store.search(
                    query=query,
                    limit=limit,
                    metadata_filter=metadata_filter,
                    min_relevance=min_relevance
                )
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {str(e)}")
                # Fallback to text-based search
                return await self._text_based_search(
                    query,
                    limit=limit,
                    metadata_filter=metadata_filter
                )
            
            # Cache results in Redis for faster future retrieval
            valid_results = []
            for result in results:
                # Validate that the result has necessary fields
                if not hasattr(result, 'content') or not result.content:
                    logger.warning(f"Skipping invalid search result: missing content")
                    continue
                    
                valid_results.append(result)
                
                # Store in Redis cache
                if isinstance(result, Memory):
                    try:
                        await self.redis_store.store(result)
                    except Exception as e:
                        logger.warning(f"Failed to cache result in Redis: {str(e)}")
            
            # Remove embeddings if not requested
            if not include_embeddings:
                for result in valid_results:
                    if hasattr(result, 'embedding'):
                        result.embedding = None
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Failed to search memories: {str(e)}")
            # Return empty list instead of raising an exception
            logger.warning("Returning empty search results due to error")
            return []

    async def _text_based_search(
        self,
        query: str,
        limit: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemorySearchResult]:
        """Perform text-based search when embeddings are unavailable."""
        results = []
        query_terms = query.lower().split() if query else []
        
        try:
            # Get memories from Redis store
            all_memories = []
            
            try:
                # Try using get_all method if available
                if hasattr(self.redis_store, 'get_all'):
                    all_memories = await self.redis_store.get_all()
                else:
                    # Fallback: Get memory IDs and retrieve individually
                    memory_keys = await self.redis_store.redis_client.keys(f"{settings.REDIS_PREFIX}memory:*")
                    for key in memory_keys:
                        memory_id = key.split(":", 2)[2]  # Extract memory ID
                        memory = await self.retrieve(memory_id)
                        if memory:
                            all_memories.append(memory)
            except Exception as e:
                logger.warning(f"Failed to retrieve memories for search: {str(e)}")
                return []  # Return empty results if retrieval fails
            
            # Search through memories
            for memory in all_memories:
                # Check metadata filter
                if metadata_filter:
                    matches_filter = True
                    for k, v in metadata_filter.items():
                        if memory.metadata.get(k) != v:
                            matches_filter = False
                            break
                    
                    if not matches_filter:
                        continue
                
                # Simple text matching
                content_lower = memory.content.lower()
                
                # Calculate relevance score
                if query_terms:
                    matches = sum(term in content_lower for term in query_terms)
                    relevance = matches / len(query_terms) if matches > 0 else 0.0
                else:
                    # All memories match an empty query
                    relevance = 1.0
                
                # Only include if there's a match or query is empty
                if relevance > 0 or not query_terms:
                    results.append(MemorySearchResult(
                        content=memory.content,
                        embedding_id=memory.embedding_id,
                        relevance_score=relevance,
                        timestamp=memory.metadata.get('timestamp', datetime.now().isoformat()),
                        metadata=memory.metadata
                    ))
            
            # Sort by relevance and apply limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            if limit and limit > 0:
                results = results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Text-based search failed: {str(e)}")
            return []  # Return empty results on error

    async def retrieve(
        self,
        memory_id: str,
        include_embedding: bool = False
    ) -> Optional[Memory]:
        """Retrieve a memory by ID."""
        if not self._initialized:
            raise RetrievalError("Memory manager not initialized")
            
        try:
            # Try Redis first for fast retrieval
            memory = await self.redis_store.retrieve(memory_id)
            
            # Try ChromaDB if not in Redis and not in fallback mode
            if not memory and not self._fallback_mode:
                memory = await self.chroma_store.retrieve(memory_id)
                if memory:
                    # Cache in Redis for future
                    await self.redis_store.store(memory)
            
            if memory and not include_embedding:
                memory.embedding = None
                
            return memory
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {str(e)}")
            raise RetrievalError(f"Failed to retrieve memory: {str(e)}")

    async def get_preferences(
        self,
        preference_type: Optional[str] = None
    ) -> List[Memory]:
        """
        Get stored user preferences.
        
        Args:
            preference_type: Optional specific preference type
            
        Returns:
            List[Memory]: List of preference memories
        """
        try:
            # Prepare metadata filter
            metadata_filter = {"type": "preference"}
            if preference_type:
                metadata_filter["preference_type"] = preference_type
            
            # Search for preferences
            return await self.search(
                query="",  # Empty query to match all preferences
                metadata_filter=metadata_filter,
                limit=100  # Get all preferences
            )
        except Exception as e:
            logger.error(f"Failed to get preferences: {str(e)}")
            raise RetrievalError(f"Failed to get preferences: {str(e)}")

    async def start_conversation(self) -> str:
        """Start a new conversation and return its ID."""
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        
        self._active_conversations[conversation_id] = {
            "start_time": datetime.now().isoformat(),
            "memories": [],
            "metadata": {
                "type": "conversation",
                "status": "active"
            }
        }
        
        return conversation_id

    async def end_conversation(self, conversation_id: str) -> None:
        """End a conversation and store summary."""
        if conversation_id not in self._active_conversations:
            return
            
        try:
            # Create end event memory
            end_memory = Memory(
                content=f"Ended conversation {conversation_id}",
                metadata={
                    "type": "conversation_event",
                    "event": "end",
                    "conversation_id": conversation_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            await self.store(end_memory)
            
            # Clean up tracking
            del self._active_conversations[conversation_id]
            
        except Exception as e:
            logger.error(f"Error ending conversation: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up memory stores and tracking."""
        try:
            # Clean up Redis store
            if self.redis_store:
                await self.redis_store.cleanup()
            
            # Clean up ChromaDB store
            if self.chroma_store:
                await self.chroma_store.cleanup()
            
            # Reset state
            self._active_conversations.clear()
            self._stored_memories.clear()
            self._failed_embeddings.clear()
            self._fallback_mode = False
            self._initialized = False
            
            logger.info("Memory manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """Get status information."""
        return {
            "initialized": self._initialized,
            "fallback_mode": self._fallback_mode,
            "failed_embeddings": len(self._failed_embeddings),
            "active_conversations": len(self._active_conversations),
            "stored_memories": len(self._stored_memories),
            "stores": {
                "redis": await self.redis_store.get_status(),
                "chroma": await self.chroma_store.get_status()
            },
            "timestamp": datetime.now().isoformat()
        }

    def __str__(self) -> str:
        """Get string representation."""
        status = "initialized" if self._initialized else "not initialized"
        mode = "fallback" if self._fallback_mode else "normal"
        return (
            f"MemoryManager({status}, {mode} mode, "
            f"memories={len(self._stored_memories)})"
        )
