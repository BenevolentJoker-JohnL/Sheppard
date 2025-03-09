import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
import os
from jsonschema import ValidationError

from src.core.memory.stats import MemoryStats
from src.core.memory.cache import LRUCache
from src.core.memory.storage.storage_manager import StorageManager
from src.core.memory.embeddings import EmbeddingManager
from src.core.memory.cleanup import CleanupManager
from src.config.config import DatabaseConfig
from src.core.memory.validation import MemoryValidator
from ..trustcall import call

logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class MemoryOperations:
    """Handles memory operations and integration"""
    
    def __init__(self, embedding_client, embedding_model, memory_config):
        self.logger = logging.getLogger(__name__)
        
        self.schema_manager = SchemaFailsafeManager()
        self.memory_validator = MemoryValidator()
        self.memory_validator.set_schema_manager(self.schema_manager)
        
        self.memory_stats = MemoryStats()
        self.memory_cache = LRUCache(memory_config["cache_size"])
        self.storage_manager = StorageManager()
        self.embedding_manager = EmbeddingManager(embedding_client, embedding_model)
        self.cleanup_manager = CleanupManager(importance_threshold=memory_config["importance_threshold"])

        self.similarity_threshold = memory_config["similarity_threshold"]
        self.retention_period = memory_config["retention_period_days"]
        self.memory_locks: Dict[str, asyncio.Lock] = {}
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        self.initialization_lock = asyncio.Lock()
        self.max_retries = 3
        self.retry_delay = 1

    async def validate_connection(self) -> bool:
        """Validate connections to all storage systems"""
        try:
            if not self.storage_manager:
                logger.error("Storage manager not initialized")
                return False

            if not await self.storage_manager.validate_connection():
                logger.error("Storage manager validation failed")
                return False

            if not self.embedding_manager:
                logger.error("Embedding manager not initialized")
                return False

            test_text = "connection_test"
            test_embedding = await self.generate_embedding(test_text)
            if test_embedding is None:
                logger.error("Embedding generation test failed")
                return False

            if not self.cleanup_manager:
                logger.error("Cleanup manager not initialized")
                return False

            logger.info("All memory connections validated successfully")
            return True

        except Exception as e:
            logger.error(f"Error validating memory connections: {str(e)}")
            return False

    async def initialize(self) -> bool:
        """Initialize memory components with improved error handling"""
        async with self.initialization_lock:
            try:
                self.logger.info("Initializing memory operations...")
                for attempt in range(self.max_retries):
                    try:
                        if await self.storage_manager.initialize():
                            break
                        if attempt == self.max_retries - 1:
                            self.logger.error("Failed to initialize storage manager")
                            return False
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            self.logger.error(f"Storage manager initialization failed: {str(e)}")
                            return False
                        await asyncio.sleep(self.retry_delay * (attempt + 1))

                for layer in ["episodic", "semantic", "contextual"]:
                    self.memory_locks[layer] = asyncio.Lock()

                self._start_background_processing()

                return True

            except Exception as e:
                self.logger.error(f"Error initializing memory operations: {str(e)}")
                return False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text with retry logic"""
        try:
            for attempt in range(self.max_retries):
                try:
                    embedding = await self.embedding_manager.generate_embedding(text)
                    if embedding:
                        self.memory_stats.record_embedding_operation(True)
                        return embedding
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            
            self.memory_stats.record_embedding_operation(False)
            return None
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            self.memory_stats.record_embedding_operation(False)
            return None

    async def store_interaction(
        self,
        user_input: str,
        response: str,
        embedding: List[float],
        memories: Dict[str, Any]
    ) -> None:
        """Store interaction in memory system with improved error handling"""
        try:
            interaction_data = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": "AI",
                "input": user_input,
                "response": response,
                "embedding": embedding,
                "entities": {},
                "topics": {},
                "state": "active",
                "importance_score": self._calculate_importance(user_input, response),
                "context_metadata": {
                    "memories_used": json.loads(json.dumps(memories, cls=DateTimeEncoder))
                }
            }

            valid_interaction = await self.memory_validator.validate_memory(interaction_data, "interaction_memory")
            
            if not valid_interaction:
                logger.error("Failed to validate interaction data")
                return

            memory_hash = self.storage_manager.generate_memory_hash(valid_interaction)

            layers = ["episodic", "semantic", "contextual"]
            tasks = []

            for layer in layers:
                task = asyncio.create_task(self._store_with_retry(valid_interaction, layer, memory_hash))
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for layer, result in zip(layers, results):
                if isinstance(result, Exception):
                    logger.error(f"Error storing in {layer} layer: {str(result)}")

        except Exception as e:
            logger.error(f"Error storing interaction: {str(e)}")

    async def _store_with_retry(self, interaction_data: Dict[str, Any], layer: str, memory_hash: str) -> None:
        """Store memory with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self.memory_locks[layer]:
                    success = await self.storage_manager.store_memory(
                        key=f"interaction_{interaction_data['timestamp']}",
                        value=interaction_data,
                        layer=layer,
                        memory_hash=memory_hash,
                        importance_score=interaction_data['importance_score']
                    )
                    
                    if success:
                        self.memory_stats.record_memory_creation("active")
                        return
                        
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to store memory in {layer} layer: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def retrieve_memories(self, user_input: str, embedding: List[float]) -> Dict[str, Any]:
        """Retrieve relevant memories with improved error handling"""
        memories = {"episodic": [], "semantic": [], "contextual": []}
        
        try:
            cache_key = hash(user_input)
            cached_result = await self.memory_cache.get_cached_response(str(cache_key))
            if cached_result:
                try:
                    loaded_memories = json.loads(cached_result)
                    if all(layer in loaded_memories for layer in memories.keys()):
                        self.memory_stats.cache_hits += 1
                        return loaded_memories
                except json.JSONDecodeError:
                    logger.warning("Failed to decode cached memories")

            self.memory_stats.cache_misses += 1
            
            retrieval_tasks = []
            for layer in memories.keys():
                task = asyncio.create_task(self._retrieve_layer_memories(layer, user_input, embedding))
                retrieval_tasks.append((layer, task))

            for layer, task in retrieval_tasks:
                try:
                    layer_memories = await task
                    if layer_memories:
                        memories[layer] = layer_memories
                except Exception as e:
                    logger.error(f"Error retrieving memories for layer {layer}: {str(e)}")
                    continue

            try:
                cached_data = json.dumps(memories, cls=DateTimeEncoder)
                await self.memory_cache.cache_response(str(cache_key), cached_data)
            except Exception as e:
                logger.error(f"Error caching memories: {str(e)}")

            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {str(e)}")
            return {"episodic": [], "semantic": [], "contextual": []}

    async def _retrieve_layer_memories(self, layer: str, user_input: str, embedding: List[float]) -> List[Dict[str, Any]]:
        """Retrieve memories for a specific layer with retry logic"""
        for attempt in range(self.max_retries):
            try:
                async with self.memory_locks[layer]:
                    memories = await self.storage_manager.retrieve_memories(user_input, embedding, layer)
                    
                    processed_memories = []
                    for memory in memories:
                        if isinstance(memory, dict):
                            processed_memory = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in memory.items()}
                            processed_memories.append(processed_memory)
                    
                    return processed_memories
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to retrieve layer memories after {self.max_retries} attempts: {str(e)}")
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return []

    def _start_background_processing(self):
        """Start background processing task"""
        if not self.is_processing:
            self.is_processing = True
            asyncio.create_task(self._process_memory_queue())

    async def _process_memory_queue(self):
        """Process queued memory operations with improved error handling"""
        while self.is_processing:
            try:
                try:
                    item = await asyncio.wait_for(self.processing_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                operation, args = item
                try:
                    if operation == "store":
                        await self._store_with_retry(*args)
                    elif operation == "update":
                        await self._update_memory_internal(*args)
                except Exception as e:
                    logger.error(f"Error processing {operation} operation: {str(e)}")
                finally:
                    self.processing_queue.task_done()

            except Exception as e:
                logger.error(f"Error in memory queue processing: {str(e)}")
                await asyncio.sleep(1)

    def _calculate_importance(self, user_input: str, response: str) -> float:
        """Calculate importance score for memory with improved scoring"""
        try:
            importance = 0.5  
            indicators = {
                "remember": 0.15, "important": 0.15, "crucial": 0.15, "key": 0.1,
                "critical": 0.15, "essential": 0.1, "priority": 0.1, "favorite": 0.15,
                "preference": 0.1, "like": 0.05, "love": 0.1, "hate": 0.1, "always": 0.1,
                "never": 0.1, "need": 0.1, "must": 0.1
            }
            
            combined_text = (user_input + " " + response).lower()
            for word, score in indicators.items():
                if word in combined_text:
                    importance += score

            content_length = len(user_input) + len(response)
            importance += min(content_length / 1000, 0.2)
            
            if "?" in user_input and len(response) > 100:
                importance += 0.1
            
            if any(pref in combined_text.lower() for pref in ["prefer", "favorite", "like", "enjoy"]):
                importance += 0.15
            
            return min(max(importance, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating importance score: {str(e)}")
            return 0.5

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics summary"""
        try:
            memory_stats = self.memory_stats.get_stats_summary()
            storage_stats = self.storage_manager.get_storage_stats()
            
            combined_stats = {
                "memory_operations": memory_stats,
                "storage": storage_stats,
                "cache": {
                    "size": len(self.memory_cache.cache),
                    "hits": self.memory_stats.cache_hits,
                    "misses": self.memory_stats.cache_misses
                },
                "embeddings": {
                    "total": self.memory_stats.embedding_count,
                    "failed": self.memory_stats.failed_embeddings
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return json.loads(json.dumps(combined_stats, cls=DateTimeEncoder))
            
        except Exception as e:
            logger.error(f"Error getting stats summary: {str(e)}")
            return {}

    async def cleanup(self) -> None:
        """Clean shutdown of memory components"""
        try:
            self.is_processing = False
            
            if not self.processing_queue.empty():
                try:
                    await asyncio.wait_for(self.processing_queue.join(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for memory queue to empty")

            cleanup_tasks = [
                self.cleanup_manager.perform_full_cleanup(),
                self.embedding_manager.shutdown(),
                self.storage_manager.cleanup()
            ]
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cleanup_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout during memory cleanup tasks")

            stats_file = os.path.join(
                DatabaseConfig.DATA_DIR,
                f"memory_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            try:
                with open(stats_file, 'w') as f:
                    json.dump(self.get_stats_summary(), f, indent=2)
            except Exception as e:
                logger.error(f"Error saving final statistics: {str(e)}")

            self.memory_cache.clear()
            self.memory_locks.clear()
            
            logger.info("Memory operations cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during memory operations shutdown: {str(e)}")

    def __str__(self) -> str:
        """String representation"""
        stats = self.get_stats_summary()
        return (
            f"MemoryOperations("
            f"embeddings={stats.get('embeddings', {}).get('total', 0)}, "
            f"cache_size={stats.get('cache', {}).get('size', 0)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        stats = self.get_stats_summary()
        return json.dumps(stats, indent=2)

