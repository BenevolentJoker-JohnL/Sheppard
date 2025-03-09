# src/core/memory/storage/redis_cache.py
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import redis.asyncio as redis
import asyncio
import hashlib

from .base import StorageBase
from .connection import ConnectionManager
from src.config.config import DatabaseConfig

logger = logging.getLogger(__name__)

class RedisCacheManager(StorageBase):
    """Redis cache implementation with improved error handling and type checking"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn_manager = connection_manager
        self.default_ttl = 3600  # 1 hour
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.batch_size = 100
        
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "operations": {
                "set": 0,
                "get": 0,
                "delete": 0,
                "batch": 0
            },
            "type_mismatches": 0
        }

    async def validate_connection(self) -> bool:
        """Validate Redis connection and functionality"""
        try:
            # Test each Redis instance
            for layer in ["ephemeral", "contextual", "episodic", "semantic", "abstracted"]:
                client = await self.conn_manager.get_redis_client(layer)
                if not client:
                    logger.error(f"Failed to get Redis client for {layer}")
                    return False

                # Test basic operations
                try:
                    # Test connection with ping
                    if not await client.ping():
                        logger.error(f"Redis ping failed for {layer}")
                        return False

                    # Test write operation
                    test_key = f"test_{layer}_{datetime.now().timestamp()}"
                    test_value = "test_value"
                    
                    # Test SET operation
                    await client.set(test_key, test_value, ex=60)
                    
                    # Test GET operation
                    retrieved_value = await client.get(test_key)
                    if retrieved_value != test_value:
                        logger.error(f"Redis GET/SET test failed for {layer}")
                        return False
                    
                    # Test DELETE operation
                    await client.delete(test_key)
                    if await client.exists(test_key):
                        logger.error(f"Redis DELETE test failed for {layer}")
                        return False
                    
                    # Test pipeline operations
                    async with client.pipeline(transaction=True) as pipe:
                        pipe.set(f"{test_key}_1", "value1", ex=60)
                        pipe.set(f"{test_key}_2", "value2", ex=60)
                        await pipe.execute()
                        
                        # Cleanup test keys
                        await client.delete(f"{test_key}_1", f"{test_key}_2")
                    
                except redis.RedisError as e:
                    logger.error(f"Redis operations test failed for {layer}: {str(e)}")
                    return False
                except Exception as e:
                    logger.error(f"Unexpected error testing Redis for {layer}: {str(e)}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating Redis connections: {str(e)}")
            return False

    async def initialize(self) -> bool:
        """Initialize Redis cache with retry logic"""
        try:
            # Test all Redis connections
            for layer in ["ephemeral", "contextual", "episodic", "semantic", "abstracted"]:
                client = await self.conn_manager.get_redis_client(layer)
                if not client:
                    logger.error(f"Failed to get Redis client for {layer}")
                    return False
                    
                # Test connection with retries
                for attempt in range(self.max_retries):
                    try:
                        await client.ping()
                        break
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            logger.error(f"Failed to connect to Redis for {layer}: {str(e)}")
                            return False
                        await asyncio.sleep(self.retry_delay)
                        
                # Clear any invalid data types during initialization
                await self._cleanup_invalid_types(client, layer)

            return True
            
        except Exception as e:
            logger.error(f"Error initializing Redis cache: {str(e)}")
            return False

    async def _cleanup_invalid_types(self, client: redis.Redis, layer: str) -> None:
        """Clean up keys with invalid types"""
        try:
            async for key in client.scan_iter(match=f"{layer}:*"):
                try:
                    key_type = await client.type(key)
                    if key_type != "string":
                        await client.delete(key)
                        self.cache_stats["type_mismatches"] += 1
                except Exception as e:
                    logger.error(f"Error checking type for key {key}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up invalid types: {str(e)}")

    async def cleanup(self) -> None:
        """No additional cleanup needed as ConnectionManager handles connections"""
        pass

    async def store_memory(
        self,
        key: str,
        value: Dict[str, Any],
        layer: str,
        memory_hash: str,
        importance_score: float
    ) -> bool:
        """Store memory in Redis cache with type validation"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return False

            # Prepare cache data
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "memory_hash": memory_hash,
                "importance_score": importance_score,
                "data": value
            }

            # Calculate TTL based on importance
            ttl = int(max(
                self.default_ttl,
                self.default_ttl * importance_score * 2
            ))

            # Store with retries
            for attempt in range(self.max_retries):
                try:
                    # Use pipeline for atomic operations
                    async with client.pipeline(transaction=True) as pipe:
                        # Check key type first if it exists
                        if await client.exists(f"{layer}:{memory_hash}"):
                            key_type = await client.type(f"{layer}:{memory_hash}")
                            if key_type != "string":
                                await client.delete(f"{layer}:{memory_hash}")
                        
                        await pipe.setex(
                            f"{layer}:{memory_hash}",
                            ttl,
                            json.dumps(cache_data)
                        )
                        await pipe.sadd(f"{layer}:keys", memory_hash)
                        await pipe.execute()
                        
                    self.cache_stats["operations"]["set"] += 1
                    return True
                    
                except redis.RedisError as e:
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to store in Redis after {self.max_retries} attempts: {str(e)}")
                        self.cache_stats["errors"] += 1
                        return False
                    await asyncio.sleep(self.retry_delay)

        except Exception as e:
            logger.error(f"Error storing memory in Redis: {str(e)}")
            self.cache_stats["errors"] += 1
            return False

    async def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        layer: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from Redis cache with type checking"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                self.cache_stats["errors"] += 1
                return []

            # Get all keys for the layer
            pattern = f"{layer}:*"
            keys = []
            
            # Use scan instead of keys for large datasets
            async for key in client.scan_iter(match=pattern):
                try:
                    # Verify key type
                    key_type = await client.type(key)
                    if key_type != "string":
                        await client.delete(key)
                        self.cache_stats["type_mismatches"] += 1
                        continue
                    keys.append(key)
                    if len(keys) >= limit * 2:  # Get more than needed for filtering
                        break
                except Exception as e:
                    logger.error(f"Error checking key type: {str(e)}")
                    continue
            
            memories = []
            async with client.pipeline(transaction=False) as pipe:
                # Batch process keys
                for i in range(0, len(keys), self.batch_size):
                    batch_keys = keys[i:i + self.batch_size]
                    for key in batch_keys:
                        pipe.get(key)
                    
                    try:
                        results = await pipe.execute()
                        for data in results:
                            if data:
                                try:
                                    memory = json.loads(data)
                                    memories.append(memory)
                                except json.JSONDecodeError:
                                    continue
                    except redis.RedisError as e:
                        logger.error(f"Error retrieving batch from Redis: {str(e)}")
                        continue

            # Sort by importance score
            memories.sort(
                key=lambda x: float(x.get('importance_score', 0)),
                reverse=True
            )
            
            # Update stats
            if memories:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            self.cache_stats["operations"]["get"] += 1

            return memories[:limit]

        except Exception as e:
            logger.error(f"Error retrieving memories from Redis: {str(e)}")
            self.cache_stats["errors"] += 1
            return []

    async def cleanup_old_memories(
        self,
        days_threshold: int,
        importance_threshold: float
    ) -> Dict[str, int]:
        """Clean up old memories from Redis cache with improved type handling"""
        cleanup_stats = {"removed": 0}
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            for layer in ["ephemeral", "contextual", "episodic", "semantic", "abstracted"]:
                client = await self.conn_manager.get_redis_client(layer)
                if not client:
                    continue

                # Use scan for efficient iteration
                async for key in client.scan_iter(match=f"{layer}:*"):
                    try:
                        # Check key type first
                        key_type = await client.type(key)
                        if key_type != "string":
                            await client.delete(key)
                            cleanup_stats["removed"] += 1
                            self.cache_stats["type_mismatches"] += 1
                            continue
                            
                        data = await client.get(key)
                        if data:
                            try:
                                memory = json.loads(data)
                                timestamp = datetime.fromisoformat(memory.get('timestamp', ''))
                                importance = float(memory.get('importance_score', 0))
                                
                                if timestamp < cutoff_date and importance < importance_threshold:
                                    async with client.pipeline(transaction=True) as pipe:
                                        await pipe.delete(key)
                                        await pipe.srem(f"{layer}:keys", key.split(':')[-1])
                                        await pipe.execute()
                                        
                                    cleanup_stats["removed"] += 1
                                    self.cache_stats["operations"]["delete"] += 1
                                    
                            except (json.JSONDecodeError, ValueError) as e:
                                # Remove corrupted data
                                await client.delete(key)
                                cleanup_stats["removed"] += 1
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {str(e)}")
                        continue

            return cleanup_stats

        except Exception as e:
            logger.error(f"Error cleaning up Redis memories: {str(e)}")
            self.cache_stats["errors"] += 1
            return cleanup_stats

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            total_ops = sum(self.cache_stats["operations"].values())
            total_accesses = self.cache_stats["hits"] + self.cache_stats["misses"]
            
            stats = {
                **self.cache_stats,
                "hit_ratio": (
                    self.cache_stats["hits"] / total_accesses
                    if total_accesses > 0 else 0
                ),
                "error_rate": (
                    self.cache_stats["errors"] / total_ops
                    if total_ops > 0 else 0
                ),
                "type_mismatch_rate": (
                    self.cache_stats["type_mismatches"] / total_ops
                    if total_ops > 0 else 0
                )
            }
            
            # Add memory usage stats
            memory_stats = {}
            for layer in ["ephemeral", "contextual", "episodic", "semantic", "abstracted"]:
                client = await self.conn_manager.get_redis_client(layer)
                if client:
                    try:
                        info = await client.info()
                        memory_stats[layer] = {
                            "used_memory": info.get("used_memory_human", "N/A"),
                            "keys": await client.scard(f"{layer}:keys")
                        }
                    except Exception as e:
                        logger.error(f"Error getting memory stats for {layer}: {str(e)}")
                        memory_stats[layer] = {"error": str(e)}
                        
            stats["memory_usage"] = memory_stats
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}

    async def clear_layer(self, layer: str) -> bool:
        """Clear all keys in a specific layer"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return False

            async with client.pipeline(transaction=True) as pipe:
                # Delete all keys matching pattern
                async for key in client.scan_iter(match=f"{layer}:*"):
                    pipe.delete(key)
                # Clear key set
                pipe.delete(f"{layer}:keys")
                await pipe.execute()
                
            return True
            
        except Exception as e:
            logger.error(f"Error clearing layer {layer}: {str(e)}")
            return False

    async def update_ttl(self, key: str, layer: str, new_ttl: int) -> bool:
        """Update TTL for a specific key"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return False

            full_key = f"{layer}:{key}"
            
            # Check key type first
            key_type = await client.type(full_key)
            if key_type != "string":
                await client.delete(full_key)
                return False
                
            exists = await client.exists(full_key)
            if exists:
                await client.expire(full_key, new_ttl)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error updating TTL for {key}: {str(e)}")
            return False

    async def set_default_ttl(self, ttl: int) -> None:
        """Set default TTL for new entries"""
        try:
            if ttl <= 0:
                raise ValueError("TTL must be positive")
            self.default_ttl = ttl
            logger.info(f"Updated default TTL to {ttl} seconds")
        except Exception as e:
            logger.error(f"Error updating default TTL: {str(e)}")

    async def get_memory_count(self, layer: str) -> int:
        """Get count of memories in a specific layer"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return 0

            count = 0
            async for key in client.scan_iter(match=f"{layer}:*"):
                if await client.type(key) == "string":
                    count += 1
            return count
            
        except Exception as e:
            logger.error(f"Error getting memory count for {layer}: {str(e)}")
            return 0

    async def get_memory_size(self, layer: str) -> int:
        """Get total size of memories in a specific layer in bytes"""
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return 0

            total_size = 0
            async for key in client.scan_iter(match=f"{layer}:*"):
                try:
                    if await client.type(key) == "string":
                        data = await client.get(key)
                        if data:
                            total_size += len(data)
                except Exception as e:
                    logger.error(f"Error getting size for key {key}: {str(e)}")
                    continue
            return total_size
            
        except Exception as e:
            logger.error(f"Error getting memory size for {layer}: {str(e)}")
            return 0

    async def validate_memory_data(self, data: Dict[str, Any]) -> bool:
        """Validate memory data structure"""
        try:
            required_fields = {'timestamp', 'memory_hash', 'importance_score', 'data'}
            if not all(field in data for field in required_fields):
                return False
                
            # Validate timestamp format
            try:
                datetime.fromisoformat(data['timestamp'])
            except ValueError:
                return False
                
            # Validate importance score
            if not isinstance(data['importance_score'], (int, float)):
                return False
            if not 0 <= data['importance_score'] <= 1:
                return False
                
            # Validate memory hash
            if not isinstance(data['memory_hash'], str):
                return False
                
            # Validate data field is dict
            if not isinstance(data['data'], dict):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating memory data: {str(e)}")
            return False

    async def repair_corrupted_data(self, layer: str) -> Dict[str, int]:
        """Attempt to repair corrupted data entries"""
        repair_stats = {
            "checked": 0,
            "corrupted": 0,
            "repaired": 0,
            "deleted": 0
        }
        
        try:
            client = await self.conn_manager.get_redis_client(layer)
            if not client:
                return repair_stats

            async for key in client.scan_iter(match=f"{layer}:*"):
                repair_stats["checked"] += 1
                try:
                    # Check key type
                    key_type = await client.type(key)
                    if key_type != "string":
                        await client.delete(key)
                        repair_stats["deleted"] += 1
                        continue
                        
                    # Get and validate data
                    data = await client.get(key)
                    if not data:
                        await client.delete(key)
                        repair_stats["deleted"] += 1
                        continue
                        
                    try:
                        memory_data = json.loads(data)
                    except json.JSONDecodeError:
                        repair_stats["corrupted"] += 1
                        await client.delete(key)
                        repair_stats["deleted"] += 1
                        continue
                        
                    # Validate data structure
                    if not await self.validate_memory_data(memory_data):
                        repair_stats["corrupted"] += 1
                        # Attempt repair
                        repaired_data = self._repair_memory_data(memory_data)
                        if repaired_data:
                            await client.set(
                                key,
                                json.dumps(repaired_data),
                                ex=self.default_ttl
                            )
                            repair_stats["repaired"] += 1
                        else:
                            await client.delete(key)
                            repair_stats["deleted"] += 1
                            
                except Exception as e:
                    logger.error(f"Error repairing key {key}: {str(e)}")
                    repair_stats["corrupted"] += 1
                    continue

            return repair_stats
            
        except Exception as e:
            logger.error(f"Error during data repair for {layer}: {str(e)}")
            return repair_stats

    def _repair_memory_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to repair corrupted memory data"""
        try:
            repaired = {
                'timestamp': data.get('timestamp', datetime.now().isoformat()),
                'memory_hash': data.get('memory_hash', hashlib.sha256(str(datetime.now().timestamp()).encode()).hexdigest()),
                'importance_score': max(0.0, min(1.0, float(data.get('importance_score', 0.5)))),
                'data': data.get('data', {}) if isinstance(data.get('data'), dict) else {}
            }
            
            # Validate repaired data
            if self.validate_memory_data(repaired):
                return repaired
            return None
            
        except Exception as e:
            logger.error(f"Error repairing memory data: {str(e)}")
            return None

    def __str__(self) -> str:
        """String representation of cache manager"""
        return (
            f"RedisCacheManager("
            f"hits={self.cache_stats['hits']}, "
            f"misses={self.cache_stats['misses']}, "
            f"errors={self.cache_stats['errors']}, "
            f"type_mismatches={self.cache_stats['type_mismatches']})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"RedisCacheManager("
            f"cache_stats={self.cache_stats}, "
            f"default_ttl={self.default_ttl}, "
            f"max_retries={self.max_retries}, "
            f"batch_size={self.batch_size})"
        )
