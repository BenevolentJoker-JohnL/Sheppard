# src/core/sheppard/interaction.py
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import asyncio
import os
import hashlib
from datetime import datetime, timedelta

from src.config.config import DatabaseConfig
from ..memory import MemoryStats
from ..trustcall import call

logger = logging.getLogger(__name__)

class InteractionHandler:
    """Handles processing of user interactions"""
    
    def __init__(self, memory_ops, response_gen, tool_manager):
        self.memory_ops = memory_ops
        self.response_gen = response_gen
        self.tool_manager = tool_manager
        self.processing_lock = asyncio.Lock()
        self.interaction_cache = {}
        self.cache_ttl = 3600  # 1 hour
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        self.interaction_stats = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "failed_interactions": 0,
            "average_response_time": 0.0,
            "tool_usage": {},
            "last_interaction": None,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_by_type": {},
            "memory_operations": {
                "store": 0,
                "retrieve": 0,
                "failed": 0
            }
        }

    async def process_interaction(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Process user input and generate response"""
        async with self.processing_lock:
            start_time = datetime.now()
            self.interaction_stats["total_interactions"] += 1
            
            try:
                # Check cache first
                cache_key = self._generate_cache_key(user_input, conversation_history)
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    self.interaction_stats["cache_hits"] += 1
                    return cached_response
                
                self.interaction_stats["cache_misses"] += 1

                # Input validation and preprocessing
                processed_input = await self._preprocess_input(user_input)
                if not processed_input:
                    self.interaction_stats["failed_interactions"] += 1
                    return "Please provide a valid input."

                # Generate embedding for input with retry logic
                user_embedding = await self._generate_embedding_with_retry(processed_input)
                if not user_embedding:
                    self.interaction_stats["failed_interactions"] += 1
                    return "I'm having trouble processing your input right now. Please try again."

                # Retrieve relevant memories
                memories = await self._retrieve_memories_with_retry(
                    processed_input,
                    user_embedding
                )

                # Analyze for tool usage
                tool_analysis = await self._analyze_tools_with_retry(
                    processed_input,
                    memories
                )

                # Generate response
                response = await self._generate_response_with_retry(
                    processed_input,
                    memories,
                    conversation_history,
                    tool_analysis
                )

                # Handle tool calls if necessary
                if tool_analysis and tool_analysis.get('use_tools'):
                    response = await self._process_tool_calls(
                        response,
                        tool_analysis
                    )

                # Store interaction in memory systems
                await self._store_interaction_with_retry(
                    processed_input,
                    response,
                    user_embedding,
                    memories
                )

                # Update statistics
                self._update_success_statistics(start_time)
                
                # Cache response
                await self._cache_response(cache_key, response)

                return response

            except Exception as e:
                self._handle_processing_error(e)
                return "I encountered an unexpected issue. Please try again."

    async def _preprocess_input(self, user_input: str) -> Optional[str]:
        """Preprocess and validate user input"""
        try:
            if not user_input or not user_input.strip():
                return None
                
            # Basic preprocessing
            processed = user_input.strip()
            
            # Remove any potentially harmful characters
            processed = ''.join(
                char for char in processed 
                if char.isprintable()
            )
            
            return processed if processed else None
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            return None

    async def _generate_embedding_with_retry(
        self,
        processed_input: str
    ) -> Optional[List[float]]:
        """Generate embedding with retry logic"""
        for attempt in range(self.max_retries):
            try:
                embedding = await self.memory_ops.generate_embedding(processed_input)
                if embedding:
                    return embedding
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                logger.error(f"Embedding generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return None

    async def _retrieve_memories_with_retry(
        self,
        processed_input: str,
        embedding: List[float]
    ) -> Dict[str, Any]:
        """Retrieve memories with retry logic"""
        for attempt in range(self.max_retries):
            try:
                memories = await self.memory_ops.retrieve_memories(
                    processed_input,
                    embedding
                )
                self.interaction_stats["memory_operations"]["retrieve"] += 1
                return memories
            except Exception as e:
                logger.error(f"Memory retrieval attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.interaction_stats["memory_operations"]["failed"] += 1
                    return {}
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return {}

    async def _analyze_tools_with_retry(
        self,
        processed_input: str,
        memories: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze tool usage with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await self.tool_manager.analyze_input(
                    processed_input,
                    memories
                )
            except Exception as e:
                logger.error(f"Tool analysis attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return {}
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return {}

    async def _generate_response_with_retry(
        self,
        processed_input: str,
        memories: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        tool_analysis: Dict[str, Any]
    ) -> str:
        """Generate response with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return await self.response_gen.generate_response(
                    processed_input,
                    memories,
                    conversation_history,
                    tool_analysis
                )
            except Exception as e:
                logger.error(f"Response generation attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    return "I'm having trouble generating a response. Please try again."
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        return "I'm having trouble generating a response. Please try again."

    async def _process_tool_calls(
        self,
        response: str,
        tool_analysis: Dict[str, Any]
    ) -> str:
        """Process tool calls with error handling"""
        try:
            processed_response = await self.tool_manager.process_response(
                response,
                tool_analysis
            )
            
            # Update tool usage statistics
            for tool in tool_analysis.get('recommended_tools', []):
                self.interaction_stats["tool_usage"][tool] = \
                    self.interaction_stats["tool_usage"].get(tool, 0) + 1
                    
            return processed_response
            
        except Exception as e:
            logger.error(f"Error processing tool calls: {str(e)}")
            return response

    async def _store_interaction_with_retry(
        self,
        processed_input: str,
        response: str,
        embedding: List[float],
        memories: Dict[str, Any]
    ) -> None:
        """Store interaction with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.memory_ops.store_interaction(
                    processed_input,
                    response,
                    embedding,
                    memories
                )
                self.interaction_stats["memory_operations"]["store"] += 1
                return
            except Exception as e:
                logger.error(f"Interaction storage attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.interaction_stats["memory_operations"]["failed"] += 1
                    break
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    def _update_success_statistics(self, start_time: datetime) -> None:
        """Update success statistics"""
        try:
            self.interaction_stats["successful_interactions"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Safe update of average response time
            if self.interaction_stats["successful_interactions"] > 0:
                current_avg = self.interaction_stats["average_response_time"]
                total_success = self.interaction_stats["successful_interactions"]
                self.interaction_stats["average_response_time"] = (
                    (current_avg * (total_success - 1) + processing_time) / total_success
                )
            
            self.interaction_stats["last_interaction"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error updating statistics: {str(e)}")

    def _handle_processing_error(self, error: Exception) -> None:
        """Handle processing errors and update statistics"""
        error_type = type(error).__name__
        self.interaction_stats["errors_by_type"][error_type] = \
            self.interaction_stats["errors_by_type"].get(error_type, 0) + 1
        self.interaction_stats["failed_interactions"] += 1
        logger.error(f"Error processing interaction: {str(error)}")

    def _generate_cache_key(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Generate cache key"""
        try:
            # Include last few interactions in cache key
            recent_history = conversation_history[-3:] if conversation_history else []
            key_components = [user_input] + [
                str(interaction)
                for interaction in recent_history
            ]
            return hashlib.sha256(
                json.dumps(key_components).encode()
            ).hexdigest()
        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return hashlib.sha256(user_input.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available"""
        try:
            if cache_key in self.interaction_cache:
                cached_item = self.interaction_cache[cache_key]
                if (datetime.now() - cached_item['timestamp']).total_seconds() < self.cache_ttl:
                    return cached_item['response']
                else:
                    del self.interaction_cache[cache_key]
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached response: {str(e)}")
            return None

    async def _cache_response(self, cache_key: str, response: str) -> None:
        """Cache response with TTL"""
        try:
            self.interaction_cache[cache_key] = {
                'response': response,
                'timestamp': datetime.now()
            }
            
            # Clean expired cache entries
            await self._clean_expired_cache()
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")

    async def _clean_expired_cache(self) -> None:
        """Clean expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = [
                k for k, v in self.interaction_cache.items()
                if (current_time - v['timestamp']).total_seconds() > self.cache_ttl
            ]
            
            for k in expired_keys:
                del self.interaction_cache[k]
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")

    async def get_interaction_summary(self) -> Dict[str, Any]:
        """Get comprehensive interaction statistics summary"""
        try:
            success_rate = 0
            if self.interaction_stats["total_interactions"] > 0:
                success_rate = (
                    self.interaction_stats["successful_interactions"] /
                    self.interaction_stats["total_interactions"] * 100
                )

            cache_hit_rate = 0
            total_cache_ops = (
                self.interaction_stats["cache_hits"] + 
                self.interaction_stats["cache_misses"]
            )
            if total_cache_ops > 0:
                cache_hit_rate = (
                    self.interaction_stats["cache_hits"] / total_cache_ops * 100
                )

            return {
                "total_interactions": self.interaction_stats["total_interactions"],
                "successful_interactions": self.interaction_stats["successful_interactions"],
                "failed_interactions": self.interaction_stats["failed_interactions"],
                "success_rate": f"{success_rate:.2f}%",
                "average_response_time": f"{self.interaction_stats['average_response_time']:.2f}s",
                "tool_usage": self.interaction_stats["tool_usage"],
                "last_interaction": self.interaction_stats["last_interaction"],
                "cache_performance": {
                    "hits": self.interaction_stats["cache_hits"],
                    "misses": self.interaction_stats["cache_misses"],
                    "hit_rate": f"{cache_hit_rate:.2f}%",
                    "current_size": len(self.interaction_cache)
                },
                "memory_operations": self.interaction_stats["memory_operations"],
                "error_distribution": self.interaction_stats["errors_by_type"]
            }
            
        except Exception as e:
            logger.error(f"Error generating interaction summary: {str(e)}")
            return {}

    async def save_interaction_stats(self, filepath: Optional[str] = None) -> bool:
        """Save interaction statistics to file"""
        try:
            if not filepath:
                filepath = os.path.join(
                    DatabaseConfig.DATA_DIR,
                    f"interaction_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(await self.get_interaction_summary(), f, indent=2)
            return True
            
        except Exception as e:
            logger.error(f"Error saving interaction stats: {str(e)}")
            return False

    async def load_interaction_stats(self, filepath: str) -> bool:
        """Load interaction statistics from file"""
        try:
            with open(filepath, 'r') as f:
                loaded_stats = json.load(f)
                
            # Convert string percentages back to numbers
            if 'success_rate' in loaded_stats:
                loaded_stats['success_rate'] = float(loaded_stats['success_rate'].rstrip('%'))
                
            if 'cache_performance' in loaded_stats:
                if 'hit_rate' in loaded_stats['cache_performance']:
                    loaded_stats['cache_performance']['hit_rate'] = \
                        float(loaded_stats['cache_performance']['hit_rate'].rstrip('%'))
            
            # Update statistics while preserving current cache
            self.interaction_stats.update({
                k: v for k, v in loaded_stats.items()
                if k != 'last_interaction'
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading interaction stats: {str(e)}")
            return False

    async def clear_cache(self) -> None:
        """Clear interaction cache"""
        try:
            self.interaction_cache.clear()
            logger.info("Interaction cache cleared")
        except Exception as e:
            logger.error(f"Error clearing interaction cache: {str(e)}")

    async def update_cache_ttl(self, new_ttl: int) -> None:
        """Update cache TTL and clean expired entries"""
        try:
            if new_ttl <= 0:
                raise ValueError("TTL must be positive")
            
            self.cache_ttl = new_ttl
            await self._clean_expired_cache()
            logger.info(f"Cache TTL updated to {new_ttl} seconds")
            
        except Exception as e:
            logger.error(f"Error updating cache TTL: {str(e)}")

    async def reset_statistics(self) -> None:
        """Reset interaction statistics"""
        try:
            # Save current stats before reset
            await self.save_interaction_stats()
            
            self.interaction_stats = {
                "total_interactions": 0,
                "successful_interactions": 0,
                "failed_interactions": 0,
                "average_response_time": 0.0,
                "tool_usage": {},
                "last_interaction": None,
                "cache_hits": 0,
                "cache_misses": 0,
                "errors_by_type": {},
                "memory_operations": {
                    "store": 0,
                    "retrieve": 0,
                    "failed": 0
                }
            }
            
            logger.info("Interaction statistics reset")
            
        except Exception as e:
            logger.error(f"Error resetting statistics: {str(e)}")

    async def cleanup(self) -> None:
        """Cleanup interaction handler resources"""
        try:
            # Save final statistics
            await self.save_interaction_stats()
            
            # Clear cache
            await self.clear_cache()
            
            # Reset statistics
            await self.reset_statistics()
            
            logger.info("Interaction handler cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during interaction handler cleanup: {str(e)}")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current handler status"""
        return {
            "initialized": True,
            "cache_size": len(self.interaction_cache),
            "cache_ttl": self.cache_ttl,
            "processing_lock": self.processing_lock.locked(),
            "last_interaction": self.interaction_stats["last_interaction"],
            "success_rate": (
                self.interaction_stats["successful_interactions"] /
                max(self.interaction_stats["total_interactions"], 1) * 100
            ),
            "memory_health": all(
                count > 0 for count in 
                self.interaction_stats["memory_operations"].values()
            )
        }

    async def validate_health(self) -> bool:
        """Validate handler health"""
        try:
            # Check essential components
            if not self.memory_ops or not self.response_gen or not self.tool_manager:
                return False
                
            # Verify memory operations
            if self.interaction_stats["memory_operations"]["failed"] > \
               self.interaction_stats["memory_operations"]["store"] * 0.5:
                return False
                
            # Check success rate
            if self.interaction_stats["total_interactions"] > 10:
                success_rate = (
                    self.interaction_stats["successful_interactions"] /
                    self.interaction_stats["total_interactions"]
                )
                if success_rate < 0.5:  # Less than 50% success
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating handler health: {str(e)}")
            return False

    def __str__(self) -> str:
        """String representation"""
        status = "🟢" if asyncio.create_task(self.validate_health()) else "🔴"
        return (
            f"InteractionHandler(status={status}, "
            f"total_interactions={self.interaction_stats['total_interactions']}, "
            f"success_rate={self.interaction_stats['successful_interactions']/max(self.interaction_stats['total_interactions'],1)*100:.1f}%, "
            f"cache_size={len(self.interaction_cache)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation"""
        return json.dumps(
            {
                "status": self.get_current_status(),
                "statistics": self.interaction_stats
            },
            indent=2,
            default=str
        )
