import logging
from collections import OrderedDict
from typing import Any, Optional, Dict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class LRUCache:
    """
    Least Recently Used (LRU) cache implementation for memory storage
    """
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache and update access statistics
        """
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Add item to cache with LRU eviction
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        """
        total_accesses = self.hits + self.misses
        hit_ratio = self.hits / total_accesses if total_accesses > 0 else 0
        return {
            "capacity": self.capacity,
            "current_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": hit_ratio
        }

    def clear(self) -> None:
        """
        Clear the cache
        """
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    async def cache_response(self, query: str, response: str) -> None:
        """
        Cache a response with timestamp
        """
        try:
            self.put(query, {
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")

    async def get_cached_response(self, query: str) -> Optional[str]:
        """
        Get cached response if available
        """
        try:
            cached_data = self.get(query)
            if cached_data:
                if isinstance(cached_data, dict):
                    return cached_data.get('response')
                return cached_data
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached response: {str(e)}")
            return None
