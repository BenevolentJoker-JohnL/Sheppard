# src/core/memory/storage/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class StorageBase(ABC):
    """Base class for storage implementations"""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize storage system"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup storage resources"""
        pass
    
    @abstractmethod
    async def store_memory(
        self,
        key: str,
        value: Dict[str, Any],
        layer: str,
        memory_hash: str,
        importance_score: float
    ) -> bool:
        """Store memory in storage system"""
        pass
    
    @abstractmethod
    async def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        layer: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from storage system"""
        pass
    
    @abstractmethod
    async def cleanup_old_memories(
        self,
        layer: str,
        days_threshold: int,
        importance_threshold: float
    ) -> Dict[str, int]:
        """Clean up old memories"""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate storage connection"""
        pass
