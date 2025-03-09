from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryStats:
    """
    Statistics tracking for memory operations
    """
    total_memories: int = 0
    memories_per_state: Dict[str, int] = field(default_factory=dict)
    last_cleanup: datetime = field(default_factory=datetime.now)
    total_unique_memories: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    embedding_count: int = 0
    failed_embeddings: int = 0
    average_importance_score: float = 0.0
    state_transitions: Dict[str, int] = field(default_factory=dict)

    def update_importance_score(self, new_score: float) -> None:
        """
        Update the running average importance score
        """
        if self.total_memories == 0:
            self.average_importance_score = new_score
        else:
            self.average_importance_score = (
                (self.average_importance_score * self.total_memories + new_score) /
                (self.total_memories + 1)
            )

    def record_state_transition(self, from_state: str, to_state: str) -> None:
        """
        Record a state transition
        """
        transition_key = f"{from_state}->{to_state}"
        self.state_transitions[transition_key] = self.state_transitions.get(transition_key, 0) + 1

    def record_memory_creation(self, state: str) -> None:
        """
        Record creation of a new memory
        """
        self.total_memories += 1
        self.memories_per_state[state] = self.memories_per_state.get(state, 0) + 1

    def record_embedding_operation(self, success: bool) -> None:
        """
        Record embedding operation result
        """
        self.embedding_count += 1
        if not success:
            self.failed_embeddings += 1

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics summary
        """
        total_cache_operations = self.cache_hits + self.cache_misses
        cache_hit_ratio = (
            self.cache_hits / total_cache_operations 
            if total_cache_operations > 0 else 0
        )
        
        embedding_success_rate = (
            (self.embedding_count - self.failed_embeddings) / self.embedding_count 
            if self.embedding_count > 0 else 0
        )

        return {
            "total_memories": self.total_memories,
            "unique_memories": self.total_unique_memories,
            "memories_per_state": self.memories_per_state,
            "cache_performance": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_ratio": cache_hit_ratio
            },
            "embedding_stats": {
                "total_embeddings": self.embedding_count,
                "failed_embeddings": self.failed_embeddings,
                "success_rate": embedding_success_rate
            },
            "average_importance": self.average_importance_score,
            "state_transitions": self.state_transitions,
            "last_cleanup": self.last_cleanup.isoformat()
        }

    def to_json(self) -> str:
        """
        Convert stats to JSON string
        """
        try:
            stats_dict = self.get_stats_summary()
            return json.dumps(stats_dict, indent=2)
        except Exception as e:
            logger.error(f"Error converting stats to JSON: {str(e)}")
            return "{}"

    def reset(self) -> None:
        """
        Reset all statistics
        """
        self.total_memories = 0
        self.memories_per_state.clear()
        self.last_cleanup = datetime.now()
        self.total_unique_memories = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.embedding_count = 0
        self.failed_embeddings = 0
        self.average_importance_score = 0.0
        self.state_transitions.clear()
