from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, List, Optional, Any

class MemoryState:
    """Memory state management"""
    def __init__(self, name: str):
        self.name = name
        self.data = {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0

    def update_access(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class Memory(BaseModel):
    """Base memory model"""
    id: str
    content: Any
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    importance_score: float = 0.0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    state: str = "default"
    layer: str
    access_count: int = 0

class UserProfile(BaseModel):
    """User profile model"""
    name: str
    age: Optional[int] = None
    interests: List[str] = Field(default_factory=list)
    occupation: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
