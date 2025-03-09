from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, client):
        self.client = client
        self.description = ""
        self.parameters = {}
        self.required = []
        self.examples = []

    @abstractmethod
    async def execute(self, **kwargs) -> Optional[str]:
        """Execute the tool with given parameters"""
        pass

    @abstractmethod
    async def validate(self) -> bool:
        """Validate tool configuration and requirements"""
        pass

    async def cleanup(self) -> None:
        """Cleanup any resources"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get tool information"""
        return {
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required,
            "examples": self.examples
        }
