"""
System initialization and component management.
File: src/core/system.py
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from src.config.settings import settings
from src.llm.client import OllamaClient
from src.memory.manager import MemoryManager
from src.research.system import ResearchSystem
from src.research.content_processor import ContentProcessor
from src.core.exceptions import (
    InitializationError,
    ResourceError,
    ConfigurationError
)

logger = logging.getLogger(__name__)

class SystemManager:
    """Manages system components and initialization."""
    
    def __init__(self):
        """Initialize system manager."""
        self.ollama_client = None
        self.memory_manager = None
        self.research_system = None
        self.content_processor = None
        self._initialized = False
        
        # Create required directories
        self._create_directories()

    def _create_directories(self) -> None:
        """Create required system directories."""
        directories = [
            settings.DATA_DIR,
            settings.LOG_DIR,
            settings.SCREENSHOT_DIR,
            settings.TEMP_DIR,
            settings.CHROMADB_PERSIST_DIRECTORY
        ]
        
        for directory in [d for d in directories if d]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> Tuple[bool, Optional[str]]:
        """Initialize system components in correct order."""
        if self._initialized:
            return True, None

        try:
            # 1. Initialize Ollama client first
            self.ollama_client = OllamaClient(
                model_name=settings.OLLAMA_MODEL,
                api_base=settings.ollama_api_base
            )
            
            # Verify model availability
            if not await self.ollama_client.verify_model_availability():
                raise ResourceError(
                    "model",
                    f"Model {settings.OLLAMA_MODEL} not available"
                )
            
            # 2. Initialize memory manager
            self.memory_manager = MemoryManager()
            self.memory_manager.set_ollama_client(self.ollama_client)
            await self.memory_manager.initialize()
            
            # 3. Initialize content processor
            self.content_processor = ContentProcessor(
                ollama_client=self.ollama_client
            )
            
            # 4. Initialize research system
            self.research_system = ResearchSystem(
                memory_manager=self.memory_manager,
                content_processor=self.content_processor,
                browser_headless=True,
                screenshot_dir=settings.SCREENSHOT_DIR,
                timeout=settings.REQUEST_TIMEOUT,
                max_retries=settings.MAX_RETRIES
            )
            
            await self.research_system.initialize()
            
            self._initialized = True
            logger.info("System initialized successfully")
            return True, None
            
        except Exception as e:
            error_msg = f"System initialization failed: {str(e)}"
            logger.error(error_msg)
            await self.cleanup()
            return False, error_msg

    async def cleanup(self) -> None:
        """Clean up system resources."""
        try:
            # Clean up in reverse order of initialization
            
            if self.research_system:
                await self.research_system.cleanup()
            
            if self.memory_manager:
                await self.memory_manager.cleanup()
            
            # Reset state
            self.ollama_client = None
            self.memory_manager = None
            self.research_system = None
            self.content_processor = None
            self._initialized = False
            
            logger.info("System cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during system cleanup: {str(e)}")

    async def get_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return {
            "initialized": self._initialized,
            "components": {
                "ollama": self.ollama_client is not None,
                "memory": self.memory_manager is not None,
                "research": self.research_system is not None,
                "content_processor": self.content_processor is not None
            },
            "models": {
                "chat": settings.OLLAMA_MODEL,
                "embedding": settings.OLLAMA_EMBED_MODEL
            },
            "gpu_enabled": settings.GPU_ENABLED,
            "settings": {
                "embedding_dimension": settings.EMBEDDING_DIMENSION,
                "chromadb_distance": settings.CHROMADB_DISTANCE_FUNC,
                "request_timeout": settings.REQUEST_TIMEOUT
            }
        }

    def __str__(self) -> str:
        """Get string representation."""
        status = "initialized" if self._initialized else "not initialized"
        components = sum(1 for c in [
            self.ollama_client,
            self.memory_manager,
            self.research_system,
            self.content_processor
        ] if c is not None)
        return f"SystemManager({status}, {components}/4 components)"


# Global system manager instance
system_manager = SystemManager()

async def initialize_system() -> Tuple[bool, Optional[str]]:
    """Initialize the system using the global manager."""
    return await system_manager.initialize()

async def cleanup_system() -> None:
    """Clean up the system using the global manager."""
    await system_manager.cleanup()
