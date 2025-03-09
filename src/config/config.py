import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from rich.console import Console
import atexit

# Initialize Rich console
console = Console()

class LogConfig:
    """Logging configuration"""
    
    LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @classmethod
    def setup_logging(cls) -> None:
        """Set up logging configuration"""
        try:
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # Set up logging
            logging.basicConfig(
                level=logging.INFO,
                format=cls.LOG_FORMAT,
                datefmt=cls.DATE_FORMAT,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(
                        os.path.join(
                            logs_dir,
                            f"sheppard_{datetime.now().strftime('%Y%m%d')}.log"
                        )
                    )
                ]
            )
            
            logger = logging.getLogger(__name__)
            logger.info("Logging system initialized")
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            sys.exit(1)

class PathConfig:
    """Path configuration"""
    
    # Get project root directory
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define directory paths
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")
    CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
    CHROMA_DIR = os.path.join(DATA_DIR, "chroma_persistence")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
    STATS_DIR = os.path.join(DATA_DIR, "stats")
    TOOLS_DIR = os.path.join(DATA_DIR, "tools")
    MEMORY_DIR = os.path.join(DATA_DIR, "memory")
    
    # Memory layer directories
    MEMORY_LAYERS = {
        "episodic": os.path.join(MEMORY_DIR, "episodic"),
        "semantic": os.path.join(MEMORY_DIR, "semantic"),
        "contextual": os.path.join(MEMORY_DIR, "contextual"),
        "general": os.path.join(MEMORY_DIR, "general"),
        "abstracted": os.path.join(MEMORY_DIR, "abstracted")
    }
    
    @classmethod
    def initialize_directories(cls) -> bool:
        """Initialize required directories"""
        try:
            # Create main directories
            directories = [
                cls.DATA_DIR,
                cls.LOGS_DIR,
                cls.CONVERSATIONS_DIR,
                cls.CHROMA_DIR,
                cls.EMBEDDINGS_DIR,
                cls.STATS_DIR,
                cls.TOOLS_DIR,
                cls.MEMORY_DIR
            ]
            
            # Add memory layer directories
            directories.extend(cls.MEMORY_LAYERS.values())
            
            # Create directories if they don't exist
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    logging.error(f"No write permission for directory: {directory}")
                    return False
                    
            logging.getLogger(__name__).info("All required directories validated")
            return True
            
        except Exception as e:
            logging.error(f"Error initializing directories: {str(e)}")
            return False
            
    @classmethod
    def cleanup_on_exit(cls) -> None:
        """Cleanup temporary files on exit"""
        try:
            logging.getLogger(__name__).info("Performing cleanup on exit")
            # Add cleanup tasks here if needed
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")

# Register cleanup on exit
atexit.register(PathConfig.cleanup_on_exit)

class MemoryConfig:
    """Memory system configuration"""
    
    # Memory configuration
    MEMORY_CONFIG = {
        "cache_size": 1000,
        "importance_threshold": 0.5,
        "similarity_threshold": 0.7,
        "retention_period_days": 30,
        "max_memories_per_layer": 10000,
        "cleanup_interval_hours": 24,
        "layers": {
            "episodic": {
                "ttl": 86400,  # 24 hours
                "max_size": 1000
            },
            "semantic": {
                "ttl": 604800,  # 7 days
                "max_size": 5000
            },
            "contextual": {
                "ttl": 3600,  # 1 hour
                "max_size": 100
            },
            "general": {
                "ttl": 2592000,  # 30 days
                "max_size": 10000
            },
            "abstracted": {
                "ttl": 7776000,  # 90 days
                "max_size": 1000
            }
        }
    }

    # Embedding configuration
    EMBEDDING_CONFIG = {
        "batch_size": 32,
        "cache_size": 1000,
        "dimension": 768  # For nomic-embed-text model
    }

    # Vector store configuration
    VECTOR_STORE_CONFIG = {
        "similarity_threshold": 0.8,
        "max_results": 10,
        "index_params": {
            "M": 16,
            "efConstruction": 200
        },
        "query_params": {
            "ef": 50
        }
    }

class DatabaseConfig:
    """Database configuration"""
    
    # PostgreSQL connection URLs
    DB_URLS = {
        "episodic_memory": "postgresql://sheppard:1234@localhost:5432/episodic_memory",
        "semantic_memory": "postgresql://sheppard:1234@localhost:5432/semantic_memory",
        "contextual_memory": "postgresql://sheppard:1234@localhost:5432/contextual_memory",
        "general_memory": "postgresql://sheppard:1234@localhost:5432/general_memory",
        "abstracted_memory": "postgresql://sheppard:1234@localhost:5432/abstracted_memory"
    }
    
    # Redis configuration
    REDIS_CONFIG = {
        "ephemeral": {"host": "localhost", "port": 6370, "db": 0},
        "contextual": {"host": "localhost", "port": 6371, "db": 0},
        "episodic": {"host": "localhost", "port": 6372, "db": 0},
        "semantic": {"host": "localhost", "port": 6373, "db": 0},
        "abstracted": {"host": "localhost", "port": 6374, "db": 0}
    }
    
    # Memory configuration reference
    MEMORY_CONFIG = MemoryConfig.MEMORY_CONFIG
    
    # Directory references from PathConfig
    DATA_DIR = PathConfig.DATA_DIR
    LOGS_DIR = PathConfig.LOGS_DIR
    CONVERSATIONS_DIR = PathConfig.CONVERSATIONS_DIR
    CHROMA_DIR = PathConfig.CHROMA_DIR
    EMBEDDINGS_DIR = PathConfig.EMBEDDINGS_DIR
    STATS_DIR = PathConfig.STATS_DIR
    TOOLS_DIR = PathConfig.TOOLS_DIR
    MEMORY_DIR = PathConfig.MEMORY_DIR
    MEMORY_LAYERS = PathConfig.MEMORY_LAYERS
    
    @classmethod
    def validate_directories(cls) -> bool:
        """Validate required directories"""
        return PathConfig.initialize_directories()

class ModelConfig:
    """Model configuration"""
    
    # Ollama host configuration
    OLLAMA_HOST = "http://localhost:11434"
    
    # Model configurations
    MODEL_CONFIG = {
        "main_chat": {
            "name": "llama3.1",
            "tag": "latest",
            "context_length": 4096
        },
        "short_context": {
            "name": "llama3.2",
            "tag": "latest",
            "context_length": 4096
        },
        "long_context": {
            "name": "mistral-nemo",
            "tag": "latest",
            "context_length": 8192
        },
        "embedding": {
            "name": "nomic-embed-text",
            "tag": "latest",
            "embedding_dim": 768
        }
    }
    
    @classmethod
    def get_model_path(cls, model_name: str) -> str:
        """Get full path for a model"""
        config = cls.MODEL_CONFIG.get(model_name)
        if config:
            return f"{config['name']}:{config['tag']}"
        return ""

class ToolConfig:
    """Tool configuration"""
    
    TOOL_CONFIG = {
        "calculator": {
            "enabled": True,
            "max_retries": 3,
            "timeout": 5
        },
        "search": {
            "enabled": True,
            "max_results": 5,
            "timeout": 10
        },
        "summarizer": {
            "enabled": True,
            "max_length": 1000,
            "timeout": 15
        },
        "sentiment": {
            "enabled": True,
            "timeout": 5
        },
        "entity_extractor": {
            "enabled": True,
            "timeout": 10
        }
    }
    
    @classmethod
    def is_tool_enabled(cls, tool_name: str) -> bool:
        """Check if a tool is enabled"""
        tool_config = cls.TOOL_CONFIG.get(tool_name, {})
        return tool_config.get("enabled", False)

# Initialize logging on module import
LogConfig.setup_logging()
