# src/system_checks.py

import os
import sys
import psutil
import platform
import logging
import subprocess
from typing import Dict, Any
import json
import httpx
import asyncio
from datetime import datetime

from src.config.config import DatabaseConfig, ModelConfig, PathConfig

logger = logging.getLogger(__name__)

def check_system_requirements() -> bool:
    """Check if system meets minimum requirements"""
    try:
        # Check Python version
        python_version = tuple(map(int, platform.python_version_tuple()))
        if python_version < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False

        # Check system memory
        memory = psutil.virtual_memory()
        if memory.total < (4 * 1024 * 1024 * 1024):  # 4GB
            logger.error("Minimum 4GB of RAM required")
            return False

        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < (5 * 1024 * 1024 * 1024):  # 5GB
            logger.error("Minimum 5GB of free disk space required")
            return False

        # Check CPU cores
        if psutil.cpu_count() < 2:
            logger.error("Minimum 2 CPU cores required")
            return False

        return True

    except Exception as e:
        logger.error(f"Error checking system requirements: {str(e)}")
        return False

def check_data_directories() -> bool:
    """Check required data directories"""
    try:
        # Initialize directories
        if not PathConfig.initialize_directories():
            return False
            
        # Check write permissions
        directories = [
            PathConfig.DATA_DIR,
            PathConfig.LOGS_DIR,
            PathConfig.CONVERSATIONS_DIR,
            PathConfig.CHROMA_DIR,
            PathConfig.EMBEDDINGS_DIR,
            PathConfig.STATS_DIR,
            PathConfig.TOOLS_DIR,
            PathConfig.MEMORY_DIR
        ]
        
        directories.extend(PathConfig.MEMORY_LAYERS.values())
        
        for directory in directories:
            if not os.path.exists(directory):
                logger.error(f"Directory does not exist: {directory}")
                return False
            if not os.access(directory, os.W_OK):
                logger.error(f"No write permission for directory: {directory}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error checking data directories: {str(e)}")
        return False

def check_ollama_service() -> bool:
    """Check if Ollama service is running"""
    try:
        response = httpx.get(f"{ModelConfig.OLLAMA_HOST}/api/tags")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error checking Ollama service: {str(e)}")
        return False

def check_model_availability() -> Dict[str, Any]:
    """Check if required models are available"""
    required_models = {
        f"{config['name']}:{config['tag']}"
        for config in ModelConfig.MODEL_CONFIG.values()
    }
    
    try:
        response = httpx.get(f"{ModelConfig.OLLAMA_HOST}/api/tags")
        if response.status_code != 200:
            return {
                "available": False,
                "error": "Failed to get model list",
                "missing_models": list(required_models)
            }

        available_models = {
            model["name"].lower()
            for model in response.json().get("models", [])
        }

        missing_models = [
            model for model in required_models
            if model.split(':')[0].lower() not in available_models
        ]

        return {
            "available": len(missing_models) == 0,
            "found_models": list(required_models - set(missing_models)),
            "missing_models": missing_models
        }

    except Exception as e:
        logger.error(f"Error checking model availability: {str(e)}")
        return {
            "available": False,
            "error": str(e),
            "missing_models": list(required_models)
        }

def get_system_info() -> Dict[str, Any]:
    """Get detailed system information"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
            "memory_total": memory.total,
            "memory_available": memory.available,
            "memory_percent": memory.percent,
            "disk_total": disk.total,
            "disk_free": disk.free,
            "disk_percent": disk.percent,
            "directories": {
                name: {
                    "exists": os.path.exists(path),
                    "writable": os.access(path, os.W_OK),
                    "size": _get_dir_size(path)
                }
                for name, path in PathConfig.MEMORY_LAYERS.items()
            },
            "ollama_status": check_ollama_service(),
            "model_status": check_model_availability()
        }

    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}

def check_system_health() -> Dict[str, Any]:
    """Check current system health"""
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "ollama_service": check_ollama_service(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error checking system health: {str(e)}")
        return {"error": str(e)}

def _get_dir_size(path: str) -> int:
    """Get directory size in bytes"""
    try:
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size
    except Exception:
        return 0

def validate_installation() -> Dict[str, bool]:
    """Validate complete installation"""
    try:
        # Check Python version
        python_version = sys.version_info >= (3, 8)

        # Check system resources
        system_resources = check_system_requirements()

        # Check Ollama service
        ollama_service = check_ollama_service()

        # Check model availability
        model_status = check_model_availability()
        required_models = model_status.get("available", False)

        # Check directories and permissions
        directories = check_data_directories()

        # Check permissions for all directories
        permissions = all(
            os.access(path, os.W_OK)
            for path in [
                PathConfig.DATA_DIR,
                PathConfig.LOGS_DIR,
                PathConfig.CONVERSATIONS_DIR,
                PathConfig.CHROMA_DIR,
                PathConfig.EMBEDDINGS_DIR,
                PathConfig.STATS_DIR,
                PathConfig.TOOLS_DIR,
                PathConfig.MEMORY_DIR,
                *PathConfig.MEMORY_LAYERS.values()
            ]
        )

        # Check database connectivity
        database_connections = _check_database_connections()

        results = {
            "python_version": python_version,
            "system_resources": system_resources,
            "ollama_service": ollama_service,
            "required_models": required_models,
            "model_details": model_status,
            "directories": directories,
            "permissions": permissions,
            "database_connections": database_connections
        }

        return results

    except Exception as e:
        logger.error(f"Error validating installation: {str(e)}")
        return {
            "python_version": False,
            "system_resources": False,
            "ollama_service": False,
            "required_models": False,
            "directories": False,
            "permissions": False,
            "database_connections": False,
            "error": str(e)
        }

def _check_database_connections() -> bool:
    """Check all database connections"""
    try:
        # Check PostgreSQL connections
        for db_url in DatabaseConfig.DB_URLS.values():
            import psycopg2
            conn = psycopg2.connect(db_url)
            conn.close()

        # Check Redis connections
        for config in DatabaseConfig.REDIS_CONFIG.values():
            import redis
            client = redis.Redis(
                host=config['host'],
                port=config['port'],
                db=config['db']
            )
            client.ping()
            client.close()

        return True

    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

async def async_validate_installation() -> Dict[str, bool]:
    """Async version of installation validation"""
    return await asyncio.to_thread(validate_installation)
