# src/core/memory/cleanup.py
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import asyncio

from .storage.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class CleanupManager:
    """Manages memory cleanup operations"""
    
    def __init__(self, importance_threshold: float):
        self.importance_threshold = importance_threshold
        self.cleanup_running = False
        self.last_cleanup = datetime.now()
        self.cleanup_stats = {
            "total_cleaned": 0,
            "last_cleanup_duration": 0,
            "errors": 0,
            "last_run": datetime.now().isoformat()
        }

    async def perform_full_cleanup(self) -> Dict[str, Any]:
        """Perform full cleanup of all storage types"""
        if self.cleanup_running:
            return self.cleanup_stats
            
        self.cleanup_running = True
        start_time = datetime.now()
        
        cleanup_results = {
            "postgresql_removed": 0,
            "chromadb_removed": 0,
            "redis_removed": 0,
            "timestamp": start_time.isoformat(),
            "layers_cleaned": []
        }
        
        try:
            storage_manager = StorageManager()
            if not await storage_manager.initialize():
                raise Exception("Failed to initialize storage manager for cleanup")
            
            try:
                layers = ["episodic", "semantic", "contextual", "general", "abstracted"]
                cleanup_tasks = []
                
                # Create cleanup tasks for each layer
                for layer in layers:
                    task = asyncio.create_task(
                        storage_manager.cleanup_old_memories(
                            layer=layer,
                            days_threshold=30,  # days threshold
                            importance_threshold=self.importance_threshold
                        )
                    )
                    cleanup_tasks.append((layer, task))
                
                # Wait for all cleanup tasks
                for layer, task in cleanup_tasks:
                    try:
                        result = await task
                        cleanup_results["postgresql_removed"] += result.get("postgresql_removed", 0)
                        cleanup_results["chromadb_removed"] += result.get("vector_store_removed", 0)
                        cleanup_results["redis_removed"] += result.get("redis_removed", 0)
                        cleanup_results["layers_cleaned"].append(layer)
                    except Exception as e:
                        logger.error(f"Error cleaning up layer {layer}: {str(e)}")
                        self.cleanup_stats["errors"] += 1
                
                # Update total stats
                cleanup_results["total_removed"] = (
                    cleanup_results["postgresql_removed"] +
                    cleanup_results["chromadb_removed"] +
                    cleanup_results["redis_removed"]
                )
                
                self.cleanup_stats["total_cleaned"] += cleanup_results["total_removed"]
                self.cleanup_stats["last_cleanup_duration"] = (
                    datetime.now() - start_time
                ).total_seconds()
                self.cleanup_stats["last_run"] = datetime.now().isoformat()
                    
            finally:
                await storage_manager.cleanup()
                
            self.last_cleanup = datetime.now()
            logger.info(f"Memory cleanup completed: {cleanup_results}")
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            self.cleanup_stats["errors"] += 1
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                **cleanup_results
            }
            
        finally:
            self.cleanup_running = False

    async def cleanup_layer(
        self,
        storage_manager: StorageManager,
        layer: str,
        cutoff_date: datetime
    ) -> Dict[str, int]:
        """Clean up a specific memory layer"""
        try:
            results = await storage_manager.cleanup_old_memories(
                layer=layer,
                days_threshold=30,
                importance_threshold=self.importance_threshold
            )
            
            return {
                "layer": layer,
                "removed": results.get("total_removed", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up layer {layer}: {str(e)}")
            return {
                "layer": layer,
                "removed": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get cleanup statistics"""
        return {
            **self.cleanup_stats,
            "last_cleanup": self.last_cleanup.isoformat(),
            "is_cleaning": self.cleanup_running,
            "importance_threshold": self.importance_threshold
        }
