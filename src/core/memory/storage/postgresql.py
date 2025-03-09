# src/core/memory/storage/postgresql.py
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import asyncio

from .base import StorageBase
from .connection import ConnectionManager
from src.config.config import DatabaseConfig

logger = logging.getLogger(__name__)

class PostgreSQLManager(StorageBase):
    """PostgreSQL storage implementation"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn_manager = connection_manager
        self.batch_size = 1000
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        self.vacuum_threshold = 10000  # rows
        
        self.storage_stats = {
            "operations": {
                "insert": 0,
                "update": 0,
                "delete": 0,
                "select": 0
            },
            "errors": {
                "insert": 0,
                "update": 0,
                "delete": 0,
                "select": 0
            },
            "latency": {
                "insert": [],
                "update": [],
                "delete": [],
                "select": []
            }
        }

    async def validate_connection(self) -> bool:
        """Validate PostgreSQL connection and schema"""
        try:
            # Test connection for each database
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    logger.error(f"Failed to connect to database: {db_name}")
                    return False
                
                try:
                    with conn.cursor() as cur:
                        # Test basic query execution
                        cur.execute("SELECT 1")
                        if cur.fetchone()[0] != 1:
                            logger.error(f"Basic query test failed for {db_name}")
                            return False
                        
                        # Verify required tables exist
                        cur.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables 
                                WHERE table_name = 'agent_interactions'
                            )
                        """)
                        if not cur.fetchone()[0]:
                            logger.error(f"Required table 'agent_interactions' not found in {db_name}")
                            return False
                        
                        # Test write permission with temporary data
                        try:
                            cur.execute("BEGIN")
                            cur.execute("""
                                INSERT INTO agent_interactions 
                                (agent_id, input, response, memory_hash, importance_score)
                                VALUES ('test', 'test', 'test', 'test', 0.0)
                                RETURNING id
                            """)
                            test_id = cur.fetchone()[0]
                            cur.execute("DELETE FROM agent_interactions WHERE id = %s", (test_id,))
                            cur.execute("ROLLBACK")
                        except Exception as e:
                            logger.error(f"Write permission test failed for {db_name}: {str(e)}")
                            return False
                        
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating PostgreSQL connection: {str(e)}")
            return False

    async def initialize(self) -> bool:
        """Initialize PostgreSQL storage"""
        try:
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    return False
                    
                try:
                    with conn.cursor() as cur:
                        # Create main table if it doesn't exist
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS agent_interactions (
                                id SERIAL PRIMARY KEY,
                                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                agent_id TEXT NOT NULL,
                                input TEXT NOT NULL,
                                response TEXT NOT NULL,
                                embedding FLOAT[],
                                entities JSONB DEFAULT '{}'::jsonb,
                                topics JSONB DEFAULT '{}'::jsonb,
                                state TEXT DEFAULT 'active',
                                memory_hash TEXT NOT NULL UNIQUE,
                                importance_score FLOAT DEFAULT 0.0,
                                last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                access_count INTEGER DEFAULT 0,
                                context_metadata JSONB DEFAULT '{}'::jsonb
                            )
                        """)
                        
                        # Create indices
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_hash ON agent_interactions(memory_hash)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON agent_interactions(timestamp)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_importance ON agent_interactions(importance_score)")
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON agent_interactions(last_accessed)")
                        
                    conn.commit()
                    
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL storage: {str(e)}")
            return False

    async def cleanup(self) -> None:
        """Cleanup is handled by ConnectionManager"""
        pass

    async def store_memory(
        self,
        key: str,
        value: Dict[str, Any],
        layer: str,
        memory_hash: str,
        importance_score: float
    ) -> bool:
        """Store memory in PostgreSQL"""
        start_time = datetime.now()
        conn = None
        
        try:
            conn = await self.conn_manager.get_pg_connection(layer)
            if not conn:
                return False

            with conn.cursor() as cur:
                # Convert embedding to PostgreSQL array format
                embedding = value.get('embedding')
                if embedding and isinstance(embedding, list):
                    embedding_array = f"{{{','.join(map(str, embedding))}}}"
                else:
                    embedding_array = None

                # Upsert memory with retry logic
                for attempt in range(self.max_retries):
                    try:
                        cur.execute("""
                            INSERT INTO agent_interactions (
                                timestamp, agent_id, input, response, embedding,
                                entities, topics, state, memory_hash, importance_score,
                                context_metadata
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (memory_hash) 
                            DO UPDATE SET
                                last_accessed = CURRENT_TIMESTAMP,
                                access_count = agent_interactions.access_count + 1,
                                importance_score = GREATEST(agent_interactions.importance_score, EXCLUDED.importance_score)
                            RETURNING id
                        """, (
                            datetime.now(),
                            value.get('agent_id', 'AI'),
                            value.get('input', ''),
                            value.get('response', ''),
                            embedding_array,
                            json.dumps(value.get('entities', {})),
                            json.dumps(value.get('topics', {})),
                            value.get('state', 'active'),
                            memory_hash,
                            importance_score,
                            json.dumps(value.get('context_metadata', {}))
                        ))
                        
                        result = cur.fetchone()
                        conn.commit()
                        
                        # Update statistics
                        self.storage_stats["operations"]["insert"] += 1
                        latency = (datetime.now() - start_time).total_seconds()
                        self.storage_stats["latency"]["insert"].append(latency)
                        
                        return bool(result)
                        
                    except psycopg2.Error as e:
                        conn.rollback()
                        if attempt == self.max_retries - 1:
                            raise
                        await asyncio.sleep(self.retry_delay)
                        
        except Exception as e:
            logger.error(f"Error storing memory in PostgreSQL: {str(e)}")
            self.storage_stats["errors"]["insert"] += 1
            if conn:
                conn.rollback()
            return False
            
        finally:
            if conn:
                self.conn_manager.return_pg_connection(layer, conn)

    async def retrieve_memories(
        self,
        query: str,
        embedding: List[float],
        layer: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories from PostgreSQL"""
        start_time = datetime.now()
        conn = None
        
        try:
            conn = await self.conn_manager.get_pg_connection(layer)
            if not conn:
                return []

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Query with cosine similarity if embedding is provided
                if embedding:
                    embedding_array = f"{{{','.join(map(str, embedding))}}}"
                    cur.execute("""
                        WITH similarity_scores AS (
                            SELECT *,
                                   1 - (embedding <=> %s::float[]) as similarity
                            FROM agent_interactions
                            WHERE state = 'active'
                            AND embedding IS NOT NULL
                        )
                        SELECT *
                        FROM similarity_scores
                        WHERE similarity > 0.5
                        ORDER BY 
                            similarity * importance_score DESC,
                            last_accessed DESC
                        LIMIT %s
                    """, (embedding_array, limit))
                else:
                    # Fallback to importance-based retrieval
                    cur.execute("""
                        SELECT *
                        FROM agent_interactions
                        WHERE state = 'active'
                        ORDER BY importance_score DESC, last_accessed DESC
                        LIMIT %s
                    """, (limit,))

                memories = cur.fetchall()
                
                # Update statistics
                self.storage_stats["operations"]["select"] += 1
                latency = (datetime.now() - start_time).total_seconds()
                self.storage_stats["latency"]["select"].append(latency)
                
                return [dict(memory) for memory in memories]
                
        except Exception as e:
            logger.error(f"Error retrieving memories from PostgreSQL: {str(e)}")
            self.storage_stats["errors"]["select"] += 1
            return []
            
        finally:
            if conn:
                self.conn_manager.return_pg_connection(layer, conn)

    async def cleanup_old_memories(
        self,
        days_threshold: int,
        importance_threshold: float
    ) -> Dict[str, int]:
        """Clean up old memories from PostgreSQL"""
        cleanup_stats = {"removed": 0}
        start_time = datetime.now()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    continue
                
                try:
                    with conn.cursor() as cur:
                        # Delete old memories with low importance
                        cur.execute("""
                            DELETE FROM agent_interactions
                            WHERE timestamp < %s
                            AND importance_score < %s
                            RETURNING id
                        """, (cutoff_date, importance_threshold))
                        
                        removed = cur.rowcount
                        cleanup_stats["removed"] += removed
                        
                        # Perform VACUUM if significant data was removed
                        if removed > self.vacuum_threshold:
                            conn.set_session(autocommit=True)
                            cur.execute("VACUUM ANALYZE agent_interactions")
                        
                        conn.commit()
                        
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
            
            # Update statistics
            self.storage_stats["operations"]["delete"] += 1
            latency = (datetime.now() - start_time).total_seconds()
            self.storage_stats["latency"]["delete"].append(latency)
            
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error cleaning up PostgreSQL memories: {str(e)}")
            self.storage_stats["errors"]["delete"] += 1
            return cleanup_stats
    def _update_operation_stats(
        self,
        operation: str,
        success: bool,
        latency: float
    ) -> None:
        """Update operation statistics"""
        try:
            # Update operation count
            self.storage_stats["operations"][operation] += 1
            
            # Update error count if failed
            if not success:
                self.storage_stats["errors"][operation] += 1
            
            # Update latency measurements
            if success and latency > 0:
                self.storage_stats["latency"][operation].append(latency)
                # Keep only last 1000 measurements
                if len(self.storage_stats["latency"][operation]) > 1000:
                    self.storage_stats["latency"][operation] = \
                        self.storage_stats["latency"][operation][-1000:]
                        
        except Exception as e:
            logger.error(f"Error updating operation stats: {str(e)}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            stats = {
                **self.storage_stats,
                "average_latency": {
                    op: sum(times) / len(times) if times else 0
                    for op, times in self.storage_stats["latency"].items()
                },
                "database_stats": await self._get_database_stats()
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting storage stats: {str(e)}")
            return {}

    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get detailed database statistics"""
        try:
            stats = {}
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    continue
                
                try:
                    with conn.cursor() as cur:
                        # Get table statistics
                        cur.execute("""
                            SELECT 
                                relname as table_name,
                                n_live_tup as row_count,
                                n_dead_tup as dead_rows,
                                pg_size_pretty(pg_relation_size(relid)) as size
                            FROM pg_stat_user_tables
                            WHERE schemaname = 'public'
                        """)
                        table_stats = cur.fetchall()
                        
                        # Get index statistics
                        cur.execute("""
                            SELECT
                                indexrelname as index_name,
                                idx_scan as usage_count,
                                pg_size_pretty(pg_relation_size(indexrelid)) as size
                            FROM pg_stat_user_indexes
                            WHERE schemaname = 'public'
                        """)
                        index_stats = cur.fetchall()
                        
                        # Get memory layer statistics
                        cur.execute("""
                            SELECT
                                state,
                                COUNT(*) as count,
                                AVG(importance_score) as avg_importance,
                                MIN(last_accessed) as oldest_access,
                                MAX(last_accessed) as newest_access
                            FROM agent_interactions
                            GROUP BY state
                        """)
                        memory_stats = cur.fetchall()
                        
                        stats[db_name] = {
                            "tables": {
                                row[0]: {
                                    "row_count": row[1],
                                    "dead_rows": row[2],
                                    "size": row[3]
                                }
                                for row in table_stats
                            },
                            "indexes": {
                                row[0]: {
                                    "usage_count": row[1],
                                    "size": row[2]
                                }
                                for row in index_stats
                            },
                            "memory_states": {
                                row[0]: {
                                    "count": row[1],
                                    "avg_importance": float(row[2]) if row[2] else 0,
                                    "oldest_access": row[3].isoformat() if row[3] else None,
                                    "newest_access": row[4].isoformat() if row[4] else None
                                }
                                for row in memory_stats
                            }
                        }
                        
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
                    
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}

    async def vacuum_analyze(self, table_name: str = 'agent_interactions') -> bool:
        """Perform VACUUM ANALYZE on specified table"""
        try:
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    continue
                
                try:
                    # VACUUM requires autocommit mode
                    conn.set_session(autocommit=True)
                    with conn.cursor() as cur:
                        cur.execute(f"VACUUM ANALYZE {table_name}")
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing VACUUM ANALYZE: {str(e)}")
            return False

    async def reindex_table(self, table_name: str = 'agent_interactions') -> bool:
        """Reindex specified table"""
        try:
            for db_name in DatabaseConfig.DB_URLS.keys():
                conn = await self.conn_manager.get_pg_connection(db_name)
                if not conn:
                    continue
                
                try:
                    with conn.cursor() as cur:
                        cur.execute(f"REINDEX TABLE {table_name}")
                        conn.commit()
                finally:
                    self.conn_manager.return_pg_connection(db_name, conn)
            
            return True
            
        except Exception as e:
            logger.error(f"Error reindexing table: {str(e)}")
            return False

    async def update_memory_state(
        self,
        memory_hash: str,
        new_state: str,
        layer: str
    ) -> bool:
        """Update memory state"""
        try:
            conn = await self.conn_manager.get_pg_connection(layer)
            if not conn:
                return False
            
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE agent_interactions
                        SET state = %s
                        WHERE memory_hash = %s
                        RETURNING id
                    """, (new_state, memory_hash))
                    
                    result = cur.fetchone()
                    conn.commit()
                    
                    return bool(result)
                    
            finally:
                self.conn_manager.return_pg_connection(layer, conn)
                
        except Exception as e:
            logger.error(f"Error updating memory state: {str(e)}")
            return False

    async def update_importance_score(
        self,
        memory_hash: str,
        new_score: float,
        layer: str
    ) -> bool:
        """Update memory importance score"""
        try:
            conn = await self.conn_manager.get_pg_connection(layer)
            if not conn:
                return False
            
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE agent_interactions
                        SET importance_score = %s
                        WHERE memory_hash = %s
                        RETURNING id
                    """, (new_score, memory_hash))
                    
                    result = cur.fetchone()
                    conn.commit()
                    
                    return bool(result)
                    
            finally:
                self.conn_manager.return_pg_connection(layer, conn)
                
        except Exception as e:
            logger.error(f"Error updating importance score: {str(e)}")
            return False

    async def get_memory_by_hash(
        self,
        memory_hash: str,
        layer: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve specific memory by hash"""
        try:
            conn = await self.conn_manager.get_pg_connection(layer)
            if not conn:
                return None
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT *
                        FROM agent_interactions
                        WHERE memory_hash = %s
                    """, (memory_hash,))
                    
                    result = cur.fetchone()
                    return dict(result) if result else None
                    
            finally:
                self.conn_manager.return_pg_connection(layer, conn)
                
        except Exception as e:
            logger.error(f"Error retrieving memory by hash: {str(e)}")
            return None

    def __str__(self) -> str:
        """String representation"""
        stats = self.storage_stats["operations"]
        return (
            f"PostgreSQLManager(insert={stats['insert']}, "
            f"update={stats['update']}, "
            f"delete={stats['delete']}, "
            f"select={stats['select']})"
        )
