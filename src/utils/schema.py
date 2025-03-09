import logging
import psycopg2
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class SchemaValidator:
    """Database schema validation and management"""
    
    def validate_and_fix_schema(self, conn, table_name: str) -> bool:
        """
        Validate and fix table schema if necessary
        """
        try:
            with conn.cursor() as cur:
                # Check for required columns
                required_columns = {
                    'state': 'TEXT',
                    'importance_score': 'FLOAT',
                    'embedding': 'FLOAT[]',
                    'metadata': 'JSONB',
                    'created_at': 'TIMESTAMP WITH TIME ZONE',
                    'last_accessed': 'TIMESTAMP WITH TIME ZONE',
                    'access_count': 'INTEGER'
                }

                # Get existing columns
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = %s
                """, (table_name,))
                
                existing_columns = {row[0]: row[1] for row in cur.fetchall()}

                # Add missing columns
                for column, data_type in required_columns.items():
                    if column not in existing_columns:
                        logger.warning(f"Adding missing column '{column}' to {table_name}")
                        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {data_type}")

                # Add indices if they don't exist
                indices = [
                    ('idx_importance', 'importance_score'),
                    ('idx_state', 'state'),
                    ('idx_created', 'created_at'),
                    ('idx_accessed', 'last_accessed'),
                    ('idx_access_count', 'access_count')
                ]

                for index_name, column in indices:
                    cur.execute("""
                        SELECT 1 
                        FROM pg_indexes 
                        WHERE tablename = %s 
                        AND indexname = %s
                    """, (table_name, index_name))
                    
                    if not cur.fetchone():
                        logger.info(f"Creating index {index_name} on {table_name}")
                        cur.execute(f"""
                            CREATE INDEX {index_name} 
                            ON {table_name} ({column})
                        """)

                conn.commit()
                logger.info(f"Schema validation complete for {table_name}")
                return True

        except Exception as e:
            logger.error(f"Error validating schema for {table_name}: {str(e)}")
            conn.rollback()
            return False

    def create_memory_table(self, conn, table_name: str) -> bool:
        """
        Create memory table with proper schema
        """
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id SERIAL PRIMARY KEY,
                        content JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        importance_score FLOAT DEFAULT 0.0,
                        embedding FLOAT[],
                        metadata JSONB DEFAULT '{}'::jsonb,
                        state TEXT DEFAULT 'default',
                        layer TEXT NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        memory_hash TEXT UNIQUE
                    )
                """.format(table_name))
                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Error creating memory table {table_name}: {str(e)}")
            conn.rollback()
            return False
