#!/usr/bin/env python3

import subprocess
import logging
from typing import List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command: str) -> bool:
    """Execute a shell command and return success status"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )
        output, error = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error executing command: {command}")
            logger.error(error.decode())
            return False
            
        logger.info(output.decode().strip())
        return True
    except Exception as e:
        logger.error(f"Exception executing command: {str(e)}")
        return False

def create_and_update_database_schema(database: str) -> bool:
    """Create and update schema for a single database"""
    logger.info(f"Creating and updating schema for {database}...")
    
    # First create the table if it doesn't exist
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS agent_interactions (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        agent_id TEXT NOT NULL,
        input TEXT NOT NULL,
        response TEXT NOT NULL,
        embedding FLOAT[],
        entities JSONB,
        topics JSONB,
        state TEXT DEFAULT 'default',
        memory_hash TEXT UNIQUE,
        importance_score FLOAT DEFAULT 0.0,
        last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 0,
        context_metadata JSONB DEFAULT '{}'::jsonb
    );
    """
    
    # Create indices
    create_indices_sql = """
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_agent_id ON agent_interactions (agent_id);
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_timestamp ON agent_interactions ("timestamp");
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_state ON agent_interactions (state);
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_importance ON agent_interactions (importance_score);
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_memory_hash ON agent_interactions (memory_hash);
    CREATE INDEX IF NOT EXISTS idx_agent_interactions_last_accessed ON agent_interactions (last_accessed);
    """
    
    # Execute create table command
    create_command = f"psql -d {database} -c \"{create_table_sql}\""
    if not run_command(f'sudo -u postgres {create_command}'):
        return False
        
    # Execute create indices command
    indices_command = f"psql -d {database} -c \"{create_indices_sql}\""
    if not run_command(f'sudo -u postgres {indices_command}'):
        return False
        
    return True

def verify_database_schema(database: str) -> bool:
    """Verify the schema update for a database"""
    logger.info(f"Verifying schema for {database}...")
    
    command = f'sudo -u postgres psql -d {database} -c "\\d agent_interactions"'
    return run_command(command)

def main():
    """Main function to update all database schemas"""
    databases = [
        "episodic_memory",
        "semantic_memory",
        "contextual_memory",
        "general_memory",
        "abstracted_memory"
    ]
    
    success = True
    
    # Create and update schemas
    for db in databases:
        if not create_and_update_database_schema(db):
            logger.error(f"Failed to create/update schema for {db}")
            success = False
    
    # Verify schemas if updates were successful
    if success:
        logger.info("\nVerifying all database schemas...")
        for db in databases:
            if not verify_database_schema(db):
                logger.error(f"Schema verification failed for {db}")
                success = False
    
    if success:
        logger.info("\n✓ All database schemas created, updated and verified successfully!")
    else:
        logger.error("\n✗ Some schema updates or verifications failed. Please check the logs above.")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        exit(exit_code)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)
