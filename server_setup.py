#!/usr/bin/env python3

import subprocess
import sys
import os
import shutil
import time

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        return False
    return output.decode().strip()

def install_packages():
    print("Installing necessary packages...")
    run_command("sudo apt update")
    run_command("sudo apt install -y redis-server postgresql")

def cleanup_redis():
    print("Cleaning up existing Redis instances...")
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    
    # Stop all Redis services and remove service files
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        service_name = f"redis-{port}.service"
        
        # Stop and disable the service if it exists
        run_command(f"sudo systemctl stop {service_name}")
        run_command(f"sudo systemctl disable {service_name}")
        
        # Remove service file
        if os.path.exists(f"/etc/systemd/system/{service_name}"):
            run_command(f"sudo rm /etc/systemd/system/{service_name}")

        # Remove config file
        config_file = f"/etc/redis/redis-{port}.conf"
        if os.path.exists(config_file):
            run_command(f"sudo rm {config_file}")

        # Clean up data directory
        data_dir = f"/var/lib/redis/{port}"
        if os.path.exists(data_dir):
            run_command(f"sudo rm -rf {data_dir}")

    # Reload systemd to recognize removed services
    run_command("sudo systemctl daemon-reload")

def setup_redis():
    print("Setting up Redis...")
    cleanup_redis()  # Clean up before setup
    
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        
        # Create data directory with proper permissions
        data_dir = f"/var/lib/redis/{port}"
        run_command(f"sudo mkdir -p {data_dir}")
        run_command(f"sudo chown redis:redis {data_dir}")
        run_command(f"sudo chmod 750 {data_dir}")
        
        # Create config file
        config_file = f"/etc/redis/redis-{port}.conf"
        config_content = f"""
        port {port}
        dir {data_dir}
        maxmemory 100mb
        maxmemory-policy allkeys-lru
        
        # Persistence configuration
        save 60 1
        save 30 10
        save 15 10000
        appendonly yes
        appendfilename "appendonly.aof"
        
        # Basic security
        protected-mode yes
        bind 127.0.0.1
        """
        
        with open("temp_redis.conf", "w") as f:
            f.write(config_content)
        run_command(f"sudo mv temp_redis.conf {config_file}")
        run_command(f"sudo chown redis:redis {config_file}")
        run_command(f"sudo chmod 644 {config_file}")
        
        # Create systemd service file
        service_file = f"/etc/systemd/system/redis-{port}.service"
        service_content = f"""[Unit]
Description=Redis In-Memory Data Store ({db_name}) on port {port}
After=network.target

[Service]
Type=notify
ExecStart=/usr/bin/redis-server {config_file} --supervised systemd
ExecStop=/usr/bin/redis-cli -p {port} shutdown
TimeoutStartSec=600
TimeoutStopSec=300
Restart=always
User=redis
Group=redis

[Install]
WantedBy=multi-user.target
"""
        with open("temp_redis.service", "w") as f:
            f.write(service_content)
        run_command(f"sudo mv temp_redis.service {service_file}")
        run_command(f"sudo chmod 644 {service_file}")
        
        # Reload systemd and start service
        run_command("sudo systemctl daemon-reload")
        run_command(f"sudo systemctl enable redis-{port}.service")
        run_command(f"sudo systemctl start redis-{port}.service")
        
        # Wait for service to start
        time.sleep(2)

def verify_redis_services():
    print("Verifying Redis services...")
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    all_services_ok = True
    
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        service_name = f"redis-{port}.service"
        
        # Check if service is enabled
        enabled = run_command(f"sudo systemctl is-enabled {service_name}")
        active = run_command(f"sudo systemctl is-active {service_name}")
        
        if enabled == "enabled" and active == "active":
            print(f"✓ Redis service for {db_name} (port {port}) is running and enabled")
            
            # Verify connection
            try:
                result = run_command(f"redis-cli -p {port} ping")
                if result == "PONG":
                    print(f"  ✓ Successfully connected to Redis on port {port}")
                else:
                    print(f"  ✗ Could not connect to Redis on port {port}")
                    all_services_ok = False
            except Exception as e:
                print(f"  ✗ Error testing Redis connection on port {port}: {str(e)}")
                all_services_ok = False
        else:
            print(f"✗ Redis service for {db_name} (port {port}) is not running properly")
            print(f"  Enabled status: {enabled}")
            print(f"  Active status: {active}")
            all_services_ok = False
    
    return all_services_ok

def database_exists(db_name):
    result = run_command(f"sudo -u postgres -i psql -tAc \"SELECT 1 FROM pg_database WHERE datname='{db_name.lower()}'\"")
    return result == "1"

def user_exists(user_name):
    result = run_command(f"sudo -u postgres -i psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='{user_name}';\"")
    return result == "1"

def drop_existing_postgres_data():
    print("Dropping existing PostgreSQL databases and tables if they exist...")

    # Drop databases if they exist
    databases = ["episodic_memory", "semantic_memory", "contextual_memory", "general_memory", "abstracted_memory"]
    for db in databases:
        if database_exists(db):
            run_command(f"sudo -u postgres -i psql -c 'REVOKE ALL PRIVILEGES ON DATABASE {db} FROM sheppard;'")
            run_command(f"sudo -u postgres -i psql -c 'DROP DATABASE IF EXISTS {db};'")
        else:
            print(f"Database '{db}' does not exist, skipping...")

    # Revoke privileges on all objects before dropping the user
    for db in databases:
        if database_exists(db):
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM sheppard;'")
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM sheppard;'")
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public FROM sheppard;'")

    # Drop the user if it exists
    if user_exists("sheppard"):
        run_command("sudo -u postgres -i psql -c 'DROP USER IF EXISTS sheppard;'")

def setup_postgresql():
    print("Setting up PostgreSQL...")
    drop_existing_postgres_data()  # Call the drop function before creating new data

    run_command("sudo systemctl enable postgresql")
    run_command("sudo systemctl start postgresql")

    # Create databases
    databases = ["episodic_memory", "semantic_memory", "contextual_memory", "general_memory", "abstracted_memory"]
    for db in databases:
        run_command(f"sudo -u postgres -i psql -c 'CREATE DATABASE {db};'")

    # Create user with superuser privileges and password
    if not user_exists("sheppard"):
        run_command("sudo -u postgres -i psql -c \"CREATE USER sheppard WITH SUPERUSER PASSWORD 'llama';\"")
    else:
        print("User 'sheppard' already exists, skipping creation...")

    # Grant all privileges to the user
    for db in databases:
        run_command(f"sudo -u postgres -i psql -c 'GRANT ALL PRIVILEGES ON DATABASE {db} TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO sheppard;'")
        run_command(f"sudo -u postgres -i psql -d {db} -c 'ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TYPES TO sheppard;'")

def create_tables():
    print("Creating PostgreSQL tables...")

    # Create tables in episodic_memory database
    run_command("""
    sudo -u postgres -i psql -d episodic_memory -c "
        CREATE TABLE IF NOT EXISTS agent_interactions (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            agent_id TEXT NOT NULL,
            input TEXT NOT NULL,
            response TEXT NOT NULL,
            entities JSONB,
            topics JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_agent_interactions_agent_id ON agent_interactions (agent_id);
        CREATE INDEX IF NOT EXISTS idx_agent_interactions_timestamp ON agent_interactions (timestamp);
    "
    """)

    # Create tables in semantic_memory database
    run_command("""
    sudo -u postgres -i psql -d semantic_memory -c "
        CREATE TABLE IF NOT EXISTS entity_relationships (
            id SERIAL PRIMARY KEY,
            entity TEXT NOT NULL,
            related_entities JSONB,
            context TEXT,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_entity_relationships_entity ON entity_relationships (entity);
    "
    """)

    # Create tables in contextual_memory database
    run_command("""
    sudo -u postgres -i psql -d contextual_memory -c "
        CREATE TABLE IF NOT EXISTS recent_contexts (
            id SERIAL PRIMARY KEY,
            agent_id TEXT NOT NULL,
            context_data JSONB,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_recent_contexts_agent_id ON recent_contexts (agent_id);
        CREATE INDEX IF NOT EXISTS idx_recent_contexts_timestamp ON recent_contexts (timestamp);
    "
    """)

    # Create tables in abstracted_memory database
    run_command("""
    sudo -u postgres -i psql -d abstracted_memory -c "
        CREATE TABLE IF NOT EXISTS abstracted_memories (
            id SERIAL PRIMARY KEY,
            agent_id TEXT NOT NULL,
            abstraction_type TEXT NOT NULL,
            content JSONB,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_abstracted_memories_agent_id ON abstracted_memories (agent_id);
        CREATE INDEX IF NOT EXISTS idx_abstracted_memories_abstraction_type ON abstracted_memories (abstraction_type);
    "
    """)

    print("All tables created successfully.")

def cleanup_temp_files():
    temp_files = ["temp_redis.conf", "temp_redis.service"]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

def main():
    success = True
    try:
        install_packages()
        setup_redis()
        if not verify_redis_services():
            print("\n⚠️ Warning: Some Redis services are not running properly!")
            success = False
        
        setup_postgresql()
        create_tables()
        
        # Clean up any temporary files
        cleanup_temp_files()
        
        if success:
            print("\n✓ Setup completed successfully!")
            print("\n✓ Redis instances are running on ports 6370-6374 and configured to start on boot.")
            print("✓ PostgreSQL is running on the default port (5432) and configured to start on boot.")
            print("✓ Databases and tables created successfully.")
            print("\n✓ All privileges have been granted to the 'sheppard' user for all databases, tables, indexes, and future objects.")
        else:
            print("\n⚠️ Setup completed with some warnings. Please check the logs above.")
    except Exception as e:
        print(f"\n✗ Error during setup: {str(e)}")
        cleanup_temp_files()  # Ensure cleanup even on error
        sys.exit(1)

if __name__ == "__main__":
    main()
