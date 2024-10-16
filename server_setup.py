#!/usr/bin/env python3

import subprocess
import sys

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        sys.exit(1)
    return output.decode().strip()

def install_packages():
    print("Installing necessary packages...")
    run_command("sudo apt update")
    run_command("sudo apt install -y redis-server postgresql")

def setup_redis():
    print("Setting up Redis...")
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        config_file = f"/etc/redis/redis-{port}.conf"
        run_command(f"sudo cp /etc/redis/redis.conf {config_file}")
        run_command(f"sudo sed -i 's/^port 6379$/port {port}/' {config_file}")
        run_command(f"sudo sed -i 's/^# maxmemory <bytes>$/maxmemory 100mb/' {config_file}")
        run_command(f"sudo sed -i 's/^# maxmemory-policy noeviction$/maxmemory-policy allkeys-lru/' {config_file}")
        
        # Configure persistence
        run_command(f"sudo sed -i 's/^save 900 1$/save 60 1/' {config_file}")
        run_command(f"sudo sed -i 's/^save 300 10$/save 30 10/' {config_file}")
        run_command(f"sudo sed -i 's/^save 60 10000$/save 15 10000/' {config_file}")
        run_command(f"sudo sed -i 's/^appendonly no$/appendonly yes/' {config_file}")
        
        # Create systemd service file
        service_file = f"/etc/systemd/system/redis-{port}.service"
        service_content = f"""[Unit]
Description=Redis In-Memory Data Store ({db_name}) on port {port}
After=network.target

[Service]
ExecStart=/usr/bin/redis-server {config_file}
ExecStop=/usr/bin/redis-cli -p {port} shutdown
Restart=always
User=redis
Group=redis

[Install]
WantedBy=multi-user.target
"""
        run_command(f"echo '{service_content}' | sudo tee {service_file} > /dev/null")
        run_command(f"sudo systemctl enable redis-{port}.service")
        run_command(f"sudo systemctl start redis-{port}.service")

def verify_redis_services():
    print("Verifying Redis services...")
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        service_name = f"redis-{port}.service"
        
        enabled_status = run_command(f"sudo systemctl is-enabled {service_name}")
        if enabled_status == "enabled":
            print(f"Redis service for {db_name} (port {port}) is enabled and will start on boot.")
        else:
            print(f"Warning: Redis service for {db_name} (port {port}) is not enabled for boot.")
        
        active_status = run_command(f"sudo systemctl is-active {service_name}")
        if active_status == "active":
            print(f"Redis service for {db_name} (port {port}) is currently running.")
        else:
            print(f"Warning: Redis service for {db_name} (port {port}) is not currently running.")

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

def setup_postgresql():
    print("Setting up PostgreSQL...")
    drop_existing_postgres_data()  # Call the drop function before creating new data

    run_command("sudo systemctl enable postgresql")
    run_command("sudo systemctl start postgresql")

    # Create databases
    databases = ["episodic_memory", "semantic_memory", "contextual_memory", "general_memory", "abstracted_memory"]
    for db in databases:
        run_command(f"sudo -u postgres -i psql -c 'CREATE DATABASE {db};'")

    # Check if user exists before creating
    if not user_exists("sheppard"):
        run_command("sudo -u postgres -i psql -c \"CREATE USER sheppard WITH SUPERUSER PASSWORD '0000';\"")
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

def main():
    install_packages()
    setup_redis()
    setup_postgresql()
    create_tables()
    verify_redis_services()
    print("\nSetup completed successfully!")
    print("\nRedis instances are running on ports 6370-6374 and configured to start on boot.")
    print("PostgreSQL is running on the default port (5432) and configured to start on boot.")
    print("Databases and tables created successfully.")
    print("\nAll privileges have been granted to the 'sheppard' user for all databases, tables, indexes, and future objects.")

if __name__ == "__main__":
    main()
