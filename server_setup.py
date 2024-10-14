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

def setup_redis(redis_instances):
    print("Setting up Redis...")
    for i, db_name in enumerate(redis_instances):
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

def verify_redis_services(redis_instances):
    print("Verifying Redis services...")
    for i, db_name in enumerate(redis_instances):
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

def drop_existing_postgres_data(user_name, databases):
    print("Dropping existing PostgreSQL databases and tables if they exist...")
    for db in databases:
        if database_exists(db):
            run_command(f"sudo -u postgres -i psql -c 'REVOKE ALL PRIVILEGES ON DATABASE {db} FROM {user_name};'")
            run_command(f"sudo -u postgres -i psql -c 'DROP DATABASE IF EXISTS {db};'")
        else:
            print(f"Database '{db}' does not exist, skipping...")

    # Revoke privileges on all objects before dropping the user if necessary
    for db in databases:
        if database_exists(db):
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM {user_name};'")
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM {user_name};'")
            run_command(f"sudo -u postgres -i psql -d {db} -c 'REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public FROM {user_name};'")

def setup_postgresql(user_name, user_password, databases):
    print("Setting up PostgreSQL...")
    drop_existing_postgres_data(user_name, databases)
    
    run_command("sudo systemctl enable postgresql")
    run_command("sudo systemctl start postgresql")
    
    # Create databases and user
    for db in databases:
        run_command(f"sudo -u postgres -i psql -c 'CREATE DATABASE {db};'")

    # Check if the user already exists before attempting to create it
    user_exists = run_command(f"sudo -u postgres -i psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='{user_name}'\"")
    if user_exists != "1":
        run_command(f"sudo -u postgres -i psql -c \"CREATE USER {user_name} WITH PASSWORD '{user_password}';\"")
    
    run_command(f"sudo -u postgres -i psql -c 'GRANT ALL PRIVILEGES ON DATABASE {', '.join(databases)} TO {user_name};'")

def create_tables():
    print("Creating PostgreSQL tables...")

    run_command("""
    sudo -u postgres -i psql -d interactionhistory -c "
        CREATE TABLE IF NOT EXISTS interactions (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions (user_id);
    "
    """)

    run_command("""
    sudo -u postgres -i psql -d embeddings -c "
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            text TEXT NOT NULL,
            embedding FLOAT[] NOT NULL,
            dimension INTEGER NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON embeddings (user_id);
    "
    """)

    run_command("""
    sudo -u postgres -i psql -d metainfo -c "
        CREATE TABLE IF NOT EXISTS metainfo (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            key VARCHAR(255) NOT NULL,
            value TEXT NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_metainfo_user_id_key ON metainfo (user_id, key);
    "
    """)

    print("All tables created successfully.")

def main():
    redis_instances = ['main', 'episodic', 'subconscious', 'global_workspace', 'stream']
    databases = ['interactionhistory', 'embeddings', 'metainfo']
    user_name = 'user_template'  # Replace with the actual user
    user_password = 'password_template'  # Replace with the actual password

    install_packages()
    setup_redis(redis_instances)
    setup_postgresql(user_name, user_password, databases)
    create_tables()
    verify_redis_services(redis_instances)
    print("\nSetup completed successfully!")
    print("\nRedis instances are running on ports 6370-6374 and configured to start on boot.")
    print("PostgreSQL is running on the default port (5432) and configured to start on boot.")
    print("Databases and tables created successfully.")

if __name__ == "__main__":
    main()
