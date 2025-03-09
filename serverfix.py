#!/usr/bin/env python3

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command):
    """Execute shell command and return output"""
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        output, error = process.communicate()
        return process.returncode, output.decode(), error.decode()
    except Exception as e:
        print(f"Error executing command: {str(e)}")
        return 1, "", str(e)

def print_status(message):
    """Print status message"""
    print(f"\n🔄 {message}")

def print_success(message):
    """Print success message"""
    print(f"✓ {message}")

def print_error(message):
    """Print error message"""
    print(f"✗ {message}")

def main():
    # Check if script is run as root
    if os.geteuid() != 0:
        print_error("Please run as root (use sudo)")
        sys.exit(1)

    print_status("Starting PostgreSQL configuration...")

    # Stop PostgreSQL service
    print_status("Stopping PostgreSQL service...")
    run_command("systemctl stop postgresql")

    # Find PostgreSQL version
    pg_path = Path("/etc/postgresql")
    pg_versions = [x for x in pg_path.iterdir() if x.is_dir()]
    if not pg_versions:
        print_error("PostgreSQL installation not found")
        sys.exit(1)
    
    pg_version = pg_versions[0].name
    pg_hba_conf = f"/etc/postgresql/{pg_version}/main/pg_hba.conf"
    postgresql_conf = f"/etc/postgresql/{pg_version}/main/postgresql.conf"

    # Backup existing configuration
    print_status("Backing up existing configuration...")
    if os.path.exists(pg_hba_conf):
        run_command(f"cp {pg_hba_conf} {pg_hba_conf}.backup")

    # Modify pg_hba.conf
    print_status("Configuring pg_hba.conf...")
    pg_hba_content = """
# PostgreSQL Client Authentication Configuration File
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all            postgres                                peer
local   all            all                                     md5
host    all            all             127.0.0.1/32            md5
host    all            all             ::1/128                 md5
"""
    with open(pg_hba_conf, 'w') as f:
        f.write(pg_hba_content.strip())

    # Set proper permissions
    run_command(f"chown postgres:postgres {pg_hba_conf}")
    run_command(f"chmod 640 {pg_hba_conf}")

    # Modify postgresql.conf
    print_status("Configuring postgresql.conf...")
    run_command(f"sed -i 's/#listen_addresses = \\'localhost\\'/listen_addresses = \\'localhost\\'/g' {postgresql_conf}")

    # Start PostgreSQL service
    print_status("Starting PostgreSQL service...")
    run_command("systemctl start postgresql")
    run_command("systemctl enable postgresql")

    # Wait for PostgreSQL to start
    time.sleep(5)

    # Remove existing databases and user
    print_status("Removing existing databases and user...")
    databases = [
        "episodic_memory",
        "semantic_memory",
        "contextual_memory",
        "general_memory",
        "abstracted_memory"
    ]
    
    for db in databases:
        run_command(f'sudo -u postgres psql -c "DROP DATABASE IF EXISTS {db};"')
    run_command('sudo -u postgres psql -c "DROP USER IF EXISTS sheppard;"')

    # Create user with new password
    print_status("Creating user 'sheppard'...")
    run_command('sudo -u postgres psql -c "CREATE USER sheppard WITH SUPERUSER PASSWORD \'llama\';"')
    run_command('sudo -u postgres psql -c "ALTER USER sheppard WITH SUPERUSER;"')

    # Create databases
    print_status("Creating databases...")
    for db in databases:
        print_status(f"Creating database: {db}")
        commands = [
            f'sudo -u postgres psql -c "CREATE DATABASE {db} WITH OWNER = sheppard;"',
            f'sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE {db} TO sheppard;"',
            f'sudo -u postgres psql -d {db} -c "GRANT ALL ON SCHEMA public TO sheppard;"'
        ]
        for cmd in commands:
            run_command(cmd)

    # Verify user and permissions
    print_status("Verifying configuration...")
    retcode, output, error = run_command('sudo -u postgres psql -c "\\du" | grep sheppard')
    if "sheppard" in output:
        print_success("User 'sheppard' created successfully")
    else:
        print_error("Failed to create user 'sheppard'")
        sys.exit(1)

    # Test connection
    print_status("Testing connection...")
    os.environ['PGPASSWORD'] = 'llama'
    retcode, output, error = run_command('psql -h localhost -U sheppard -d episodic_memory -c "\\l"')
    if retcode == 0:
        print_success("Connection test successful")
    else:
        print_error("Connection test failed")
        print(error)
        sys.exit(1)

    # Restart PostgreSQL
    print_status("Restarting PostgreSQL...")
    run_command("systemctl restart postgresql")

    # Create environment file
    print_status("Creating environment file...")
    env_content = """
DB_USER=sheppard
DB_PASSWORD=llama
DB_HOST=localhost
DB_PORT=5432
"""
    with open('.env', 'w') as f:
        f.write(env_content.strip())

    print_success("PostgreSQL configuration completed successfully!")
    print_status("You can now run the Python application.")
    print_success("Setup complete! The .env file has been created with database credentials.")

if __name__ == "__main__":
    main()
