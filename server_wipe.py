# server_wipe.py
import subprocess
import sys
import os
import shutil

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(error.decode())
        return False
    return output.decode().strip()

def cleanup_redis():
    print("\n🧹 Cleaning up Redis instances...")
    redis_dbs = ['ephemeral', 'contextual', 'episodic', 'semantic', 'abstracted']
    
    for i, db_name in enumerate(redis_dbs):
        port = 6370 + i
        service_name = f"redis-{port}.service"
        print(f"\nCleaning Redis instance: {db_name} (port {port})")
        
        # Stop and disable the service if it exists
        print(f"  ◦ Stopping service {service_name}...")
        run_command(f"sudo systemctl stop {service_name}")
        run_command(f"sudo systemctl disable {service_name}")
        
        # Remove service file
        service_path = f"/etc/systemd/system/{service_name}"
        if os.path.exists(service_path):
            print(f"  ◦ Removing service file {service_path}")
            run_command(f"sudo rm {service_path}")

        # Remove config file
        config_file = f"/etc/redis/redis-{port}.conf"
        if os.path.exists(config_file):
            print(f"  ◦ Removing config file {config_file}")
            run_command(f"sudo rm {config_file}")

        # Clean up data directory
        data_dir = f"/var/lib/redis/{port}"
        if os.path.exists(data_dir):
            print(f"  ◦ Removing data directory {data_dir}")
            run_command(f"sudo rm -rf {data_dir}")

    # Reload systemd to recognize removed services
    print("\n  ◦ Reloading systemd daemon...")
    run_command("sudo systemctl daemon-reload")
    print("✓ Redis cleanup completed")

def cleanup_chromadb():
    print("\n🧹 Cleaning up ChromaDB...")
    chroma_paths = [
        os.path.expanduser("~/.cache/chroma"),
        "/path/to/persist",  # Default path from your setup
        "chroma.sqlite3",
        "data/chroma_persistence"  # Added this path
    ]
    
    for path in chroma_paths:
        if os.path.exists(path):
            print(f"  ◦ Removing ChromaDB data: {path}")
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
    print("✓ ChromaDB cleanup completed")

    # Also remove any leftover index files
    print("  ◦ Cleaning up any remaining ChromaDB files...")
    run_command("find . -name '*.bin' -type f -delete")
    run_command("find . -name '*.pkl' -type f -delete")
    run_command("find . -name 'chroma-*' -type d -exec rm -rf {} +")

def database_exists(db_name):
    result = run_command(f"sudo -u postgres -i psql -tAc \"SELECT 1 FROM pg_database WHERE datname='{db_name.lower()}'\"")
    return result == "1"

def cleanup_postgresql():
    print("\n🧹 Cleaning up PostgreSQL...")
    databases = ["episodic_memory", "semantic_memory", "contextual_memory", "general_memory", "abstracted_memory"]
    
    # Drop databases but preserve user and permissions
    for db in databases:
        if database_exists(db):
            print(f"\nCleaning database: {db}")
            
            # Forcefully disconnect all users
            run_command(f"""sudo -u postgres -i psql -c "
                SELECT pg_terminate_backend(pid) 
                FROM pg_stat_activity 
                WHERE datname = '{db}' AND pid <> pg_backend_pid();"
            """)
            
            # Drop the database
            print(f"  ◦ Dropping database {db}...")
            run_command(f"sudo -u postgres -i psql -c 'DROP DATABASE IF EXISTS {db};'")
    
    print("✓ PostgreSQL cleanup completed")

def main():
    try:
        print("🚀 Starting database cleanup process...")
        
        # Clean Redis
        cleanup_redis()
        
        # Clean ChromaDB
        cleanup_chromadb()
        
        # Clean PostgreSQL
        cleanup_postgresql()
        
        print("\n✨ All cleanup operations completed successfully!")
        print("\nThe following has been cleaned:")
        print("  • Redis instances (ports 6370-6374)")
        print("  • ChromaDB data and cache")
        print("  • PostgreSQL databases (preserving user and permissions)")
        print("\n⚠️  Note: You may need to run the setup script again to recreate the databases.")
        
    except Exception as e:
        print(f"\n❌ Error during cleanup: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
