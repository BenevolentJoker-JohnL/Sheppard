#!/usr/bin/env python3
"""
Main entry point for the Chat System application.
Handles initialization and running of the chat system with proper cleanup.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess
import nest_asyncio
from dataclasses import dataclass, field
import json
import traceback
import re
from typing import Dict, Any, List, Optional, Tuple, Union, Set  # Add proper typing imports
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.panel import Panel

# Local imports
from src.config.logging import setup_logging
from src.config.settings import settings
from src.core.chat import ChatApp
from src.core.commands import CommandHandler
from src.core.exceptions import InitializationError
from src.llm.client import OllamaClient
from src.memory.manager import MemoryManager
from src.memory.models import Memory
from src.research.models import (
    ResearchTask,
    TaskStatus,
    ResearchType,
    ResearchFinding,
    ValidationLevel
)
from src.research.config import (
    ResearchConfig,
    BrowserConfig,
    NavigationConfig,
    ScrapingConfig,
    FirecrawlConfig,
    ContentProcessingConfig
)
from src.research.system import ResearchSystem
from src.research.content_processor import ContentProcessor
from src.research.firecrawl_config import FirecrawlConfig
from src.utils.console import console

# Disable WebDriver Manager logging
os.environ['WDM_LOG'] = str(logging.NOTSET)
os.environ['WDM_LOG_LEVEL'] = '0'

# Configure custom log formatter
class NoMarkupFormatter(logging.Formatter):
    """Custom formatter to handle rich markup in log messages."""
    def format(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.replace("[", "\\[").replace("]", "\\]")
        return super().format(record)

# Configure rich handler with proper error handling
rich_handler = RichHandler(
    console=console,
    rich_tracebacks=True,
    markup=False,
    show_time=True,
    show_level=True,
    enable_link_path=True
)
rich_handler.setFormatter(NoMarkupFormatter("%(message)s"))

# Configure logging with proper error handling
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich_handler]
)

# Install rich traceback handler with proper configuration
install_rich_traceback(console=console, show_locals=True, width=None, extra_lines=3)

logger = logging.getLogger(__name__)

def create_directory_structure(base_dir: Path) -> Dict[str, Path]:
    """Create necessary directory structure with proper error handling."""
    directories = {
        'data': base_dir / 'data',
        'logs': base_dir / 'logs',
        'screenshots': base_dir / 'screenshots',
        'temp': base_dir / 'temp',
        'chroma': base_dir / 'chroma_storage',
        'downloads': base_dir / 'data' / 'downloads',
        'research_results': base_dir / 'data' / 'research_results',
        'conversations': base_dir / 'data' / 'conversations'
    }

    created_dirs = {}
    for name, path in directories.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
            created_dirs[name] = path
        except Exception as e:
            logger.error(f"Failed to create directory {name} at {path}: {str(e)}")
            raise InitializationError(f"Failed to create directory: {name}")

    return created_dirs

def get_chrome_version() -> Optional[str]:
    """Get Chrome version with proper error handling."""
    try:
        result = subprocess.run(
            ["google-chrome", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip().split()[-1]
    except FileNotFoundError:
        logger.warning("Google Chrome is not installed or not in PATH")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get Chrome version: {e.stderr}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting Chrome version: {str(e)}")
        return None

async def initialize_firecrawl() -> Optional[FirecrawlConfig]:
    """Initialize Firecrawl with proper error handling."""
    try:
        api_key = os.getenv('FIRECRAWL_API_KEY')
        if not api_key:
            logger.warning("FIRECRAWL_API_KEY not found in environment")
            return None

        # Create proper FirecrawlConfig instance
        firecrawl_config = FirecrawlConfig(
            api_key=api_key,
            base_url="https://api.firecrawl.dev",
            version="v1",
            formats=["markdown"],
            max_pages=100,
            timeout=300,
            poll_interval=30,
            exclude_paths=[],
            request_headers={
                "User-Agent": "Sheppard-Research-Bot/1.0",
                "Accept": "application/json"
            },
            scrape_options={
                "waitUntil": "networkidle0",
                "timeout": 30000,
                "removeScripts": True,
                "removeStyles": True,
                "removeTracking": True
            },
            retries=3,
            retry_delay=1.0,
            concurrent_limit=5
        )

        return firecrawl_config

    except Exception as e:
        logger.error(f"Failed to initialize Firecrawl config: {str(e)}")
        return None

async def verify_system_requirements() -> Tuple[bool, Optional[str]]:
    """Verify system requirements with proper error handling."""
    try:
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            return False, "Python 3.8 or higher is required"

        # Check Chrome installation
        chrome_version = get_chrome_version()
        if not chrome_version:
            return False, "Google Chrome is required but not found"

        # Check for required environment variables
        required_env_vars = ['OLLAMA_API_HOST', 'OLLAMA_API_PORT']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            return False, f"Missing required environment variables: {', '.join(missing_vars)}"

        return True, None

    except Exception as e:
        logger.error(f"System requirements verification failed: {str(e)}")
        return False, f"System verification failed: {str(e)}"

async def initialize_components(base_dir: Path) -> Tuple[Optional[ChatApp], Optional[str]]:
    """Initialize system components with proper error handling."""
    try:
        # Verify system requirements
        requirements_met, error = await verify_system_requirements()
        if not requirements_met:
            return None, error

        # Create directory structure
        try:
            directories = create_directory_structure(base_dir)
        except Exception as e:
            return None, f"Failed to create directory structure: {str(e)}"

        # Set up logging
        log_path = directories['logs'] / f"chat_{datetime.now():%Y%m%d}.log"
        setup_logging(level=settings.get('LOG_LEVEL', 'INFO'), log_file=str(log_path))

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing system components...", total=100)

            try:
                # Initialize Ollama client (10%)
                ollama_client = OllamaClient(
                    model_name=settings.OLLAMA_MODEL,
                    api_base=settings.ollama_api_base
                )
                if not ollama_client:
                    raise InitializationError("Failed to initialize Ollama client")
                progress.update(task, advance=10)

                # Initialize memory manager (20%)
                memory_manager = MemoryManager()
                if not memory_manager:
                    raise InitializationError("Failed to initialize memory manager")
                memory_manager.set_ollama_client(ollama_client)
                await memory_manager.initialize()
                progress.update(task, advance=20)

                # Initialize Firecrawl config (10%)
                firecrawl_config = await initialize_firecrawl()
                progress.update(task, advance=10)

                # Initialize research config (20%)
                browser_config = BrowserConfig(
                    headless=True,
                    window_size=(1920, 1080),
                    screenshot_dir=directories['screenshots'],
                    download_dir=directories['downloads'],
                    timeout=30,
                    retry_attempts=3,
                    retry_delay=1.0
                )

                research_config = ResearchConfig(
                    browser=browser_config,
                    navigation=NavigationConfig(max_depth=3),
                    scraping=ScrapingConfig(user_agent="Sheppard-Research-Bot/1.0"),
                    content=ContentProcessingConfig(),
                    firecrawl=firecrawl_config,
                    max_concurrent_tasks=5,
                    task_timeout=600,
                    embed_findings=True,
                    save_results=True,
                    results_dir=directories['research_results'],
                    log_level="INFO"
                )
                progress.update(task, advance=20)

                # Initialize content processor (10%)
                content_processor = ContentProcessor(
                    ollama_client=ollama_client,
                    firecrawl_config=firecrawl_config,
                    chunk_size=settings.get('CHUNK_SIZE', 1000),
                    chunk_overlap=settings.get('CHUNK_OVERLAP', 100)
                )
                progress.update(task, advance=10)

                # Initialize research system (20%)
                research_system = ResearchSystem(
                    memory_manager=memory_manager,
                    ollama_client=ollama_client,
                    config=research_config
                )

                # If Firecrawl is configured, set it in the content processor
                if firecrawl_config and research_system.firecrawl:
                    content_processor.set_firecrawl_client(research_system.firecrawl)

                await research_system.initialize()
                progress.update(task, advance=20)

                # Initialize chat app (10%)
                # Create ChatApp instance first, then initialize with the systems
                chat_app = ChatApp()
                await chat_app.initialize(
                    memory_system=memory_manager,
                    research_system=research_system,
                    llm_system=ollama_client
                )
                progress.update(task, advance=10)

                # Final completion
                progress.update(task, completed=100)
                console.print("[green]System initialization completed successfully[/green]")

                return chat_app, None

            except Exception as e:
                error_msg = f"Component initialization failed: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                return None, error_msg

    except Exception as e:
        error_msg = f"System initialization failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, error_msg

async def cleanup_resources() -> None:
    """Clean up async resources with proper error handling."""
    try:
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete with timeout
        if tasks:
            await asyncio.wait(tasks, timeout=5.0)
            
        # Force cancel any remaining tasks
        remaining = [t for t in tasks if not t.done()]
        for task in remaining:
            try:
                task.cancel()
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"Error cancelling task: {str(e)}")

    except Exception as e:
        logger.error(f"Error during resource cleanup: {str(e)}")

async def run_chat(chat_app: ChatApp) -> None:
    """Run the main chat loop with proper error handling."""
    command_handler = CommandHandler(console=console, chat_app=chat_app)
    command_handler.show_welcome()

    while True:
        try:
            # Display user prompt
            console.print("\nYou: ", end="", style="green")
            user_input = input().strip()

            # Handle exit commands
            if user_input.lower() in {"exit", "quit", "bye"}:
                break

            # Handle commands
            if user_input.startswith("/"):
                await command_handler.handle_command(user_input)
                continue

            # Display user message
            console.print(
                Panel(
                    user_input,
                    title="User",
                    border_style="green"
                )
            )
            
            # Extract and store any preferences from user input
            try:
                await chat_app._extract_and_store_preferences(user_input)
            except Exception as e:
                logger.warning(f"Error extracting preferences: {str(e)}")

            # Process chat input with response buffering
            response_buffer = []
            metadata_buffer = None

            try:
                async for response in chat_app.process_input(user_input):
                    if response and response.content:
                        response_buffer.append(response.content)
                        if not metadata_buffer and response.metadata:
                            metadata_buffer = response.metadata

                # Display complete buffered response
                if response_buffer:
                    complete_response = "".join(response_buffer)
                    console.print(
                        Panel(
                            complete_response,
                            title="Sheppard",
                            border_style="blue"
                        )
                    )
            except Exception as e:
                logger.error(f"Error processing chat input: {str(e)}")
                console.print(
                    Panel(
                        f"An error occurred: {str(e)}",
                        title="Error",
                        border_style="red"
                    )
                )

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            break
        except EOFError:
            logger.info("Received EOF")
            break
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            console.print(
                Panel(
                    f"An error occurred: {str(e)}",
                    title="Error",
                    border_style="red"
                )
            )

async def async_main() -> int:
    """Main application entry point with proper error handling."""
    try:
        # Apply nest_asyncio to support asyncio in Jupyter notebooks
        nest_asyncio.apply()

        # Get base directory
        base_dir = Path(__file__).parent

        # Initialize components
        chat_app, error = await initialize_components(base_dir)
        if error:
            console.print(
                Panel(
                    f"Initialization Error: {error}",
                    title="Error",
                    border_style="red"
                )
            )
            return 1

        # Run chat loop
        await run_chat(chat_app)

        # Clean up resources
        await cleanup_resources()
        return 0

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        await cleanup_resources()
        return 0
    except Exception as e:
        logger.critical("Fatal error: %s", str(e))
        console.print(
            Panel(
                f"Fatal error: {str(e)}\n{traceback.format_exc()}",
                title="Fatal Error",
                border_style="red"
            )
        )
        await cleanup_resources()
        return 1

def main() -> int:
    """Synchronous wrapper for async_main with proper error handling."""
    try:
        # Configure console for better error display
        console.record = True
        
        # Run the async main function
        return asyncio.run(async_main())
    
    except KeyboardInterrupt:
        console.print("\nShutting down gracefully...")
        return 0
    except asyncio.CancelledError:
        console.print("\nAsync operation cancelled, shutting down...")
        return 0
    except Exception as e:
        logger.critical("Fatal error: %s", str(e))
        console.print(
            Panel(
                f"Fatal error: {str(e)}\n{traceback.format_exc()}",
                title="Fatal Error",
                border_style="red"
            )
        )
        return 1
    finally:
        # Ensure console is reset
        console.record = False

if __name__ == "__main__":
    sys.exit(main())
