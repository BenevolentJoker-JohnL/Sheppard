import asyncio
import logging
import os
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.table import Table
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style

from src.core.sheppard import Sheppard
from src.config.config import console, DatabaseConfig
from src.core.memory import StorageManager, MemoryStats

logger = logging.getLogger(__name__)

async def get_user_input(prompt="You: "):
    """Get user input with styled prompt"""
    style = Style.from_dict({
        'prompt': 'cyan bold',
    })
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)

async def cleanup(sheppard=None):
    """Cleanup function to handle shutdown"""
    if sheppard:
        try:
            await sheppard.shutdown()
        except Exception as e:
            logger.error(f"Error during Sheppard shutdown: {str(e)}")

def display_help():
    """Display help information"""
    help_text = """
    Available Commands:
    - 'exit': End the conversation
    - 'save': Save the current conversation
    - 'stats': View memory statistics
    - 'help': Display this help message
    - 'clear': Clear the screen
    """
    console.print(Panel(help_text, title="Help", style="bold blue"))

async def main():
    """Main application loop"""
    sheppard = None
    try:
        # Initialize Sheppard
        sheppard = Sheppard()
        init_success = await sheppard.initialize()
        
        if not init_success:
            console.print(Panel(
                "Failed to initialize Sheppard. Please check the logs.",
                title="Error",
                style="bold red"
            ))
            return

        # Welcome message
        console.print(Panel(
            "Welcome to Sheppard AI Assistant. Type 'help' for available commands.", 
            title="Welcome", 
            style="bold green"
        ))

        # Main conversation loop
        running = True
        while running:
            try:
                user_input = await get_user_input()
                command = user_input.lower().strip()

                if command == 'exit':
                    running = False
                    continue
                    
                elif command == 'help':
                    display_help()
                    continue
                    
                elif command == 'clear':
                    console.clear()
                    continue
                    
                # Process normal input
                response = await sheppard.process_input(user_input)
                
                if response:
                    console.print(Panel(
                        Markdown(response),
                        title="Sheppard's Response",
                        border_style="blue"
                    ))
                    logger.info(f"Successful interaction - Input length: {len(user_input)}, Response length: {len(response)}")
                else:
                    console.print(Panel(
                        "No response generated.",
                        title="Sheppard's Response",
                        border_style="red"
                    ))

            except KeyboardInterrupt:
                console.print("\nUse 'exit' to close the application safely.", style="yellow")
                continue
                
            except Exception as e:
                logger.error(f"Error in conversation loop: {str(e)}", exc_info=True)
                console.print(Panel(
                    f"An error occurred: {str(e)}\nPlease try again.",
                    title="Error",
                    style="bold red"
                ))

    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}", exc_info=True)
        console.print(f"A fatal error occurred: {str(e)}", style="bold red")
    finally:
        await cleanup(sheppard)
        console.print("\nThank you for using Sheppard AI Assistant. Goodbye!", style="bold green")

def setup_logging():
    """Configure logging"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'sheppard_{datetime.now().strftime("%Y%m%d")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

if __name__ == "__main__":
    try:
        # Setup logging
        setup_logging()
        
        # Create and get event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run main application
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        console.print("\nApplication terminated by user.", style="yellow")
    except Exception as e:
        console.print(f"An unexpected error occurred: {str(e)}", style="bold red")
        logger.exception("Unexpected error in main execution")
    finally:
        # Clean up the event loop
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
        except Exception as e:
            logger.error(f"Error closing event loop: {str(e)}")
