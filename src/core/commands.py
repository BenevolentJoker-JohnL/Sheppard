"""
Command handling system for chat functionality.
File: src/core/commands.py
"""

import logging
import shlex
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.research.models import ResearchType
from src.research.exceptions import BrowserError
from src.core.constants import (
    WELCOME_TEXT,
    COMMANDS,
    HELP_CATEGORIES,
    ERROR_MESSAGES,
    STYLES
)
from src.core.exceptions import CommandError

logger = logging.getLogger(__name__)

class CommandHandler:
    """Handles chat commands and provides help system."""
    
    def __init__(
        self,
        console: Console,
        chat_app: Any  # Avoid circular import
    ):
        """
        Initialize command handler.
        
        Args:
            console: Rich console for output
            chat_app: Reference to main chat app
        """
        self.console = console
        self.chat_app = chat_app
        self.command_history: List[Dict[str, Any]] = []

    def show_welcome(self) -> None:
        """Display welcome message with system capabilities."""
        self.console.print(Panel(
            Markdown(WELCOME_TEXT),
            title="Welcome to Sheppard Agency",
            border_style=STYLES['title']
        ))

    async def handle_command(self, input_text: str) -> bool:
        """
        Process a potential command input.
        
        Args:
            input_text: User input text
            
        Returns:
            bool: Whether input was handled as a command
        """
        if not input_text.startswith('/'):
            return False
            
        try:
            # Parse command and arguments
            parts = shlex.split(input_text)
            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []
            
            # Check if command exists
            if command not in COMMANDS:
                self.console.print(
                    ERROR_MESSAGES['command_not_found'],
                    style=STYLES['error']
                )
                return True
            
            # Record command in history
            self.command_history.append({
                'command': command,
                'args': args,
                'timestamp': datetime.now().isoformat()
            })
            
            # Handle command
            await self._dispatch_command(command, args)
            return True
            
        except Exception as e:
            logger.error(f"Command handling error: {str(e)}")
            self.console.print(
                f"Error executing command: {str(e)}",
                style=STYLES['error']
            )
            return True

    async def _handle_help(self, *args) -> None:
        """Handle help command."""
        if args:
            # Show help for specific command
            command = f"/{args[0]}" if not args[0].startswith('/') else args[0]
            if command not in COMMANDS:
                self.console.print(
                    ERROR_MESSAGES['command_not_found'],
                    style=STYLES['error']
                )
                return

            cmd_info = COMMANDS[command]
            help_text = f"""# {command}

## Description
{cmd_info['description']}

## Usage
`{cmd_info['usage']}`

## Examples
{chr(10).join(f"- `{example}`" for example in cmd_info['examples'])}

Category: {HELP_CATEGORIES[cmd_info['category']]}
"""
            self.console.print(Panel(
                Markdown(help_text),
                title=f"Help: {command}",
                border_style=STYLES['info']
            ))
            return

        # Show all commands grouped by category
        table = Table(title="Available Commands")
        table.add_column("Command", style=STYLES['command'])
        table.add_column("Description")
        table.add_column("Category", style=STYLES['info'])
        
        # Group commands by category
        categorized = {}
        for cmd, info in COMMANDS.items():
            category = info['category']
            if category not in categorized:
                categorized[category] = []
            categorized[category].append((cmd, info))
        
        # Add commands to table by category
        for category in sorted(categorized.keys()):
            for cmd, info in sorted(categorized[category]):
                table.add_row(
                    cmd,
                    info['description'],
                    HELP_CATEGORIES[category]
                )

        self.console.print(table)

    async def _handle_research(
        self,
        *args,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle research command."""
        if not args:
            self.console.print(
                ERROR_MESSAGES['invalid_usage'].format(command='research'),
                style=STYLES['error']
            )
            return

        # Get topic by joining all non-option args
        topic_args = [arg for arg in args if not arg.startswith('--')]
        topic = ' '.join(topic_args)

        # Parse options
        depth = 3  # Default depth 
        headless = False  # Default to visible browser

        # Parse optional args
        for arg in args:
            if arg.startswith('--depth='):
                try:
                    depth = int(arg.split('=')[1]) 
                    depth = max(1, min(depth, 5))  # Clamp between 1-5
                except ValueError:
                    self.console.print(
                        "Invalid depth value. Using default.",
                        style=STYLES['warning']
                    )
            elif arg == '--headless':
                headless = True

        if not topic:
            self.console.print(
                "Please provide a research topic.", 
                style=STYLES['error']
            )
            return

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(
                description=f"[cyan]Researching: {topic}",
                total=100
            )

            try:
                # Perform research - REMOVED headless parameter
                results = await self.chat_app.research_system.research_topic(
                    topic=topic,
                    research_type=ResearchType.WEB_SEARCH,
                    depth=depth,
                    progress_callback=lambda p: progress.update(task, completed=p * 100)
                )

                if results.get('findings'):
                    # Import the formatter function
                    from src.research.processors import format_research_results
                    
                    # Format the findings into a markdown string
                    formatted_findings = format_research_results(results)
                    
                    self.console.print(Panel(
                        Markdown(formatted_findings),
                        title=f"Research Results: {topic}",
                        border_style=STYLES['success']
                    ))
                else:
                    self.console.print(
                        "No significant findings were discovered.",
                        style=STYLES['warning']
                    )

            except Exception as e:
                self.console.print(
                    ERROR_MESSAGES['research_failed'].format(error=str(e)),
                    style=STYLES['error']
                )
    async def _handle_memory(self, *args) -> None:
        """Handle memory command."""
        if not args:
            self.console.print(
                ERROR_MESSAGES['invalid_usage'].format(command='memory'),
                style=STYLES['error']
            )
            return

        action = args[0]
        if action == 'search':
            if len(args) < 2:
                self.console.print("Please provide a search query.", style=STYLES['error'])
                return

            query = ' '.join(args[1:])
            results = await self.chat_app.memory_manager.search(query)
            
            if results:
                table = Table(title=f"Memory Search Results: {query}")
                table.add_column("Content")
                table.add_column("Relevance")
                table.add_column("Date")

                for result in results:
                    table.add_row(
                        result.content[:100] + "..." if len(result.content) > 100 else result.content,
                        f"{result.metadata.get('relevance_score', 0):.2f}",
                        result.metadata.get('timestamp', 'Unknown')
                    )

                self.console.print(table)
            else:
                self.console.print("No matching memories found.", style=STYLES['warning'])

        elif action == 'clear':
            confirm = '--confirm' in args
            if not confirm:
                self.console.print(
                    "Are you sure? Use '/memory clear --confirm' to confirm.",
                    style=STYLES['warning']
                )
                return

            await self.chat_app.memory_manager.cleanup()
            self.console.print("Memory cleared successfully.", style=STYLES['success'])

    async def _handle_status(self, *args) -> None:
        """Handle status command."""
        try:
            status = await self.chat_app.get_system_status()
            
            table = Table(title="System Status")
            table.add_column("Component")
            table.add_column("Status")
            table.add_column("Details")

            # System status
            table.add_row(
                "System",
                "✓" if status['system']['initialized'] else "✗",
                f"GPU: {'Enabled' if status['system']['gpu_enabled'] else 'Disabled'}"
            )

            # Models status
            table.add_row(
                "Models",
                "Active",
                f"Chat: {status['models']['chat_model']}\nEmbed: {status['models']['embed_model']}"
            )

            # Memory status
            table.add_row(
                "Memory",
                "Active",
                f"Context tokens: {status['memory']['max_context_tokens']}\nPreferences: {status['memory']['preferences_count']}"
            )

            # Research status
            table.add_row(
                "Research",
                "✓" if status['research']['browser_available'] else "✗",
                f"Timeout: {status['research']['browser_timeout']}s"
            )

            self.console.print(table)

        except Exception as e:
            self.console.print(
                f"Error getting system status: {str(e)}",
                style=STYLES['error']
            )

    async def _handle_settings(self, *args) -> None:
        """Handle settings command."""
        try:
            if not args:
                # Show all settings
                settings = await self.chat_app.get_settings()
                
                table = Table(title="Current Settings")
                table.add_column("Setting")
                table.add_column("Value")
                table.add_column("Description")

                for key, value in settings.items():
                    description = self.chat_app.get_setting_description(key)
                    table.add_row(
                        key,
                        str(value),
                        description or ""
                    )

                self.console.print(table)
                return

            # Update setting
            setting = args[0]
            if len(args) < 2:
                # Show specific setting
                value = await self.chat_app.get_setting(setting)
                if value is not None:
                    description = self.chat_app.get_setting_description(setting)
                    self.console.print(Panel(
                        f"{setting} = {value}\n\n{description}",
                        title=f"Setting: {setting}",
                        border_style=STYLES['info']
                    ))
                else:
                    self.console.print(f"Unknown setting: {setting}", style=STYLES['error'])
                return

            # Update setting
            value = args[1]
            await self.chat_app.update_setting(setting, value)
            self.console.print(
                f"Updated {setting} to {value}",
                style=STYLES['success']
            )

        except Exception as e:
            self.console.print(
                f"Error handling settings: {str(e)}",
                style=STYLES['error']
            )

    async def _handle_clear(self, *args) -> None:
        """Handle clear command."""
        confirm = '--confirm' in args
        if not confirm:
            self.console.print(
                "Are you sure? Use '/clear --confirm' to confirm.",
                style=STYLES['warning']
            )
            return

        self.console.clear()
        self.show_welcome()
        self.console.print("Chat history cleared.", style=STYLES['success'])

    async def _handle_save(self, *args) -> None:
        """Handle save command."""
        if len(args) < 2:
            self.console.print(
                ERROR_MESSAGES['invalid_usage'].format(command='save'),
                style=STYLES['error']
            )
            return

        save_type = args[0]
        filename = args[1]

        try:
            if save_type == 'chat':
                filepath = await self.chat_app.save_chat_history(filename)
                self.console.print(
                    f"Chat history saved to: {filepath}",
                    style=STYLES['success']
                )
            elif save_type == 'research':
                # Implement research saving logic
                self.console.print(
                    "Research saving not yet implemented.",
                    style=STYLES['warning']
                )
            else:
                self.console.print(
                    f"Unknown save type: {save_type}",
                    style=STYLES['error']
                )

        except Exception as e:
            self.console.print(
                f"Error saving {save_type}: {str(e)}",
                style=STYLES['error']
            )

    async def _handle_browse(self, *args) -> None:
        """Handle browse command."""
        if not args:
            self.console.print(
                ERROR_MESSAGES['invalid_usage'].format(command='browse'),
                style=STYLES['error']
            )
            return

        url = args[0]
        headless = '--headless' in args

        try:
            content = await self.chat_app.research_system.browse_url(
                url=url,
                headless=headless
            )

            if content:
                self.console.print(Panel(
                    content[:500] + "..." if len(content) > 500 else content,
                    title=f"Content from {url}",
                    border_style=STYLES['success']
                ))
            else:
                self.console.print(
                    "No content extracted from URL.",
                    style=STYLES['warning']
                )

        except Exception as e:
            self.console.print(
                ERROR_MESSAGES['browser_error'].format(error=str(e)),
                style=STYLES['error']
            )
    
    async def _handle_preferences(self, *args) -> None:
        """Handle preferences command."""
        if not args:
            # Show all preferences
            prefs = await self.chat_app.get_all_preferences()
            
            table = Table(title="User Preferences")
            table.add_column("Preference")
            table.add_column("Value")
            table.add_column("Type")
            table.add_column("Last Updated")

            for pref in prefs:
                table.add_row(
                    pref['key'],
                    str(pref['value']),
                    pref['type'],
                    pref['timestamp']
                )

            self.console.print(table)
            return

        action = args[0]

        if action == 'list':
            # List available preference types
            category = args[1] if len(args) > 1 else None
            prefs = await self.chat_app.list_preferences(category)

            table = Table(title="Available Preferences")
            table.add_column("Key")
            table.add_column("Type")
            table.add_column("Description")

            for pref in prefs:
                table.add_row(
                    pref['key'],
                    pref['type'],
                    pref['description']
                )

            self.console.print(table)

        elif action == 'set':
            if len(args) < 3:
                self.console.print(
                    ERROR_MESSAGES['invalid_usage'].format(command='preferences set'),
                    style=STYLES['error']
                )
                return

            key = args[1]
            value = args[2]

            try:
                await self.chat_app.set_preference(key, value)
                self.console.print(
                    f"Preference {key} set to: {value}",
                    style=STYLES['success']
                )
            except Exception as e:
                self.console.print(
                    f"Error setting preference: {str(e)}",
                    style=STYLES['error']
                )

        elif action == 'clear':
            confirm = '--confirm' in args
            if not confirm:
                self.console.print(
                    "Are you sure? Use '/preferences clear --confirm' to confirm.",
                    style=STYLES['warning']
                )
                return

            await self.chat_app.clear_preferences()
            self.console.print(
                "All preferences cleared.",
                style=STYLES['success']
            )

        else:
            self.console.print(
                f"Unknown preference action: {action}",
                style=STYLES['error']
            )

    async def _dispatch_command(self, command: str, args: List[str]) -> None:
        """
        Dispatch command to appropriate handler.
        
        Args:
            command: Command to dispatch
            args: Command arguments
        """
        handlers = {
            '/help': self._handle_help,
            '/research': self._handle_research,
            '/memory': self._handle_memory,
            '/status': self._handle_status,
            '/settings': self._handle_settings,
            '/clear': self._handle_clear,
            '/save': self._handle_save,
            '/browse': self._handle_browse,
            '/preferences': self._handle_preferences
        }
        
        handler = handlers.get(command)
        if handler:
            await handler(*args)
        else:
            raise CommandError(f"No handler for command: {command}")

    async def verify_handler_integrity(self) -> Dict[str, bool]:
        """
        Verify integrity of command handler.
        
        Returns:
            Dict[str, bool]: Status of each handler
        """
        integrity = {
            'console': self.console is not None,
            'chat_app': self.chat_app is not None,
            'handlers': True
        }
        
        # Verify all command handlers exist
        for command in COMMANDS:
            handler_name = f"_handle_{command[1:]}"  # Remove leading '/'
            if not hasattr(self, handler_name):
                integrity['handlers'] = False
                logger.error(f"Missing handler for command: {command}")
        
        return integrity
