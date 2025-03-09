"""
Enhanced constants for welcome messages, system text, and configuration.
File: src/core/constants.py
"""

# System Operation Constants
MAX_RETRY_ATTEMPTS = 3
REQUEST_TIMEOUT = 30  # seconds
MAX_CONCURRENT_TASKS = 5

WELCOME_TEXT = """# 🤖 Welcome to Sheppard AI

## Your Personal AI Research Assistant

Sheppard AI combines advanced memory systems with research capabilities 
to help you find, analyze, and remember information.

### Key Features:

- 🔍 **Intelligent Web Research** - Browse and analyze online content
- 💾 **Persistent Memory** - Remember past conversations and preferences
- 🧠 **Context Awareness** - Build on previous interactions naturally
- 🔄 **Powerful Commands** - Access system features with simple commands
- 🎨 **Preference Learning** - Adapts to your communication style

### Getting Started:
- Type `/help` for available commands
- Start research with `/research topic`
- Search your conversation history with `/memory search query`
- Configure with `/settings`

### Pro Tips:
- All commands start with `/`
- Use natural language for best results
- Share your preferences to get more personalized responses
"""

COMMANDS = {
    '/help': {
        'description': 'Show available commands and usage information',
        'usage': '/help [command]',
        'examples': [
            '/help',
            '/help research'
        ],
        'category': 'general'
    },

    '/research': {
        'description': 'Start a research task on a topic with live web browsing',
        'usage': '/research <topic> [--depth=<1-5>] [--headless]',
        'examples': [
            '/research "current AI developments"',
            '/research "climate change" --depth=3'
        ],
        'category': 'research'
    },
    '/status': {
        'description': 'Show current system status and active components',
        'usage': '/status [component]',
        'examples': [
            '/status',
            '/status research'
        ],
        'category': 'system'
    },
    '/clear': {
        'description': 'Clear chat history and start fresh',
        'usage': '/clear [--confirm]',
        'examples': [
            '/clear',
            '/clear --confirm'
        ],
        'category': 'general'
    },
    '/memory': {
        'description': 'Search or manage chat memory and context',
        'usage': '/memory <action> [options]',
        'examples': [
            '/memory search "python code"',
            '/memory clear --all'
        ],
        'category': 'memory'
    },
    '/settings': {
        'description': 'View or modify chat settings',
        'usage': '/settings [setting] [value]',
        'examples': [
            '/settings',
            '/settings temperature 0.7'
        ],
        'category': 'system'
    },
    '/browse': {
        'description': 'Open browser to research specific URL',
        'usage': '/browse <url> [--headless]',
        'examples': [
            '/browse https://example.com',
            '/browse https://example.com --headless'
        ],
        'category': 'research'
    },
    '/save': {
        'description': 'Save current research or conversation',
        'usage': '/save [type] [filename]',
        'examples': [
            '/save research my_findings',
            '/save chat chat_log.txt'
        ],
        'category': 'general'
    },
    '/preferences': {
        'description': 'View or update user preferences',
        'usage': '/preferences [action] [key] [value]',
        'examples': [
            '/preferences list',
            '/preferences set theme dark'
        ],
        'category': 'system'
    },
    '/exit': {
        'description': 'Exit the Sheppard AI application',
        'usage': '/exit',
        'examples': [
            '/exit'
        ],
        'category': 'general'
    }
}

HELP_CATEGORIES = {
    'general': 'Basic chat commands',
    'research': 'Web research and browsing',
    'memory': 'Memory and context management',
    'system': 'System settings and status'
}

ERROR_MESSAGES = {
    'command_not_found': "Command not found. Type '/help' to see available commands.",
    'invalid_usage': "Invalid command usage. See '/help {command}' for correct usage.",
    'research_failed': "Research operation failed: {error}",
    'browser_error': "Browser operation failed: {error}",
    'memory_error': "Memory operation failed: {error}",
    'settings_error': "Settings operation failed: {error}",
    'initialization_error': "System initialization failed: {error}"
}

# Research System Constants
RESEARCH_SETTINGS = {
    'max_depth': 3,
    'max_pages': 5,
    'timeout': 30,
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Visual styling constants for Rich
STYLES = {
    'title': 'bold #2196f3',  # Bold Blue
    'command': '#00bcd4',     # Teal
    'error': 'bold red',
    'warning': 'yellow',
    'success': 'bold green',
    'info': '#2196f3',        # Blue
    'progress': '#e91e63'     # Pink
}

# Research-specific styling
RESEARCH_STYLES = {
    'title': 'bold #2196f3',      # Bold Blue
    'finding': '#4caf50',         # Green
    'source': '#2196f3',          # Blue
    'task': '#ff9800',            # Orange
    'progress': '#00bcd4',        # Teal
    'url': '#03a9f4 underline',   # Light Blue Underlined
    'separator': '#78909c',       # Blue Grey
    'summary': 'bold #4caf50'     # Bold Green
}
# Merge with main STYLES dictionary
STYLES.update(RESEARCH_STYLES)

# Default HTTP headers
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
