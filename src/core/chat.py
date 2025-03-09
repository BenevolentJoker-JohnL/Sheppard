import logging
import json
import asyncio
import re
from typing import Dict, Any, AsyncGenerator, Optional, List, Union, Set, TYPE_CHECKING
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from src.memory.models import Memory, MemoryType
# Import only what's needed directly
from src.research.models import ResearchType
from src.research.models import ChatResponse, ChatMetadata, ResponseType
from src.research.models import Message, MessageRole, MessageMetadata
from src.research.models import Persona, PersonaType
from src.research.models import User, UserPreferences

# For classes only needed for type hints
if TYPE_CHECKING:
    from src.research.models import ResearchResult

from src.research.processors import format_research_results
from src.utils.exceptions import (
    ChatInitError,
    PersonaNotFoundError,
    UnauthorizedError,
    ValidationError,
    ResearchError
)
from src.utils.validation import (
    validate_message_content,
    validate_metadata,
    validate_user_preferences
)
from src.utils.constants import (
    MAX_MESSAGE_LENGTH,
    MAX_CONTEXT_MESSAGES,
    DEFAULT_RESPONSE_TYPE,
    SYSTEM_PERSONA_ID
)

logger = logging.getLogger(__name__)

class ChatState(Enum):
    """Enumeration of possible chat states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    CLOSED = "closed"

@dataclass
class ChatConfig:
    """Configuration settings for ChatApp."""
    max_message_length: int = MAX_MESSAGE_LENGTH
    max_context_messages: int = MAX_CONTEXT_MESSAGES
    default_response_type: ResponseType = DEFAULT_RESPONSE_TYPE
    enable_memory: bool = True
    enable_research: bool = True
    enable_personas: bool = True
    default_persona_id: str = SYSTEM_PERSONA_ID
    auto_save_context: bool = True
    debug_mode: bool = False

class ChatContext:
    """Manages chat context and state."""
    def __init__(self, config: ChatConfig):
        self.config = config
        self.state = ChatState.INITIALIZING
        self.messages: List[Message] = []
        self.active_users: Set[str] = set()
        self.metadata: Dict[str, Any] = {}
        
    def add_message(self, message: Message) -> None:
        """Add message to context while maintaining max length."""
        self.messages.append(message)
        if len(self.messages) > self.config.max_context_messages:
            self.messages.pop(0)
            
    def clear_messages(self) -> None:
        """Clear message history."""
        self.messages = []
        
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """Update context metadata."""
        self.metadata.update(metadata)
        
    def add_active_user(self, user_id: str) -> None:
        """Add user to active users set."""
        self.active_users.add(user_id)
        
    def remove_active_user(self, user_id: str) -> None:
        """Remove user from active users set."""
        self.active_users.discard(user_id)

class ChatApp:
    """Main chat application class."""
    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()
        self.context = ChatContext(self.config)
        self._initialized = False
        
        # Core systems
        self.memory_system = None
        self.research_system = None 
        self.llm_system = None
        
        # User and persona management
        self.users: Dict[str, User] = {}
        self.personas: Dict[str, Persona] = {}
        self.current_persona: Optional[Persona] = None
        
        # Async locks
        self._processing_lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        
    @property
    def is_ready(self) -> bool:
        """Check if chat app is initialized and ready."""
        return (self._initialized and 
                self.context.state == ChatState.READY)

    async def initialize(
        self,
        memory_system = None,
        research_system = None,
        llm_system = None,
        personas: Optional[Dict[str, Persona]] = None,
        users: Optional[Dict[str, User]] = None
    ) -> None:
        """Initialize chat application with required subsystems."""
        async with self._init_lock:
            if self._initialized:
                logger.warning("ChatApp already initialized")
                return
                
            try:
                # Initialize core systems
                self.memory_system = memory_system
                self.research_system = research_system
                self.llm_system = llm_system
                
                # Validate and set personas
                if personas:
                    for persona_id, persona in personas.items():
                        if not isinstance(persona, Persona):
                            raise ValidationError(f"Invalid persona object for ID: {persona_id}")
                    self.personas = personas
                    
                    # Set default persona
                    default_persona = next(
                        (p for p in personas.values() if p.persona_type == PersonaType.DEFAULT),
                        None
                    )
                    if default_persona:
                        self.current_persona = default_persona
                    elif self.config.default_persona_id in personas:
                        self.current_persona = personas[self.config.default_persona_id]
                
                # Initialize user registry
                if users:
                    for user_id, user in users.items():
                        if not isinstance(user, User):
                            raise ValidationError(f"Invalid user object for ID: {user_id}")
                    self.users = users
                
                # Verify required systems
                if self.config.enable_memory and not self.memory_system:
                    raise ChatInitError("Memory system required but not provided")
                if self.config.enable_research and not self.research_system:
                    raise ChatInitError("Research system required but not provided")
                if not self.llm_system:
                    raise ChatInitError("LLM system is required")
                
                # Initialize complete
                self._initialized = True
                self.context.state = ChatState.READY
                logger.info("ChatApp initialized successfully")
                
            except Exception as e:
                self.context.state = ChatState.ERROR
                logger.error(f"Failed to initialize ChatApp: {str(e)}")
                raise ChatInitError(f"Initialization failed: {str(e)}")

    async def process_input(
        self,
        user_input: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[ChatResponse, None]:
        """Process user input with context integration."""
        if not self._initialized:
            raise RuntimeError("ChatApp not initialized")
            
        # Acquire processing lock
        async with self._processing_lock:
            try:
                self.context.state = ChatState.PROCESSING

                # Validate input
                if not user_input:
                    yield ChatResponse(
                        content="Empty input received.",
                        response_type=ResponseType.ERROR
                    )
                    return
                
                if len(user_input) > self.config.max_message_length:
                    yield ChatResponse(
                        content=f"Input exceeds maximum length of {self.config.max_message_length} characters.",
                        response_type=ResponseType.ERROR
                    )
                    return

                # Validate user if ID provided
                if user_id:
                    try:
                        user = await self.validate_user_access(user_id)
                        metadata = metadata or {}
                        metadata["user"] = asdict(user)
                    except UnauthorizedError as e:
                        yield ChatResponse(
                            content=str(e),
                            response_type=ResponseType.ERROR
                        )
                        return

                # Handle commands
                if user_input.startswith("/"):
                    if user_input.startswith("/clear"):
                        self.context.clear_messages()
                        yield ChatResponse(
                            content="Chat context cleared.",
                            response_type=ResponseType.SYSTEM
                        )
                        return

                    if user_input.startswith("/persona "):
                        persona_id = user_input[len("/persona "):].strip()
                        if self.set_persona(persona_id):
                            yield ChatResponse(
                                content=f"Switched to persona: {self.current_persona.name}",
                                response_type=ResponseType.SYSTEM
                            )
                        else:
                            yield ChatResponse(
                                content=f"Persona '{persona_id}' not found",
                                response_type=ResponseType.ERROR
                            )
                        return
                    
                    if user_input.startswith("/research "):
                        research_topic = user_input[len("/research "):].strip()
                        if research_topic:
                            try:
                                async for response in self.perform_research(research_topic, metadata):
                                    yield response
                                return
                            except ResearchError as e:
                                yield ChatResponse(
                                    content=f"Research error: {str(e)}",
                                    response_type=ResponseType.ERROR
                                )
                                return
                        else:
                            yield ChatResponse(
                                content="Please provide a research topic.",
                                response_type=ResponseType.ERROR
                            )
                            return

                    if user_input.startswith("/status"):
                        try:
                            status = await self.get_system_status()
                            status_text = json.dumps(status, indent=2)
                            yield ChatResponse(
                                content=f"System Status:\n```json\n{status_text}\n```",
                                response_type=ResponseType.SYSTEM
                            )
                            return
                        except Exception as e:
                            yield ChatResponse(
                                content=f"Error getting system status: {str(e)}",
                                response_type=ResponseType.ERROR
                            )
                            return
                
                # Get relevant memories
                memory_context = ""
                if self.memory_system and self.config.enable_memory:
                    try:
                        # Search for relevant memories
                        relevant_memories = await self.memory_system.search(
                            user_input,
                            limit=5,
                            metadata_filter={"type": "conversation"}
                        )
                        
                        # Get user preferences
                        try:
                            preferences = await self.memory_system.search(
                                "",
                                metadata_filter={"type": "preference"},
                                limit=10
                            )
                        except Exception as pref_error:
                            logger.warning(f"Error retrieving preferences: {str(pref_error)}")
                            preferences = []  # Use empty list if preferences can't be retrieved
                        
                        # If we have memories, format them as context
                        if relevant_memories or preferences:
                            # Build context string
                            context_parts = []
                            
                            # Add memory context
                            if relevant_memories:
                                context_parts.append("Relevant past conversations:")
                                for memory in relevant_memories:
                                    if hasattr(memory, 'content') and memory.content:
                                        context_parts.append(f"- {memory.content}")
                                context_parts.append("")
                            
                            # Add preference context
                            if preferences:
                                context_parts.append("User preferences:")
                                for pref in preferences:
                                    if hasattr(pref, 'content') and pref.content:
                                        context_parts.append(f"- {pref.content}")
                            
                            memory_context = "\n".join(context_parts)
                            
                            # Add context to metadata
                            metadata = metadata or {}
                            metadata["memory_context"] = memory_context
                    except Exception as e:
                        logger.warning(f"Error retrieving memory context: {str(e)}")
                        # Continue without memory context if retrieval fails
                        memory_context = ""
                
                # Prepare messages with memory context
                messages = []
                if memory_context:
                    messages.append({
                        "role": "system",
                        "content": f"Use this context from previous interactions when it's relevant to the current conversation: {memory_context}"
                    })
                
                # Add user message
                messages.append({
                    "role": "user", 
                    "content": user_input
                })

                # Generate response using LLM
                try:
                    response_content = ""
                    async for response in self.llm_system.chat(
                        messages=messages,
                        stream=True,
                        persona=self.current_persona,
                        metadata=metadata
                    ):
                        # Append to response content for later storage
                        if response.content:
                            response_content += response.content
                        yield response
                
                    # Store interaction in memory
                    if self.memory_system and self.config.enable_memory:
                        try:
                            await self._store_interaction(user_input, response_content)
                            
                            # Check for and store preferences
                            await self._extract_and_store_preferences(user_input)
                        except Exception as e:
                            logger.warning(f"Error storing interaction in memory: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Error generating LLM response: {str(e)}")
                    yield ChatResponse(
                        content="I encountered an error while processing your request.",
                        response_type=ResponseType.ERROR
                    )
                    return
            
            finally:
                self.context.state = ChatState.READY

    async def perform_research(
        self,
        topic: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[ChatResponse, None]:
        """
        Perform research on a given topic.
        
        Args:
            topic: Research topic
            metadata: Optional metadata
            
        Yields:
            ChatResponse: Response messages
        """
        if not self.research_system:
            yield ChatResponse(
                content="Research system not available.",
                response_type=ResponseType.ERROR
            )
            return
            
        try:
            # Yield thinking response
            yield ChatResponse(
                content=f"Researching '{topic}'...",
                response_type=ResponseType.THINKING
            )
            
            # Track progress
            progress_percent = 0.0
            
            # Define progress callback
            def progress_callback(progress: float):
                nonlocal progress_percent
                progress_percent = progress
            
            # Perform research
            results = await self.research_system.research_topic(
                topic=topic,
                research_type=ResearchType.WEB_SEARCH,
                depth=3,
                progress_callback=progress_callback,
                metadata=metadata
            )
            
            # Format results
            formatted_results = format_research_results(results)
            
            # Create result response
            response = ChatResponse(
                content=formatted_results,
                response_type=ResponseType.RESEARCH,
                metadata=ChatMetadata(
                    sources=results.get('sources', []),
                    context_type="research"
                )
            )
            
            # Store research in memory if available
            if self.memory_system and self.config.enable_memory:
                try:
                    # Format memory content
                    memory_content = f"Research on '{topic}':\n\n"
                    if results.get('summary'):
                        memory_content += f"Summary: {results['summary']}\n\n"
                    
                    memory_content += "Key findings:\n"
                    for finding in results.get('findings', [])[:5]:  # Limit to top 5 findings
                        if isinstance(finding, dict):
                            memory_content += f"- {finding.get('summary', 'No summary')}\n"
                        else:
                            memory_content += f"- {finding}\n"
                    
                    # Store memory
                    await self.memory_system.store(
                        Memory(
                            content=memory_content,
                            metadata={
                                "type": "research_results",
                                "topic": topic,
                                "timestamp": datetime.now().isoformat(),
                                "sources": [
                                    s.get('url') for s in results.get('sources', [])
                                    if isinstance(s, dict) and 'url' in s
                                ]
                            }
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to store research in memory: {str(e)}")
            
            yield response
            
        except Exception as e:
            logger.error(f"Research error: {str(e)}")
            yield ChatResponse(
                content=f"An error occurred during research: {str(e)}",
                response_type=ResponseType.ERROR
            )

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information.
        
        Returns:
            Dict[str, Any]: System status
        """
        try:
            status = {
                "system": {
                    "initialized": self._initialized,
                    "state": self.context.state.value,
                    "gpu_enabled": getattr(self.llm_system, "gpu_enabled", False),
                    "timestamp": datetime.now().isoformat()
                },
                "models": {
                    "chat_model": getattr(self.llm_system, "model_name", "unknown"),
                    "embed_model": getattr(self.llm_system, "embed_model", "unknown"),
                },
                "memory": {
                    "enabled": self.config.enable_memory,
                    "preferences_count": 0,
                    "max_context_tokens": self.config.max_context_messages,
                },
                "research": {
                    "enabled": self.config.enable_research,
                    "browser_available": False,
                    "browser_timeout": 0
                }
            }
            
            # Add memory status if available
            if self.memory_system:
                memory_status = await self.memory_system.get_status()
                status["memory"].update(memory_status)
            
            # Add research status if available
            if self.research_system:
                research_status = await self.research_system.get_status()
                status["research"].update({
                    "browser_available": research_status.get("components", {}).get("browser", False),
                    "browser_timeout": research_status.get("browser_timeout", 30)
                })
            
            # Add user and persona info
            status["users"] = {
                "count": len(self.users),
                "active": len(self.context.active_users)
            }
            
            status["personas"] = {
                "count": len(self.personas),
                "current": self.current_persona.id if self.current_persona else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def save_chat_history(self, filename: str) -> str:
        """
        Save chat history to file.
        
        Args:
            filename: Filename to save history to
            
        Returns:
            str: Path to saved file
        """
        try:
            from pathlib import Path
            
            # Create directory if it doesn't exist
            save_dir = Path("chat_history")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Add extension if not provided
            if not filename.endswith('.json'):
                filename += '.json'
                
            filepath = save_dir / filename
            
            # Format history
            history = []
            for message in self.context.messages:
                history.append({
                    "role": message.role.value,
                    "content": message.content,
                    "timestamp": message.metadata.timestamp
                })
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "messages": history,
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "message_count": len(history),
                        "persona": self.current_persona.id if self.current_persona else None
                    }
                }, f, indent=2)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Failed to save chat history: {str(e)}")
            raise

    async def get_all_preferences(self) -> List[Dict[str, Any]]:
        """
        Get all stored user preferences.
        
        Returns:
            List[Dict[str, Any]]: List of preference objects
        """
        try:
            if not self.memory_system:
                return []
                
            # Search for preferences
            preference_memories = await self.memory_system.search(
                "",
                metadata_filter={"type": "preference"},
                limit=100
            )
            
            # Format preferences
            preferences = []
            for memory in preference_memories:
                if not hasattr(memory, 'metadata'):
                    continue
                    
                pref_type = memory.metadata.get('preference_type')
                pref_value = memory.metadata.get('preference_value')
                
                if pref_type and pref_value:
                    preferences.append({
                        "key": pref_type,
                        "value": pref_value,
                        "type": "string",
                        "timestamp": memory.metadata.get('timestamp', datetime.now().isoformat())
                    })
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to get preferences: {str(e)}")
            return []
            
    async def set_preference(self, key: str, value: Any) -> None:
        """
        Set a user preference.
        
        Args:
            key: Preference key
            value: Preference value
        """
        try:
            if not self.memory_system:
                raise ValueError("Memory system not available")
                
            # Store preference
            await self.memory_system.store(
                Memory(
                    content=f"User preference: {key} = {value}",
                    metadata={
                        "type": "preference",
                        "preference_type": key,
                        "preference_value": value,
                        "timestamp": datetime.now().isoformat()
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Failed to set preference: {str(e)}")
            raise
            
    async def clear_preferences(self) -> None:
        """Clear all user preferences."""
        try:
            if not self.memory_system:
                raise ValueError("Memory system not available")
                
            # Search for preferences
            preference_memories = await self.memory_system.search(
                "",
                metadata_filter={"type": "preference"},
                limit=1000
            )
            
            # Delete each preference
            for memory in preference_memories:
                if hasattr(memory, 'embedding_id'):
                    await self.memory_system.delete(memory.embedding_id)
            
        except Exception as e:
            logger.error(f"Failed to clear preferences: {str(e)}")
            raise

    async def list_preferences(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available preference types.
        
        Args:
            category: Optional category filter
            
        Returns:
            List[Dict[str, Any]]: Available preferences
        """
        # Define available preferences
        preferences = [
            {
                "key": "theme",
                "type": "string",
                "description": "UI theme (light/dark)",
                "category": "appearance"
            },
            {
                "key": "notifications",
                "type": "boolean",
                "description": "Enable notifications",
                "category": "system"
            },
            {
                "key": "language",
                "type": "string",
                "description": "Interface language",
                "category": "appearance"
            },
            {
                "key": "max_message_length",
                "type": "integer",
                "description": "Maximum message length",
                "category": "system"
            },
            {
                "key": "auto_save",
                "type": "boolean",
                "description": "Auto-save conversations",
                "category": "system"
            },
            {
                "key": "research_depth",
                "type": "integer",
                "description": "Research depth level (1-5)",
                "category": "research"
            },
            {
                "key": "color",
                "type": "string",
                "description": "Favorite color",
                "category": "personal"
            },
            {
                "key": "food",
                "type": "string",
                "description": "Favorite food",
                "category": "personal"
            },
            {
                "key": "destination",
                "type": "string",
                "description": "Favorite destination",
                "category": "personal"
            },
            {
                "key": "flavor",
                "type": "string", 
                "description": "Favorite flavor",
                "category": "personal"
            }
        ]
        
        # Filter by category if provided
        if category:
            return [p for p in preferences if p["category"] == category]
        
        return preferences

    async def get_setting(self, setting: str) -> Any:
        """
        Get a specific system setting.
        
        Args:
            setting: Setting key
            
        Returns:
            Any: Setting value
        """
        settings = await self.get_settings()
        return settings.get(setting)
        
    async def get_settings(self) -> Dict[str, Any]:
        """
        Get all system settings.
        
        Returns:
            Dict[str, Any]: System settings
        """
        settings = {
            "max_message_length": self.config.max_message_length,
            "max_context_messages": self.config.max_context_messages,
            "default_response_type": self.config.default_response_type.value,
            "enable_memory": self.config.enable_memory,
            "enable_research": self.config.enable_research,
            "enable_personas": self.config.enable_personas,
            "default_persona_id": self.config.default_persona_id,
            "auto_save_context": self.config.auto_save_context,
            "debug_mode": self.config.debug_mode
        }
        
        return settings
        
    def get_setting_description(self, setting: str) -> str:
        """
        Get description for a setting.
        
        Args:
            setting: Setting key
            
        Returns:
            str: Setting description
        """
        descriptions = {
            "max_message_length": "Maximum length of messages in characters",
            "max_context_messages": "Maximum number of messages to keep in context",
            "default_response_type": "Default response type (normal, system, research, etc.)",
            "enable_memory": "Enable memory system for context storage",
            "enable_research": "Enable research capabilities",
            "enable_personas": "Enable persona switching",
            "default_persona_id": "Default persona ID",
            "auto_save_context": "Automatically save context to memory",
            "debug_mode": "Enable debug mode"
        }
        
        return descriptions.get(setting, "No description available")
        
    async def update_setting(self, setting: str, value: Any) -> None:
        """
        Update a system setting.
        
        Args:
            setting: Setting key
            value: New value
        """
        try:
            # Convert value to appropriate type
            if setting in ["enable_memory", "enable_research", "enable_personas", "auto_save_context", "debug_mode"]:
                if isinstance(value, str):
                    value = value.lower() in ["true", "1", "yes", "y"]
                else:
                    value = bool(value)
            elif setting in ["max_message_length", "max_context_messages"]:
                value = int(value)
            elif setting == "default_response_type":
                value = ResponseType(value)
                
            # Update setting
            setattr(self.config, setting, value)
            
        except Exception as e:
            logger.error(f"Failed to update setting: {str(e)}")
            raise ValidationError(f"Invalid value for setting {setting}: {str(e)}")

    async def validate_user_access(
        self,
        user_id: str,
        required_permissions: Optional[Set[str]] = None
    ) -> User:
        """Validate user access and permissions."""
        if user_id not in self.users:
            raise UnauthorizedError(f"User not found: {user_id}")
            
        user = self.users[user_id]
        if required_permissions:
            missing_permissions = required_permissions - set(user.permissions)
            if missing_permissions:
                raise UnauthorizedError(
                    f"User lacks required permissions: {missing_permissions}"
                )
        return user
        
    def set_persona(self, persona_id: str) -> bool:
        """Set the active persona for the chat."""
        if persona_id in self.personas:
            self.current_persona = self.personas[persona_id]
            return True
        return False

    async def update_user_preferences(
        self,
        user_id: str,
        preferences: UserPreferences
    ) -> None:
        """Update user preferences."""
        if user_id not in self.users:
            raise ValidationError(f"User not found: {user_id}")
            
        try:
            validate_user_preferences(preferences)
            self.users[user_id].preferences = preferences
        except ValidationError as e:
            logger.error(f"Invalid user preferences: {str(e)}")
            raise

    async def add_user(self, user: User) -> None:
        """Add new user to chat system."""
        if not isinstance(user, User):
            raise ValidationError("Invalid user object")
            
        if user.id in self.users:
            raise ValidationError(f"User ID already exists: {user.id}")
            
        self.users[user.id] = user
        self.context.add_active_user(user.id)

    async def remove_user(self, user_id: str) -> None:
        """Remove user from chat system."""
        if user_id not in self.users:
            raise ValidationError(f"User not found: {user_id}")
            
        del self.users[user_id]
        self.context.remove_active_user(user_id)

    async def close(self) -> None:
        """Clean up resources when shutting down."""
        try:
            # Close core systems
            if self.memory_system:
                await self.memory_system.close()
            if self.research_system:
                await self.research_system.close()
            if self.llm_system:
                await self.llm_system.close()
                
            # Clear context and state
            self.context.clear_messages()
            self.context.state = ChatState.CLOSED
            self._initialized = False
            
            logger.info("ChatApp shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ChatApp shutdown: {str(e)}")
            raise

    async def _handle_error(
        self,
        error: Exception,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """Handle errors and generate appropriate response."""
        error_message = str(error)
        error_type = type(error).__name__
        
        # Log error
        logger.error(
            f"Error occurred - Type: {error_type}, Message: {error_message}",
            extra={
                "user_id": user_id,
                "metadata": metadata,
                "error_type": error_type
            }
        )
        
        # Generate user-friendly response based on error type
        if isinstance(error, UnauthorizedError):
            return ChatResponse(
                content="You don't have permission to perform this action.",
                response_type=ResponseType.ERROR,
                metadata={"error_type": "unauthorized"}
            )
        elif isinstance(error, ValidationError):
            return ChatResponse(
                content="Invalid input or parameters provided.",
                response_type=ResponseType.ERROR,
                metadata={"error_type": "validation"}
            )
        elif isinstance(error, ResearchError):
            return ChatResponse(
                content="An error occurred while performing research.",
                response_type=ResponseType.ERROR,
                metadata={"error_type": "research"}
            )
        else:
            # Generic error response
            return ChatResponse(
                content="An unexpected error occurred. Please try again later.",
                response_type=ResponseType.ERROR,
                metadata={"error_type": "internal"}
            )

    async def _validate_and_process_metadata(
        self,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate and process metadata dictionary."""
        if not metadata:
            return {}
            
        try:
            # Validate metadata structure
            validate_metadata(metadata)
            
            # Add system metadata
            processed_metadata = {
                **metadata,
                "timestamp": datetime.utcnow().isoformat(),
                "chat_state": self.context.state.value,
                "persona_id": self.current_persona.id if self.current_persona else None
            }
            
            return processed_metadata
            
        except ValidationError as e:
            logger.warning(f"Invalid metadata: {str(e)}")
            return {}

    async def _store_interaction(self, user_input: str, response_content: str) -> None:
        """Store a user-assistant interaction in memory."""
        try:
            # Format interaction text
            interaction = f"""User: {user_input}
Assistant: {response_content}
Timestamp: {datetime.now().isoformat()}"""
            # Prepare metadata
            metadata = {
                "type": "conversation",
                "timestamp": datetime.now().isoformat(),
                "user_query": user_input,
                "response_summary": response_content[:100] + "..." if len(response_content) > 100 else response_content
            }
            
            # Store memory
            await self.memory_system.store(
                Memory(
                    content=interaction,
                    metadata=metadata
                )
            )
        except Exception as e:
            logger.warning(f"Failed to store interaction: {str(e)}")

    async def _extract_and_store_preferences(self, content: str) -> None:
        """Extract and store preferences from text content."""
        try:
            # Define preference patterns
            preference_patterns = {
                "color": r"favorite\s+color\s+(?:is|:)\s+(\w+)",
                "destination": r"favorite\s+destination\s+(?:is|:)\s+([A-Za-z\s]+)",
                "flavor": r"favorite\s+flavor\s+(?:is|:)\s+([A-Za-z\s]+)",
                "food": r"favorite\s+food\s+(?:is|:)\s+([A-Za-z\s]+)"
            }
            
            # Extract preferences
            extracted_preferences = {}
            for pref_type, pattern in preference_patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    extracted_preferences[pref_type] = match.group(1).strip()
            
            # Store each preference
            for pref_type, value in extracted_preferences.items():
                # Store as memory
                await self.memory_system.store(
                    Memory(
                        content=f"User's favorite {pref_type} is {value}",
                        metadata={
                            "type": "preference",
                            "preference_type": pref_type,
                            "preference_value": value,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to extract and store preferences: {str(e)}")

    async def _should_store_in_memory(
        self,
        message_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Determine if message should be stored in memory."""
        if not self.config.enable_memory or not self.config.auto_save_context:
            return False
            
        # Don't store system messages or error responses
        if message_type in ["system", "error"]:
            return False
            
        # Check metadata for storage flags
        if metadata and not metadata.get("store_in_memory", True):
            return False
            
        return True

    def _get_default_response_type(self, content: str) -> ResponseType:
        """Determine default response type based on content."""
        if not content:
            return ResponseType.ERROR
            
        # Check content characteristics
        if content.startswith(("Error:", "Failed:", "Invalid:")):
            return ResponseType.ERROR
        elif content.startswith(("System:", "Info:", "Notice:")):
            return ResponseType.SYSTEM
        elif "research findings" in content.lower():
            return ResponseType.RESEARCH
        else:
            return self.config.default_response_type

    def __repr__(self) -> str:
        """String representation of ChatApp instance."""
        return (
            f"ChatApp(initialized={self._initialized}, "
            f"state={self.context.state.value}, "
            f"active_users={len(self.context.active_users)}, "
            f"current_persona={self.current_persona.id if self.current_persona else None})"
        )
