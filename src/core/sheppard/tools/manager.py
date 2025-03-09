# src/core/sheppard/tools/manager.py

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import asyncio
import os

from src.core.trust_call import TrustCall, SchemaFailsafeManager
from src.config.config import DatabaseConfig

logger = logging.getLogger(__name__)

class ToolManager:
    """Manages tool registration and execution with TrustCall"""
    
    def __init__(self, client):
        self.client = client
        self.trust_call = TrustCall(client)
        self.tool_stats = {}
        self._initialization_lock = asyncio.Lock()
        self._initialized = False
        self._tool_registry: Dict[str, Callable] = {}
        
        # Schema definitions for basic tools
        self.tool_schemas = {
            "calculator": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
            "search": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            },
            "summarizer": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to summarize"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length in words",
                        "minimum": 10,
                        "default": 100
                    }
                },
                "required": ["text"]
            }
        }

    async def initialize(self) -> bool:
        """Initialize tool manager with TrustCall"""
        if self._initialized:
            return True

        async with self._initialization_lock:
            try:
                # Register all tool schemas first
                for name, schema in self.tool_schemas.items():
                    if not await self._register_tool_with_schema(name, schema):
                        logger.error(f"Failed to register schema for tool: {name}")
                        return False
                    
                    # Initialize stats
                    self.tool_stats[name] = {
                        "calls": 0,
                        "errors": 0,
                        "last_used": None,
                        "avg_response_time": 0.0
                    }

                self._initialized = True
                return True
                
            except Exception as e:
                logger.error(f"Error initializing tool manager: {str(e)}")
                return False

    async def _register_tool_with_schema(self, name: str, schema: Dict[str, Any]) -> bool:
        """Register tool schema and implementation"""
        try:
            # Register schema first
            if not self.trust_call.register_schema(name, schema):
                logger.error(f"Failed to register schema for tool: {name}")
                return False

            # Create tool implementation
            tool_impl = await self._create_tool_implementation(name)
            
            # Register tool implementation
            if not await self._register_tool_implementation(name, tool_impl):
                logger.error(f"Failed to register tool implementation: {name}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error registering tool {name}: {str(e)}")
            return False

    async def _create_tool_implementation(self, tool_name: str) -> Callable:
        """Create a callable tool implementation"""
        async def tool_impl(**kwargs):
            start_time = datetime.now()
            result = None
            
            try:
                # Validate parameters
                valid_params = await self.trust_call.validate_schema_call(
                    tool_name,
                    kwargs
                )
                
                if not valid_params:
                    logger.error(f"Invalid parameters for tool {tool_name}")
                    return None

                # Execute appropriate tool function
                if tool_name == "calculator":
                    result = await self._execute_calculator(valid_params)
                elif tool_name == "search":
                    result = await self._execute_search(valid_params)
                elif tool_name == "summarizer":
                    result = await self._execute_summarizer(valid_params)
                
                if result is not None:
                    # Update stats only on successful execution
                    self.tool_stats[tool_name]["calls"] += 1
                    self.tool_stats[tool_name]["last_used"] = datetime.now().isoformat()
                    
                    # Update average response time
                    exec_time = (datetime.now() - start_time).total_seconds()
                    current_avg = self.tool_stats[tool_name]["avg_response_time"]
                    total_calls = self.tool_stats[tool_name]["calls"]
                    self.tool_stats[tool_name]["avg_response_time"] = (
                        (current_avg * (total_calls - 1) + exec_time) / total_calls
                    )
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                self.tool_stats[tool_name]["errors"] += 1
                return None

        return tool_impl

    async def _register_tool_implementation(self, name: str, implementation: Callable) -> bool:
        """Register tool implementation"""
        try:
            self._tool_registry[name] = implementation
            
            # Use wrapper to convert async function to sync for TrustCall
            def sync_wrapper(**kwargs):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(implementation(**kwargs))

            if not self.trust_call.register_tool(name, sync_wrapper):
                logger.error(f"Failed to register tool with TrustCall: {name}")
                return False

            return True
            
        except Exception as e:
            logger.error(f"Error registering tool implementation {name}: {str(e)}")
            return False

    async def _execute_calculator(self, params: Dict[str, Any]) -> Optional[str]:
        """Execute calculator tool"""
        try:
            expression = params["expression"]
            safe_chars = set("0123456789+-*/() .")
            cleaned_expr = ''.join(c for c in expression if c in safe_chars)
            safe_dict = {
                "abs": abs, "float": float, "int": int,
                "max": max, "min": min, "pow": pow,
                "round": round, "sum": sum
            }
            result = eval(cleaned_expr, {"__builtins__": {}}, safe_dict)
            return str(result)
        except Exception as e:
            logger.error(f"Calculator error: {str(e)}")
            return None

    async def _execute_search(self, params: Dict[str, Any]) -> Optional[str]:
        """Execute search tool"""
        try:
            query = params["query"]
            max_results = params.get("max_results", 5)
            return f"Web search results for: {query} (limited to {max_results} results)"
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return None

    async def _execute_summarizer(self, params: Dict[str, Any]) -> Optional[str]:
        """Execute summarizer tool"""
        try:
            text = params["text"]
            max_length = params.get("max_length", 100)
            
            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": f"Summarize the following text in {max_length} words or less:"
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            
            if isinstance(response, dict) and 'message' in response:
                return response['message']['content']
            return None
        except Exception as e:
            logger.error(f"Summarizer error: {str(e)}")
            return None

    async def analyze_input(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if tools should be used for input"""
        if not self._initialized:
            return {"use_tools": False, "recommended_tools": [], "reasoning": "Tool manager not initialized"}

        try:
            prompt = (
                "Analyze if any tools should be used for this input. "
                f"Available tools: {', '.join(self.tool_schemas.keys())}.\n\n"
                f"Input: {user_input}\n"
                f"Context: {json.dumps(context)}\n\n"
                "Return a JSON object with the following structure:\n"
                "{\n"
                '  "use_tools": boolean,\n'
                '  "recommended_tools": [string],\n'
                '  "reasoning": string\n'
                "}"
            )

            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a tool analysis expert. Analyze inputs and recommend appropriate tools."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            if isinstance(response, dict) and 'message' in response:
                try:
                    return json.loads(response['message']['content'])
                except json.JSONDecodeError:
                    logger.error("Failed to parse tool analysis response")
                    
            return {
                "use_tools": False,
                "recommended_tools": [],
                "reasoning": "Failed to get valid analysis"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing for tools: {str(e)}")
            return {
                "use_tools": False,
                "recommended_tools": [],
                "reasoning": f"Error during analysis: {str(e)}"
            }

    async def execute_tool(self, name: str, **kwargs) -> Optional[Any]:
        """Execute a registered tool"""
        if not self._initialized or name not in self._tool_registry:
            return None

        try:
            tool_impl = self._tool_registry[name]
            return await tool_impl(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            return None

    async def validate_health(self) -> bool:
        """Validate tool manager health"""
        if not self._initialized:
            return False
            
        try:
            # Check if tools are registered
            if not self.tool_schemas or not self._tool_registry:
                return False
                
            # Validate basic tool execution
            test_results = []
            
            # Test calculator
            calc_result = await self._execute_calculator({"expression": "2+2"})
            test_results.append(calc_result == "4")
            
            # Test search
            search_result = await self._execute_search({"query": "test", "max_results": 1})
            test_results.append(isinstance(search_result, str))
            
            # Test summarizer
            summary_result = await self._execute_summarizer({"text": "Test sentence.", "max_length": 5})
            test_results.append(isinstance(summary_result, str))
            
            return all(test_results)
            
        except Exception as e:
            logger.error(f"Health validation error: {str(e)}")
            return False

    async def shutdown(self) -> None:
        """Cleanup tool manager resources"""
        try:
            if not self._initialized:
                return

            # Save final statistics
            stats_file = os.path.join(
                DatabaseConfig.STATS_DIR,
                f"tool_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(stats_file, 'w') as f:
                json.dump(self.tool_stats, f, indent=2)
            
            self._initialized = False
            self._tool_registry.clear()
            
        except Exception as e:
            logger.error(f"Error during tool manager shutdown: {str(e)}")
