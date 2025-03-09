import functools
import inspect
import asyncio
import logging
from typing import Callable, TypeVar, ParamSpec, Dict, Any, Optional
from datetime import datetime
import json
from jsonschema import validate, ValidationError
from jsonschema.exceptions import SchemaError
import jsonpatch

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')

class SchemaManager:
    """Manages schema validation and updates"""
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.schema_versions: Dict[str, int] = {}

    def register_schema(self, name: str, schema: Dict[str, Any]) -> bool:
        """Register a new schema or update existing one"""
        try:
            # Validate the schema itself
            if not self._validate_schema_structure(schema):
                logger.error(f"Invalid schema structure for {name}")
                return False

            self.schemas[name] = schema
            self.schema_versions[name] = self.schema_versions.get(name, 0) + 1
            return True

        except Exception as e:
            logger.error(f"Error registering schema {name}: {str(e)}")
            return False

    def _validate_schema_structure(self, schema: Dict[str, Any]) -> bool:
        """Validate schema structure"""
        try:
            required_fields = {'type', 'properties'}
            if not all(field in schema for field in required_fields):
                return False

            # Validate schema can be used for validation
            validate({}, schema)  # Test with empty object
            return True

        except SchemaError:
            return False
        except Exception as e:
            logger.error(f"Error validating schema structure: {str(e)}")
            return False

    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate data against registered schema"""
        try:
            if schema_name not in self.schemas:
                logger.error(f"Schema {schema_name} not found")
                return None

            schema = self.schemas[schema_name]
            validate(data, schema)
            return data

        except ValidationError as e:
            logger.error(f"Validation error for schema {schema_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return None

    def update_schema(self, name: str, patch: Dict[str, Any]) -> bool:
        """Update existing schema using JSON Patch"""
        try:
            if name not in self.schemas:
                logger.error(f"Schema {name} not found")
                return False

            current_schema = self.schemas[name]
            patch_obj = jsonpatch.JsonPatch([patch])
            updated_schema = patch_obj.apply(current_schema)

            if not self._validate_schema_structure(updated_schema):
                logger.error(f"Invalid updated schema for {name}")
                return False

            self.schemas[name] = updated_schema
            self.schema_versions[name] += 1
            return True

        except Exception as e:
            logger.error(f"Error updating schema {name}: {str(e)}")
            return False

class TrustCall:
    """Manages trusted function calls with schema validation"""
    def __init__(self, client=None):
        self.schema_manager = SchemaManager()
        self.client = client
        self._registered_tools: Dict[str, Callable] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def register_schema(self, name: str, schema: Dict[str, Any]) -> bool:
        """Register a schema for validation"""
        return self.schema_manager.register_schema(name, schema)

    def register_tool(self, name: str, tool_func: Callable) -> bool:
        """Register a tool function with its schema"""
        try:
            if name not in self.schema_manager.schemas:
                logger.error(f"No schema registered for tool {name}")
                return False

            self._registered_tools[name] = tool_func
            self._locks[name] = asyncio.Lock()
            return True

        except Exception as e:
            logger.error(f"Error registering tool {name}: {str(e)}")
            return False

    async def validate_schema_call(self, schema_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate parameters against schema"""
        return self.schema_manager.validate_data(schema_name, params)

    async def call_tool(self, name: str, **params) -> Any:
        """Call a registered tool with validated parameters"""
        if name not in self._registered_tools:
            raise ValueError(f"Tool {name} not registered")

        lock = self._locks.get(name, asyncio.Lock())
        
        async with lock:
            try:
                validated_params = await self.validate_schema_call(name, params)
                if not validated_params:
                    raise ValueError(f"Invalid parameters for tool {name}")

                tool_func = self._registered_tools[name]
                return await tool_func(**validated_params)

            except Exception as e:
                logger.error(f"Error calling tool {name}: {str(e)}")
                raise

    async def update_tool_schema(self, name: str, patch: Dict[str, Any]) -> bool:
        """Update a tool's schema"""
        return self.schema_manager.update_schema(name, patch)

def call(func: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator for function wrapping with error handling.
    Supports both async and sync functions.
    """
    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = datetime.now()
        try:
            # Get current event loop or create new one
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            result = await func(*args, **kwargs)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)}", 
                exc_info=True
            )
            raise

    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Error in {func.__name__} after {execution_time:.3f}s: {str(e)}", 
                exc_info=True
            )
            raise

    # Return appropriate wrapper based on function type
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper
