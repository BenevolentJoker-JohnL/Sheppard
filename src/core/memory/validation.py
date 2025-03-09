# src/core/memory/validation.py
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from jsonschema import validate as validate_schema, ValidationError

logger = logging.getLogger(__name__)

class ValidationSchemas:
    """Validation schema management for memory systems"""
    
    def __init__(self):
        """Initialize validation schemas"""
        self.schemas = {}
        self.initialize_schemas()
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "repairs_attempted": 0,
            "repairs_successful": 0
        }

    def initialize_schemas(self):
        """Initialize base validation schemas"""
        # Common fields for all memory types
        common_fields = {
            "timestamp": {"type": "string", "format": "date-time"},
            "importance_score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "memory_hash": {"type": "string"},
            "state": {
                "type": "string",
                "enum": ["active", "archived", "deleted"]
            }
        }

        # Base memory schema
        self.schemas["base_memory"] = {
            "type": "object",
            "properties": {
                **common_fields,
                "content": {"type": "string"},
                "metadata": {"type": "object"}
            },
            "required": ["content", "memory_hash", "importance_score"]
        }

        # Interaction memory schema
        self.schemas["interaction_memory"] = {
            "type": "object",
            "properties": {
                **common_fields,
                "input": {"type": "string"},
                "response": {"type": "string"},
                "embedding": {
                    "type": "array",
                    "items": {"type": "number"}
                },
                "entities": {"type": "object"},
                "topics": {"type": "object"},
                "context_metadata": {"type": "object"}
            },
            "required": [
                "input",
                "response",
                "embedding",
                "state",
                "importance_score",
                "memory_hash"
            ]
        }

        # Episodic memory schema 
        self.schemas["episodic_memory"] = {
            "type": "object",
            "properties": {
                **common_fields,
                "experience": {"type": "string"},
                "context": {"type": "object"},
                "emotions": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "embedding": {
                    "type": "array", 
                    "items": {"type": "number"}
                }
            },
            "required": [
                "experience",
                "memory_hash",
                "importance_score",
                "embedding"
            ]
        }

        # Semantic memory schema
        self.schemas["semantic_memory"] = {
            "type": "object",
            "properties": {
                **common_fields,
                "concept": {"type": "string"},
                "definition": {"type": "string"},
                "relationships": {"type": "object"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "embedding": {
                    "type": "array",
                    "items": {"type": "number"}
                }
            },
            "required": [
                "concept",
                "definition", 
                "memory_hash",
                "importance_score",
                "embedding"
            ]
        }

    async def validate_memory(
        self,
        memory_data: Dict[str, Any],
        schema_type: str = "base_memory"
    ) -> Optional[Dict[str, Any]]:
        """Validate memory data against schema"""
        try:
            self.validation_stats["total_validations"] += 1
            
            if schema_type not in self.schemas:
                logger.error(f"Unknown schema type: {schema_type}")
                self.validation_stats["failed_validations"] += 1
                return None
                
            schema = self.schemas[schema_type]
            
            # Add timestamp if not present
            if 'timestamp' not in memory_data:
                memory_data['timestamp'] = datetime.now().isoformat()
                
            try:
                validate_schema(memory_data, schema)
                self.validation_stats["successful_validations"] += 1
                return memory_data
            except ValidationError:
                # Attempt repair
                self.validation_stats["repairs_attempted"] += 1
                try:
                    repaired_data = self._repair_memory_data(memory_data, schema)
                    if repaired_data:
                        self.validation_stats["repairs_successful"] += 1
                        self.validation_stats["successful_validations"] += 1
                        return repaired_data
                except Exception as e:
                    logger.error(f"Error repairing memory data: {str(e)}")
                
                self.validation_stats["failed_validations"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error validating memory: {str(e)}")
            self.validation_stats["failed_validations"] += 1
            return None

    def _repair_memory_data(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Attempt to repair invalid memory data"""
        try:
            repaired_data = {}
            
            # Handle required fields
            for field in schema.get('required', []):
                if field not in data or data[field] is None:
                    # Set defaults
                    if field == 'embedding':
                        repaired_data[field] = [0.0] * 768
                    elif field == 'importance_score':
                        repaired_data[field] = 0.5  
                    elif field == 'state':
                        repaired_data[field] = 'active'
                    elif field in ['input', 'response', 'content', 'experience', 'concept', 'definition']:
                        repaired_data[field] = ''
                    elif field == 'memory_hash':
                        repaired_data[field] = str(hash(str(datetime.now())))
                else:
                    repaired_data[field] = data[field]
            
            # Copy non-required fields if valid
            for field, value in data.items():
                if field not in repaired_data:
                    field_schema = schema['properties'].get(field, {})
                    field_type = field_schema.get('type')
                    
                    if field_type == 'object' and not isinstance(value, dict):
                        repaired_data[field] = {}
                    elif field_type == 'array' and not isinstance(value, list):
                        repaired_data[field] = []
                    elif field_type == 'string' and not isinstance(value, str):
                        repaired_data[field] = str(value)
                    elif field_type == 'number' and not isinstance(value, (int, float)):
                        repaired_data[field] = 0.0
                    else:
                        repaired_data[field] = value
            
            # Validate repaired data
            validate_schema(repaired_data, schema)
            return repaired_data
            
        except Exception as e:
            logger.error(f"Error repairing memory data: {str(e)}")
            return None

    def get_schema(self, schema_type: str) -> Optional[Dict[str, Any]]:
        """Get a specific schema"""
        return self.schemas.get(schema_type)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return self.validation_stats.copy()
