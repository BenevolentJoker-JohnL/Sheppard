import logging
from typing import Optional
from .base import BaseTool

logger = logging.getLogger(__name__)

class CalculatorTool(BaseTool):
    """Tool for basic calculations"""
    
    def __init__(self, client):
        super().__init__(client)
        self.description = "Performs basic mathematical calculations"
        self.parameters = {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate",
                "required": True
            }
        }
        self.required = ["expression"]
        self.examples = [
            {
                "input": {"expression": "2 + 2"},
                "output": "4"
            }
        ]

    async def execute(self, **kwargs) -> Optional[str]:
        """Execute calculation"""
        try:
            expression = kwargs.get("expression", "")
            if not expression:
                logger.error("No expression provided")
                return None

            # Clean expression for safety
            safe_chars = set("0123456789+-*/() .")
            cleaned_expression = ''.join(c for c in expression if c in safe_chars)
            
            # Define safe operations
            safe_dict = {
                "abs": abs,
                "float": float,
                "int": int,
                "max": max,
                "min": min,
                "pow": pow,
                "round": round,
                "sum": sum
            }

            result = eval(cleaned_expression, {"__builtins__": {}}, safe_dict)
            return str(result)
            
        except Exception as e:
            logger.error(f"Error in calculation: {str(e)}")
            return None

    async def validate(self) -> bool:
        """Validate tool configuration"""
        try:
            test_result = await self.execute(expression="2 + 2")
            return test_result == "4"
        except Exception as e:
            logger.error(f"Calculator validation failed: {str(e)}")
            return False
