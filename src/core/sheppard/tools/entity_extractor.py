import logging
import json
from typing import Optional
from .base import BaseTool

logger = logging.getLogger(__name__)

class EntityExtractorTool(BaseTool):
    """Tool for named entity extraction"""
    
    def __init__(self, client):
        super().__init__(client)
        self.description = "Extracts named entities from text"
        self.parameters = {
            "text": {
                "type": "string",
                "description": "Text to analyze",
                "required": True
            }
        }
        self.required = ["text"]
        self.examples = [
            {
                "input": {"text": "John works at Microsoft in Seattle."},
                "output": (
                    '{"entities": {'
                    '"people": ["John"], '
                    '"organizations": ["Microsoft"], '
                    '"locations": ["Seattle"]}}'
                )
            }
        ]

    async def execute(self, **kwargs) -> Optional[str]:
        """Execute entity extraction"""
        try:
            text = kwargs.get("text")
            if not text:
                logger.error("No text provided for entity extraction")
                return None

            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract named entities (people, organizations, locations) "
                            "from the following text. Return as JSON with categories."
                        )
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ]
            )
            
            if isinstance(response, dict) and 'message' in response:
                try:
                    entities_data = json.loads(response['message']['content'])
                    return json.dumps(entities_data, indent=2)
                except json.JSONDecodeError:
                    return response['message']['content']
            return None
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return None

    async def validate(self) -> bool:
        """Validate tool configuration"""
        try:
            test_result = await self.execute(
                text="John works at Microsoft in Seattle."
            )
            return bool(test_result)
        except Exception as e:
            logger.error(f"Entity extractor validation failed: {str(e)}")
            return False
