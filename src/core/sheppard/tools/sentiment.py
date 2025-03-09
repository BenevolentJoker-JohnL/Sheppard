import logging
import json
from typing import Optional
from .base import BaseTool

logger = logging.getLogger(__name__)

class SentimentTool(BaseTool):
    """Tool for sentiment analysis"""
    
    def __init__(self, client):
        super().__init__(client)
        self.description = "Analyzes sentiment in text"
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
                "input": {"text": "I love this product!"},
                "output": '{"sentiment": "positive", "confidence": 0.95}'
            }
        ]

    async def execute(self, **kwargs) -> Optional[str]:
        """Execute sentiment analysis"""
        try:
            text = kwargs.get("text")
            if not text:
                logger.error("No text provided for sentiment analysis")
                return None

            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze the sentiment of the following text. "
                            "Return a JSON object with sentiment (positive/negative/neutral) "
                            "and confidence score."
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
                    sentiment_data = json.loads(response['message']['content'])
                    return json.dumps(sentiment_data, indent=2)
                except json.JSONDecodeError:
                    return response['message']['content']
            return None
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return None

    async def validate(self) -> bool:
        """Validate tool configuration"""
        try:
            test_result = await self.execute(
                text="This is a test sentence."
            )
            return bool(test_result)
        except Exception as e:
            logger.error(f"Sentiment analyzer validation failed: {str(e)}")
            return False
