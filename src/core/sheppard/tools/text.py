# src/core/sheppard/tools/text.py
from typing import Dict, Any
import json
from .base import BaseTool
import logging

logger = logging.getLogger(__name__)

class SummarizationTool(BaseTool):
    """Tool for text summarization"""

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "default": ""
                },
                "max_length": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1
                }
            },
            "required": ["text"]
        }

    async def execute(self, text: str, max_length: int = 100) -> str:
        """Summarize text using short context model"""
        try:
            max_length = max(1, min(max_length, 1000))
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
                self.update_stats(True)
                return response['message']['content']
            return "Unable to generate summary."
            
        except Exception as e:
            logger.error(f"Error in summarize_text: {str(e)}")
            self.update_stats(False)
            return f"Error generating summary: {str(e)}"

class SentimentTool(BaseTool):
    """Tool for sentiment analysis"""

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "default": ""
                }
            },
            "required": ["text"]
        }

    async def execute(self, text: str) -> str:
        """Analyze sentiment of text"""
        try:
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
                    self.update_stats(True)
                    return json.dumps(sentiment_data, indent=2)
                except json.JSONDecodeError:
                    return response['message']['content']
                    
            return "Unable to analyze sentiment."
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            self.update_stats(False)
            return f"Error analyzing sentiment: {str(e)}"

class EntityExtractionTool(BaseTool):
    """Tool for named entity extraction"""

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "default": ""
                }
            },
            "required": ["text"]
        }

    async def execute(self, text: str) -> str:
        """Extract named entities from text"""
        try:
            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract and list the named entities (people, places, organizations) "
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
                    self.update_stats(True)
                    return json.dumps(entities_data, indent=2)
                except json.JSONDecodeError:
                    return response['message']['content']
                    
            return "Unable to extract entities."
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            self.update_stats(False)
            return f"Error extracting entities: {str(e)}"
