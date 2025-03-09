import logging
from typing import Optional
from .base import BaseTool

logger = logging.getLogger(__name__)

class SummarizerTool(BaseTool):
    """Tool for text summarization"""
    
    def __init__(self, client):
        super().__init__(client)
        self.description = "Summarizes text content using LLM models"
        self.parameters = {
            "text": {
                "type": "string",
                "description": "Text to summarize",
                "required": True
            },
            "max_length": {
                "type": "integer",
                "description": "Maximum length of summary (in words)",
                "required": False,
                "default": None
            }
        }
        self.required = ["text"]
        self.examples = [
            {
                "input": {"text": "Long article...", "max_length": 100},
                "output": "Concise summary..."
            }
        ]

    async def execute(self, **kwargs) -> Optional[str]:
        """Execute text summarization"""
        try:
            text = kwargs.get("text")
            max_length = kwargs.get("max_length")
            
            if not text:
                logger.error("No text provided for summarization")
                return None
            
            # Create summarization prompt
            prompt = "Summarize the following text concisely"
            if max_length:
                prompt += f" in {max_length} words or less"
            prompt += f":\n\n{text}"
            
            # Use short context model for quick response
            response = await self.client.chat(
                model="llama3.2:latest",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    }
                ]
            )
            
            if isinstance(response, dict) and 'message' in response:
                return response['message']['content']
            return None
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return None

    async def validate(self) -> bool:
        """Validate tool configuration"""
        try:
            # Test with short sample text
            test_result = await self.execute(
                text="This is a test sentence for validation."
            )
            return bool(test_result)
        except Exception as e:
            logger.error(f"Summarizer validation failed: {str(e)}")
            return False
