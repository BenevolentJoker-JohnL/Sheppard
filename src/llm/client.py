"""
Enhanced wrapper for Ollama API with proper error handling.
File: src/llm/client.py
"""

import json
import logging
import aiohttp
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List, Union
from datetime import datetime

from src.config.settings import settings
from src.llm.models import ChatMessage, ChatResponse
from src.llm.exceptions import (
    ModelNotFoundError,
    APIError,
    TokenLimitError,
    ValidationError,
    EmbeddingError,
    TimeoutError,
    StreamError,
    ServiceUnavailableError
)

logger = logging.getLogger(__name__)

class OllamaClient:
    """Enhanced wrapper for Ollama API with schema validation and error handling."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        validate_schemas: bool = True
    ):
        """Initialize Ollama client."""
        self.model_name = model_name or settings.OLLAMA_MODEL
        self.embed_model = settings.OLLAMA_EMBED_MODEL
        self.api_base = api_base or settings.ollama_api_base
        self.validate_schemas = validate_schemas
        self.session = None
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        self._retry_count = 0
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 1.0

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def _handle_api_error(self, response: aiohttp.ClientResponse, operation: str) -> None:
        """Handle API error response."""
        try:
            error_text = await response.text()
            error_data = json.loads(error_text)
            error_msg = error_data.get('error', 'Unknown API error')
        except:
            error_msg = f"API call failed with status {response.status}"
        
        logger.error(f"{operation} failed: {error_msg}")
        
        if response.status == 404:
            raise ModelNotFoundError(self.model_name)
        elif response.status == 413:
            raise TokenLimitError("Input too long")
        elif response.status == 408:
            raise TimeoutError("Request timed out")
        elif response.status == 503:
            raise ServiceUnavailableError("Service temporarily unavailable")
        elif response.status >= 500:
            if self._retry_count < self.MAX_RETRIES:
                self._retry_count += 1
                await asyncio.sleep(self.RETRY_DELAY * self._retry_count)
                return await self._retry_operation(operation)
            else:
                raise APIError(f"Server error after {self.MAX_RETRIES} retries: {error_msg}")
        else:
            raise APIError(f"API error: {error_msg}")

    async def generate_embedding(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Generate embeddings using mxbai-embed-large model."""
        try:
            await self._ensure_session()
            
            # Handle empty text
            if not text or not text.strip():
                logger.debug("Empty text provided, returning zero vector")
                return [0.0] * self.embedding_dimension
            
            # Prepare request
            url = f"{self.api_base}/api/embeddings"
            payload = {
                'model': self.embed_model,
                'prompt': text.strip()
            }
            
            # Make request
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    await self._handle_api_error(response, "embedding")
                
                result = await response.json()
                
                # Check for errors
                if 'error' in result:
                    error_msg = result.get('error', 'Unknown API error')
                    logger.error(f"API returned error: {error_msg}")
                    raise APIError(f"API returned error: {error_msg}")
                
                # Extract embedding
                embedding = result.get('embedding')
                if not embedding:
                    raise APIError("No embedding field in API response")
                
                # Verify dimension
                if len(embedding) != self.embedding_dimension:
                    error_msg = (
                        f"Generated embedding dimension {len(embedding)} does not match "
                        f"expected dimension {self.embedding_dimension}"
                    )
                    logger.error(error_msg)
                    raise ValidationError(error_msg)
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")

    async def embeddings(self, prompt: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings using the specified model.
        
        This method provides backward compatibility with older code.
        
        Args:
            prompt: Text to embed
            model: Optional model override
            
        Returns:
            List[float]: Embedding vector
        """
        return await self.generate_embedding(prompt)

    async def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, Any]]],
        stream: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        persona: Optional[Any] = None,  # Added persona parameter
        metadata: Optional[Dict[str, Any]] = None  # Added metadata parameter
    ) -> AsyncGenerator[ChatResponse, None]:
        """Send a chat request to the Ollama API."""
        try:
            await self._ensure_session()
            
            # Convert messages to dict format
            message_dicts = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    message_dicts.append(msg.model_dump())
                elif hasattr(msg, 'model_dump'):  # For other Pydantic models
                    message_dicts.append(msg.model_dump())
                elif hasattr(msg, 'to_dict'):     # For classes with to_dict method
                    message_dicts.append(msg.to_dict())
                elif isinstance(msg, dict):       # Already a dictionary
                    message_dicts.append(msg)
                else:
                    # Convert to a basic dict with role and content
                    message_dicts.append({
                        "role": getattr(msg, 'role', 'user'),
                        "content": getattr(msg, 'content', str(msg))
                    })
            
            # Prepare request
            url = f"{self.api_base}/api/chat"
            payload = {
                'model': self.model_name,
                'messages': message_dicts,
                'stream': stream,
                'options': {
                    'temperature': temperature or settings.DEFAULT_TEMPERATURE,
                    'top_p': top_p or settings.DEFAULT_TOP_P,
                    'num_predict': max_tokens or settings.MAX_TOKENS
                }
            }
            
            # Make request
            async with self.session.post(url, json=payload) as response:
                if response.status != 200:
                    await self._handle_api_error(response, "chat")
                
                # Handle streamed response
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            
                            if 'error' in chunk:
                                error_msg = chunk['error']
                                if 'model not found' in error_msg.lower():
                                    raise ModelNotFoundError(self.model_name)
                                elif 'token limit' in error_msg.lower():
                                    raise TokenLimitError(0, 0)
                                else:
                                    raise APIError(f"API returned error: {error_msg}")
                            
                            if 'message' in chunk and 'content' in chunk['message']:
                                yield ChatResponse(
                                    content=chunk['message']['content'],
                                    role="assistant",
                                    done=chunk.get('done', False),
                                    metadata={
                                        'eval_count': chunk.get('eval_count'),
                                        'eval_duration': chunk.get('eval_duration'),
                                        'total_duration': chunk.get('total_duration'),
                                        'load_duration': chunk.get('load_duration')
                                    }
                                )
                                
                        except json.JSONDecodeError:
                            continue
                    
        except Exception as e:
            logger.error(f"Error in chat request: {str(e)}")
            raise

    async def summarize_text(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> Optional[str]:
        """
        Summarize text content using chat completion.
        
        Args:
            text: Text to summarize
            max_length: Optional maximum summary length
            
        Returns:
            Optional[str]: Generated summary
        """
        try:
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of the following text, focusing on the key points and main ideas:

{text}
"""
            if max_length:
                prompt += f"\nLimit the summary to approximately {max_length} characters."

            # Create message for chat
            messages = [{"role": "user", "content": prompt}]
            summary = ""

            # Get summary through chat completion
            async for response in self.chat(
                messages=messages,
                stream=True,
                temperature=0.3,  # Lower temperature for more focused summary
                max_tokens=max_length,
            ):
                if response and response.content:
                    summary += response.content

            return summary.strip() if summary else None

        except Exception as e:
            logger.warning(f"Summarization failed: {str(e)}")
            raise

    async def extract_keywords(
        self,
        text: str,
        max_keywords: int = 10
    ) -> List[str]:
        """
        Extract key terms and phrases from text.
        
        Args:
            text: Text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List[str]: Extracted keywords
        """
        try:
            # Create keyword extraction prompt
            prompt = f"""Please extract the most important keywords and key phrases from the following text.
Return them as a Python list of strings, with no additional formatting or explanation.
Limit the response to {max_keywords} most relevant terms.

Text:
{text}

Format the response as a valid Python list like this: ["keyword1", "keyword2", ...]"""

            # Create message for chat
            messages = [{"role": "user", "content": prompt}]
            keywords_text = ""

            # Get keywords through chat completion
            async for response in self.chat(
                messages=messages,
                stream=True,
                temperature=0.3,
                max_tokens=200
            ):
                if response and response.content:
                    keywords_text += response.content

            # Parse the response as a Python list
            try:
                # Clean up the response and evaluate it as a Python expression
                keywords_text = keywords_text.strip()
                if keywords_text.startswith("[") and keywords_text.endswith("]"):
                    keywords = eval(keywords_text)
                    if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
                        return keywords[:max_keywords]
                return []
            except Exception as e:
                logger.error(f"Failed to parse keywords response: {str(e)}")
                return []

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {str(e)}")
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            self.session = None

    def __str__(self) -> str:
        """String representation."""
        return f"OllamaClient(model={self.model_name}, embed_model={self.embed_model})"
