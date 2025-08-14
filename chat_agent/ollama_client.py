"""
Ollama LLM Client for local language model integration.
"""

import json
import logging
import requests
from typing import Dict, List, Any, Optional, Iterator, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OllamaModel(Enum):
    """Available Ollama models."""
    LLAMA2 = "llama2"
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"
    CODELLAMA = "codellama"
    MISTRAL = "mistral"
    NEURAL_CHAT = "neural-chat"
    STARLING = "starling-lm"
    VICUNA = "vicuna"
    ORCA_MINI = "orca-mini"


@dataclass
class OllamaMessage:
    """Represents a message in the conversation."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[str] = None


@dataclass
class OllamaResponse:
    """Represents a response from Ollama."""
    message: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class OllamaClient:
    """
    Client for interacting with Ollama local LLM server.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: Union[str, OllamaModel] = OllamaModel.LLAMA2,
        timeout: int = 60
    ):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model.value if isinstance(model, OllamaModel) else model
        self.timeout = timeout
        self.session = requests.Session()
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama server at {self.base_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to Ollama server: {e}")
            logger.info("Make sure Ollama is running with: ollama serve")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the Ollama server.
        
        Returns:
            List of available models
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get('models', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull a model if it's not already available.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {"name": model_name}
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=data,
                timeout=300  # Longer timeout for model download
            )
            response.raise_for_status()
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[List[int]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[OllamaResponse, Iterator[OllamaResponse]]:
        """
        Generate text using the Ollama model.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            context: Optional context for conversation continuity
            stream: Whether to stream the response
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            OllamaResponse or iterator of responses if streaming
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        if context:
            data["context"] = context
            
        # Add optional parameters
        options = {}
        for key in ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed']:
            if key in kwargs:
                options[key] = kwargs[key]
        
        if options:
            data["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_response(response)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def chat(
        self,
        messages: List[OllamaMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[OllamaResponse, Iterator[OllamaResponse]]:
        """
        Chat with the model using conversation history.
        
        Args:
            messages: List of conversation messages
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            OllamaResponse or iterator of responses if streaming
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content} 
                for msg in messages
            ],
            "stream": stream
        }
        
        # Add optional parameters
        options = {}
        for key in ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed']:
            if key in kwargs:
                options[key] = kwargs[key]
        
        if options:
            data["options"] = options
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=data,
                timeout=self.timeout,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_chat_response(response)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def _handle_response(self, response: requests.Response) -> OllamaResponse:
        """Handle non-streaming response."""
        data = response.json()
        return OllamaResponse(
            message=data.get('response', ''),
            model=data.get('model', self.model),
            done=data.get('done', True),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )
    
    def _handle_chat_response(self, response: requests.Response) -> OllamaResponse:
        """Handle chat response."""
        data = response.json()
        message_content = ""
        if 'message' in data:
            message_content = data['message'].get('content', '')
        
        return OllamaResponse(
            message=message_content,
            model=data.get('model', self.model),
            done=data.get('done', True),
            total_duration=data.get('total_duration'),
            load_duration=data.get('load_duration'),
            prompt_eval_count=data.get('prompt_eval_count'),
            prompt_eval_duration=data.get('prompt_eval_duration'),
            eval_count=data.get('eval_count'),
            eval_duration=data.get('eval_duration')
        )
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[OllamaResponse]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    
                    # Handle both generate and chat streaming responses
                    message_content = ""
                    if 'response' in data:
                        message_content = data['response']
                    elif 'message' in data:
                        message_content = data['message'].get('content', '')
                    
                    yield OllamaResponse(
                        message=message_content,
                        model=data.get('model', self.model),
                        done=data.get('done', False),
                        total_duration=data.get('total_duration'),
                        load_duration=data.get('load_duration'),
                        prompt_eval_count=data.get('prompt_eval_count'),
                        prompt_eval_duration=data.get('prompt_eval_duration'),
                        eval_count=data.get('eval_count'),
                        eval_duration=data.get('eval_duration')
                    )
                except json.JSONDecodeError:
                    continue
    
    def embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings for text (if supported by model).
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        data = {
            "model": self.model,
            "prompt": text
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/embeddings",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('embedding', [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Embeddings generation failed: {e}")
            return []
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model (defaults to current model)
            
        Returns:
            Model information
        """
        model = model_name or self.model
        data = {"name": model}
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info for {model}: {e}")
            return {}
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available, False otherwise
        """
        models = self.list_models()
        return any(model['name'] == model_name for model in models)
    
    def set_model(self, model: Union[str, OllamaModel]) -> None:
        """
        Switch to a different model.
        
        Args:
            model: Model name or OllamaModel enum
        """
        self.model = model.value if isinstance(model, OllamaModel) else model
        logger.info(f"Switched to model: {self.model}")
    
    def health_check(self) -> bool:
        """
        Check if Ollama server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
