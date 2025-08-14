"""
Chat Agent Package using Ollama LLM

A conversational AI agent that integrates with the existing search agent
and provides chat capabilities using Ollama local LLM.
"""

__version__ = "1.0.0"
__author__ = "Search Agent Team"

from .chat_agent import ChatAgent
from .ollama_client import OllamaClient
from .conversation_manager import ConversationManager

__all__ = ["ChatAgent", "OllamaClient", "ConversationManager"]
