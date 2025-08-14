"""
Main Chat Agent implementation integrating Ollama LLM with Search Agent.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Iterator
from datetime import datetime

from .ollama_client import OllamaClient, OllamaMessage, OllamaModel, OllamaResponse
from .conversation_manager import ConversationManager
from src.search_agent import SearchAgent
from src.utils import setup_logging

logger = logging.getLogger(__name__)


class ChatAgent:
    """
    Chat Agent that combines Ollama LLM with the existing Search Agent
    for enhanced conversational AI capabilities.
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        ollama_model: Union[str, OllamaModel] = OllamaModel.LLAMA2,
        enable_search: bool = True,
        conversation_storage_path: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the Chat Agent.
        
        Args:
            ollama_url: Ollama server URL
            ollama_model: Model to use for chat
            enable_search: Whether to enable search agent integration
            conversation_storage_path: Path to store conversations
            system_prompt: Default system prompt
        """
        setup_logging()
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            storage_path=conversation_storage_path
        )
        
        # Initialize search agent if enabled
        self.search_agent = None
        self.enable_search = enable_search
        if enable_search:
            try:
                self.search_agent = SearchAgent()
                logger.info("Search agent integration enabled")
            except Exception as e:
                logger.warning(f"Could not initialize search agent: {e}")
                self.enable_search = False
        
        # Default system prompt
        self.default_system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Create default session if none exists
        if not self.conversation_manager.sessions:
            self.conversation_manager.create_session(
                title="Default Chat Session",
                metadata={"created_by": "ChatAgent", "model": str(ollama_model)}
            )
        
        logger.info("Chat Agent initialized successfully")
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        base_prompt = """You are a helpful and intelligent AI assistant. You provide accurate, informative, and engaging responses to user questions and requests.

Key guidelines:
- Be helpful, honest, and harmless
- Provide clear and well-structured responses
- If you're unsure about something, acknowledge it
- Use examples when helpful
- Be conversational but professional"""

        if self.enable_search:
            base_prompt += """
- You have access to a knowledge base through search capabilities
- When relevant information might be available in the knowledge base, I will provide it to help you answer questions
- Use both your training knowledge and provided search results to give comprehensive answers"""
        
        return base_prompt
    
    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_search: bool = True,
        stream: bool = False,
        temperature: float = 0.7,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Chat with the agent.
        
        Args:
            message: User message
            session_id: Session ID (uses current if not provided)
            use_search: Whether to use search augmentation
            stream: Whether to stream the response
            temperature: Model temperature
            **kwargs: Additional model parameters
            
        Returns:
            Response string or iterator of response chunks if streaming
        """
        try:
            # Get or create session
            if session_id:
                self.conversation_manager.set_current_session(session_id)
            elif not self.conversation_manager.get_current_session():
                session_id = self.conversation_manager.create_session()
            
            # Add user message to conversation
            self.conversation_manager.add_message("user", message)
            
            # Prepare messages for the model
            messages = self._prepare_messages_for_model(use_search, message)
            
            # Generate response
            if stream:
                return self._stream_chat_response(messages, temperature, **kwargs)
            else:
                return self._generate_chat_response(messages, temperature, **kwargs)
                
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            
            # Still add the error to conversation for context
            try:
                self.conversation_manager.add_message("assistant", error_msg)
            except:
                pass
            
            return error_msg
    
    def _prepare_messages_for_model(
        self, 
        use_search: bool, 
        current_message: str
    ) -> List[OllamaMessage]:
        """Prepare messages for the model including search results if needed."""
        messages = []
        
        # Add system prompt
        messages.append(OllamaMessage(
            role="system",
            content=self.default_system_prompt
        ))
        
        # Get conversation context
        context_messages = self.conversation_manager.get_conversation_context(
            max_messages=10  # Limit context to last 10 messages
        )
        
        # Add search results if enabled and relevant
        search_context = ""
        if use_search and self.enable_search and self.search_agent:
            search_context = self._get_search_context(current_message)
        
        # Add conversation history (excluding the current message we just added)
        for msg in context_messages[:-1]:  # Exclude the last message (current user message)
            messages.append(msg)
        
        # Add current message with search context if available
        final_message = current_message
        if search_context:
            final_message = f"""User Question: {current_message}

Relevant Information from Knowledge Base:
{search_context}

Please provide a comprehensive answer using both your knowledge and the information from the knowledge base above."""
        
        messages.append(OllamaMessage(
            role="user",
            content=final_message
        ))
        
        return messages
    
    def _get_search_context(self, query: str) -> str:
        """Get relevant information from the search agent."""
        try:
            search_results = self.search_agent.search(
                query=query,
                k=3,  # Get top 3 results
                return_sources=True
            )
            
            if search_results['metadata']['search_successful'] and search_results['sources']:
                context_parts = []
                for i, source in enumerate(search_results['sources'][:3], 1):
                    score = source.get('similarity_score', 0)
                    content = source['content'][:500]  # Limit content length
                    context_parts.append(f"Source {i} (Relevance: {score:.2f}):\n{content}")
                
                return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.warning(f"Search augmentation failed: {e}")
        
        return ""
    
    def _generate_chat_response(
        self, 
        messages: List[OllamaMessage], 
        temperature: float,
        **kwargs
    ) -> str:
        """Generate a non-streaming chat response."""
        response = self.ollama_client.chat(
            messages=messages,
            stream=False,
            temperature=temperature,
            **kwargs
        )
        
        response_text = response.message
        
        # Add assistant response to conversation
        self.conversation_manager.add_message("assistant", response_text)
        
        return response_text
    
    def _stream_chat_response(
        self, 
        messages: List[OllamaMessage], 
        temperature: float,
        **kwargs
    ) -> Iterator[str]:
        """Generate a streaming chat response."""
        full_response = ""
        
        for chunk in self.ollama_client.chat(
            messages=messages,
            stream=True,
            temperature=temperature,
            **kwargs
        ):
            chunk_text = chunk.message
            full_response += chunk_text
            yield chunk_text
        
        # Add complete assistant response to conversation
        self.conversation_manager.add_message("assistant", full_response)
    
    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            title: Optional title for the conversation
            
        Returns:
            New session ID
        """
        session_id = self.conversation_manager.create_session(
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            metadata={
                "created_by": "ChatAgent",
                "model": self.ollama_client.model,
                "search_enabled": self.enable_search
            }
        )
        
        logger.info(f"Started new conversation: {session_id}")
        return session_id
    
    def get_conversation_history(
        self, 
        session_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history.
        
        Args:
            session_id: Session ID (uses current if not provided)
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = self.conversation_manager.get_messages(
            session_id=session_id,
            limit=limit
        )
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for msg in messages
        ]
    
    def list_conversations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        sessions = self.conversation_manager.list_sessions(limit=limit)
        
        return [
            self.conversation_manager.get_session_summary(session.session_id)
            for session in sessions
        ]
    
    def switch_conversation(self, session_id: str) -> bool:
        """
        Switch to a different conversation.
        
        Args:
            session_id: Session ID to switch to
            
        Returns:
            True if successful, False otherwise
        """
        return self.conversation_manager.set_current_session(session_id)
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self.conversation_manager.delete_session(session_id)
    
    def clear_current_conversation(self) -> bool:
        """
        Clear the current conversation.
        
        Returns:
            True if successful, False otherwise
        """
        return self.conversation_manager.clear_session()
    
    def search_conversations(
        self, 
        query: str,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search through conversation history.
        
        Args:
            query: Search query
            session_id: Session ID to search in (all sessions if not provided)
            
        Returns:
            List of matching messages with context
        """
        results = self.conversation_manager.search_messages(
            query=query,
            session_id=session_id
        )
        
        return [
            {
                "session_id": sid,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp
            }
            for sid, msg in results
        ]
    
    def export_conversation(
        self, 
        session_id: str, 
        format: str = "json"
    ) -> Optional[str]:
        """
        Export a conversation in various formats.
        
        Args:
            session_id: Session ID to export
            format: Export format ('json', 'txt', 'md')
            
        Returns:
            Exported data as string or None if session not found
        """
        return self.conversation_manager.export_session(session_id, format)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available Ollama models.
        
        Returns:
            List of available models
        """
        return self.ollama_client.list_models()
    
    def switch_model(self, model: Union[str, OllamaModel]) -> bool:
        """
        Switch to a different Ollama model.
        
        Args:
            model: Model name or OllamaModel enum
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = model.value if isinstance(model, OllamaModel) else model
            
            # Check if model is available
            if not self.ollama_client.is_model_available(model_name):
                logger.warning(f"Model {model_name} not available, attempting to pull...")
                if not self.ollama_client.pull_model(model_name):
                    return False
            
            self.ollama_client.set_model(model)
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat agent statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "current_model": self.ollama_client.model,
            "search_enabled": self.enable_search,
            "ollama_health": self.ollama_client.health_check(),
            "total_sessions": len(self.conversation_manager.sessions),
            "current_session": self.conversation_manager.current_session_id,
        }
        
        # Add search agent stats if available
        if self.search_agent:
            try:
                search_stats = self.search_agent.get_stats()
                stats["search_agent"] = search_stats
            except Exception as e:
                logger.warning(f"Could not get search agent stats: {e}")
        
        # Add session stats
        current_session = self.conversation_manager.get_current_session()
        if current_session:
            stats["current_session_stats"] = self.conversation_manager.get_session_summary(
                current_session.session_id
            )
        
        return stats
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Health status of each component
        """
        health = {
            "ollama": self.ollama_client.health_check(),
            "conversation_manager": True,  # Always available
            "search_agent": False
        }
        
        if self.search_agent:
            try:
                # Try a simple operation to test search agent
                self.search_agent.get_stats()
                health["search_agent"] = True
            except Exception:
                health["search_agent"] = False
        
        return health
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a new system prompt.
        
        Args:
            prompt: New system prompt
        """
        self.default_system_prompt = prompt
        logger.info("Updated system prompt")
    
    def add_documents_to_search(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[List[str]]:
        """
        Add documents to the search agent knowledge base.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            
        Returns:
            List of document IDs or None if search not enabled
        """
        if not self.enable_search or not self.search_agent:
            logger.warning("Search agent not available")
            return None
        
        try:
            return self.search_agent.add_documents(documents, metadata)
        except Exception as e:
            logger.error(f"Failed to add documents to search: {e}")
            return None
