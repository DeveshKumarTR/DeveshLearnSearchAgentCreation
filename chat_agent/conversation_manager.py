"""
Conversation Manager for handling chat sessions and history.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .ollama_client import OllamaMessage

logger = logging.getLogger(__name__)


@dataclass
class ConversationSession:
    """Represents a conversation session."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    title: str
    messages: List[OllamaMessage]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "title": self.title,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp
                }
                for msg in self.messages
            ],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationSession':
        """Create from dictionary."""
        messages = [
            OllamaMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=msg.get("timestamp")
            )
            for msg in data.get("messages", [])
        ]
        
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            title=data["title"],
            messages=messages,
            metadata=data.get("metadata", {})
        )


class ConversationManager:
    """
    Manages conversation sessions, history, and persistence.
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_sessions: int = 100,
        max_messages_per_session: int = 1000
    ):
        """
        Initialize conversation manager.
        
        Args:
            storage_path: Path to store conversation history
            max_sessions: Maximum number of sessions to keep
            max_messages_per_session: Maximum messages per session
        """
        self.storage_path = Path(storage_path) if storage_path else Path("conversations")
        self.storage_path.mkdir(exist_ok=True)
        
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        
        # In-memory cache of active sessions
        self.sessions: Dict[str, ConversationSession] = {}
        self.current_session_id: Optional[str] = None
        
        # Load existing sessions
        self._load_sessions()
        
    def create_session(
        self,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversation session.
        
        Args:
            title: Optional title for the session
            metadata: Optional metadata
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        session = ConversationSession(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            title=title or f"Conversation {now.strftime('%Y-%m-%d %H:%M')}",
            messages=[],
            metadata=metadata or {}
        )
        
        self.sessions[session_id] = session
        self.current_session_id = session_id
        
        # Save to disk
        self._save_session(session)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            ConversationSession or None if not found
        """
        return self.sessions.get(session_id)
    
    def set_current_session(self, session_id: str) -> bool:
        """
        Set the current active session.
        
        Args:
            session_id: Session ID to set as current
            
        Returns:
            True if successful, False if session not found
        """
        if session_id in self.sessions:
            self.current_session_id = session_id
            logger.info(f"Set current session to: {session_id}")
            return True
        return False
    
    def get_current_session(self) -> Optional[ConversationSession]:
        """
        Get the current active session.
        
        Returns:
            Current ConversationSession or None
        """
        if self.current_session_id:
            return self.sessions.get(self.current_session_id)
        return None
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """
        Add a message to a session.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            session_id: Session ID (uses current if not provided)
            timestamp: Message timestamp
            
        Returns:
            True if successful, False otherwise
        """
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            logger.warning(f"Session not found: {target_session_id}")
            return False
        
        session = self.sessions[target_session_id]
        
        # Check message limit
        if len(session.messages) >= self.max_messages_per_session:
            logger.warning(f"Session {target_session_id} has reached message limit")
            return False
        
        message = OllamaMessage(
            role=role,
            content=content,
            timestamp=timestamp or datetime.now().isoformat()
        )
        
        session.messages.append(message)
        session.updated_at = datetime.now()
        
        # Save to disk
        self._save_session(session)
        
        return True
    
    def get_messages(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
        role_filter: Optional[str] = None
    ) -> List[OllamaMessage]:
        """
        Get messages from a session.
        
        Args:
            session_id: Session ID (uses current if not provided)
            limit: Maximum number of messages to return
            role_filter: Filter by message role
            
        Returns:
            List of messages
        """
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return []
        
        session = self.sessions[target_session_id]
        messages = session.messages
        
        # Apply role filter
        if role_filter:
            messages = [msg for msg in messages if msg.role == role_filter]
        
        # Apply limit
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_conversation_context(
        self,
        session_id: Optional[str] = None,
        max_messages: int = 20
    ) -> List[OllamaMessage]:
        """
        Get recent conversation context for the model.
        
        Args:
            session_id: Session ID (uses current if not provided)
            max_messages: Maximum number of messages to include
            
        Returns:
            List of recent messages for context
        """
        return self.get_messages(
            session_id=session_id,
            limit=max_messages
        )
    
    def clear_session(self, session_id: Optional[str] = None) -> bool:
        """
        Clear messages from a session.
        
        Args:
            session_id: Session ID (uses current if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        target_session_id = session_id or self.current_session_id
        
        if not target_session_id or target_session_id not in self.sessions:
            return False
        
        session = self.sessions[target_session_id]
        session.messages = []
        session.updated_at = datetime.now()
        
        # Save to disk
        self._save_session(session)
        
        logger.info(f"Cleared session: {target_session_id}")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session completely.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        # Remove from memory
        del self.sessions[session_id]
        
        # Remove from disk
        session_file = self.storage_path / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
        
        # Update current session if necessary
        if self.current_session_id == session_id:
            self.current_session_id = None
        
        logger.info(f"Deleted session: {session_id}")
        return True
    
    def list_sessions(
        self,
        limit: Optional[int] = None,
        sort_by: str = "updated_at"
    ) -> List[ConversationSession]:
        """
        List conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            sort_by: Field to sort by ('created_at', 'updated_at', 'title')
            
        Returns:
            List of conversation sessions
        """
        sessions = list(self.sessions.values())
        
        # Sort sessions
        if sort_by == "created_at":
            sessions.sort(key=lambda s: s.created_at, reverse=True)
        elif sort_by == "updated_at":
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
        elif sort_by == "title":
            sessions.sort(key=lambda s: s.title)
        
        # Apply limit
        if limit:
            sessions = sessions[:limit]
        
        return sessions
    
    def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        role_filter: Optional[str] = None
    ) -> List[Tuple[str, OllamaMessage]]:
        """
        Search for messages containing a query.
        
        Args:
            query: Search query
            session_id: Session ID to search in (searches all if not provided)
            role_filter: Filter by message role
            
        Returns:
            List of tuples (session_id, message)
        """
        results = []
        query_lower = query.lower()
        
        sessions_to_search = {}
        if session_id and session_id in self.sessions:
            sessions_to_search[session_id] = self.sessions[session_id]
        else:
            sessions_to_search = self.sessions
        
        for sid, session in sessions_to_search.items():
            for message in session.messages:
                if role_filter and message.role != role_filter:
                    continue
                
                if query_lower in message.content.lower():
                    results.append((sid, message))
        
        return results
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session summary
        """
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        # Count messages by role
        role_counts = {}
        for message in session.messages:
            role_counts[message.role] = role_counts.get(message.role, 0) + 1
        
        total_chars = sum(len(msg.content) for msg in session.messages)
        
        return {
            "session_id": session_id,
            "title": session.title,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "message_count": len(session.messages),
            "role_counts": role_counts,
            "total_characters": total_chars,
            "metadata": session.metadata
        }
    
    def export_session(self, session_id: str, format: str = "json") -> Optional[str]:
        """
        Export a session in various formats.
        
        Args:
            session_id: Session ID to export
            format: Export format ('json', 'txt', 'md')
            
        Returns:
            Exported data as string or None if session not found
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if format == "json":
            return json.dumps(session.to_dict(), indent=2)
        
        elif format == "txt":
            lines = [f"Conversation: {session.title}"]
            lines.append(f"Created: {session.created_at}")
            lines.append(f"Updated: {session.updated_at}")
            lines.append("-" * 50)
            
            for message in session.messages:
                lines.append(f"\n[{message.role.upper()}]")
                lines.append(message.content)
            
            return "\n".join(lines)
        
        elif format == "md":
            lines = [f"# {session.title}"]
            lines.append(f"**Created:** {session.created_at}")
            lines.append(f"**Updated:** {session.updated_at}")
            lines.append("")
            
            for message in session.messages:
                if message.role == "user":
                    lines.append(f"## User")
                elif message.role == "assistant":
                    lines.append(f"## Assistant")
                else:
                    lines.append(f"## {message.role.title()}")
                
                lines.append(message.content)
                lines.append("")
            
            return "\n".join(lines)
        
        return None
    
    def _save_session(self, session: ConversationSession) -> None:
        """Save session to disk."""
        try:
            session_file = self.storage_path / f"{session.session_id}.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
    
    def _load_sessions(self) -> None:
        """Load sessions from disk."""
        try:
            for session_file in self.storage_path.glob("*.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    session = ConversationSession.from_dict(data)
                    self.sessions[session.session_id] = session
                    
                except Exception as e:
                    logger.error(f"Failed to load session from {session_file}: {e}")
            
            # Clean up old sessions if too many
            if len(self.sessions) > self.max_sessions:
                sessions_by_date = sorted(
                    self.sessions.values(),
                    key=lambda s: s.updated_at
                )
                
                sessions_to_remove = sessions_by_date[:-self.max_sessions]
                for session in sessions_to_remove:
                    self.delete_session(session.session_id)
            
            logger.info(f"Loaded {len(self.sessions)} conversation sessions")
            
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up sessions older than specified days.
        
        Args:
            days_old: Delete sessions older than this many days
            
        Returns:
            Number of sessions deleted
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        sessions_to_delete = []
        
        for session in self.sessions.values():
            if session.updated_at < cutoff_date:
                sessions_to_delete.append(session.session_id)
        
        deleted_count = 0
        for session_id in sessions_to_delete:
            if self.delete_session(session_id):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count
