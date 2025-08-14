"""
WebSocket consumers for real-time chat.
"""

import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.shortcuts import get_object_or_404
from .models import ConversationSession, ChatMessage
from .views import get_chat_agent

logger = logging.getLogger(__name__)


class ChatConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for real-time chat."""
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.session_id = self.scope['url_route']['kwargs']['session_id']
        self.room_group_name = f'chat_{self.session_id}'
        
        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        
        await self.accept()
        logger.info(f"WebSocket connected for session {self.session_id}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        logger.info(f"WebSocket disconnected for session {self.session_id}")
    
    async def receive(self, text_data):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(text_data)
            message_type = data.get('type', 'chat')
            
            if message_type == 'chat':
                await self.handle_chat_message(data)
            elif message_type == 'typing':
                await self.handle_typing_indicator(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON received in WebSocket")
            await self.send_error("Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error(f"Error processing message: {str(e)}")
    
    async def handle_chat_message(self, data):
        """Handle chat message."""
        try:
            message = data.get('message', '').strip()
            use_search = data.get('use_search', True)
            
            if not message:
                await self.send_error("Message cannot be empty")
                return
            
            # Verify session exists
            session = await self.get_session()
            if not session:
                await self.send_error("Session not found")
                return
            
            # Save user message
            user_message = await self.save_message(session, 'user', message, {'use_search': use_search})
            
            # Send user message to group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': {
                        'id': str(user_message.id),
                        'role': 'user',
                        'content': message,
                        'timestamp': user_message.timestamp.isoformat()
                    }
                }
            )
            
            # Generate AI response
            await self.generate_ai_response(session, message, use_search)
            
        except Exception as e:
            logger.error(f"Error handling chat message: {e}")
            await self.send_error(f"Error processing chat: {str(e)}")
    
    async def generate_ai_response(self, session, message, use_search):
        """Generate AI response using the chat agent."""
        try:
            # Get chat agent
            chat_agent = await database_sync_to_async(get_chat_agent)()
            if not chat_agent:
                await self.send_error("Chat agent not available")
                return
            
            # Switch to correct session
            agent_session_id = session.metadata.get('agent_session_id')
            if agent_session_id:
                await database_sync_to_async(chat_agent.switch_conversation)(agent_session_id)
            
            # Generate streaming response
            response_chunks = []
            
            async for chunk in self.stream_chat_response(chat_agent, message, use_search):
                response_chunks.append(chunk)
                
                # Send chunk to group
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'chat_chunk',
                        'chunk': chunk
                    }
                )
            
            # Save complete response
            full_response = ''.join(response_chunks)
            assistant_message = await self.save_message(
                session, 
                'assistant', 
                full_response, 
                {'use_search': use_search, 'streamed': True}
            )
            
            # Send completion signal
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_complete',
                    'message': {
                        'id': str(assistant_message.id),
                        'role': 'assistant',
                        'content': full_response,
                        'timestamp': assistant_message.timestamp.isoformat()
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            await self.send_error(f"Error generating response: {str(e)}")
    
    async def stream_chat_response(self, chat_agent, message, use_search):
        """Stream chat response from the agent."""
        def _generate():
            return chat_agent.chat(
                message=message,
                use_search=use_search,
                stream=True,
                temperature=0.7
            )
        
        # Run the streaming chat in a thread to avoid blocking
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the chat generation in a thread
            future = executor.submit(_generate)
            
            # Stream chunks as they become available
            generator = await loop.run_in_executor(None, lambda: future.result())
            
            for chunk in generator:
                yield chunk
    
    async def handle_typing_indicator(self, data):
        """Handle typing indicator."""
        is_typing = data.get('is_typing', False)
        
        # Broadcast typing indicator to group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'typing_indicator',
                'is_typing': is_typing,
                'user': self.scope.get('user', {}).get('username', 'Anonymous')
            }
        )
    
    async def chat_message(self, event):
        """Send chat message to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'message',
            'message': event['message']
        }))
    
    async def chat_chunk(self, event):
        """Send chat chunk to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'chunk',
            'chunk': event['chunk']
        }))
    
    async def chat_complete(self, event):
        """Send chat completion signal to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'complete',
            'message': event['message']
        }))
    
    async def typing_indicator(self, event):
        """Send typing indicator to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'typing',
            'is_typing': event['is_typing'],
            'user': event['user']
        }))
    
    async def send_error(self, error_message):
        """Send error message to WebSocket."""
        await self.send(text_data=json.dumps({
            'type': 'error',
            'error': error_message
        }))
    
    @database_sync_to_async
    def get_session(self):
        """Get conversation session from database."""
        try:
            return ConversationSession.objects.get(id=self.session_id, is_active=True)
        except ConversationSession.DoesNotExist:
            return None
    
    @database_sync_to_async
    def save_message(self, session, role, content, metadata):
        """Save message to database."""
        return ChatMessage.objects.create(
            session=session,
            role=role,
            content=content,
            metadata=metadata
        )
