"""
Chat App views.
"""

import json
import logging
import time
from typing import Dict, Any
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils.decorators import method_decorator
from django.views import View
from django.core.paginator import Paginator
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
import sys
import os

# Add project root to path to import chat agent
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from chat_agent.chat_agent import ChatAgent
from chat_agent.ollama_client import OllamaModel
from .models import ConversationSession, ChatMessage, SearchQuery, SystemMetrics

logger = logging.getLogger(__name__)

# Global chat agent instance
_chat_agent = None


def get_chat_agent():
    """Get or create chat agent instance."""
    global _chat_agent
    if _chat_agent is None:
        try:
            config = settings.CHAT_AGENT_CONFIG
            _chat_agent = ChatAgent(
                ollama_url=config['OLLAMA_URL'],
                ollama_model=config['OLLAMA_MODEL'],
                enable_search=config['ENABLE_SEARCH'],
                conversation_storage_path=config['CONVERSATION_STORAGE_PATH']
            )
            logger.info("Chat agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chat agent: {e}")
            _chat_agent = None
    return _chat_agent


def index(request):
    """Main chat interface."""
    return render(request, 'chat_app/index.html')


@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """Health check endpoint."""
    chat_agent = get_chat_agent()
    
    if chat_agent:
        health = chat_agent.health_check()
        stats = chat_agent.get_stats()
        
        # Save metrics to database
        try:
            SystemMetrics.objects.create(
                ollama_health=health.get('ollama', False),
                search_agent_health=health.get('search_agent', False),
                active_sessions=stats.get('total_sessions', 0),
                total_messages=0,  # Could be calculated from database
                metadata={
                    'stats': stats,
                    'health': health
                }
            )
        except Exception as e:
            logger.warning(f"Failed to save metrics: {e}")
        
        return Response({
            'status': 'healthy' if all(health.values()) else 'degraded',
            'health': health,
            'stats': stats
        })
    else:
        return Response({
            'status': 'unhealthy',
            'error': 'Chat agent not available'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@api_view(['POST'])
@permission_classes([AllowAny])
def create_conversation(request):
    """Create a new conversation session."""
    try:
        data = request.data
        title = data.get('title', 'New Conversation')
        
        # Create Django model
        session = ConversationSession.objects.create(
            title=title,
            user=request.user if request.user.is_authenticated else None,
            metadata={'created_via': 'api'}
        )
        
        # Create session in chat agent
        chat_agent = get_chat_agent()
        if chat_agent:
            agent_session_id = chat_agent.start_new_conversation(title)
            session.metadata['agent_session_id'] = agent_session_id
            session.save()
        
        return Response({
            'session_id': str(session.id),
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'message_count': 0
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        return Response({
            'error': 'Failed to create conversation'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def list_conversations(request):
    """List conversation sessions."""
    try:
        queryset = ConversationSession.objects.filter(is_active=True)
        
        # Add user filter if authenticated
        if request.user.is_authenticated:
            queryset = queryset.filter(user=request.user)
        
        # Pagination
        page_size = request.GET.get('page_size', 20)
        paginator = Paginator(queryset, page_size)
        page_number = request.GET.get('page', 1)
        page_obj = paginator.get_page(page_number)
        
        conversations = []
        for session in page_obj:
            conversations.append({
                'session_id': str(session.id),
                'title': session.title,
                'created_at': session.created_at.isoformat(),
                'updated_at': session.updated_at.isoformat(),
                'message_count': session.message_count,
                'last_message': session.last_message.content[:100] if session.last_message else None
            })
        
        return Response({
            'conversations': conversations,
            'total': paginator.count,
            'page': page_obj.number,
            'has_next': page_obj.has_next(),
            'has_previous': page_obj.has_previous()
        })
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        return Response({
            'error': 'Failed to list conversations'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_conversation(request, session_id):
    """Get conversation details and messages."""
    try:
        session = get_object_or_404(ConversationSession, id=session_id, is_active=True)
        
        # Get messages
        messages = session.messages.all()
        message_data = []
        
        for msg in messages:
            message_data.append({
                'id': str(msg.id),
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat(),
                'metadata': msg.metadata
            })
        
        return Response({
            'session_id': str(session.id),
            'title': session.title,
            'created_at': session.created_at.isoformat(),
            'updated_at': session.updated_at.isoformat(),
            'messages': message_data,
            'metadata': session.metadata
        })
        
    except Exception as e:
        logger.error(f"Failed to get conversation {session_id}: {e}")
        return Response({
            'error': 'Conversation not found'
        }, status=status.HTTP_404_NOT_FOUND)


@csrf_exempt
@require_http_methods(["POST"])
def chat_message(request, session_id):
    """Send a chat message and get response."""
    try:
        # Parse request data
        data = json.loads(request.body)
        message = data.get('message', '').strip()
        use_search = data.get('use_search', True)
        stream = data.get('stream', False)
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        # Get or create session
        try:
            session = ConversationSession.objects.get(id=session_id, is_active=True)
        except ConversationSession.DoesNotExist:
            return JsonResponse({'error': 'Session not found'}, status=404)
        
        # Get chat agent
        chat_agent = get_chat_agent()
        if not chat_agent:
            return JsonResponse({'error': 'Chat agent not available'}, status=503)
        
        # Save user message to database
        user_message = ChatMessage.objects.create(
            session=session,
            role='user',
            content=message,
            metadata={'use_search': use_search}
        )
        
        start_time = time.time()
        
        try:
            # Switch to the correct session in chat agent
            agent_session_id = session.metadata.get('agent_session_id')
            if agent_session_id:
                chat_agent.switch_conversation(agent_session_id)
            
            # Generate response
            if stream:
                return _stream_chat_response(chat_agent, message, session, use_search, start_time)
            else:
                response_text = chat_agent.chat(
                    message=message,
                    use_search=use_search,
                    stream=False,
                    temperature=0.7
                )
                
                # Save assistant response to database
                assistant_message = ChatMessage.objects.create(
                    session=session,
                    role='assistant',
                    content=response_text,
                    metadata={
                        'use_search': use_search,
                        'response_time': time.time() - start_time
                    }
                )
                
                # Record search query if search was used
                if use_search:
                    SearchQuery.objects.create(
                        session=session,
                        query=message,
                        search_successful=True,
                        execution_time=time.time() - start_time
                    )
                
                return JsonResponse({
                    'message_id': str(assistant_message.id),
                    'response': response_text,
                    'timestamp': assistant_message.timestamp.isoformat(),
                    'response_time': time.time() - start_time
                })
                
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            
            # Save error message
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            assistant_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=error_message,
                metadata={'error': True, 'error_message': str(e)}
            )
            
            return JsonResponse({
                'message_id': str(assistant_message.id),
                'response': error_message,
                'timestamp': assistant_message.timestamp.isoformat(),
                'error': True
            }, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error(f"Chat message failed: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)


def _stream_chat_response(chat_agent, message, session, use_search, start_time):
    """Handle streaming chat response."""
    def generate():
        try:
            full_response = ""
            
            for chunk in chat_agent.chat(
                message=message,
                use_search=use_search,
                stream=True,
                temperature=0.7
            ):
                full_response += chunk
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Save complete response to database
            assistant_message = ChatMessage.objects.create(
                session=session,
                role='assistant',
                content=full_response,
                metadata={
                    'use_search': use_search,
                    'response_time': time.time() - start_time,
                    'streamed': True
                }
            )
            
            # Send completion signal
            yield f"data: {json.dumps({'complete': True, 'message_id': str(assistant_message.id)})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    response = StreamingHttpResponse(
        generate(),
        content_type='text/event-stream'
    )
    response['Cache-Control'] = 'no-cache'
    response['Connection'] = 'keep-alive'
    return response


@api_view(['DELETE'])
@permission_classes([AllowAny])
def delete_conversation(request, session_id):
    """Delete a conversation session."""
    try:
        session = get_object_or_404(ConversationSession, id=session_id)
        
        # Delete from chat agent
        chat_agent = get_chat_agent()
        if chat_agent:
            agent_session_id = session.metadata.get('agent_session_id')
            if agent_session_id:
                chat_agent.delete_conversation(agent_session_id)
        
        # Mark as inactive instead of deleting
        session.is_active = False
        session.save()
        
        return Response({'message': 'Conversation deleted successfully'})
        
    except Exception as e:
        logger.error(f"Failed to delete conversation {session_id}: {e}")
        return Response({
            'error': 'Failed to delete conversation'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
@permission_classes([AllowAny])
def get_models(request):
    """Get available Ollama models."""
    try:
        chat_agent = get_chat_agent()
        if not chat_agent:
            return Response({'error': 'Chat agent not available'}, status=503)
        
        models = chat_agent.get_available_models()
        
        return Response({
            'models': models,
            'current_model': chat_agent.ollama_client.model
        })
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return Response({
            'error': 'Failed to get models'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def switch_model(request):
    """Switch to a different Ollama model."""
    try:
        data = request.data
        model_name = data.get('model')
        
        if not model_name:
            return Response({'error': 'Model name is required'}, status=400)
        
        chat_agent = get_chat_agent()
        if not chat_agent:
            return Response({'error': 'Chat agent not available'}, status=503)
        
        success = chat_agent.switch_model(model_name)
        
        if success:
            return Response({
                'message': f'Switched to model: {model_name}',
                'current_model': chat_agent.ollama_client.model
            })
        else:
            return Response({
                'error': f'Failed to switch to model: {model_name}'
            }, status=400)
            
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        return Response({
            'error': 'Failed to switch model'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([AllowAny])
def add_documents(request):
    """Add documents to the search agent knowledge base."""
    try:
        data = request.data
        documents = data.get('documents', [])
        metadata = data.get('metadata', [])
        
        if not documents:
            return Response({'error': 'Documents are required'}, status=400)
        
        chat_agent = get_chat_agent()
        if not chat_agent:
            return Response({'error': 'Chat agent not available'}, status=503)
        
        doc_ids = chat_agent.add_documents_to_search(documents, metadata)
        
        if doc_ids:
            return Response({
                'message': f'Added {len(doc_ids)} documents to knowledge base',
                'document_ids': doc_ids
            })
        else:
            return Response({
                'error': 'Failed to add documents or search not enabled'
            }, status=400)
            
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        return Response({
            'error': 'Failed to add documents'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
