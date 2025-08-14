"""
Chat App URL configuration.
"""

from django.urls import path
from . import views

app_name = 'chat_app'

urlpatterns = [
    # Web interface
    path('', views.index, name='index'),
    
    # API endpoints
    path('api/health/', views.health_check, name='health_check'),
    path('api/conversations/', views.list_conversations, name='list_conversations'),
    path('api/conversations/create/', views.create_conversation, name='create_conversation'),
    path('api/conversations/<uuid:session_id>/', views.get_conversation, name='get_conversation'),
    path('api/conversations/<uuid:session_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('api/conversations/<uuid:session_id>/chat/', views.chat_message, name='chat_message'),
    
    # Model management
    path('api/models/', views.get_models, name='get_models'),
    path('api/models/switch/', views.switch_model, name='switch_model'),
    
    # Knowledge base management
    path('api/documents/add/', views.add_documents, name='add_documents'),
]
