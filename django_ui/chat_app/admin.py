"""
Chat App admin configuration.
"""

from django.contrib import admin
from .models import ConversationSession, ChatMessage, SearchQuery, SystemMetrics


@admin.register(ConversationSession)
class ConversationSessionAdmin(admin.ModelAdmin):
    list_display = ['title', 'user', 'created_at', 'updated_at', 'message_count', 'is_active']
    list_filter = ['created_at', 'is_active', 'user']
    search_fields = ['title', 'user__username']
    readonly_fields = ['id', 'created_at', 'message_count']
    
    def message_count(self, obj):
        return obj.message_count
    message_count.short_description = 'Messages'


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ['session', 'role', 'content_preview', 'timestamp']
    list_filter = ['role', 'timestamp', 'session']
    search_fields = ['content', 'session__title']
    readonly_fields = ['id', 'timestamp']
    
    def content_preview(self, obj):
        return obj.content[:100] + "..." if len(obj.content) > 100 else obj.content
    content_preview.short_description = 'Content Preview'


@admin.register(SearchQuery)
class SearchQueryAdmin(admin.ModelAdmin):
    list_display = ['session', 'query_preview', 'results_count', 'search_successful', 'timestamp', 'execution_time']
    list_filter = ['search_successful', 'timestamp']
    search_fields = ['query', 'session__title']
    readonly_fields = ['id', 'timestamp']
    
    def query_preview(self, obj):
        return obj.query[:100] + "..." if len(obj.query) > 100 else obj.query
    query_preview.short_description = 'Query Preview'


@admin.register(SystemMetrics)
class SystemMetricsAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'ollama_health', 'search_agent_health', 'active_sessions', 'total_messages', 'response_time']
    list_filter = ['ollama_health', 'search_agent_health', 'timestamp']
    readonly_fields = ['id', 'timestamp']
