"""
Chat App models.
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import uuid


class ConversationSession(models.Model):
    """Model for storing conversation sessions."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    title = models.CharField(max_length=200, default="New Conversation")
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = "Conversation Session"
        verbose_name_plural = "Conversation Sessions"
    
    def __str__(self):
        return f"{self.title} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
    
    @property
    def message_count(self):
        return self.messages.count()
    
    @property
    def last_message(self):
        return self.messages.last()


class ChatMessage(models.Model):
    """Model for storing individual chat messages."""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
        ('system', 'System'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ConversationSession, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    timestamp = models.DateTimeField(default=timezone.now)
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['timestamp']
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"
    
    def __str__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"{self.role}: {content_preview}"


class SearchQuery(models.Model):
    """Model for tracking search queries and results."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(ConversationSession, on_delete=models.CASCADE, related_name='searches')
    query = models.TextField()
    results_count = models.IntegerField(default=0)
    search_successful = models.BooleanField(default=False)
    timestamp = models.DateTimeField(default=timezone.now)
    execution_time = models.FloatField(null=True, blank=True)  # in seconds
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Search Query"
        verbose_name_plural = "Search Queries"
    
    def __str__(self):
        query_preview = self.query[:50] + "..." if len(self.query) > 50 else self.query
        return f"Search: {query_preview}"


class SystemMetrics(models.Model):
    """Model for storing system metrics and health data."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    timestamp = models.DateTimeField(default=timezone.now)
    ollama_health = models.BooleanField(default=False)
    search_agent_health = models.BooleanField(default=False)
    active_sessions = models.IntegerField(default=0)
    total_messages = models.IntegerField(default=0)
    memory_usage = models.FloatField(null=True, blank=True)  # in MB
    response_time = models.FloatField(null=True, blank=True)  # in seconds
    metadata = models.JSONField(default=dict, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "System Metrics"
        verbose_name_plural = "System Metrics"
    
    def __str__(self):
        return f"Metrics {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
