"""
Chat App Django application.
"""

from django.apps import AppConfig


class ChatAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'chat_app'
    verbose_name = 'Chat Application'
    
    def ready(self):
        """Initialize the chat agent when the app is ready."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Chat App is ready")
