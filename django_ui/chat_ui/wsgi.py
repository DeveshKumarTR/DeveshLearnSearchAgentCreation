"""
WSGI config for chat_ui project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chat_ui.settings')

application = get_wsgi_application()
