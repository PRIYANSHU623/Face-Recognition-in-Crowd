# """
# ASGI config for FaceRecognitionSystem project.

# It exposes the ASGI callable as a module-level variable named ``application``.

# For more information on this file, see
# https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
# """

# import os

# from django.core.asgi import get_asgi_application

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FaceRecognitionSystem.settings')

# application = get_asgi_application()

"""
ASGI config for face_recognition_system project.
Exposes the ASGI callable as a module-level variable named ``application``.

NOTE: Make sure routing.py exists in face_recognition_system/ folder
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FaceRecognitionSystem.settings')

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
django_asgi_app = get_asgi_application()

# Import routing after django setup
# If you see a warning here, make sure routing.py exists in face_recognition_system/ folder
try:
    from FaceRecognitionSystem.routing import websocket_urlpatterns
except ImportError:
    # If routing.py doesn't exist yet, create it with the content from routing_py artifact
    print("WARNING: face_recognition_system/routing.py not found!")
    print("Please create it with the WebSocket URL patterns.")
    websocket_urlpatterns = []

from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from channels.security.websocket import AllowedHostsOriginValidator

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(
                websocket_urlpatterns
            )
        )
    ),
})