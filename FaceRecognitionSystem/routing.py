"""
WebSocket URL routing configuration.
"""

from django.urls import re_path
from recognition import consumers

websocket_urlpatterns = [
    re_path(r'ws/recognition/$', consumers.RecognitionConsumer.as_asgi()),
]