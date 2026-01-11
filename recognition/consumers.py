"""
WebSocket consumer for real-time updates.
"""

import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging

logger = logging.getLogger(__name__)


class RecognitionConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer to send real-time face recognition updates to frontend.
    """
    
    async def connect(self):
        """Handle WebSocket connection."""
        self.group_name = 'recognition_updates'
        
        # Join group
        await self.channel_layer.group_add(
            self.group_name,
            self.channel_name
        )
        
        await self.accept()
        logger.info(f"WebSocket connected: {self.channel_name}")
    
    async def disconnect(self, close_code):
        """Handle WebSocket disconnection."""
        # Leave group
        await self.channel_layer.group_discard(
            self.group_name,
            self.channel_name
        )
        logger.info(f"WebSocket disconnected: {self.channel_name}")
    
    async def receive(self, text_data):
        """Handle messages from WebSocket (currently not used)."""
        pass
    
    async def frame_update(self, event):
        """
        Send video frame update to WebSocket.
        
        Args:
            event: Dictionary with 'camera_number' and 'frame' (base64 encoded)
        """
        await self.send(text_data=json.dumps({
            'type': 'frame',
            'camera_number': event['camera_number'],
            'frame': event['frame']
        }))
    
    async def detection_alert(self, event):
        """
        Send detection alert to WebSocket.
        
        Args:
            event: Dictionary with detection details
        """
        await self.send(text_data=json.dumps({
            'type': 'detection',
            'person_name': event['person_name'],
            'camera_number': event['camera_number'],
            'confidence': event['confidence'],
            'timestamp': event['timestamp'],
            'detection_id': event['detection_id']
        }))
    
    async def camera_status(self, event):
        """
        Send camera status update to WebSocket.
        
        Args:
            event: Dictionary with camera status
        """
        await self.send(text_data=json.dumps({
            'type': 'camera_status',
            'cameras': event['cameras']
        }))