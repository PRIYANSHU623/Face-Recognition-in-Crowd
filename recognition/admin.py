"""
Django admin configuration for face recognition models.
"""

from django.contrib import admin
from .models import Person, DetectionLog, CameraStatus


@admin.register(Person)
class PersonAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active', 'created_at', 'updated_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name']
    readonly_fields = ['created_at', 'updated_at', 'face_embedding']


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ['person_name', 'camera_number', 'detection_time', 'confidence_score']
    list_filter = ['person_name', 'camera_number', 'detection_time']
    search_fields = ['person_name']
    date_hierarchy = 'detection_time'
    readonly_fields = ['detection_time']


@admin.register(CameraStatus)
class CameraStatusAdmin(admin.ModelAdmin):
    list_display = ['camera_number', 'is_active', 'resolution', 'last_checked']
    list_filter = ['is_active']