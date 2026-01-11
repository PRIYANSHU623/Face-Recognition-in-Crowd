"""
Django models for face recognition system.
"""

from django.db import models
from django.utils import timezone


class Person(models.Model):
    """
    Model to store known persons with their reference photos.
    """
    name = models.CharField(max_length=100, unique=True)
    uploaded_image = models.ImageField(upload_to='known_faces/')
    face_embedding = models.JSONField(null=True, blank=True)  # Store face embeddings
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'People'

    def __str__(self):
        return self.name


class DetectionLog(models.Model):
    """
    Model to log face detection events.
    """
    person_name = models.CharField(max_length=100)
    camera_number = models.IntegerField()
    detection_time = models.DateTimeField(default=timezone.now)
    confidence_score = models.FloatField(default=0.0)  # Distance/confidence metric
    snapshot_image = models.ImageField(upload_to='detections/', null=True, blank=True)
    
    class Meta:
        ordering = ['-detection_time']
        indexes = [
            models.Index(fields=['-detection_time']),
            models.Index(fields=['person_name']),
            models.Index(fields=['camera_number']),
        ]

    def __str__(self):
        return f"{self.person_name} - Camera {self.camera_number} - {self.detection_time}"


class CameraStatus(models.Model):
    """
    Model to track camera status and availability.
    """
    camera_number = models.IntegerField(unique=True)
    is_active = models.BooleanField(default=False)
    last_checked = models.DateTimeField(auto_now=True)
    resolution = models.CharField(max_length=50, blank=True)
    
    class Meta:
        ordering = ['camera_number']
        verbose_name_plural = 'Camera Statuses'

    def __str__(self):
        status = "Active" if self.is_active else "Inactive"
        return f"Camera {self.camera_number} - {status}"