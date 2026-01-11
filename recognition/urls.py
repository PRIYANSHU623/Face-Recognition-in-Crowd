"""
URL configuration for recognition app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_person, name='upload_person'),
    path('delete/<int:person_id>/', views.delete_person, name='delete_person'),
    path('live/', views.live_view, name='live_view'),
    path('api/start-cameras/', views.start_cameras, name='start_cameras'),
    path('api/stop-cameras/', views.stop_cameras, name='stop_cameras'),
    path('api/detections/', views.get_detections, name='get_detections'),
    path('api/status/', views.get_system_status, name='get_system_status'),
]