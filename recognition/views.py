"""
Django views for face recognition system.
"""

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .models import Person, DetectionLog, CameraStatus
from .face_recognition_engine import FaceRecognitionEngine
from .camera_manager import CameraManager
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)

# Global instances (in production, use proper singleton or dependency injection)
camera_manager = CameraManager()
# face_engine = FaceRecognitionEngine(threshold=0.7)#++++++++++++++++++++++++++++++++
try:
    face_engine = FaceRecognitionEngine(
        threshold=0.6, 
        model_name='Facenet512',
        use_yolo=True  # ← ENABLE YOLOv8
    )
except Exception as e:
    logger.warning(f"Failed to load with YOLOv8: {e}")
    # Fallback to MTCNN
    face_engine = FaceRecognitionEngine(
        threshold=0.6,
        model_name='Facenet512',
        use_yolo=False
    )

#-----------------------------------------------------------------------------------------
camera_manager.set_face_engine(face_engine)


def index(request):
    """
    Main page - upload known faces and manage system.
    """
    persons = Person.objects.filter(is_active=True)
    recent_detections = DetectionLog.objects.all()[:10]
    
    context = {
        'persons': persons,
        'recent_detections': recent_detections,
        'system_running': camera_manager.running
    }
    
    return render(request, 'recognition/index.html', context)


def upload_person(request):
    """
    Handle uploading a new known person.
    """
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        image = request.FILES.get('image')
        
        if not name or not image:
            messages.error(request, 'Please provide both name and image.')
            return redirect('index')
        
        try:
            # Check if person already exists
            if Person.objects.filter(name=name).exists():
                messages.error(request, f'Person with name "{name}" already exists.')
                return redirect('index')
            
            # Create person
            person = Person.objects.create(
                name=name,
                uploaded_image=image
            )
            
            # Extract face embedding
            image_path = person.uploaded_image.path
            embedding = face_engine.extract_embedding(image_path=image_path)
            
            if embedding is not None:
                # Save embedding as list (JSON serializable)
                person.face_embedding = embedding.tolist()
                person.save()
                
                # Update camera manager's known embeddings
                update_known_embeddings()
                
                messages.success(request, f'Successfully added {name} to the system.')
                logger.info(f"Added new person: {name}")
            else:
                # Delete person if no face detected
                person.delete()
                messages.error(request, 'No face detected in the uploaded image. Please try again.')
                
        except Exception as e:
            logger.error(f"Error uploading person: {e}")
            messages.error(request, f'Error uploading person: {str(e)}')
        
        return redirect('index')
    
    return redirect('index')


def delete_person(request, person_id):
    """
    Delete a known person.
    """
    try:
        person = Person.objects.get(id=person_id)
        name = person.name
        person.delete()
        
        # Update known embeddings
        update_known_embeddings()
        
        messages.success(request, f'Successfully deleted {name}.')
        logger.info(f"Deleted person: {name}")
    except Person.DoesNotExist:
        messages.error(request, 'Person not found.')
    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        messages.error(request, f'Error deleting person: {str(e)}')
    
    return redirect('index')


def live_view(request):
    """
    Live camera view page.
    """
    context = {
        'system_running': camera_manager.running
    }
    return render(request, 'recognition/live.html', context)


def start_cameras(request):
    print("Hi , camera is running")
    """
    Start all available cameras.
    """
    if request.method == 'POST':
        try:
            if camera_manager.running:
                return JsonResponse({
                    'success': False,
                    'message': 'Cameras are already running'
                })
            
            # Update known embeddings before starting
            update_known_embeddings()
            
            # Start cameras
            available_cameras = camera_manager.start_all_cameras()
            
            if available_cameras:
                # Update camera status in database
                for cam_num in available_cameras:
                    CameraStatus.objects.update_or_create(
                        camera_number=cam_num,
                        defaults={'is_active': True}
                    )
                
                logger.info(f"Started cameras: {available_cameras}")
                return JsonResponse({
                    'success': True,
                    'cameras': available_cameras,
                    'message': f'Successfully started {len(available_cameras)} camera(s)'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'No cameras detected'
                })
                
        except Exception as e:
            logger.error(f"Error starting cameras: {e}")
            return JsonResponse({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})


def stop_cameras(request):
    """
    Stop all cameras.
    """
    if request.method == 'POST':
        try:
            camera_manager.stop_all_cameras()
            
            # Update camera status in database
            CameraStatus.objects.all().update(is_active=False)
            
            logger.info("Stopped all cameras")
            return JsonResponse({
                'success': True,
                'message': 'All cameras stopped'
            })
            
        except Exception as e:
            logger.error(f"Error stopping cameras: {e}")
            return JsonResponse({
                'success': False,
                'message': f'Error: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})


def get_detections(request):
    """
    Get recent detection logs (AJAX endpoint).
    """
    try:
        limit = int(request.GET.get('limit', 20))
        detections = DetectionLog.objects.all()[:limit]
        
        data = []
        for detection in detections:
            data.append({
                'id': detection.id,
                'person_name': detection.person_name,
                'camera_number': detection.camera_number,
                'timestamp': detection.detection_time.isoformat(),
                'confidence': round(detection.confidence_score, 2)
            })
        
        return JsonResponse({
            'success': True,
            'detections': data
        })
        
    except Exception as e:
        logger.error(f"Error getting detections: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        })


def get_system_status(request):
    """
    Get current system status (AJAX endpoint).
    """
    try:
        return JsonResponse({
            'success': True,
            'running': camera_manager.running,
            'active_cameras': list(camera_manager.cameras.keys()),
            'known_persons': Person.objects.filter(is_active=True).count()
        })
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        })

def update_known_embeddings():
    """
    Update the camera manager with current known face embeddings.
    """
    try:
        persons = Person.objects.filter(is_active=True, face_embedding__isnull=False)
        embeddings = {}
        
        for person in persons:
            if person.face_embedding:
                embeddings[person.name] = np.array(person.face_embedding)
        
        camera_manager.update_known_embeddings(embeddings)
        logger.info(f"Updated known embeddings: {len(embeddings)} persons")
        
    except Exception as e:
        logger.error(f"Error updating known embeddings: {e}")
        
        
        
