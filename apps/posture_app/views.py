"""
Django Views - Posture Coach (Simplified & Bulletproof)
"""
import base64
import json
import logging
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

logger = logging.getLogger('posture_coach')

# ─────────────────────────────────────────────────────────────
# SIMPLE ENGINE CACHE
# ─────────────────────────────────────────────────────────────
_engines = {}

def _get_engine(session_key, exercise):
    from apps.ml_engine.inference_engine import get_or_create_engine
    profiles_dir = Path(settings.ML_CONFIG['PROFILES_DIR'])
    return get_or_create_engine(session_key, exercise, profiles_dir)

def _profiles_dir():
    return Path(settings.ML_CONFIG['PROFILES_DIR'])


# ─────────────────────────────────────────────────────────────
# PAGE VIEWS
# ─────────────────────────────────────────────────────────────
def index(request):
    exercises = settings.ML_CONFIG.get('SUPPORTED_EXERCISES', ['squat','pushup','barbell_curl','hammer_curl','shoulder_press'])
    exercise_labels = {
        'squat':          'Squat',
        'pushup':         'Push-Up',
        'barbell_curl':   'Barbell Curl',
        'hammer_curl':    'Hammer Curl',
        'shoulder_press': 'Shoulder Press',
    }
    context = {
        'exercises': [(e, exercise_labels.get(e, e.title())) for e in exercises],
        'default_exercise': request.session.get('exercise', 'squat'),
        'score_threshold_good': settings.ML_CONFIG.get('SCORE_THRESHOLD_GOOD', 75),
        'score_threshold_warning': settings.ML_CONFIG.get('SCORE_THRESHOLD_WARNING', 50),
    }
    return render(request, 'posture_app/index.html', context)


def session_summary(request):
    score_history = request.session.get('score_history', [])
    exercise = request.session.get('exercise', 'squat')
    reps = request.session.get('last_reps', 0)
    session_start = request.session.get('session_start', time.time())
    duration_s = int(time.time() - session_start)

    if score_history:
        avg_score = sum(score_history) / len(score_history)
        min_score = min(score_history)
        max_score = max(score_history)
        good_frames = sum(1 for s in score_history if s >= settings.ML_CONFIG.get('SCORE_THRESHOLD_GOOD', 75))
        pct_good = round(100 * good_frames / len(score_history), 1)
    else:
        avg_score = min_score = max_score = 0.0
        pct_good = 0.0

    exercise_labels = {
        'squat':'Squat','pushup':'Push-Up','barbell_curl':'Barbell Curl',
        'hammer_curl':'Hammer Curl','shoulder_press':'Shoulder Press',
    }

    context = {
        'exercise': exercise,
        'exercise_label': exercise_labels.get(exercise, exercise.title()),
        'reps': reps,
        'avg_score': round(avg_score, 1),
        'min_score': round(min_score, 1),
        'max_score': round(max_score, 1),
        'pct_good': pct_good,
        'duration_s': duration_s,
        'total_frames': len(score_history),
        'score_history_json': json.dumps(score_history[-200:]),
    }
    return render(request, 'posture_app/session_summary.html', context)


# ─────────────────────────────────────────────────────────────
# API: INFER
# ─────────────────────────────────────────────────────────────
@csrf_exempt
@require_POST
def api_infer(request):
    try:
        body = json.loads(request.body)
        frame_b64 = body.get('frame', '')
        exercise  = body.get('exercise', 'squat').lower().replace('-','_').replace(' ','_')

        supported = settings.ML_CONFIG.get('SUPPORTED_EXERCISES',
            ['squat','pushup','barbell_curl','hammer_curl','shoulder_press'])
        if exercise not in supported:
            exercise = 'squat'

        # Decode frame
        frame = _decode_frame(frame_b64)
        if frame is None:
            return JsonResponse(_no_pose_response("Could not decode frame"))

        # Get engine
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key or 'default'

        engine = _get_engine(session_key, exercise)

        # Run inference
        result = engine.process_frame_b64(frame_b64, draw_skeleton=False)

        # Ensure all expected keys exist
        result.setdefault('feature_scores', {})
        result.setdefault('rep_just_counted', False)
        result.setdefault('landmarks', None)
        result.setdefault('worst_features', [])

        # Save session data
        request.session['exercise'] = exercise
        if 'session_start' not in request.session:
            request.session['session_start'] = time.time()

        if result.get('pose_detected'):
            history = request.session.get('score_history', [])
            history.append(round(result.get('score', 0), 1))
            if len(history) > 500:
                history = history[-500:]
            request.session['score_history'] = history
            request.session['last_reps'] = result.get('reps', 0)

        request.session.modified = True
        return JsonResponse(result)

    except Exception as e:
        logger.error("api_infer error: %s\n%s", e, traceback.format_exc())
        return JsonResponse({
            'pose_detected': False,
            'score': 0,
            'status': 'Server Error',
            'color': 'grey',
            'reps': 0,
            'rep_state': 'idle',
            'rep_just_counted': False,
            'feedback': [f'Server error: {str(e)}'],
            'feature_scores': {},
            'landmarks': None,
            'worst_features': [],
            'error': str(e),
        }, status=200)  # Return 200 so JS can read the error message


# ─────────────────────────────────────────────────────────────
# API: RESET SESSION
# ─────────────────────────────────────────────────────────────
@csrf_exempt
def api_reset_session(request):
    try:
        exercise = request.GET.get('exercise', 'squat')
        request.session['exercise'] = exercise
        request.session['session_start'] = time.time()
        request.session['score_history'] = []
        request.session['last_reps'] = 0
        request.session.modified = True

        if request.session.session_key:
            from apps.ml_engine.inference_engine import get_or_create_engine
            engine = get_or_create_engine(
                request.session.session_key, exercise, _profiles_dir()
            )
            engine.reset_session()

        return JsonResponse({'status': 'ok', 'exercise': exercise})
    except Exception as e:
        logger.error("Reset error: %s", e)
        return JsonResponse({'status': 'ok'})


# ─────────────────────────────────────────────────────────────
# API: UPLOAD VIDEO
# ─────────────────────────────────────────────────────────────
@csrf_exempt
@require_POST
def api_upload(request):
    try:
        video_file = request.FILES.get('video')
        exercise   = request.POST.get('exercise', 'squat')

        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)

        import tempfile, os
        suffix = Path(video_file.name).suffix or '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in video_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            from apps.ml_engine.inference_engine import InferenceEngine
            engine = InferenceEngine(exercise, _profiles_dir())
            result = engine.process_video_file(tmp_path)
        finally:
            os.unlink(tmp_path)

        return JsonResponse(result)

    except Exception as e:
        logger.error("Upload error: %s\n%s", e, traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _decode_frame(b64_data):
    try:
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        raw  = base64.b64decode(b64_data)
        arr  = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Frame decode error: %s", e)
        return None


def _no_pose_response(msg="No pose detected"):
    return {
        'pose_detected': False,
        'score': 0,
        'status': msg,
        'color': 'grey',
        'reps': 0,
        'rep_state': 'idle',
        'rep_just_counted': False,
        'feedback': [msg],
        'feature_scores': {},
        'landmarks': None,
        'worst_features': [],
    }


# ─────────────────────────────────────────────────────────────
# API: PROFILE STATUS (used by health check)
# ─────────────────────────────────────────────────────────────
def api_profile_status(request):
    try:
        profiles_dir = _profiles_dir()
        supported = settings.ML_CONFIG.get('SUPPORTED_EXERCISES',
            ['squat','pushup','barbell_curl','hammer_curl','shoulder_press'])
        status = {}
        for ex in supported:
            profile_path = profiles_dir / f'{ex}_profile.json'
            status[ex] = {
                'has_profile': profile_path.exists(),
                'path': str(profile_path),
            }
        return JsonResponse({'profiles': status, 'profiles_dir': str(profiles_dir)})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)