"""
URL patterns for Posture Coach application.

REST Endpoints
──────────────
GET  /                          → Main coaching interface
POST /api/infer/                → Live frame inference endpoint
POST /api/upload/               → Video upload + batch analysis
GET  /api/session/reset/        → Reset session (new exercise)
GET  /session-summary/          → Session summary page
"""
from django.urls import path
from apps.posture_app import views

urlpatterns = [
    # Pages
    path('',                    views.index,           name='index'),
    path('session-summary/',    views.session_summary, name='session_summary'),

    # API endpoints
    path('api/infer/',          views.api_infer,       name='api_infer'),
    path('api/upload/',         views.api_upload,      name='api_upload'),
    path('api/session/reset/',  views.api_reset_session, name='api_reset_session'),
    path('api/profile/status/', views.api_profile_status, name='api_profile_status'),
]
