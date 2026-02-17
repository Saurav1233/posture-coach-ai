@echo off
REM ═══════════════════════════════════════════════════════════════
REM  PostureCoach AI — Start Development Server
REM ═══════════════════════════════════════════════════════════════
call venv\Scripts\activate.bat
echo Starting PostureCoach AI server on http://localhost:8000
echo Press Ctrl+C to stop.
echo.
python manage.py runserver 0.0.0.0:8000
