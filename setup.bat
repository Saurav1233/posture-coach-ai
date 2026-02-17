@echo off
REM ═══════════════════════════════════════════════════════════════
REM  PostureCoach AI — Windows Setup Script
REM  Python 3.10+ required
REM ═══════════════════════════════════════════════════════════════

echo.
echo  [1/7] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo.
echo  [2/7] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo  [3/7] Installing dependencies...
pip install -r requirements.txt

echo.
echo  [4/7] Running Django migrations...
python manage.py migrate

echo.
echo  [5/7] Creating static files directory...
python manage.py collectstatic --noinput

echo.
echo  [6/7] Generating demo posture profiles...
python training/scripts/generate_demo_profiles.py

echo.
echo  [7/7] Creating log directory...
if not exist logs mkdir logs

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   Setup complete!
echo.
echo   To START the server, run:
echo       run_server.bat
echo.
echo   To TRAIN on your videos:
echo       python training\scripts\train.py --exercise squat --video_dir training\data\raw_videos\squat --cross_validate
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pause
