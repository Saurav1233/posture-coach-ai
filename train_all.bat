@echo off
REM ═══════════════════════════════════════════════════════════════
REM  Train posture profiles for ALL 5 exercises
REM  Place your MP4 videos in:
REM    training\data\raw_videos\squat\
REM    training\data\raw_videos\pushup\
REM    training\data\raw_videos\barbell_curl\
REM    training\data\raw_videos\hammer_curl\
REM    training\data\raw_videos\shoulder_press\
REM ═══════════════════════════════════════════════════════════════
call venv\Scripts\activate.bat

echo Training: Squat
python training\scripts\train.py --exercise squat --video_dir training\data\raw_videos\squat --cross_validate

echo Training: Push-Up
python training\scripts\train.py --exercise pushup --video_dir training\data\raw_videos\pushup --cross_validate

echo Training: Barbell Curl
python training\scripts\train.py --exercise barbell_curl --video_dir training\data\raw_videos\barbell_curl --cross_validate

echo Training: Hammer Curl
python training\scripts\train.py --exercise hammer_curl --video_dir training\data\raw_videos\hammer_curl --cross_validate

echo Training: Shoulder Press
python training\scripts\train.py --exercise shoulder_press --video_dir training\data\raw_videos\shoulder_press --cross_validate

echo.
echo All exercises trained! Profiles saved to training\data\posture_profiles\
pause
