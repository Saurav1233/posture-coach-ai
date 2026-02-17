@echo off
echo =============================================
echo  PostureCoach AI - Training All Exercises
echo =============================================
echo.
echo Make sure you have MP4 videos in:
echo   training\data\raw_videos\squat\
echo   training\data\raw_videos\pushup\
echo   training\data\raw_videos\barbell_curl\
echo   training\data\raw_videos\hammer_curl\
echo   training\data\raw_videos\shoulder_press\
echo.

call venv\Scripts\activate.bat

echo [1/5] Training: Squat
if exist training\data\raw_videos\squat\*.mp4 (
    python training\scripts\train.py --exercise squat --video_dir training\data\raw_videos\squat
) else (
    echo   SKIPPED - no videos found in training\data\raw_videos\squat\
)

echo.
echo [2/5] Training: Push-Up
if exist training\data\raw_videos\pushup\*.mp4 (
    python training\scripts\train.py --exercise pushup --video_dir training\data\raw_videos\pushup
) else (
    echo   SKIPPED - no videos found in training\data\raw_videos\pushup\
)

echo.
echo [3/5] Training: Barbell Curl
if exist training\data\raw_videos\barbell_curl\*.mp4 (
    python training\scripts\train.py --exercise barbell_curl --video_dir training\data\raw_videos\barbell_curl
) else (
    echo   SKIPPED - no videos found in training\data\raw_videos\barbell_curl\
)

echo.
echo [4/5] Training: Hammer Curl
if exist training\data\raw_videos\hammer_curl\*.mp4 (
    python training\scripts\train.py --exercise hammer_curl --video_dir training\data\raw_videos\hammer_curl
) else (
    echo   SKIPPED - no videos found in training\data\raw_videos\hammer_curl\
)

echo.
echo [5/5] Training: Shoulder Press
if exist training\data\raw_videos\shoulder_press\*.mp4 (
    python training\scripts\train.py --exercise shoulder_press --video_dir training\data\raw_videos\shoulder_press
) else (
    echo   SKIPPED - no videos found in training\data\raw_videos\shoulder_press\
)

echo.
echo =============================================
echo  Training Complete!
echo  Profiles saved to: training\data\posture_profiles\
echo =============================================
pause