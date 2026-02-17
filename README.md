# PostureCoach AI — Complete System Documentation

## Table of Contents
1. [Project Overview](#overview)
2. [System Architecture Flowchart](#flowchart)
3. [Full Folder Structure](#structure)
4. [ML Pipeline Design](#ml-pipeline)
5. [Mathematical Justification](#math)
6. [Academic Validation](#validation)
7. [Installation & Setup](#setup)
8. [Training Your Own Models](#training)
9. [Design Decisions](#decisions)

---

## 1. Project Overview {#overview}

PostureCoach AI is a production-grade, Django-based web application that provides
real-time exercise posture analysis using computer vision and statistical modeling.

**Key characteristics:**
- Uses MediaPipe Pose for landmark extraction (33 joints, 3D coordinates)
- Deviation-based one-class learning (no incorrect posture data required)
- 5 exercises with custom biomechanical feature sets
- State-machine repetition counting (noise-robust)
- Clean 3-layer architecture (Frontend / Django / ML Engine)

---

## 2. System Architecture Flowchart {#flowchart}

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                           │
│                                                                     │
│  ┌──────────┐  getUserMedia   ┌──────────────┐                      │
│  │  Webcam  │ ──────────────► │  <video>     │                      │
│  └──────────┘                 │  element     │                      │
│                               └──────┬───────┘                      │
│                                      │ drawImage                    │
│                               ┌──────▼───────┐                      │
│                               │  <canvas>    │ toDataURL (JPEG)     │
│                               └──────┬───────┘                      │
│                                      │ base64 JPEG                  │
│               ┌──────────────────────▼──────────────────────────┐   │
│               │             coach.js (FrameSender)              │   │
│               │  POST /api/infer/ every 80ms (12 fps)           │   │
│               └──────────────────────┬──────────────────────────┘   │
└──────────────────────────────────────┼──────────────────────────────┘
                                       │ HTTP POST
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DJANGO BACKEND (Layer 2)                        │
│                                                                     │
│  ┌────────────────────┐                                             │
│  │   views.py         │ api_infer() — validates, calls engine       │
│  │   api_infer()      │ session management, score history           │
│  └──────────┬─────────┘                                             │
└─────────────┼───────────────────────────────────────────────────────┘
              │ calls
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ML ENGINE (Layer 3)                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   InferenceEngine                           │    │
│  │                                                             │    │
│  │   base64 JPEG ──► cv2.imdecode ──► BGR frame               │    │
│  │                                        │                   │    │
│  │                               ┌────────▼──────────┐        │    │
│  │                               │  PoseExtractor    │        │    │
│  │                               │  (MediaPipe Pose) │        │    │
│  │                               │  33 landmarks     │        │    │
│  │                               └────────┬──────────┘        │    │
│  │                                        │                   │    │
│  │                               ┌────────▼──────────┐        │    │
│  │                               │  Normalization    │        │    │
│  │                               │  torso_center     │        │    │
│  │                               │  torso_length     │        │    │
│  │                               └────────┬──────────┘        │    │
│  │                                        │ norm_lm (33,3)    │    │
│  │              ┌─────────────────────────┼────────────────┐  │    │
│  │              │                         │                │  │    │
│  │    ┌─────────▼──────────┐   ┌──────────▼──────────┐    │  │    │
│  │    │  Feature Module    │   │   draw_skeleton()   │    │  │    │
│  │    │  (per exercise)    │   │   color = f(score)  │    │  │    │
│  │    │  joint angles      │   │   → JPEG b64        │    │  │    │
│  │    │  alignment         │   └────────────────────-┘    │  │    │
│  │    │  symmetry          │                               │  │    │
│  │    └─────────┬──────────┘                               │  │    │
│  │              │ features dict                             │  │    │
│  │    ┌─────────▼──────────┐   ┌──────────────────────┐    │  │    │
│  │    │  PostureScorer     │   │   RepCounter         │    │  │    │
│  │    │                    │   │   (state machine)    │    │  │    │
│  │    │  z_i=(x_i-μ_i)/σ_i│   │   EMA smoothing      │    │  │    │
│  │    │  score = f(z)      │   │   hold_frames guard  │    │  │    │
│  │    │  feedback msgs     │   │   → reps++           │    │  │    │
│  │    └─────────┬──────────┘   └──────────┬───────────┘    │  │    │
│  │              └────────────────┬─────────┘               │  │    │
│  │                               │ result dict             │  │    │
│  └───────────────────────────────┼─────────────────────────┘  │    │
│                                  │                             │    │
└──────────────────────────────────┼─────────────────────────────────┘
                                   │ JSON response
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     BROWSER (UI Update)                             │
│                                                                     │
│   annotated_frame → <img>  (base64 JPEG with skeleton)             │
│   score           → SVG ring animation + color                     │
│   reps            → bounce animation + flash banner                │
│   feedback        → feedback panel (color-coded)                   │
│   feature_scores  → mini bar chart (joint breakdown)               │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline Flowchart
```
  MP4 Videos (correct posture only)
         │
         ▼
  PoseExtractor (MediaPipe, complexity=2)
  Extract 33 landmarks per frame (skip=3)
         │
         ▼
  Normalize by torso center + torso length
         │
         ▼
  Feature Module (per exercise)
  Compute: angles, alignment, symmetry
         │
         ▼
  Save to CSV  ────────────────────────────┐
  (features_csv/exercise_features.csv)     │
         │                                 │
         ▼                                 │
  Build Profile:                           │
  μ_i = mean(feature_i)                   │
  σ_i = std(feature_i)                    │
  (IQR-trimmed for robustness)            │
         │                                 │
         ▼                                 ▼
  Save JSON Profile         Leave-One-Video-Out CV
  posture_profiles/         (validates stability)
  exercise_profile.json
```

---

## 3. Full Folder Structure {#structure}

```
posture_coach/
├── manage.py                          # Django entry point
├── requirements.txt                   # Python dependencies
├── setup.bat                          # Windows one-click setup
├── run_server.bat                     # Windows server start
├── train_all.bat                      # Train all 5 exercises
├── .env.example                       # Environment template
│
├── config/                            # Django project config
│   ├── __init__.py
│   ├── settings.py                    # All settings incl. ML_CONFIG
│   ├── urls.py                        # Root URL routing
│   └── wsgi.py
│
├── apps/
│   ├── __init__.py
│   │
│   ├── ml_engine/                     # ML Layer (no Django deps)
│   │   ├── __init__.py
│   │   ├── pose_extractor.py          # MediaPipe wrapper + normalization
│   │   ├── biomechanics.py            # Pure geometry functions
│   │   ├── posture_scorer.py          # Deviation scoring engine
│   │   ├── rep_counter.py             # State machine rep counter
│   │   ├── inference_engine.py        # Orchestration layer
│   │   │
│   │   └── features/                  # Per-exercise feature definitions
│   │       ├── __init__.py            # Feature registry
│   │       ├── squat_features.py
│   │       ├── pushup_features.py
│   │       ├── barbell_curl_features.py
│   │       ├── hammer_curl_features.py
│   │       └── shoulder_press_features.py
│   │
│   └── posture_app/                   # Django App
│       ├── __init__.py
│       ├── apps.py
│       ├── urls.py                    # URL patterns
│       ├── views.py                   # HTTP handlers (4 endpoints)
│       │
│       ├── templates/posture_app/
│       │   ├── index.html             # Main coaching interface
│       │   └── session_summary.html   # Post-session summary
│       │
│       └── static/posture_app/
│           ├── css/
│           │   └── style.css          # Dark theme design system
│           └── js/
│               ├── coach.js           # Webcam + inference loop + UI
│               └── summary_chart.js   # Canvas score chart
│
├── training/
│   ├── scripts/
│   │   ├── train.py                   # Main training script
│   │   └── generate_demo_profiles.py # Demo profiles (no videos needed)
│   │
│   └── data/
│       ├── raw_videos/                # Place MP4 files here
│       │   ├── squat/
│       │   ├── pushup/
│       │   ├── barbell_curl/
│       │   ├── hammer_curl/
│       │   └── shoulder_press/
│       ├── features_csv/              # Auto-generated by training
│       └── posture_profiles/          # JSON profiles (auto-generated)
│
├── media/
│   ├── uploads/                       # Temp video uploads
│   └── sessions/
│
└── logs/
    └── posture_coach.log
```

---

## 4. ML Pipeline Design {#ml-pipeline}

### A. Pose Extraction (pose_extractor.py)
- Uses MediaPipe Pose (model complexity=1 for inference, 2 for training)
- Extracts 33 landmarks in image-normalized coordinates (x,y,z ∈ [0,1])
- Filters frames with visibility < 0.55 on torso landmarks

**Normalization:**
```
torso_center = mean(L_shoulder, R_shoulder, L_hip, R_hip)
torso_length = mean(|each_torso_pt - torso_center|)
norm_lm = (lm - torso_center) / torso_length
```

### B. Feature Engineering (features/*.py)
Each exercise has a dedicated module defining:
- Biomechanically relevant joint angles (joint_angle_2d)
- Alignment/lean angles (vertical_deviation_angle)
- Bilateral symmetry ratios (bilateral_symmetry_ratio)
- Exercise-specific indicators (valgus, hip sag, elbow flare, etc.)

### C. Posture Profiling (posture_scorer.py)
Training computes μ and σ for each feature across all correct-posture frames.
The profile is stored as a JSON file per exercise.

### D. Inference Scoring
For each frame at inference time:
- Compute z-score per feature: z_i = |x_i - μ_i| / σ_i
- Clamp at Z_MAX=3.0
- Weighted combination: score = 100 × (1 - Σ(w_i × z_i/Z_MAX) / Σw_i)

---

## 5. Mathematical Justification {#math}

### Posture Score Formula

```
z_i      = |x_i - μ_i| / σ_i              [standard score]
cz_i     = min(z_i, 3.0)                  [clamped z-score]
fs_i     = 100 × (1 - cz_i / 3.0)         [feature score ∈ [0,100]]
score    = 100 × (1 - Σ(w_i × cz_i/3) / Σw_i)   [global score]
```

**Score interpretation:**
| z-score | Deviation | Score Contribution |
|---------|-----------|-------------------|
| 0.0     | On mean   | 100               |
| 0.5     | ½σ        | 83                |
| 1.0     | 1σ (68%)  | 67                |
| 2.0     | 2σ (95%)  | 33                |
| ≥3.0    | 3σ (99.7%)| 0                 |

**Why weighted mean of z-scores?**
This is equivalent to the Mahalanobis distance in diagonal covariance form,
where weights replace inverse-variance weighting. This is interpretable,
auditable, and directly maps to individual joint feedback.

---

## 6. Academic Validation {#validation}

### Without Negative Samples — How We Validate

1. **Leave-One-Video-Out CV** (automated in train.py --cross_validate)
   - Train profile on N-1 videos, score N-th video
   - Correct posture should score ≥ 75 on the holdout
   - A low CV score gap (< 5 pts std) proves the profile generalizes

2. **Synthetic Perturbation Testing** (conceptual)
   - Artificially perturb joint angles by 1σ, 2σ, 3σ
   - Verify score drops by ~33, ~67, ~100 points respectively
   - This validates the mathematical relationship is working correctly

3. **Calibration Check**
   - Plot histogram of scores on all training frames
   - Should cluster tightly near 75-100 for correct posture
   - A wide or low distribution suggests poor feature engineering

4. **Expert Review**
   - Have a physiotherapist or coach review frames with scores 40-60
   - These are the "borderline" cases — their feedback calibrates thresholds

---

## 7. Installation & Setup {#setup}

### Prerequisites
- Windows 10/11
- Python 3.10 (exact version recommended for MediaPipe compatibility)
- Webcam

### Quick Start (Windows)
```batch
git clone <repo> posture_coach
cd posture_coach
setup.bat
run_server.bat
```
Open http://localhost:8000 in Chrome or Edge.

### Manual Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python training/scripts/generate_demo_profiles.py
python manage.py runserver
```

---

## 8. Training Your Own Models {#training}

### Prepare Videos
Place MP4 files in the correct exercise folder:
```
training/data/raw_videos/squat/*.mp4
training/data/raw_videos/pushup/*.mp4
... etc
```
**Recommendations:**
- 5–15 videos per exercise
- 30–90 seconds each (full range of motion, multiple reps)
- Good lighting, full body visible
- Only correct posture (by definition of this system)

### Run Training
```bash
# Train one exercise with cross-validation
python training/scripts/train.py \
  --exercise squat \
  --video_dir training/data/raw_videos/squat \
  --cross_validate

# Train all exercises at once
train_all.bat
```

### Profiles
Training outputs JSON profiles to `training/data/posture_profiles/`.
The server loads these automatically on next request (cached in memory).

To force reload after retraining:
```python
from apps.ml_engine.posture_scorer import clear_scorer_cache
clear_scorer_cache()
```

---

## 9. Key Design Decisions {#decisions}

| Decision | Rationale |
|----------|-----------|
| Deviation-based (not binary classification) | Only correct posture data available. One-class statistical modeling is the only valid approach. |
| Gaussian profile (mean + std) | Interpretable, computationally trivial, directly maps to z-score feedback. Equivalent to diagonal Gaussian envelope / one-class model. |
| IQR-trimmed statistics | MediaPipe occasionally outputs spurious landmarks. Tukey fences remove these outliers without discarding valid data. |
| Torso normalization | Makes the system robust to different body sizes and camera distances. Without this, a tall person and a short person would get different scores for identical technique. |
| EMA smoothing on rep counter | Prevents jitter-driven false rep counts at state boundaries. Alpha=0.25 provides a good balance between responsiveness and stability. |
| Hold frames (3) in state machine | Requires 3 consecutive frames in the new state before transitioning. Prevents single-frame noise from triggering rep counts. |
| Server-side skeleton rendering | Avoids sending full landmark coordinates to the browser on every frame. The annotated JPEG is smaller and more secure. |
| Per-exercise feature weights | Biomechanically motivated — injury risk features (knee valgus, trunk lean) are weighted 3x higher than cosmetic features. |
| Feature registry pattern | Allows adding new exercises without modifying any existing code. Just add a new feature module and register it. |
| Session-based rep persistence | Rep counts survive page refreshes. The counter state is serialized to Django session (JSONable dict). |
