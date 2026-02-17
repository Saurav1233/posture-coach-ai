/**
 * PostureCoach AI — Main Client-Side Controller v4
 * Beautiful colored skeleton like the reference image
 * Each body segment has its own color
 */
"use strict";

const CFG = window.POSTURE_CONFIG;
const RING_CIRCUMFERENCE = 314;
const JPEG_QUALITY = 0.7;

// ── Colored skeleton segments (like reference image) ──────────
// Each connection has its own color: red=spine, yellow=arms, blue=legs etc
const SKELETON_SEGMENTS = [
  // HEAD
  { connections: [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8]], color: '#ffffff', width: 2 },
  // SPINE / TORSO (red)
  { connections: [[11,12],[11,23],[12,24],[23,24]], color: '#ff4444', width: 4 },
  // LEFT ARM (yellow)
  { connections: [[11,13],[13,15]], color: '#ffdd00', width: 4 },
  // RIGHT ARM (yellow)
  { connections: [[12,14],[14,16]], color: '#ffdd00', width: 4 },
  // LEFT HAND extras
  { connections: [[15,17],[15,19],[15,21]], color: '#ffaa00', width: 2 },
  // RIGHT HAND extras
  { connections: [[16,18],[16,20],[16,22]], color: '#ffaa00', width: 2 },
  // LEFT LEG (blue/purple)
  { connections: [[23,25],[25,27]], color: '#4488ff', width: 4 },
  // RIGHT LEG (blue)
  { connections: [[24,26],[26,28]], color: '#44aaff', width: 4 },
  // LEFT FOOT (green)
  { connections: [[27,29],[29,31],[27,31]], color: '#44dd44', width: 3 },
  // RIGHT FOOT (green)
  { connections: [[28,30],[30,32],[28,32]], color: '#44dd44', width: 3 },
];

// Joint dot colors by group
const JOINT_COLORS = {
  head:       { indices: [0,1,2,3,4,5,6,7,8,9,10], color: '#ffffff' },
  shoulders:  { indices: [11,12],                   color: '#ff6644' },
  elbows:     { indices: [13,14],                   color: '#ffdd00' },
  wrists:     { indices: [15,16],                   color: '#ffaa00' },
  hips:       { indices: [23,24],                   color: '#ff4488' },
  knees:      { indices: [25,26],                   color: '#4488ff' },
  ankles:     { indices: [27,28],                   color: '#44aaff' },
  feet:       { indices: [29,30,31,32],             color: '#44dd44' },
  hands:      { indices: [17,18,19,20,21,22],       color: '#ffcc44' },
};

// Posture quality colors for overall skeleton tint
const QUALITY_COLORS = {
  green:  null,    // use segment colors (good form)
  yellow: '#fbbf24',
  red:    '#ff4444',
  grey:   '#888888',
};

// ── State ──────────────────────────────────────────
let isRunning      = false;
let requestPending = false;
let lastResult     = null;
let currentLandmarks = null;
let currentColor     = 'green';
let rafId          = null;
let inferTimer     = null;
let fpsCounter     = 0;
let fpsLastTime    = Date.now();

// ── DOM ────────────────────────────────────────────
const elVideo        = document.getElementById('webcam-video');
const elCapture      = document.getElementById('capture-canvas');
const elBtnStart     = document.getElementById('btn-start');
const elBtnStop      = document.getElementById('btn-stop');
const elBtnReset     = document.getElementById('btn-reset-reps');
const elExercise     = document.getElementById('exercise-select');
const elCameraStatus = document.getElementById('camera-status');
const elScoreVal     = document.getElementById('score-value');
const elRingFill     = document.getElementById('score-ring-fill');
const elStatus       = document.getElementById('posture-status');
const elRepCount     = document.getElementById('rep-count');
const elRepBadge     = document.getElementById('rep-state-badge');
const elFeedback     = document.getElementById('feedback-list');
const elFeatures     = document.getElementById('feature-bars');
const elFpsDisplay   = document.getElementById('fps-display');
const elOverlay      = document.getElementById('camera-overlay');
const elRepFlash     = document.getElementById('rep-flash');
const elUploadInput  = document.getElementById('video-upload-input');
const elUploadStatus = document.getElementById('upload-status');
const elUploadResults= document.getElementById('upload-results');
const captureCtx     = elCapture.getContext('2d');

// ── Create display canvas ──────────────────────────
const elDisplay = document.createElement('canvas');
elDisplay.id = 'display-canvas';
elDisplay.style.cssText = `
  position:absolute; top:0; left:0;
  width:100%; height:100%;
  border-radius:18px; z-index:3; display:block;
`;
document.getElementById('video-container').appendChild(elDisplay);
const dispCtx = elDisplay.getContext('2d');

// Hide old img
const oldImg = document.getElementById('annotated-frame');
if (oldImg) oldImg.style.display = 'none';


// ════════════════════════════════════════════════════
// SKELETON DRAWING — Beautiful colored joints
// ════════════════════════════════════════════════════

function drawSkeleton(landmarks, qualityColor, W, H) {
  if (!landmarks || landmarks.length < 33) return;

  // Build pixel points array
  const pts = landmarks.map(lm => ({
    x: lm[0] * W,
    y: lm[1] * H,
    v: lm.length > 3 ? lm[3] : 1.0,
  }));

  const useSegmentColors = (qualityColor === 'green' || qualityColor === null);

  // ── Draw connections with segment colors ──────────
  SKELETON_SEGMENTS.forEach(seg => {
    const lineColor = useSegmentColors ? seg.color : (QUALITY_COLORS[qualityColor] || '#aaa');

    seg.connections.forEach(([a, b]) => {
      if (!pts[a] || !pts[b]) return;
      if (pts[a].v < 0.2 || pts[b].v < 0.2) return;

      // Glow effect — draw wider semi-transparent line first
      dispCtx.beginPath();
      dispCtx.moveTo(pts[a].x, pts[a].y);
      dispCtx.lineTo(pts[b].x, pts[b].y);
      dispCtx.strokeStyle = lineColor + '44';  // 27% opacity glow
      dispCtx.lineWidth = seg.width + 4;
      dispCtx.lineCap = 'round';
      dispCtx.stroke();

      // Main line
      dispCtx.beginPath();
      dispCtx.moveTo(pts[a].x, pts[a].y);
      dispCtx.lineTo(pts[b].x, pts[b].y);
      dispCtx.strokeStyle = lineColor;
      dispCtx.lineWidth = seg.width;
      dispCtx.lineCap = 'round';
      dispCtx.stroke();
    });
  });

  // ── Draw joint dots ────────────────────────────────
  Object.values(JOINT_COLORS).forEach(group => {
    const dotColor = useSegmentColors ? group.color : (QUALITY_COLORS[qualityColor] || '#aaa');

    group.indices.forEach(i => {
      if (!pts[i] || pts[i].v < 0.2) return;

      // Outer glow ring
      dispCtx.beginPath();
      dispCtx.arc(pts[i].x, pts[i].y, 9, 0, Math.PI * 2);
      dispCtx.fillStyle = dotColor + '33';
      dispCtx.fill();

      // White border
      dispCtx.beginPath();
      dispCtx.arc(pts[i].x, pts[i].y, 6, 0, Math.PI * 2);
      dispCtx.fillStyle = 'white';
      dispCtx.fill();

      // Colored fill
      dispCtx.beginPath();
      dispCtx.arc(pts[i].x, pts[i].y, 4, 0, Math.PI * 2);
      dispCtx.fillStyle = dotColor;
      dispCtx.fill();
    });
  });
}

function drawHUD(result, W, H) {
  if (!result) return;
  const score  = Math.round(result.score || 0);
  const reps   = result.reps || 0;
  const status = result.status || '';
  const col    = QUALITY_COLORS[result.color] || '#00e87a';
  const scoreColor = result.color === 'green' ? '#00e87a' :
                     result.color === 'yellow' ? '#fbbf24' : '#ff4444';

  // Background pill
  dispCtx.fillStyle = 'rgba(0,0,0,0.6)';
  dispCtx.beginPath();
  dispCtx.roundRect(12, 12, 160, 85, 10);
  dispCtx.fill();

  dispCtx.font = 'bold 22px Segoe UI,sans-serif';
  dispCtx.fillStyle = scoreColor;
  dispCtx.fillText('Score: ' + score, 22, 42);

  dispCtx.font = 'bold 20px Segoe UI,sans-serif';
  dispCtx.fillStyle = '#fff';
  dispCtx.fillText('Reps:  ' + reps, 22, 66);

  dispCtx.font = 'bold 13px Segoe UI,sans-serif';
  dispCtx.fillStyle = scoreColor;
  dispCtx.fillText(status, 22, 87);
}


// ════════════════════════════════════════════════════
// RENDER LOOP — 60fps smooth display
// ════════════════════════════════════════════════════
function renderLoop() {
  if (!isRunning) return;
  rafId = requestAnimationFrame(renderLoop);

  const W = elDisplay.offsetWidth  || 640;
  const H = elDisplay.offsetHeight || 480;
  elDisplay.width  = W;
  elDisplay.height = H;

  // Draw webcam feed
  if (elVideo.readyState >= 2) {
    dispCtx.drawImage(elVideo, 0, 0, W, H);
  } else {
    dispCtx.fillStyle = '#111';
    dispCtx.fillRect(0, 0, W, H);
  }

  // Draw skeleton on top
  if (currentLandmarks) {
    drawSkeleton(currentLandmarks, currentColor, W, H);
  }

  // Draw HUD
  if (lastResult && lastResult.pose_detected) {
    drawHUD(lastResult, W, H);
  }

  // FPS counter
  fpsCounter++;
  const now = Date.now();
  if (now - fpsLastTime >= 1000) {
    elFpsDisplay.textContent = fpsCounter + ' fps';
    fpsCounter = 0;
    fpsLastTime = now;
  }
}


// ════════════════════════════════════════════════════
// INFERENCE LOOP — sends frames to server
// ════════════════════════════════════════════════════
async function inferenceLoop() {
  if (!isRunning || requestPending) return;
  if (!elVideo || elVideo.readyState < 2) return;

  requestPending = true;
  try {
    elCapture.width  = 640;
    elCapture.height = 480;
    captureCtx.drawImage(elVideo, 0, 0, 640, 480);
    const b64 = elCapture.toDataURL('image/jpeg', JPEG_QUALITY);

    const resp = await fetch(CFG.api_infer_url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': CFG.csrf_token,
      },
      body: JSON.stringify({
        frame:         b64,
        exercise:      elExercise.value,
        draw_skeleton: false,
      }),
    });

    if (!resp.ok) { console.error('Server error:', resp.status); return; }

    const data = await resp.json();
    lastResult = data;

    if (data.pose_detected && data.landmarks) {
      currentLandmarks = data.landmarks;
      currentColor     = data.color || 'green';
    } else {
      currentLandmarks = null;
    }

    updateUI(data);

  } catch (err) {
    console.error('Inference error:', err);
  } finally {
    requestPending = false;
  }
}


// ════════════════════════════════════════════════════
// CAMERA
// ════════════════════════════════════════════════════
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    elVideo.srcObject = stream;
    await elVideo.play();

    await new Promise(resolve => {
      if (elVideo.readyState >= 2) { resolve(); return; }
      elVideo.addEventListener('loadeddata', resolve, { once: true });
    });

    elOverlay.style.display  = 'none';
    elFpsDisplay.classList.remove('hidden');
    elCameraStatus.className = 'status-pill status-live';
    elCameraStatus.textContent = 'Live';

    isRunning   = true;
    fpsLastTime = Date.now();

    await resetSession();
    renderLoop();
    inferTimer = setInterval(inferenceLoop, 100);

    console.log('✅ Camera + inference started');

  } catch (err) {
    console.error('Camera error:', err);
    alert('Camera error: ' + err.message);
    elBtnStart.disabled = false;
    elBtnStop.disabled  = true;
  }
}

function stopCamera() {
  isRunning = false;
  clearInterval(inferTimer); inferTimer = null;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (elVideo.srcObject) {
    elVideo.srcObject.getTracks().forEach(t => t.stop());
    elVideo.srcObject = null;
  }
  dispCtx.clearRect(0, 0, elDisplay.width, elDisplay.height);
  elOverlay.style.display    = 'flex';
  elFpsDisplay.classList.add('hidden');
  elCameraStatus.className   = 'status-pill status-idle';
  elCameraStatus.textContent = 'Camera Off';
  currentLandmarks = null;
  lastResult       = null;
  setTimeout(() => { window.location.href = '/session-summary/'; }, 500);
}


// ════════════════════════════════════════════════════
// UI UPDATE
// ════════════════════════════════════════════════════
function updateUI(data) {
  const score     = data.score || 0;
  const color     = data.color || 'grey';
  const reps      = data.reps  || 0;
  const scoreHex  = color === 'green' ? '#00e87a' : color === 'yellow' ? '#fbbf24' : color === 'red' ? '#ff4444' : '#aaa';

  // Score ring
  elScoreVal.textContent = data.pose_detected ? Math.round(score) : '--';
  elScoreVal.style.color = scoreHex;
  elRingFill.style.strokeDashoffset = RING_CIRCUMFERENCE * (1 - score / 100);
  elRingFill.style.stroke = scoreHex;

  // Status
  const statusCls = { green:'status-good', yellow:'status-warn', red:'status-poor' };
  elStatus.textContent = data.pose_detected
    ? (data.status || 'Analyzing...')
    : '⚠️ Show full body — step back!';
  elStatus.className = 'posture-status ' + (statusCls[color] || 'status-idle');

  // Reps
  const prevReps = parseInt(elRepCount.textContent) || 0;
  elRepCount.textContent = reps;
  elRepBadge.textContent = data.rep_state || 'idle';
  if (data.rep_just_counted && reps > prevReps) triggerRepAnim();

  // Feedback
  renderFeedback(data.feedback || [], color, data.pose_detected);

  // Feature bars
  if (data.feature_scores) renderFeatureBars(data.feature_scores);
}

function renderFeedback(msgs, color, detected) {
  elFeedback.innerHTML = '';
  const cls = { green:'feedback-good', yellow:'feedback-warn', red:'feedback-poor', grey:'feedback-neutral' }[color] || 'feedback-neutral';

  if (!detected) {
    addLi('⚠️ Step back — show your full body (head to hips)', 'feedback-warn');
    return;
  }
  if (!msgs.length) {
    addLi('✅ Excellent form! Keep it up!', 'feedback-good');
    return;
  }
  msgs.forEach(m => addLi(m, cls));
}

function addLi(text, cls) {
  const li = document.createElement('li');
  li.className = 'feedback-item ' + cls;
  li.textContent = text;
  elFeedback.appendChild(li);
}

function renderFeatureBars(scores) {
  if (!Object.keys(scores).length) return;
  elFeatures.innerHTML = '';
  Object.entries(scores).slice(0, 8).forEach(([name, val]) => {
    const pct   = Math.max(0, Math.min(100, val));
    const color = pct >= 75 ? '#00e87a' : pct >= 50 ? '#fbbf24' : '#f05252';
    elFeatures.innerHTML += `
      <div class="feature-row">
        <div class="feature-row-header">
          <span class="feature-name">${name.replace(/_/g,' ')}</span>
          <span class="feature-val">${Math.round(pct)}</span>
        </div>
        <div class="feature-bar-track">
          <div class="feature-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>`;
  });
}

function triggerRepAnim() {
  elRepCount.classList.remove('bounce');
  void elRepCount.offsetWidth;
  elRepCount.classList.add('bounce');
  setTimeout(() => elRepCount.classList.remove('bounce'), 500);
  elRepFlash.classList.remove('hidden','show');
  void elRepFlash.offsetWidth;
  elRepFlash.classList.add('show');
  setTimeout(() => {
    elRepFlash.classList.remove('show');
    setTimeout(() => elRepFlash.classList.add('hidden'), 300);
  }, 1000);
}


// ════════════════════════════════════════════════════
// SESSION RESET
// ════════════════════════════════════════════════════
async function resetSession() {
  try {
    await fetch(`${CFG.api_reset_url}?exercise=${elExercise.value}`, {
      headers: { 'X-CSRFToken': CFG.csrf_token },
    });
  } catch(e) {}
  elRepCount.textContent   = '0';
  elScoreVal.textContent   = '--';
  elScoreVal.style.color   = '#00e87a';
  elRingFill.style.strokeDashoffset = RING_CIRCUMFERENCE;
  elRingFill.style.stroke  = '#00e87a';
  elStatus.textContent     = 'Waiting...';
  elStatus.className       = 'posture-status status-idle';
  elFeedback.innerHTML     = '<li class="feedback-item feedback-neutral">Get into position and start your exercise!</li>';
  elFeatures.innerHTML     = '<div class="feature-placeholder">Waiting for pose detection...</div>';
  currentLandmarks = null;
  lastResult = null;
}


// ════════════════════════════════════════════════════
// VIDEO UPLOAD
// ════════════════════════════════════════════════════
async function handleUpload(file) {
  elUploadStatus.textContent = 'Analyzing ' + file.name + '...';
  elUploadStatus.classList.remove('hidden');
  elUploadResults.classList.add('hidden');
  const fd = new FormData();
  fd.append('video', file);
  fd.append('exercise', elExercise.value);
  try {
    const resp = await fetch(CFG.api_upload_url, {
      method:'POST', headers:{'X-CSRFToken':CFG.csrf_token}, body:fd
    });
    const data = await resp.json();
    elUploadStatus.classList.add('hidden');
    if (data.error) { elUploadStatus.textContent='Error: '+data.error; elUploadStatus.classList.remove('hidden'); return; }
    const sc = data.average_score >= 75 ? '#00e87a' : '#fbbf24';
    elUploadResults.innerHTML = `<h4>Analysis Complete</h4>
      <p>Frames: <strong>${data.total_frames_analyzed||0}</strong></p>
      <p>Reps: <strong>${data.total_reps||0}</strong></p>
      <p>Avg Score: <strong>${data.average_score||0}/100</strong></p>
      <p>Result: <strong style="color:${sc}">${data.overall_status||'N/A'}</strong></p>`;
    elUploadResults.classList.remove('hidden');
  } catch(err) { elUploadStatus.textContent='Upload failed: '+err.message; }
}


// ════════════════════════════════════════════════════
// EVENT LISTENERS
// ════════════════════════════════════════════════════
elBtnStart.addEventListener('click', async () => {
  elBtnStart.disabled = true; elBtnStop.disabled = false;
  await startCamera();
});
elBtnStop.addEventListener('click', () => {
  elBtnStart.disabled = false; elBtnStop.disabled = true;
  stopCamera();
});
elBtnReset.addEventListener('click', resetSession);
elExercise.addEventListener('change', () => { if (isRunning) resetSession(); });
elUploadInput.addEventListener('change', e => { if (e.target.files[0]) handleUpload(e.target.files[0]); });
window.addEventListener('beforeunload', e => { if (isRunning) { e.preventDefault(); e.returnValue=''; } });