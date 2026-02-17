/**
 * Session Summary Score Chart
 * ============================
 * Renders the score-over-time line chart on the summary page
 * using native Canvas 2D API (no external charting library dependency).
 */

"use strict";

(function () {
  const canvas = document.getElementById('score-chart');
  if (!canvas || !window.history || !history.length) return;

  const scores = window.history;
  if (!scores || !scores.length) return;

  const ctx = canvas.getContext('2d');
  const W = canvas.parentElement.clientWidth - 40;
  const H = 160;
  canvas.width  = W;
  canvas.height = H;

  const PAD = { top: 16, right: 16, bottom: 28, left: 40 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top  - PAD.bottom;

  const minY = 0;
  const maxY = 100;

  // ── Helpers ──────────────────────────────────────────────
  function xPos(i) {
    return PAD.left + (i / Math.max(scores.length - 1, 1)) * chartW;
  }
  function yPos(v) {
    return PAD.top + chartH - ((v - minY) / (maxY - minY)) * chartH;
  }
  function scoreColor(v) {
    if (v >= 75) return '#00e87a';
    if (v >= 50) return '#fbbf24';
    return '#f05252';
  }

  // ── Grid lines ────────────────────────────────────────────
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  [25, 50, 75, 100].forEach(gridY => {
    const y = yPos(gridY);
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + chartW, y);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '10px Segoe UI, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(gridY, PAD.left - 6, y + 4);
  });

  // ── Threshold lines ───────────────────────────────────────
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = 'rgba(0,232,122,0.3)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD.left, yPos(75));
  ctx.lineTo(PAD.left + chartW, yPos(75));
  ctx.stroke();
  ctx.setLineDash([]);

  // ── Area fill (gradient under line) ──────────────────────
  if (scores.length > 1) {
    const gradient = ctx.createLinearGradient(0, PAD.top, 0, PAD.top + chartH);
    gradient.addColorStop(0, 'rgba(59,130,246,0.3)');
    gradient.addColorStop(1, 'rgba(59,130,246,0.0)');

    ctx.beginPath();
    ctx.moveTo(xPos(0), yPos(scores[0]));
    scores.forEach((v, i) => i > 0 && ctx.lineTo(xPos(i), yPos(v)));
    ctx.lineTo(xPos(scores.length - 1), PAD.top + chartH);
    ctx.lineTo(xPos(0), PAD.top + chartH);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
  }

  // ── Line ─────────────────────────────────────────────────
  ctx.beginPath();
  ctx.lineWidth = 2;
  scores.forEach((v, i) => {
    if (i === 0) ctx.moveTo(xPos(0), yPos(v));
    else ctx.lineTo(xPos(i), yPos(v));
  });
  ctx.strokeStyle = '#3b82f6';
  ctx.stroke();

  // ── Colored dots at key points ────────────────────────────
  const step = Math.max(1, Math.floor(scores.length / 20));
  scores.forEach((v, i) => {
    if (i % step !== 0 && i !== scores.length - 1) return;
    ctx.beginPath();
    ctx.arc(xPos(i), yPos(v), 3, 0, Math.PI * 2);
    ctx.fillStyle = scoreColor(v);
    ctx.fill();
  });

  // ── X-axis labels ─────────────────────────────────────────
  ctx.fillStyle = 'rgba(255,255,255,0.25)';
  ctx.font = '10px Segoe UI, sans-serif';
  ctx.textAlign = 'center';
  const nLabels = Math.min(5, scores.length);
  for (let l = 0; l < nLabels; l++) {
    const i = Math.round(l * (scores.length - 1) / Math.max(nLabels - 1, 1));
    ctx.fillText(i, xPos(i), H - 6);
  }
  ctx.fillText('frames', PAD.left + chartW / 2, H);

})();
