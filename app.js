import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const cv = document.getElementById("stage");
const ctx = cv.getContext("2d");
const startEl = document.getElementById("start");
const msg = document.getElementById("msg");

let W = 0, H = 0, DPR = 1;
function resize() {
  DPR = Math.min(devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  cv.width = W * DPR; cv.height = H * DPR;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
addEventListener("resize", resize); resize();

/* ---------------- Audio ---------------- */
let ac = null;
const PENTA = [0, 2, 4, 7, 9];
function note(freq, vol) {
  if (!ac) return;
  const t = ac.currentTime;
  const g = ac.createGain(), lp = ac.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.setValueAtTime(3500, t);
  lp.frequency.exponentialRampToValueAtTime(500, t + 1.4);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(Math.max(0.02, vol), t + 0.008);
  g.gain.exponentialRampToValueAtTime(0.0001, t + 1.6);
  [["triangle", 1, 1], ["sine", 2, 0.35]].forEach(([type, m, a]) => {
    const o = ac.createOscillator(); o.type = type; o.frequency.value = freq * m;
    const og = ac.createGain(); og.gain.value = a;
    o.connect(og).connect(lp); o.start(t); o.stop(t + 1.7);
  });
  lp.connect(g).connect(ac.destination);
}
function freqFromLength(len) {            // corde courte = note aiguë
  const x = 1 - Math.min(Math.max(len / W, 0.08), 1);   // 0..0.92
  const step = Math.round(x * 5 * PENTA.length);         // ~5 octaves max
  const oct = Math.floor(step / 5), deg = PENTA[step % 5];
  return 130.81 * Math.pow(2, (oct * 12 + deg) / 12);
}

/* ---------------- Corde (équation d'onde 1D) ---------------- */
const N = 140;
const u = new Float32Array(N + 1), v = new Float32Array(N + 1);
let energy = 0, hue = 190, presence = 0;
const sparks = [];

function pluck(t, amp) {
  const k = Math.max(2, Math.min(N - 2, Math.round(t * N)));
  const w = Math.max(3, Math.min(k, N - k));
  for (let i = 1; i < N; i++) {
    const f = Math.max(0, 1 - Math.abs(i - k) / w);
    u[i] += f * amp;
  }
}
function stepString() {
  for (let s = 0; s < 3; s++) {
    for (let i = 1; i < N; i++) v[i] += 0.38 * (u[i - 1] - 2 * u[i] + u[i + 1]);
    let e = 0;
    for (let i = 1; i < N; i++) { v[i] *= 0.9965; u[i] += v[i]; e += Math.abs(u[i]); }
    energy = e / N;
  }
}

/* ---------------- Entrées : mains ou pointeur ---------------- */
let landmarker = null, useCam = false;
let frame = { hands: [], anchors: null, tips: [] };
const TIPS = [12, 16, 20];                 // doigts qui grattent (pouce/index = ancres)

function toScreen(l) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh);
  return { x: (1 - l.x) * vw * s + (W - vw * s) / 2, y: l.y * vh * s + (H - vh * s) / 2 };
}
function readHands(now) {
  const r = landmarker.detectForVideo(video, now);
  const hands = r.landmarks.map(h => h.map(toScreen))
    .sort((a, b) => a[0].x - b[0].x);       // gauche -> droite à l'écran
  let anchors = null;
  if (hands.length >= 2) {
    anchors = [hands[0], hands[hands.length - 1]].map(h => ({
      x: (h[4].x + h[8].x) / 2, y: (h[4].y + h[8].y) / 2 }));
  }
  const tips = [];
  hands.forEach((h, hi) => TIPS.forEach(i => tips.push({ id: hi + "-" + i, ...h[i] })));
  frame = { hands, anchors, tips };
}
const ptr = { x: -1, y: -1, down: false };
function readPointer() {
  frame = { hands: [], anchors: [{ x: W * .15, y: H * .42 }, { x: W * .85, y: H * .42 }],
            tips: ptr.x < 0 ? [] : [{ id: "ptr", x: ptr.x, y: ptr.y }] };
}
addEventListener("pointermove", e => { ptr.x = e.clientX; ptr.y = e.clientY; });
addEventListener("pointerdown", e => { ptr.x = e.clientX; ptr.y = e.clientY; });

/* ---------------- Détection du passage d'un doigt ---------------- */
const prevSide = new Map();
function detectPlucks(A, B) {
  const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  for (const p of frame.tips) {
    const d = (p.x - A.x) * nx + (p.y - A.y) * ny;
    const t = ((p.x - A.x) * ux + (p.y - A.y) * uy) / L;
    const prev = prevSide.get(p.id);
    prevSide.set(p.id, d);
    if (prev === undefined || t < 0.03 || t > 0.97) continue;
    if (Math.sign(prev) !== Math.sign(d) && Math.abs(prev - d) < 160) {
      const speed = Math.abs(prev - d);
      const amp = Math.min(46, 10 + speed * 1.6) * Math.sign(d - prev) * -1;
      pluck(t, amp);
      note(freqFromLength(L), Math.min(0.35, 0.06 + speed / 120));
      hue = (hue + 47) % 360;
      for (let i = 0; i < 14; i++) sparks.push({
        x: A.x + dx * t, y: A.y + dy * t,
        vx: (Math.random() - .5) * 3, vy: (Math.random() - .5) * 3, life: 1, h: hue });
    }
  }
}

/* ---------------- Dessin ---------------- */
const BONES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
function drawHands() {
  ctx.lineWidth = 3; ctx.strokeStyle = "#22c33b"; ctx.lineCap = "round";
  for (const h of frame.hands) {
    ctx.beginPath();
    for (const [a, b] of BONES) { ctx.moveTo(h[a].x, h[a].y); ctx.lineTo(h[b].x, h[b].y); }
    ctx.stroke();
    ctx.fillStyle = "#e02020";
    for (const p of h) { ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 7); ctx.fill(); }
  }
}
function drawString(A, B, a) {
  const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    pts.push([A.x + dx * t + nx * u[i], A.y + dy * t + ny * u[i]]);
  }
  const path = () => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); };
  const glow = Math.min(1, energy / 6);

  ctx.save(); ctx.globalAlpha = a;
  // graduations
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 1.5;
  const ticks = 14;
  for (let k = 0; k <= ticks; k++) {
    const t = k / ticks, i = Math.round(t * N);
    const x = pts[i][0], y = pts[i][1], s = k % 2 ? 4 : 7;
    ctx.beginPath(); ctx.moveTo(x - nx * s, y - ny * s); ctx.lineTo(x + nx * s, y + ny * s); ctx.stroke();
  }
  // halo + coeur
  ctx.globalCompositeOperation = "lighter"; ctx.lineJoin = "round"; ctx.lineCap = "round";
  for (const [w, al, l] of [[26, .10, 55], [14, .18, 60], [7, .35, 70]]) {
    ctx.strokeStyle = `hsla(${hue},100%,${l}%,${al * (0.4 + glow * 1.6)})`;
    ctx.lineWidth = w; path(); ctx.stroke();
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5; path(); ctx.stroke();
  // extrémités
  for (const p of [A, B]) {
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.stroke();
  }
  // étincelles le long de la corde quand elle vibre
  if (glow > .05 && Math.random() < glow * 2) {
    const i = (Math.random() * N) | 0;
    sparks.push({ x: pts[i][0], y: pts[i][1], vx: (Math.random() - .5) * 1.5,
      vy: (Math.random() - .5) * 1.5 + .3, life: 1, h: hue + Math.random() * 60 });
  }
  ctx.restore();
}
function drawSparks() {
  ctx.save(); ctx.globalCompositeOperation = "lighter";
  for (let i = sparks.length - 1; i >= 0; i--) {
    const s = sparks[i];
    s.x += s.vx; s.y += s.vy; s.vy += 0.03; s.life -= 0.018;
    if (s.life <= 0) { sparks.splice(i, 1); continue; }
    ctx.fillStyle = `hsla(${s.h},100%,75%,${s.life})`;
    ctx.fillRect(s.x, s.y, 2.5, 2.5);
  }
  ctx.restore();
}

/* ---------------- Boucle ---------------- */
let last = -1;
function loop(now) {
  requestAnimationFrame(loop);
  ctx.clearRect(0, 0, W, H);
  if (useCam && video.readyState >= 2) {
    const vw = video.videoWidth, vh = video.videoHeight, s = Math.max(W / vw, H / vh);
    ctx.save(); ctx.translate(W, 0); ctx.scale(-1, 1);
    ctx.drawImage(video, (W - vw * s) / 2, (H - vh * s) / 2, vw * s, vh * s);
    ctx.restore();
    if (video.currentTime !== last) { last = video.currentTime; readHands(now); }
  } else {
    ctx.fillStyle = "#0b0b12"; ctx.fillRect(0, 0, W, H);
    readPointer();
  }
  drawHands();
  const have = !!frame.anchors;
  presence += ((have ? 1 : 0) - presence) * 0.2;
  if (have) {
    detectPlucks(frame.anchors[0], frame.anchors[1]);
    stepString();
  } else { u.fill(0); v.fill(0); prevSide.clear(); }
  if (presence > 0.02 && have) drawString(frame.anchors[0], frame.anchors[1], presence);
  drawSparks();
}

/* ---------------- Démarrage ---------------- */
document.getElementById("go").onclick = async () => {
  ac = new (window.AudioContext || window.webkitAudioContext)();
  ac.resume();
  msg.textContent = "Chargement…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia(
      { video: { facingMode: "user", width: 1280, height: 720 }, audio: false });
    video.srcObject = stream; await video.play();
    const files = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    const make = delegate => HandLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate },
      runningMode: "VIDEO", numHands: 2 });
    try { landmarker = await make("GPU"); } catch { landmarker = await make("CPU"); }
    useCam = true;
  } catch (e) {
    console.warn(e);
    alert("Caméra ou modèle indisponible (" + (e && e.message || e) + ") : mode souris/doigt activé (glisse sur la corde).");
  }
  startEl.classList.add("hidden");
};
requestAnimationFrame(loop);
