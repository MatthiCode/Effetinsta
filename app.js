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
function note(freq, vol, bright = 1) {
  if (!ac) return;
  const t = ac.currentTime;
  const g = ac.createGain(), lp = ac.createBiquadFilter();
  lp.type = "lowpass"; lp.frequency.setValueAtTime(Math.min(9000, freq * 6 * bright), t);
  lp.frequency.exponentialRampToValueAtTime(Math.max(300, freq * 1.5), t + 1.4);
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
  return 65.41 * Math.pow(2, (oct * 12 + deg) / 12);   // do grave (corde longue) -> aigu (corde courte)
}

/* Voix continue : un son tenu tant que la corde vibre, de même hauteur que sa longueur */
function setVoice(S, freq, level) {
  if (!ac) return;
  const t = ac.currentTime;
  if (level < 0.002) {
    if (S.voice) {
      const vc = S.voice; S.voice = null;
      vc.g.gain.setTargetAtTime(0, t, 0.08);
      vc.o1.stop(t + 0.6); vc.o2.stop(t + 0.6);
    }
    return;
  }
  if (!S.voice) {
    const g = ac.createGain(); g.gain.value = 0;
    const o1 = ac.createOscillator(), o2 = ac.createOscillator();
    o1.type = "sine"; o2.type = "triangle";
    const g2 = ac.createGain(); g2.gain.value = 0.25;
    o1.connect(g); o2.connect(g2).connect(g); g.connect(ac.destination);
    o1.start(); o2.start();
    S.voice = { o1, o2, g };
  }
  const vc = S.voice;
  vc.o1.frequency.setTargetAtTime(freq, t, 0.02);
  vc.o2.frequency.setTargetAtTime(freq * 2, t, 0.02);
  vc.g.gain.setTargetAtTime(level, t, 0.04);
}
function updateVoice(S) {
  const L = Math.hypot(S.B.x - S.A.x, S.B.y - S.A.y);
  setVoice(S, freqFromLength(L), Math.min(0.09, S.energy * 0.01));
}
function silence(S) { setVoice(S, 0, 0); }

/* ---------------- Corde (équation d'onde 1D) ---------------- */
const N = 140;
const MAX_STRINGS = 200;
const strings = [];                                  // cordes posées
let nextId = 0;
function newString(A, B) {
  const id = nextId++;
  return { id, A, B, u: new Float32Array(N + 1), v: new Float32Array(N + 1),
           energy: 0, hue: (id * 47 + 190) % 360, placedAt: performance.now() };
}
const sparks = [];

/* Sauvegarde locale : les traits restent après rechargement (coordonnées relatives à l'écran) */
const KEY = "corde-de-mains";
function save() {
  try {
    localStorage.setItem(KEY, JSON.stringify(strings.map(S =>
      [S.A.x / W, S.A.y / H, S.B.x / W, S.B.y / H])));
  } catch {}
}
function restore() {
  try {
    for (const [ax, ay, bx, by] of JSON.parse(localStorage.getItem(KEY) || "[]"))
      strings.push(newString({ x: ax * W, y: ay * H }, { x: bx * W, y: by * H }));
  } catch {}
}

function pluck(S, t, amp) {
  const u = S.u;
  const k = Math.max(2, Math.min(N - 2, Math.round(t * N)));
  const w = Math.max(3, Math.min(k, N - k));
  for (let i = 1; i < N; i++) {
    const f = Math.max(0, 1 - Math.abs(i - k) / w);
    u[i] += f * amp;
  }
}
function stepString(S) {
  const { u, v } = S;
  for (let s = 0; s < 3; s++) {
    for (let i = 1; i < N; i++) v[i] += 0.38 * (u[i - 1] - 2 * u[i] + u[i + 1]);
    let e = 0;
    for (let i = 1; i < N; i++) { v[i] *= 0.9965; u[i] += v[i]; e += Math.abs(u[i]); }
    S.energy = e / N;
  }
}

/* ---------------- Entrées : mains ou pointeur ---------------- */
let landmarker = null, useCam = false;
let frame = { hands: [], sides: null, tips: [] };
const TIPS = [4, 8, 12, 16, 20];           // tous les bouts de doigts peuvent gratter la corde posée
const pin = [false, false];                // pincement avec hystérésis (gauche, droite)
const line = { mode: "none", A: null, B: null, idle: 0 };  // none | drawing (la corde en cours de tracé)
const draft = newString({ x: 0, y: 0 }, { x: 0, y: 0 });

function toScreen(l) {
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh);
  return { x: (1 - l.x) * vw * s + (W - vw * s) / 2, y: l.y * vh * s + (H - vh * s) / 2 };
}
function readHands(now) {
  const r = landmarker.detectForVideo(video, now);
  const hands = r.landmarks.map(h => h.map(toScreen))
    .sort((a, b) => a[0].x - b[0].x);       // gauche -> droite à l'écran
  let sides = null;
  if (hands.length >= 2) {
    sides = [hands[0], hands[hands.length - 1]].map((h, k) => {
      const size = Math.hypot(h[0].x - h[9].x, h[0].y - h[9].y) || 1;
      const r = Math.hypot(h[4].x - h[8].x, h[4].y - h[8].y) / size;
      pin[k] = pin[k] ? r < 0.42 : r < 0.28;
      return { pinch: pin[k], mid: { x: (h[4].x + h[8].x) / 2, y: (h[4].y + h[8].y) / 2 } };
    });
  } else { pin[0] = pin[1] = false; }
  const tips = [];
  hands.forEach((h, hi) => TIPS.forEach(i => tips.push({ id: hi + "-" + i, ...h[i] })));
  frame = { hands, sides, tips };
}
const ptr = { x: -1, y: -1, down: false };
function readPointer() {            // repli sans caméra : glisser = tracer, survoler = gratter
  frame = { hands: [], sides: null,
            tips: ptr.x < 0 || ptr.down ? [] : [{ id: "ptr", x: ptr.x, y: ptr.y }] };
}
addEventListener("pointermove", e => {
  ptr.x = e.clientX; ptr.y = e.clientY;
  if (!useCam && ptr.down) line.B = { x: ptr.x, y: ptr.y };
});
addEventListener("pointerdown", e => {
  if (useCam || e.target.closest("button") || !startEl.classList.contains("hidden")) return;
  ptr.x = e.clientX; ptr.y = e.clientY; ptr.down = true;
  line.mode = "drawing"; line.A = { x: ptr.x, y: ptr.y }; line.B = { ...line.A };
});
addEventListener("pointerup", () => {
  if (useCam || !ptr.down) return;
  ptr.down = false; settle();
});

/* ---------------- Tracer / poser la corde ---------------- */
function settle() {
  const L = Math.hypot(line.B.x - line.A.x, line.B.y - line.A.y);
  line.mode = "none";
  if (L < 80) return;
  const S = newString(line.A, line.B);
  strings.push(S);
  if (strings.length > MAX_STRINGS) silence(strings.shift());
  pluck(S, 0.5, 10); note(freqFromLength(L), 0.1);
  save();
}
function updateLine() {
  if (!useCam) return;
  const s = frame.sides;
  if (line.mode !== "drawing") {
    if (s && s[0].pinch && s[1].pinch) {      // deux mains pincées : on commence à tracer
      line.mode = "drawing"; line.idle = 0;
      line.A = { ...s[0].mid }; line.B = { ...s[1].mid };
    }
    return;
  }
  if (s && s[0].pinch) line.A = { ...s[0].mid };
  if (s && s[1].pinch) line.B = { ...s[1].mid };
  const any = s && (s[0].pinch || s[1].pinch);
  line.idle = any ? 0 : line.idle + 1;
  if (line.idle > 5) settle();                // plus aucune main ne pince : la corde se pose
}

/* ---------------- Détection du passage d'un doigt ---------------- */
const prevSide = new Map();
function detectPlucks(S) {
  const { A, B } = S;
  const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  for (const p of frame.tips) {
    const d = (p.x - A.x) * nx + (p.y - A.y) * ny;
    const t = ((p.x - A.x) * ux + (p.y - A.y) * uy) / L;
    const key = S.id + ":" + p.id, prev = prevSide.get(key);
    prevSide.set(key, d);
    if (prev === undefined || t < 0.05 || t > 0.95) continue;
    if (performance.now() - S.placedAt < 400) continue;   // laisse les doigts se dégager
    if (Math.sign(prev) !== Math.sign(d) && Math.abs(prev - d) < 160 && Math.abs(prev - d) > 3) {
      const speed = Math.abs(prev - d);
      const amp = Math.min(46, 10 + speed * 1.6) * Math.sign(d - prev) * -1;
      pluck(S, t, amp);
      note(freqFromLength(L), Math.min(0.3, 0.05 + speed / 140), 0.6 + Math.abs(t - 0.5) * 2.4);
      S.hue = (S.hue + 47) % 360;
      for (let i = 0; i < 14; i++) sparks.push({
        x: A.x + dx * t, y: A.y + dy * t,
        vx: (Math.random() - .5) * 3, vy: (Math.random() - .5) * 3, life: 1, h: S.hue });
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
function drawString(S, a) {
  const { A, B, u, hue } = S;
  const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1;
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    pts.push([A.x + dx * t + nx * u[i], A.y + dy * t + ny * u[i]]);
  }
  const path = () => { ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); };
  const glow = Math.min(1, S.energy / 6);

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
  updateLine();
  for (const S of strings) { detectPlucks(S); stepString(S); updateVoice(S); drawString(S, 1); }
  if (line.mode === "drawing") {
    draft.A = line.A; draft.B = line.B; drawString(draft, 1);
    setVoice(draft, freqFromLength(Math.hypot(line.B.x - line.A.x, line.B.y - line.A.y)), 0.04);
  } else silence(draft);
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

document.getElementById("clear").onclick = () => { strings.forEach(silence); strings.length = 0; prevSide.clear(); save(); };
restore();
