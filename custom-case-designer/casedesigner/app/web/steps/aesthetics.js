// Шаг «Эстетика»: фото анфас с улыбкой и с ретрактором, пары точек «фото — скан», камера, горизонт, средняя линия,
// точки лица и размеры зубов по лицу (сервер — app/steps/aesthetics.py).
import { $, busy, choose, fmt, hide, render, state, toast, viewer } from '../core.js';
import { get, post, run } from '../api.js';
import { icons } from '../icons.js';

const KINDS = { smile: 'С улыбкой', retracted: 'С ретрактором' };
// Точки лица по порядку отметки: пары справа и слева пациента (на фото анфас правая сторона пациента — слева).
const FACE_STEPS = [
  ['pupil_right', 'правый зрачок'], ['pupil_left', 'левый зрачок'],
  ['endocanthion_right', 'внутренний угол правого глаза'], ['endocanthion_left', 'внутренний угол левого глаза'],
  ['ala_right', 'правое крыло носа'], ['ala_left', 'левое крыло носа'],
];
const COLORS = { pair: '#5ec8ff', pending: '#ffd166', midline: '#ff8fab', horizon: '#9be564', face: '#c3a6ff', overlay: 'rgba(94, 200, 255, 0.55)',
  scan: 'rgba(80, 220, 255, 0.95)', photo: 'rgba(255, 214, 0, 0.9)' };
const ROTATE_DEG_PER_PX = 0.1; // ПКМ: поворот скана, как в exocad Smile Design
const NO_ADJ = () => ({ pitch: 0, yaw: 0, roll: 0, dx: 0, dy: 0, scale: 1 });

const aes = {
  kind: 'smile', mode: null, // null | pair | horizon | midline | face
  pending: { px: null, point: null }, horizon: [], faceStep: 0, overlay: true,
  view: null, // { k, ox, oy } — пиксели фото → точки окна; null — по размеру окна
  image: null, imageKey: '', drag: null, down3d: null,
  adj: NO_ADJ(), adjTimer: null, // ручная поправка скана поверх фото — пока тянут мышью (сервер получает по отпусканию)
};

// ---------- камера фото в окне: поправка и проекция, как aesthetic.adjust_camera / Camera.project ----------
function mat(a, b) { return a.map((r) => b[0].map((_, j) => r.reduce((s, x, k) => s + x * b[k][j], 0))); }
function rot(axis, deg) {
  const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
  if (axis === 0) return [[1, 0, 0], [0, c, -s], [0, s, c]];
  if (axis === 1) return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}
const mulv = (M, v) => M.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);

export function adjustCamera(cam, d) {
  if (!cam || !d) return cam;
  const sgn = cam.mirrored ? -1 : 1;
  const Q = mat(rot(2, sgn * d.roll), mat(rot(1, sgn * d.yaw), rot(0, d.pitch))); // как Rotation.from_euler('xyz')
  const A = cam.anchor;
  const RA = mulv(cam.R, A), QRA = mulv(Q, RA);
  const R = mat(Q, cam.R);
  const t = [0, 1, 2].map((i) => RA[i] + cam.t[i] - QRA[i]);
  const ca = RA.map((x, i) => x + cam.t[i]);
  const ua = cam.K[0][0] * ca[0] / ca[2] + cam.K[0][2], va = cam.K[1][1] * ca[1] / ca[2] + cam.K[1][2];
  const K = cam.K.map((r) => [...r]);
  K[0][0] *= d.scale; K[1][1] *= d.scale;
  K[0][2] = cam.K[0][2] + (1 - d.scale) * (ua - cam.K[0][2]) + sgn * d.dx;
  K[1][2] = cam.K[1][2] + (1 - d.scale) * (va - cam.K[1][2]) + d.dy;
  return { ...cam, K, R, t };
}

export function projectPoints(cam, pts) {
  const [[f, , cx], [, g, cy]] = cam.K;
  const W = cam.size?.[0] ?? 0;
  return pts.map((p) => {
    const c = [0, 1, 2].map((i) => cam.R[i][0] * p[0] + cam.R[i][1] * p[1] + cam.R[i][2] * p[2] + cam.t[i]);
    const u = f * c[0] / c[2] + cx, v = g * c[1] / c[2] + cy;
    return [cam.mirrored ? W - u : u, v];
  });
}

const adjusting = () => Object.entries(aes.adj).some(([k, v]) => (k === 'scale' ? Math.abs(v - 1) > 1e-6 : Math.abs(v) > 1e-6));

async function commitAdjust() {
  if (!adjusting()) return;
  const manual = aes.adj;
  try {
    state.aesthetics = await post('aesthetics/adjust', { kind: aes.kind, manual });
  } catch (e) { toast(e.message); }
  aes.adj = NO_ADJ();
  render();
}

const info = () => state.aesthetics;
const photo = () => info()?.photos?.[aes.kind] || null;

export async function prepareAesthetics() {
  try { state.aesthetics = await get('aesthetics'); }
  catch (e) { toast(e.message); }
  render();
  autoRefine();
}

async function send(path, body) {
  try {
    state.aesthetics = await post(path, { kind: aes.kind, ...body });
  } catch (e) {
    toast(e.message);
  }
  render();
  autoRefine();
}

// Доводка нейросетью — фоновой задачей (десятки секунд): окно не ждёт, по готовности обновляется само.
let autoRunning = false;
async function autoRefine() {
  if (autoRunning || !Object.values(info()?.photos || {}).some((p) => p?.auto_pending)) return;
  autoRunning = true;
  try {
    const result = await run('aesthetics/auto', {});
    if (result) state.aesthetics = result;
  } catch (e) { toast(e.message); }
  autoRunning = false;
  if (state.step === 'aesthetics') render();
  if (Object.values(info()?.photos || {}).some((p) => p?.auto_pending)) setTimeout(autoRefine, 500); // пары меняли, пока считали
}

// ---------- окно фото ----------
function photoView() {
  let el = $('#viewPhoto');
  if (el) return el;
  el = document.createElement('section');
  el.className = 'view photo';
  el.id = 'viewPhoto';
  el.innerHTML = '<div class="view-label"><b>Фото</b> <span class="pos"></span></div><canvas></canvas><div class="view-hint"></div><div class="empty"></div>';
  $('.main').appendChild(el);
  const canvas = $('canvas', el);
  new ResizeObserver(() => drawPhoto()).observe(el);
  canvas.addEventListener('wheel', (e) => {
    if (!aes.image || !photo()) return;
    e.preventDefault();
    if (e.shiftKey && photo().camera) { // Shift + колесо — масштаб скана поверх фото
      aes.adj.scale *= Math.exp(-(e.deltaY || e.deltaX) * 0.0005);
      clearTimeout(aes.adjTimer);
      aes.adjTimer = setTimeout(commitAdjust, 400);
      drawPhoto();
      return;
    }
    const v = currentView(canvas);
    const f = Math.exp(-e.deltaY * 0.0015);
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    aes.view = { k: v.k * f, ox: mx - (mx - v.ox) * f, oy: my - (my - v.oy) * f };
    drawPhoto();
  }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 && e.button !== 1 && e.button !== 2) return;
    const v = currentView(canvas);
    // ПКМ — повернуть скан, Shift + ЛКМ — сдвинуть скан (поверх фото, как в exocad Smile Design)
    const adjust = photo()?.camera && (e.button === 2 ? 'rotate' : e.button === 0 && e.shiftKey ? 'move' : null);
    if (e.button === 2 && !adjust) return;
    aes.drag = { x: e.clientX, y: e.clientY, ox: v.ox, oy: v.oy, moved: false, button: e.button, adjust,
      start: { ...aes.adj }, k: v.k };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    const d = aes.drag;
    if (!d) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true;
    if (d.moved && d.adjust) {
      const mx = e.clientX - d.x, my = e.clientY - d.y;
      if (d.adjust === 'rotate') aes.adj = { ...d.start, yaw: d.start.yaw + mx * ROTATE_DEG_PER_PX, pitch: d.start.pitch - my * ROTATE_DEG_PER_PX };
      else aes.adj = { ...d.start, dx: d.start.dx + mx / d.k, dy: d.start.dy + my / d.k };
      drawPhoto();
      return;
    }
    if (d.moved) {
      aes.view = { ...currentView(canvas), ox: d.ox + e.clientX - d.x, oy: d.oy + e.clientY - d.y };
      drawPhoto();
    }
  });
  canvas.addEventListener('pointerup', (e) => {
    const d = aes.drag;
    aes.drag = null;
    if (d?.adjust) { if (d.moved) commitAdjust(); return; }
    if (!d || d.moved || d.button !== 0) return;
    const r = canvas.getBoundingClientRect();
    const v = currentView(canvas);
    const p = photo();
    if (!p) return;
    const u = (e.clientX - r.left - v.ox) / v.k, w = (e.clientY - r.top - v.oy) / v.k;
    if (u < 0 || w < 0 || u > p.size[0] || w > p.size[1]) return;
    photoClick([round1(u), round1(w)]);
  });
  canvas.addEventListener('dblclick', () => { aes.view = null; drawPhoto(); });
  return el;
}

const round1 = (x) => Math.round(x * 10) / 10;

function currentView(canvas) {
  const p = photo();
  if (aes.view) return aes.view;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!p) return { k: 1, ox: 0, oy: 0 };
  const k = Math.min(cw / p.size[0], ch / p.size[1]) * 0.96;
  return { k, ox: (cw - p.size[0] * k) / 2, oy: (ch - p.size[1] * k) / 2 };
}

function loadImage(p) {
  const key = `${aes.kind}-${p.version}-${state.incognito ? 1 : 0}-${state.incognito ? p.pairs.length : 0}`;
  if (key === aes.imageKey) return;
  aes.imageKey = key;
  const img = new Image();
  img.onload = () => { if (aes.imageKey === key) { aes.image = img; drawPhoto(); } };
  img.onerror = () => toast('Не удалось показать фото');
  img.src = `/api/aesthetics/${aes.kind}/image?k=${encodeURIComponent(key)}`;
}

export function drawPhoto() {
  const el = photoView();
  const canvas = $('canvas', el);
  const p = photo();
  $('.empty', el).innerHTML = p ? '' : `<span>Загрузите фото ${KINDS[aes.kind].toLowerCase()}</span>`;
  $('.pos', el).textContent = p ? KINDS[aes.kind].toLowerCase() : '';
  $('.view-hint', el).textContent = p ? hint() : '';
  $('.view-hint', el).hidden = !p;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, cw, ch);
  canvas.style.cursor = aes.mode ? 'crosshair' : 'grab';
  if (!p) { aes.image = null; aes.imageKey = ''; return; }
  loadImage(p);
  if (!aes.image) return;
  const v = currentView(canvas);
  const at = (q) => [v.ox + q[0] * v.k, v.oy + q[1] * v.k];
  g.drawImage(aes.image, v.ox, v.oy, p.size[0] * v.k, p.size[1] * v.k);
  const cam = p.camera ? adjustCamera({ ...p.camera, size: p.size }, aes.adj) : null;
  if (aes.overlay && cam) {
    const o = p.overlay || {};
    const poly = (pts, color, width) => {
      g.strokeStyle = color; g.lineWidth = width; g.beginPath();
      pts.forEach((q, i) => { const [x, y] = at(q); if (i) g.lineTo(x, y); else g.moveTo(x, y); });
      g.closePath(); g.stroke();
    };
    if (!adjusting()) (o.photo || []).forEach((c) => poly(c.px, COLORS.photo, 1.2)); // зубы на фото (нейросеть)
    (o.scan || []).forEach((c) => poly(projectPoints(cam, c.points), COLORS.scan, 1.4)); // зубы скана
    if (o.points?.length) {
      g.fillStyle = COLORS.overlay;
      for (const q of projectPoints(cam, o.points)) { const [x, y] = at(q); g.fillRect(x - 0.8, y - 0.8, 1.6, 1.6); }
    }
  }
  // горизонт — линия через центр кадра под углом horizon_deg
  const h = (p.horizon_deg * Math.PI) / 180;
  const c = at([p.size[0] / 2, p.size[1] / 2]);
  const L = p.size[0] * v.k * 0.6;
  line(g, [c[0] - L * Math.cos(h), c[1] - L * Math.sin(h)], [c[0] + L * Math.cos(h), c[1] + L * Math.sin(h)], COLORS.horizon, [6, 5]);
  if (p.midline_px.length >= 2) {
    const pts = p.midline_px.map(at);
    line(g, pts[0], pts[pts.length - 1], COLORS.midline, []);
  }
  p.midline_px.forEach((q) => dot(g, at(q), COLORS.midline, 3.5));
  aes.horizon.forEach((q) => dot(g, at(q), COLORS.horizon, 4));
  Object.values(p.face || {}).forEach((q) => dot(g, at(q), COLORS.face, 4));
  p.pairs.forEach((pair, i) => { dot(g, at(pair.px), COLORS.pair, 5); label(g, at(pair.px), String(i + 1)); });
  if (aes.pending.px) dot(g, at(aes.pending.px), COLORS.pending, 6);
}

function line(g, a, b, color, dash) {
  g.save();
  g.strokeStyle = color;
  g.lineWidth = 1.5;
  g.setLineDash(dash);
  g.beginPath();
  g.moveTo(...a);
  g.lineTo(...b);
  g.stroke();
  g.restore();
}

function dot(g, [x, y], color, r) {
  g.beginPath();
  g.arc(x, y, r, 0, Math.PI * 2);
  g.fillStyle = color;
  g.fill();
  g.lineWidth = 1.5;
  g.strokeStyle = 'rgba(0,0,0,.7)';
  g.stroke();
}

function label(g, [x, y], text) {
  g.font = '600 11px system-ui, sans-serif';
  g.lineWidth = 3;
  g.strokeStyle = 'rgba(0,0,0,.8)';
  g.strokeText(text, x + 7, y - 6);
  g.fillStyle = '#fff';
  g.fillText(text, x + 7, y - 6);
}

function hint() {
  if (aes.mode === 'pair') {
    if (aes.pending.px && !aes.pending.point) return 'теперь — та же точка на зубах в 3D';
    if (aes.pending.point && !aes.pending.px) return 'теперь — та же точка на фото';
    return 'точка на фото и та же точка на зубах в 3D';
  }
  if (aes.mode === 'horizon') return aes.horizon.length ? 'вторая точка горизонтали' : 'две точки на истинной горизонтали';
  if (aes.mode === 'midline') return 'точки средней линии лица: переносица, кончик носа, фильтрум, подбородок';
  if (aes.mode === 'face') return FACE_STEPS[aes.faceStep]?.[1] || '';
  return photo()?.camera
    ? 'колесо — масштаб окна, перетаскивание — сдвиг фото; скан: ПКМ — повернуть, Shift + ЛКМ — сдвинуть, Shift + колесо — масштаб'
    : 'колесо — масштаб, перетаскивание — сдвиг, двойной щелчок — весь кадр';
}

function photoClick(px) {
  if (aes.mode === 'pair') {
    aes.pending.px = px;
    return aes.pending.point ? submitPair() : drawPhoto();
  }
  if (aes.mode === 'horizon') {
    aes.horizon.push(px);
    if (aes.horizon.length < 2) return drawPhoto();
    let [dx, dy] = [aes.horizon[1][0] - aes.horizon[0][0], aes.horizon[1][1] - aes.horizon[0][1]];
    if (dx < 0) { dx = -dx; dy = -dy; }
    aes.horizon = [];
    aes.mode = null;
    return send('aesthetics/set', { horizon_deg: Math.round((Math.atan2(dy, dx) * 180) / Math.PI * 100) / 100 });
  }
  if (aes.mode === 'midline') {
    const pts = [...(photo().midline_px || []), px].slice(-8);
    return send('aesthetics/set', { midline_px: pts });
  }
  if (aes.mode === 'face') {
    const [key] = FACE_STEPS[aes.faceStep] || [];
    if (!key) return;
    aes.faceStep += 1;
    if (aes.faceStep >= FACE_STEPS.length) aes.mode = null;
    return send('aesthetics/set', { face: { [key]: px } });
  }
}

async function submitPair() {
  const { px, point } = aes.pending;
  aes.pending = { px: null, point: null };
  viewer.remove('lm:aes:pending');
  await send('aesthetics/pair', { px, point });
}

// ---------- точка на скане в 3D ----------
export function aestheticPointerDown(e) {
  aes.down3d = e.button === 0 ? [e.clientX, e.clientY] : null;
}

export function aestheticPick(e) {
  const started = aes.down3d;
  aes.down3d = null;
  if (state.step !== 'aesthetics' || aes.mode !== 'pair' || !started || e.button !== 0) return;
  if (e.target !== viewer.renderer.domElement) return;
  if (Math.hypot(e.clientX - started[0], e.clientY - started[1]) > 4) return;
  const upper = state.scans.find((s) => s.jaw === 'upper' && s.registered && s.role !== 'bite');
  if (!upper) return toast('Нет совмещённого скана верхней челюсти');
  const point = viewer.pick(upper.id, e.clientX, e.clientY);
  if (!point) return toast('Щёлкните по зубам верхнего скана');
  aes.pending.point = point;
  viewer.setMarker('aes:pending', point, COLORS.pending);
  if (aes.pending.px) return submitPair();
  render();
}

function drawMarkers() {
  for (const key of [...viewer.objects.keys()]) {
    if (key.startsWith('lm:aes:pair')) viewer.remove(key);
  }
  if (state.step !== 'aesthetics') return;
  (photo()?.pairs || []).forEach((pair, i) => viewer.setMarker(`aes:pair${i}`, pair.point, COLORS.pair));
}

export function endAesthetics() {
  aes.mode = null;
  aes.pending = { px: null, point: null };
  aes.horizon = [];
  viewer.remove('lm:aes:pending');
  for (const key of [...viewer.objects.keys()]) {
    if (key.startsWith('lm:aes:')) viewer.remove(key);
  }
}

// ---------- действия панели ----------
export async function aestheticAction(action, arg) {
  const p = photo();
  if (action === 'kind') {
    endAesthetics();
    aes.kind = arg;
    aes.view = null;
    return render();
  }
  if (action === 'load') {
    const paths = await choose('photo', `Фото ${KINDS[aes.kind].toLowerCase()} (JPEG, PNG)`);
    if (!paths) return;
    endAesthetics();
    aes.view = null;
    return send('aesthetics/photo', { path: paths[0] });
  }
  if (action === 'remove') { endAesthetics(); return send('aesthetics/photo/remove', {}); }
  if (action === 'mode') {
    const next = aes.mode === arg ? null : arg;
    endAesthetics();
    aes.mode = next;
    if (next === 'face') aes.faceStep = Math.max(0, FACE_STEPS.findIndex(([k]) => !p?.face?.[k]));
    return render();
  }
  if (action === 'unpair') return send('aesthetics/pair/remove', { index: arg === 'all' ? null : Number(arg) });
  if (action === 'overlay') { aes.overlay = !aes.overlay; return render(); }
  if (action === 'refine') {
    return busy('Уточнение по зубам', async (progress) => {
      state.aesthetics = await run('aesthetics/refine', { kind: aes.kind }, progress);
    }).then(render);
  }
  if (action === 'resetcam') { aes.adj = NO_ADJ(); return send('aesthetics/camera/reset', {}); }
  if (action === 'horizon') {
    const h = arg === 'zero' ? 0 : Math.round(((p?.horizon_deg || 0) + Number(arg)) * 100) / 100;
    return send('aesthetics/set', { horizon_deg: Math.max(-45, Math.min(45, h)) });
  }
  if (action === 'midreset') return send('aesthetics/set', { midline_px: [] });
  if (action === 'facereset') return send('aesthetics/set', { face: Object.fromEntries(FACE_STEPS.map(([k]) => [k, null])) });
}

// поля панели: расстояние до камеры, доводка моделью (событие change — app.js)
export function aestheticChange(e) {
  const t = e.target;
  if (t.matches('[data-aes-distance]')) return send('aesthetics/set', { distance_mm: Number(t.value) || 0 });
  if (t.matches('[data-aes-auto]')) return send('aesthetics/set', { auto: t.checked });
}

const side = (mm, right, left) => (mm > 0 ? right : left);

function cameraHtml(p) {
  if (!p.camera) {
    const why = p.fit_error && !p.fit_error.startsWith('пар точек') ? `<div class="warning warn">${icons.warn}<span>${hide(p.fit_error)}</span></div>` : '';
    return `${why}<p class="muted small">Камера фото — после ${p.need} пар (как в exocad: две сходные точки, лучше подальше друг от друга —
      например, бугры клыков); дальше доводит нейросеть по зубам.</p>`;
  }
  const c = p.camera;
  const a = p.auto_info;
  const auto = c.fixed ? (a?.manual ? 'Поправлено вручную — камера запомнена.' : 'Уточнено по зубам — камера запомнена.')
    : p.auto_pending ? 'Доводка по зубам… (нейросеть, до полуминуты; пока — по парам)'
    : a?.used ? `Доводка по зубам ${a.teeth.join(', ')}: совпадение ${fmt(a.iou_before, 2)} → ${fmt(a.iou_after, 2)}, край ${fmt(a.edge_px, 1)} px.`
      : a?.error ? `Без доводки: ${hide(a.error)}.` : '';
  return `<div class="metrics">
      <div class="metric"><span>По парам, px</span><b class="${c.check ? 'warn' : 'ok'}">${fmt(c.residual_px, 1)}</b></div>
      <div class="metric"><span>До камеры, мм</span><b>${fmt(c.distance_mm, 0)}</b></div>
      <div class="metric"><span>Фокус, мм (35)</span><b>${fmt(c.focal_35mm, 0)}</b></div></div>
    ${auto ? `<p class="muted small">${auto}</p>` : ''}
    ${c.check ? '<div class="warning warn">' + icons.warn + '<span>Точки пар легли далеко — проверьте пары (номера на фото и точки в 3D).</span></div>' : ''}
    ${c.mirrored ? '<p class="muted small">Фото зеркальное (фронтальная камера) — учтено.</p>' : ''}`;
}

function alignHtml(p) {
  return `<div class="label">Совмещение</div>
    <div class="aes-row"><span>Расстояние до камеры</span><b>${fmt(p.distance_mm, 0)} мм${p.distance_set ? '' : ' (как exocad)'}</b></div>
    <input type="range" class="rehab-range" min="300" max="3000" step="50" value="${p.distance_mm}" data-aes-distance>
    <label class="joint-toggle ${p.auto_ready ? '' : 'disabled'}"><input type="checkbox" data-aes-auto ${p.auto ? 'checked' : ''} ${p.auto_ready ? '' : 'disabled'}>
      <span>Доводка нейросетью по зубам</span><small>${p.auto_ready ? 'SAM 2.1: контуры каждого зуба' : 'нет модели SAM 2.1 (models/sam2.1-tiny)'}</small></label>
    ${p.camera ? `<div class="row" style="gap:4px;margin-top:4px">
      <button class="btn ghost grow" data-aes="refine" ${p.auto_ready ? '' : 'disabled'}>Уточнить по зубам</button>
      ${p.camera.fixed ? '<button class="btn ghost" data-aes="resetcam" title="Снова по парам и доводке">Сбросить поправку</button>' : ''}</div>
      <p class="muted small">Поправить вручную — поверх фото: ПКМ — повернуть скан, Shift + ЛКМ — сдвинуть, Shift + колесо — масштаб.</p>` : ''}`;
}

function aestheticHtml(p) {
  const a = p.aesthetic;
  if (!a) return '';
  const rows = [`<div class="aes-row"><span>Окклюзионная горизонталь к истинной</span><b>${fmt(Math.abs(a.roll_deg), 1)}°${a.roll_deg ? side(a.roll_deg, ', выше справа', ', выше слева') : ''}</b></div>`];
  if (a.midline_offset_mm !== null) rows.push(`<div class="aes-row"><span>Средняя линия зубов от лица</span><b>${fmt(Math.abs(a.midline_offset_mm), 1)} мм${a.midline_offset_mm ? side(a.midline_offset_mm, ' вправо', ' влево') : ''}</b></div>`);
  if (a.facial_midline_tilt_deg !== null) rows.push(`<div class="aes-row"><span>Наклон средней линии лица</span><b>${fmt(Math.abs(a.facial_midline_tilt_deg), 1)}°</b></div>`);
  for (const [name, deg] of Object.entries(a.hints || {})) rows.push(`<div class="aes-row"><span>Линия ${name} к горизонту</span><b>${fmt(deg, 1)}°</b></div>`);
  return `<div class="label">По фото (стороны — пациента)</div>${rows.join('')}${(a.notes || []).map((n) => `<p class="muted small">${hide(n)}</p>`).join('')}`;
}

function guideHtml(p) {
  const widths = Object.entries(p.widths || {});
  if (!widths.length) return '';
  return `<div class="label">Размеры зубов по лицу — ориентир</div>
    ${widths.map(([k, v]) => `<div class="aes-row"><span>${k}</span><b>${fmt(v, 1)} мм</b></div>`).join('')}
    ${(p.guide || []).map((g) => `<div class="aes-row" title="${g.why}"><span>${g.name}</span><b>${fmt(g.mm, 1)} мм</b></div>`).join('')}
    <p class="muted small">Ориентир — средние соотношения из литературы, не правило: для выбора библиотеки зубов в Smile Design.
      Глубина точек лица — средний профиль; точнее — со сканом лица.</p>`;
}

export function renderAesthetics() {
  queueMicrotask(() => { drawPhoto(); drawMarkers(); });
  const data = info();
  const p = photo();
  const tabs = `<div class="seg" style="margin-bottom:10px">${Object.entries(KINDS).map(([k, t]) => {
    const cam = data?.photos?.[k]?.camera;
    return `<button class="${k === aes.kind ? 'on' : ''}" data-aes="kind" data-arg="${k}">${t}${cam ? ' ✓' : ''}</button>`;
  }).join('')}</div>`;
  const lead = `<h2>Эстетика</h2><p class="lead">Фото анфас с улыбкой и с ретрактором. Каждое совмещается со сканом верхней челюсти
    по точкам на зубах. По фото — эстетическая плоскость монтажа в шаге «Артикулятор» и «Вид Smile Design» в exocad.</p>`;
  if (!data) return `${lead}<p class="muted small">Загружаю…</p>`;
  if (!data.upper) {
    return `${lead}<div class="warning warn">${icons.warn}<span>Сначала поставьте скан верхней челюсти (шаг «Сканы»).</span></div>`;
  }
  const missing = data.missing?.length ? `<div class="warning warn">${icons.warn}<span>Фото кейса не найдено на месте: ${data.missing.join(', ')} — загрузите заново.</span></div>` : '';
  if (!p) {
    return `${lead}${missing}${tabs}<button class="btn primary wide" data-aes="load">${icons.plus}Загрузить фото ${KINDS[aes.kind].toLowerCase()}…</button>
      <p class="muted small">Анфас, голова прямо, фото с телефона как есть (поворот и фокус — по EXIF).</p>`;
  }
  const mode = (m, title, on) => `<button class="btn ${aes.mode === m ? 'on primary' : ''} grow" data-aes="mode" data-arg="${m}">${aes.mode === m ? on : title}</button>`;
  const pairs = p.pairs.map((_x, i) => `<button class="badge" data-aes="unpair" data-arg="${i}" title="Убрать пару ${i + 1}">${i + 1} ×</button>`).join(' ');
  const facePoints = FACE_STEPS.filter(([k]) => p.face?.[k]).length;
  const mount = data.mount_kind ? `<p class="muted small">${data.mounted_by_photo ? '✓ Модели смонтированы по фото.' : 'Монтаж по фото: в шаге «Артикулятор» — эстетическая плоскость.'}
      ${data.mount_kind === 'retracted' ? ' Фото с улыбкой нет — по фото с ретрактором.' : ''}</p>` : '';
  return `${lead}${missing}${tabs}
    <div class="card"><div class="card-head">${icons.eye}<h3>Фото ${KINDS[aes.kind].toLowerCase()}</h3>
      <button class="btn ghost sm" data-aes="load">Заменить</button>
      <button class="btn icon ghost" data-aes="remove" title="Убрать фото">${icons.trash}</button></div>
      ${(p.notes || []).map((n) => `<p class="muted small" style="margin:0 0 6px">${hide(n)}</p>`).join('')}
      <div class="label" style="margin-top:4px">Пары «фото — скан»: ${p.pairs.length} из ${p.need}+</div>
      <div class="row">${mode('pair', `${icons.target}Ставить пары`, 'Готово')}
        ${p.pairs.length ? '<button class="btn ghost" data-aes="unpair" data-arg="all">Убрать все</button>' : ''}</div>
      ${aes.mode === 'pair' ? '<p class="muted small">Щёлкните точку на фото и ту же точку на зубах в 3D (порядок любой). Хватит двух подальше друг от друга — бугры клыков; остальное доводит нейросеть.</p>' : ''}
      ${pairs ? `<div class="row" style="flex-wrap:wrap;gap:4px;margin-top:6px">${pairs}</div>` : ''}
      ${cameraHtml(p)}${alignHtml(p)}
      ${p.camera ? `<button class="btn ghost wide sm" data-aes="overlay">${aes.overlay ? 'Скрыть' : 'Показать'} наложение скана</button>` : ''}
      <div class="label">Горизонт: ${fmt(p.horizon_deg, 1)}°</div>
      <div class="row"><button class="btn ghost" data-aes="horizon" data-arg="-0.5">−0,5°</button>
        <button class="btn ghost" data-aes="horizon" data-arg="0.5">+0,5°</button>
        ${mode('horizon', 'По двум точкам', 'Отмена')}
        <button class="btn ghost" data-aes="horizon" data-arg="zero" title="Истинная горизонталь — край кадра">0</button></div>
      <div class="label">Средняя линия лица: ${p.midline_px.length ? `${p.midline_px.length} точ.` : 'нет'}</div>
      <div class="row">${mode('midline', 'Отметить', 'Готово')}
        ${p.midline_px.length ? '<button class="btn ghost" data-aes="midreset">Сбросить</button>' : ''}</div>
      <div class="label">Точки лица: ${facePoints} из ${FACE_STEPS.length}</div>
      <div class="row">${mode('face', 'Зрачки, углы глаз, нос', 'Готово')}
        ${facePoints ? '<button class="btn ghost" data-aes="facereset">Сбросить</button>' : ''}</div>
      ${aestheticHtml(p)}${guideHtml(p)}</div>${mount}`;
}
