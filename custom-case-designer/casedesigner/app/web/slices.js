// Срез КТ: картинка с сервера, контуры сканов и структур поверх, перекрестие других срезов.
// Колесо — масштаб к точке под курсором (назад до конца — вписать), нажатое колесо или правая
// кнопка — сдвиг, ползунок внизу — срезы (колесо над ползунком тоже листает срезы).
// Ручная поправка выбранного скана — манипулятор на срезе (кольцо с центром на скане): тянуть внутри
// кольца — сдвиг в плоскости среза, за кольцо — поворот вокруг центра; стрелки — точный сдвиг,
// Ctrl+←/→ — точный поворот (с Shift — крупнее). Контур скана двигается сразу, без ожидания сервера.
import { get, post } from './api.js';

export const AXES = {
  axial: { title: 'Аксиальный', color: '#5b8cff' },
  coronal: { title: 'Корональный', color: '#3ecf8e' },
  sagittal: { title: 'Сагиттальный', color: '#f5b84b' },
};
const NORMAL_AXIS = { axial: 2, coronal: 1, sagittal: 0 };
const MAX_ZOOM = 40;
const MAX_PIXELS = 2048; // сторона картинки видимой части среза (как на сервере)
const CACHE = 48; // столько последних картинок срезов держится в памяти
const STEP_MM = [0.05, 0.25]; // стрелки: сдвиг, мм (с Shift — второе)
const STEP_DEG = [0.1, 0.5]; // стрелки: поворот, градусы
const RING = 46; // радиус кольца манипулятора, пиксели
const RING_GRIP = 9; // полоса кольца, за которую поворачивают
const IDENTITY = [1, 0, 0, 0, 1, 0]; // плоское преобразование (u, v): [a, b, c, d, e, f] — u' = a·u + b·v + c, v' = d·u + e·v + f

function compose(m, n) { // сначала n, потом m
  return [m[0] * n[0] + m[1] * n[3], m[0] * n[1] + m[1] * n[4], m[0] * n[2] + m[1] * n[5] + m[2],
    m[3] * n[0] + m[4] * n[3], m[3] * n[1] + m[4] * n[4], m[3] * n[2] + m[4] * n[5] + m[5]];
}
function rotation(angle, [pu, pv]) {
  const c = Math.cos(angle), s = Math.sin(angle);
  return [c, -s, pu - c * pu + s * pv, s, c, pv - s * pu - c * pv];
}
const shift = (du, dv) => [1, 0, du, 0, 1, dv];
const applyM = (m, u, v) => [m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5]];

let hovered = null; // вид под мышью — ему достаются стрелки

export class SliceView {
  constructor(el, axis, app) {
    this.el = el;
    this.axis = axis;
    this.app = app;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    el.innerHTML = `<div class="view-label"><i class="dot" style="background:${AXES[axis].color}"></i><b>${AXES[axis].title}</b><span class="pos"></span></div>
      <div class="empty"><span>КТ не открыт</span></div>
      <input type="range" class="slice-slider" title="Срез" hidden style="accent-color:${AXES[axis].color}">`;
    el.appendChild(this.canvas);
    this.posLabel = el.querySelector('.pos');
    this.emptyEl = el.querySelector('.empty');
    this.slider = el.querySelector('.slice-slider');
    this.image = null; // { canvas, ia, ib, ja, jb } — картинка и её край в пикселях полного среза
    this.overlays = [];
    this.cache = new Map();
    this.imageSeq = 0;
    this.overlaySeq = 0;
    this.imageKey = null;
    this.overlayKey = null;
    this.direction = 0; // куда листали последний раз — туда подгружается следующий срез
    this.zoom = 1;
    this.pan = [0, 0]; // сдвиг изображения, пиксели холста
    this.drag = null; // перетаскивание манипулятора: { id, mode, start, now, centre }
    this.pending = null; // { id, m } — поправка, которую сервер ещё не вернул контуром
    new ResizeObserver(() => { this.draw(); this.refreshSoon(); }).observe(el);
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.canvas.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); }); // без автопрокрутки
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('pointerenter', () => { hovered = this; });
    this.canvas.addEventListener('pointerleave', () => { if (hovered === this) hovered = null; });
    this.canvas.addEventListener('pointermove', (e) => this.onHover(e));
    this.canvas.addEventListener('dblclick', (e) => {
      const p = this.geometry && this.toWorld(e.offsetX, e.offsetY);
      if (p) this.app.placeLandmark(p);
    });
    this.slider.addEventListener('input', () => this.setPos(Number(this.slider.value)));
    this.slider.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.geometry) this.step(-Math.sign(e.deltaY) * (e.shiftKey ? 5 : 1));
    }, { passive: false });
  }

  async setCt(info) {
    this.geometry = info ? await get(`ct/geometry?axis=${this.axis}`) : null;
    this.sliceStep = info ? Math.min(...info.spacing) : 0.5;
    this.emptyEl.hidden = !!info;
    this.slider.hidden = !info;
    if (info) Object.assign(this.slider, { min: this.geometry.range[0], max: this.geometry.range[1], step: this.sliceStep });
    this.zoom = 1;
    this.pan = [0, 0];
    this.image = null;
    this.overlays = [];
    this.cache.clear();
    this.imageKey = this.overlayKey = null;
    this.draw();
  }

  get pos() { return this.app.cursor[NORMAL_AXIS[this.axis]]; }

  setPos(value) {
    const [lo, hi] = this.geometry.range;
    const cursor = [...this.app.cursor];
    const next = Math.min(hi, Math.max(lo, value));
    this.direction = Math.sign(next - this.pos) || this.direction;
    cursor[NORMAL_AXIS[this.axis]] = next;
    this.app.setCursor(cursor);
  }

  step(n) { this.setPos(this.pos + n * this.sliceStep); }

  onWheel(e) {
    e.preventDefault();
    if (!this.geometry) return;
    const before = this.layout();
    const i = (e.offsetX - before.x0) / before.scale, j = (e.offsetY - before.y0) / before.scale;
    this.zoom = Math.min(MAX_ZOOM, Math.max(1, this.zoom * Math.exp(-e.deltaY * 0.0015)));
    if (this.zoom === 1) this.pan = [0, 0];
    else { // точка под курсором остаётся на месте
      const after = this.layout();
      this.pan[0] += e.offsetX - (after.x0 + i * after.scale);
      this.pan[1] += e.offsetY - (after.y0 + j * after.scale);
    }
    this.draw();
    this.refreshSoon();
  }

  onPointerDown(e) {
    if (!this.geometry) return;
    if (e.button === 0) return this.onLeftDown(e);
    if (e.button !== 1 && e.button !== 2) return;
    e.preventDefault();
    try { this.canvas.setPointerCapture(e.pointerId); } catch { /* указатель уже отпущен */ }
    const start = [e.clientX, e.clientY], pan = [...this.pan];
    const move = (ev) => {
      this.pan = [pan[0] + ev.clientX - start[0], pan[1] + ev.clientY - start[1]];
      this.draw();
      this.refreshSoon();
    };
    const up = () => {
      this.canvas.removeEventListener('pointermove', move);
      this.canvas.removeEventListener('pointerup', up);
      this.canvas.style.cursor = '';
    };
    this.canvas.addEventListener('pointermove', move);
    this.canvas.addEventListener('pointerup', up);
    this.canvas.style.cursor = 'grabbing';
  }

  // Центр манипулятора на срезе (мм по осям среза) и что под точкой холста: сдвиг, поворот или ничего.
  manipulator() {
    const t = this.app.manipTarget?.();
    if (!t || !this.geometry) return null;
    const g = this.geometry; // центр берётся из 3D-вида — там поправка уже применена
    return { ...t, centre: [t.centre[g.u_axis], t.centre[g.v_axis]] };
  }

  hit(t, cx, cy) {
    const [x, y] = this.toCanvas(...t.centre);
    const r = Math.hypot(cx - x, cy - y);
    return r <= RING - RING_GRIP ? 'move' : Math.abs(r - RING) <= RING_GRIP ? 'rotate' : null;
  }

  // Левая кнопка: на манипуляторе — тянуть скан, иначе щелчок ставит перекрестие.
  onLeftDown(e) {
    const t = this.manipulator();
    const mode = t && this.hit(t, e.offsetX, e.offsetY);
    try { this.canvas.setPointerCapture(e.pointerId); } catch { /* указатель уже отпущен */ }
    const start = this.toPlane(e.offsetX, e.offsetY);
    const move = (ev) => {
      if (!mode) return;
      const r = this.canvas.getBoundingClientRect();
      this.drag = { id: t.id, mode, start, now: this.toPlane(ev.clientX - r.left, ev.clientY - r.top), centre: t.centre };
      this.draw();
    };
    const up = (ev) => {
      this.canvas.removeEventListener('pointermove', move);
      this.canvas.removeEventListener('pointerup', up);
      if (this.drag) {
        const m = this.dragMatrix(this.drag);
        this.drag = null;
        this.commit(t.id, m);
      } else if (!mode) {
        const p = this.toWorld(ev.offsetX, ev.offsetY);
        if (p) this.app.setCursor(p);
      }
    };
    this.canvas.addEventListener('pointermove', move);
    this.canvas.addEventListener('pointerup', up);
  }

  onHover(e) {
    if (e.buttons) return;
    const t = this.manipulator();
    const mode = t && this.hit(t, e.offsetX, e.offsetY);
    this.canvas.style.cursor = mode === 'move' ? 'move' : mode === 'rotate' ? 'grab' : '';
  }

  dragMatrix(d) {
    if (d.mode === 'rotate') {
      const a0 = Math.atan2(d.start[1] - d.centre[1], d.start[0] - d.centre[0]);
      const a1 = Math.atan2(d.now[1] - d.centre[1], d.now[0] - d.centre[0]);
      return rotation(a1 - a0, d.centre);
    }
    return shift(d.now[0] - d.start[0], d.now[1] - d.start[1]);
  }

  // Поправка в плоскости среза → сразу на контур (до ответа сервера) и в положение скана.
  commit(id, m) {
    this.pending = { id, m: compose(m, this.pending?.id === id ? this.pending.m : IDENTITY) };
    this.draw();
    this.app.moveScan(id, this.worldMatrix(m));
  }

  // Стрелки в виде под мышью: сдвиг по экрану, Ctrl+←/→ — поворот вокруг центра манипулятора.
  static onKey(e) {
    const view = hovered;
    if (!view?.geometry || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) return false;
    const t = view.manipulator();
    if (!t) return false;
    e.preventDefault();
    const g = view.geometry, big = e.shiftKey ? 1 : 0;
    if (e.ctrlKey || e.metaKey) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return true;
      // по часовой стрелке на экране — →; знак зависит от того, куда на экране смотрят оси среза
      const screen = (e.key === 'ArrowRight' ? 1 : -1) * STEP_DEG[big] * Math.PI / 180;
      view.commit(t.id, rotation(screen * Math.sign(g.du * g.dv), t.centre));
    } else {
      const step = STEP_MM[big];
      const du = e.key === 'ArrowRight' ? step : e.key === 'ArrowLeft' ? -step : 0;
      const dv = e.key === 'ArrowDown' ? step : e.key === 'ArrowUp' ? -step : 0;
      view.commit(t.id, shift(du * Math.sign(g.du), dv * Math.sign(g.dv)));
    }
    return true;
  }

  // Плоская поправка матрицей 4×4 в мм пациента (по строкам): оси u, v среза, нормаль не меняется.
  worldMatrix(m) {
    const g = this.geometry, u = g.u_axis, v = g.v_axis;
    const M = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
    M[u][u] = m[0]; M[u][v] = m[1]; M[u][3] = m[2];
    M[v][u] = m[3]; M[v][v] = m[4]; M[v][3] = m[5];
    return M;
  }

  // Сервер вернул контуры с новым положением — живая поправка больше не нужна.
  settled(id) { if (this.pending?.id === id) { this.pending = null; this.draw(); } }

  layout() {
    const g = this.geometry;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    const scale = Math.min(w / g.width, h / g.height) * 0.98 * this.zoom;
    return { scale, x0: (w - g.width * scale) / 2 + this.pan[0], y0: (h - g.height * scale) / 2 + this.pan[1] };
  }

  // мм по осям среза → пиксели холста
  toCanvas(u, v) {
    const g = this.geometry, L = this.layout();
    return [L.x0 + ((u - g.u0) / g.du) * L.scale, L.y0 + ((v - g.v0) / g.dv) * L.scale];
  }

  // пиксели холста → мм по осям среза (без проверки границ)
  toPlane(cx, cy) {
    const g = this.geometry, L = this.layout();
    return [g.u0 + ((cx - L.x0) / L.scale) * g.du, g.v0 + ((cy - L.y0) / L.scale) * g.dv];
  }

  toWorld(cx, cy) {
    const g = this.geometry, L = this.layout();
    const i = (cx - L.x0) / L.scale, j = (cy - L.y0) / L.scale;
    if (i < 0 || j < 0 || i > g.width || j > g.height) return null;
    const p = [...this.app.cursor];
    p[g.u_axis] = g.u0 + i * g.du;
    p[g.v_axis] = g.v0 + j * g.dv;
    return p;
  }

  refreshSoon() {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this.refresh(), 60);
  }

  // Что показать: видимая часть среза в разрешении экрана. Ключ одинаков — картинка уже есть.
  request(pos) {
    const [lvl, wid] = this.app.window;
    const g = this.geometry, L = this.layout();
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    // Видимая часть в пикселях полного среза (центр пикселя i — в u0 + i·du, края — ±0.5).
    const ia = Math.max(-0.5, -L.x0 / L.scale), ib = Math.min(g.width - 0.5, (w - L.x0) / L.scale);
    const ja = Math.max(-0.5, -L.y0 / L.scale), jb = Math.min(g.height - 0.5, (h - L.y0) / L.scale);
    if (!(ib > ia && jb > ja && w && h)) return null;
    const px = (n) => Math.max(1, Math.min(MAX_PIXELS, Math.round(n * L.scale * dpr)));
    const q = new URLSearchParams({
      axis: this.axis, pos: pos.toFixed(2), level: lvl, width: wid, format: 'raw',
      ua: (g.u0 + ia * g.du).toFixed(3), ub: (g.u0 + ib * g.du).toFixed(3),
      va: (g.v0 + ja * g.dv).toFixed(3), vb: (g.v0 + jb * g.dv).toFixed(3), cols: px(ib - ia), rows: px(jb - ja),
    });
    return { key: q.toString(), ia, ib, ja, jb };
  }

  async fetchImage(req) {
    const hit = this.cache.get(req.key);
    if (hit) {
      this.cache.delete(req.key); // свежий — в конец
      this.cache.set(req.key, hit);
      return hit;
    }
    const buf = await get(`ct/slice?${req.key}`);
    const [cols, rows] = new Uint32Array(buf, 0, 2);
    const gray = new Uint8Array(buf, 8, cols * rows);
    const rgba = new Uint8ClampedArray(cols * rows * 4);
    for (let k = 0, m = 0; k < gray.length; k++, m += 4) {
      const v = gray[k];
      rgba[m] = v; rgba[m + 1] = v; rgba[m + 2] = v; rgba[m + 3] = 255;
    }
    const c = document.createElement('canvas');
    c.width = cols; c.height = rows;
    c.getContext('2d').putImageData(new ImageData(rgba, cols, rows), 0, 0);
    const image = { canvas: c, ia: req.ia, ib: req.ib, ja: req.ja, jb: req.jb };
    this.cache.set(req.key, image);
    while (this.cache.size > CACHE) this.cache.delete(this.cache.keys().next().value);
    return image;
  }

  // Обновить картинку и контуры — каждое само по себе: срез появляется, не дожидаясь контуров,
  // а вид, у которого ничего не поменялось (листают другой срез), ничего не запрашивает.
  async refresh() {
    if (!this.geometry) return;
    this.draw();
    const pos = this.pos;
    const req = this.request(pos);
    if (req && req.key !== this.imageKey) {
      this.imageKey = req.key;
      const seq = ++this.imageSeq;
      const cached = this.cache.get(req.key);
      if (cached) { this.image = cached; this.draw(); }
      const image = cached || await this.fetchImage(req).catch(() => null);
      if (image && seq === this.imageSeq) { this.image = image; this.draw(); }
      if (seq === this.imageSeq) this.prefetch(pos);
    }
    const visible = this.app.visibleKeys();
    const okey = `${pos.toFixed(3)}|${visible.join(',')}|${this.app.overlayVersion?.() ?? 0}`;
    if (okey !== this.overlayKey) {
      this.overlayKey = okey;
      const seq = ++this.overlaySeq;
      const pendingAt = this.pending;
      const ov = await post('overlays', { axis: this.axis, pos, visible }).catch(() => null);
      if (ov && seq === this.overlaySeq) {
        this.overlays = ov;
        if (this.pending === pendingAt && this.app.settledScan?.(pendingAt?.id)) this.pending = null;
        this.draw();
      }
    }
  }

  // Следующий срез по ходу листания — заранее, чтобы показать его сразу.
  prefetch(pos) {
    if (!this.direction) return;
    const [lo, hi] = this.geometry.range;
    const next = pos + this.direction * this.sliceStep;
    if (next < lo || next > hi) return;
    const req = this.request(next);
    if (req && !this.cache.has(req.key)) this.fetchImage(req).catch(() => {});
  }

  draw() {
    const c = this.canvas, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.geometry) return;
    this.posLabel.textContent = `${this.pos.toFixed(1)} мм`;
    this.slider.value = this.pos;
    const g = this.geometry, L = this.layout();
    if (this.image) {
      const { canvas, ia, ib, ja, jb } = this.image;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(canvas, L.x0 + ia * L.scale, L.y0 + ja * L.scale, (ib - ia) * L.scale, (jb - ja) * L.scale);
    }
    const live = {}; // скан → плоская поправка, которую ещё не отразил сервер (и текущее перетаскивание)
    if (this.pending) live[this.pending.id] = this.pending.m;
    if (this.drag) live[this.drag.id] = compose(this.dragMatrix(this.drag), live[this.drag.id] || IDENTITY);
    for (const o of this.overlays) {
      const m = live[o.id];
      ctx.strokeStyle = o.color;
      ctx.lineWidth = m ? o.width + 0.6 : o.width;
      ctx.beginPath();
      const s = o.segments;
      for (let k = 0; k < s.length; k += 4) {
        let a = [s[k], s[k + 1]], b = [s[k + 2], s[k + 3]];
        if (m) { a = applyM(m, ...a); b = applyM(m, ...b); }
        const [ax, ay] = this.toCanvas(a[0], a[1]);
        const [bx, by] = this.toCanvas(b[0], b[1]);
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }
    // ориентиры рядом с плоскостью среза
    for (const l of this.app.landmarkPoints?.() || []) {
      const dist = Math.abs(l.point[g.normal_axis] - this.pos);
      if (dist > 3) continue;
      const [px, py] = this.toCanvas(l.point[g.u_axis], l.point[g.v_axis]);
      ctx.globalAlpha = 1 - dist / 4;
      ctx.fillStyle = l.suggested ? '#f5b84b' : '#3ecf8e';
      ctx.strokeStyle = l.selected ? '#ffffff' : 'rgba(0,0,0,.6)';
      ctx.lineWidth = l.selected ? 2 : 1;
      ctx.beginPath(); ctx.arc(px, py, l.selected ? 5 : 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.font = '11px "Segoe UI", system-ui';
      ctx.fillText(l.key.replace('_', ' '), px + 7, py - 6);
      ctx.globalAlpha = 1;
    }
    // перекрестие: положения двух других срезов
    const cur = this.app.cursor;
    const [cx, cy] = this.toCanvas(cur[g.u_axis], cur[g.v_axis]);
    const other = Object.keys(AXES).filter((a) => a !== this.axis);
    const byAxis = Object.fromEntries(other.map((a) => [NORMAL_AXIS[a], AXES[a].color]));
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.7;
    ctx.strokeStyle = byAxis[g.u_axis];
    ctx.beginPath(); ctx.moveTo(cx, L.y0); ctx.lineTo(cx, L.y0 + g.height * L.scale); ctx.stroke();
    ctx.strokeStyle = byAxis[g.v_axis];
    ctx.beginPath(); ctx.moveTo(L.x0, cy); ctx.lineTo(L.x0 + g.width * L.scale, cy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    this.drawManipulator(ctx);
  }

  // Манипулятор: кольцо (поворот) с ручкой и центр-крестик (сдвиг).
  drawManipulator(ctx) {
    const t = this.manipulator();
    if (!t) return;
    let centre = t.centre, angle = 0;
    if (this.drag) {
      const m = this.dragMatrix(this.drag);
      centre = applyM(m, ...this.drag.centre);
      angle = Math.atan2(m[3], m[0]);
    }
    const [x, y] = this.toCanvas(...centre);
    const g = this.geometry, screenAngle = -angle * Math.sign(g.du * g.dv);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,.9)';
    ctx.fillStyle = 'rgba(255,255,255,.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, RING, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 0.12;
    ctx.beginPath(); ctx.arc(x, y, RING - RING_GRIP, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    const kx = x + RING * Math.sin(screenAngle), ky = y - RING * Math.cos(screenAngle); // ручка поворота
    ctx.beginPath(); ctx.arc(kx, ky, 5, 0, Math.PI * 2); ctx.fill();
    ctx.beginPath(); // крестик сдвига
    ctx.moveTo(x - 9, y); ctx.lineTo(x + 9, y); ctx.moveTo(x, y - 9); ctx.lineTo(x, y + 9);
    ctx.stroke();
    ctx.restore();
  }
}
