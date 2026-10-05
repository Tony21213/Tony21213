// Срез КТ: картинка с сервера, контуры сканов и структур поверх, перекрестие других срезов.
// Колесо — масштаб к точке под курсором (назад до конца — вписать), нажатое колесо или правая
// кнопка — сдвиг, ползунок внизу — срезы (колесо над ползунком тоже листает срезы).
import { get, post } from './api.js';

export const AXES = {
  axial: { title: 'Аксиальный', color: '#5b8cff' },
  coronal: { title: 'Корональный', color: '#3ecf8e' },
  sagittal: { title: 'Сагиттальный', color: '#f5b84b' },
};
const NORMAL_AXIS = { axial: 2, coronal: 1, sagittal: 0 };
const MAX_ZOOM = 40;
const MAX_PIXELS = 2048; // сторона картинки видимой части среза (как на сервере)

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
    this.image = null; // { img, ia, ib, ja, jb } — картинка и её край в пикселях полного среза
    this.overlays = [];
    this.seq = 0;
    this.zoom = 1;
    this.pan = [0, 0]; // сдвиг изображения, пиксели холста
    new ResizeObserver(() => { this.draw(); this.refreshSoon(); }).observe(el);
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    this.canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.canvas.addEventListener('mousedown', (e) => { if (e.button === 1) e.preventDefault(); }); // без автопрокрутки
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    this.canvas.addEventListener('dblclick', (e) => {
      const p = this.geometry && this.toWorld(e.offsetX, e.offsetY);
      if (p) this.app.placeLandmark(p);
    });
    this.slider.addEventListener('input', () => this.setPos(Number(this.slider.value)));
    this.slider.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.geometry) this.setPos(this.pos - Math.sign(e.deltaY) * this.sliceStep * (e.shiftKey ? 5 : 1));
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
    this.draw();
  }

  get pos() { return this.app.cursor[NORMAL_AXIS[this.axis]]; }

  setPos(value) {
    const [lo, hi] = this.geometry.range;
    const cursor = [...this.app.cursor];
    cursor[NORMAL_AXIS[this.axis]] = Math.min(hi, Math.max(lo, value));
    this.app.setCursor(cursor);
  }

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
    if (!this.geometry || (e.button !== 1 && e.button !== 2)) return;
    e.preventDefault();
    this.canvas.setPointerCapture(e.pointerId);
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

  onClick(e) {
    if (!this.geometry) return;
    const p = this.toWorld(e.offsetX, e.offsetY);
    if (p) this.app.setCursor(p);
  }

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

  // Загрузить видимую часть среза в разрешении экрана и контуры (старые ответы отбрасываются).
  async refresh() {
    if (!this.geometry) return;
    const seq = ++this.seq;
    const pos = this.pos;
    const [lvl, wid] = this.app.window;
    const g = this.geometry, L = this.layout();
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    // Видимая часть в пикселях полного среза (центр пикселя i — в u0 + i·du, края — ±0.5).
    const ia = Math.max(-0.5, -L.x0 / L.scale), ib = Math.min(g.width - 0.5, (w - L.x0) / L.scale);
    const ja = Math.max(-0.5, -L.y0 / L.scale), jb = Math.min(g.height - 0.5, (h - L.y0) / L.scale);
    const overlays = post('overlays', { axis: this.axis, pos, visible: this.app.visibleKeys() }).catch(() => []);
    let image = null;
    if (ib > ia && jb > ja && w && h) {
      const px = (n) => Math.max(1, Math.min(MAX_PIXELS, Math.round(n * L.scale * dpr)));
      const q = new URLSearchParams({
        axis: this.axis, pos: pos.toFixed(2), level: lvl, width: wid,
        ua: (g.u0 + ia * g.du).toFixed(3), ub: (g.u0 + ib * g.du).toFixed(3),
        va: (g.v0 + ja * g.dv).toFixed(3), vb: (g.v0 + jb * g.dv).toFixed(3), cols: px(ib - ia), rows: px(jb - ja),
      });
      const img = new Image();
      img.src = `/api/ct/slice?${q}`;
      if (await img.decode().then(() => true, () => false)) image = { img, ia, ib, ja, jb };
    }
    const ov = await overlays;
    if (seq !== this.seq) return;
    if (image) this.image = image;
    this.overlays = ov;
    this.draw();
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
      const { img, ia, ib, ja, jb } = this.image;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, L.x0 + ia * L.scale, L.y0 + ja * L.scale, (ib - ia) * L.scale, (jb - ja) * L.scale);
    }
    for (const o of this.overlays) {
      ctx.strokeStyle = o.color;
      ctx.lineWidth = o.width;
      ctx.beginPath();
      const s = o.segments;
      for (let k = 0; k < s.length; k += 4) {
        const [ax, ay] = this.toCanvas(s[k], s[k + 1]);
        const [bx, by] = this.toCanvas(s[k + 2], s[k + 3]);
        ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }
    // ориентиры рядом с плоскостью среза
    for (const l of this.app.landmarkPoints?.() || []) {
      const d = Math.abs(l.point[g.normal_axis] - this.pos);
      if (d > 3) continue;
      const [px, py] = this.toCanvas(l.point[g.u_axis], l.point[g.v_axis]);
      ctx.globalAlpha = 1 - d / 4;
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
  }
}
