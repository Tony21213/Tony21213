// Срез КТ: картинка с сервера, контуры сканов и структур поверх, перекрестие других срезов.
import { get, post } from './api.js';

export const AXES = {
  axial: { title: 'Аксиальный', color: '#5b8cff' },
  coronal: { title: 'Корональный', color: '#3ecf8e' },
  sagittal: { title: 'Сагиттальный', color: '#f5b84b' },
};
const NORMAL_AXIS = { axial: 2, coronal: 1, sagittal: 0 };

export class SliceView {
  constructor(el, axis, app) {
    this.el = el;
    this.axis = axis;
    this.app = app;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    el.innerHTML = `<div class="view-label"><i class="dot" style="background:${AXES[axis].color}"></i><b>${AXES[axis].title}</b><span class="pos"></span></div>
      <div class="empty"><span>КТ не открыт</span></div>`;
    el.appendChild(this.canvas);
    this.posLabel = el.querySelector('.pos');
    this.emptyEl = el.querySelector('.empty');
    this.image = null;
    this.overlays = [];
    this.seq = 0;
    this.zoom = 1;
    new ResizeObserver(() => this.draw()).observe(el);
    this.canvas.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    this.canvas.addEventListener('dblclick', (e) => {
      const p = this.geometry && this.toWorld(e.offsetX, e.offsetY);
      if (p) this.app.placeLandmark(p);
    });
  }

  async setCt(info) {
    this.geometry = info ? await get(`ct/geometry?axis=${this.axis}`) : null;
    this.emptyEl.hidden = !!info;
    this.image = null;
    this.overlays = [];
    this.draw();
  }

  get pos() { return this.app.cursor[NORMAL_AXIS[this.axis]]; }

  onWheel(e) {
    e.preventDefault();
    if (!this.geometry) return;
    if (e.ctrlKey) {
      this.zoom = Math.min(8, Math.max(1, this.zoom * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
      return this.draw();
    }
    const step = (e.shiftKey ? 2 : 0.5) * Math.sign(e.deltaY);
    const [lo, hi] = this.geometry.range;
    const cursor = [...this.app.cursor];
    cursor[NORMAL_AXIS[this.axis]] = Math.min(hi, Math.max(lo, this.pos - step));
    this.app.setCursor(cursor);
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
    return { scale, x0: (w - g.width * scale) / 2, y0: (h - g.height * scale) / 2 };
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

  // Загрузить картинку и контуры для текущего положения (старые ответы отбрасываются).
  async refresh() {
    if (!this.geometry) return;
    const seq = ++this.seq;
    const pos = this.pos;
    const [lvl, wid] = this.app.window;
    const img = new Image();
    img.src = `/api/ct/slice?axis=${this.axis}&pos=${pos.toFixed(2)}&level=${lvl}&width=${wid}`;
    const overlays = post('overlays', { axis: this.axis, pos, visible: this.app.visibleKeys() }).catch(() => []);
    await img.decode().catch(() => null);
    const ov = await overlays;
    if (seq !== this.seq) return;
    this.image = img;
    this.overlays = ov;
    this.draw();
  }

  draw() {
    const c = this.canvas, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!this.geometry) return;
    this.posLabel.textContent = `${this.pos.toFixed(1)} мм`;
    const g = this.geometry, L = this.layout();
    if (this.image) {
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(this.image, L.x0, L.y0, g.width * L.scale, g.height * L.scale);
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
