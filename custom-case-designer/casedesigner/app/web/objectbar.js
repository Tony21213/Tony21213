// Панель объектов окна 3D (как в exocad): видимость и прозрачность каждого объекта и группы — на любом шаге.
// Структуры КТ — те же флажки, что в шаге «КТ» (видимые попадают в экспорт); сканы, поверхность КТ и суставы —
// скрываются только на экране (viewer.userHidden), и шаги программы их обратно не включают.
import { $, app, render, scanLabel, state, toggleStructures, viewer } from './core.js';
import { icons } from './icons.js';

const EXTRA = { ct_teeth: 'Поверхность КТ', ct_bone: 'Кость по КТ' };
const JOINT = { head: ['Головка', 'правая', 'левая'], slope: ['Скат бугорка', 'правый', 'левый'] };
const STORE = 'ccd.objectBar';

const bar = { open: new Set(['Сканы']), collapsed: false, dragging: false, el: null };
try {
  const saved = JSON.parse(localStorage.getItem(STORE) || '{}');
  bar.collapsed = !!saved.collapsed;
  if (Array.isArray(saved.open)) bar.open = new Set(saved.open);
} catch { /* нет хранилища — по умолчанию */ }

function remember() {
  try { localStorage.setItem(STORE, JSON.stringify({ collapsed: bar.collapsed, open: [...bar.open] })); } catch { /* не важно */ }
}

const shown = (key) => viewer.objects.has(key) && viewer.wanted.get(key) !== false;
const opacity = (key) => viewer.objects.get(key)?.material.opacity ?? 1;
const isStructure = (key) => state.structures.some((s) => s.key === key);
const visible = (key) => (isStructure(key) ? state.visible.has(key) : !viewer.userHidden.has(key));

// Группы: [{ title, items: [{ key, name, color }] }] — только объекты, которые сейчас есть в окне 3D.
function groups() {
  const out = [];
  const scans = state.scans.filter((s) => shown(s.id)).map((s) => ({ key: s.id, name: scanLabel(s), color: s.color }));
  if (scans.length) out.push({ title: 'Сканы', items: scans });
  const byGroup = {};
  for (const s of state.structures) {
    if (viewer.objects.has(s.key)) (byGroup[s.group] ??= []).push({ key: s.key, name: s.name, color: s.color });
  }
  for (const [title, items] of Object.entries(byGroup)) out.push({ title, items });
  const extra = Object.entries(EXTRA).filter(([k]) => shown(k)).map(([key, name]) => ({ key, name, color: '#e9e2d2' }));
  if (extra.length) out.push({ title: 'КТ', items: extra });
  const joints = [];
  for (const part of ['head', 'slope']) for (const [i, side] of [[1, 'right'], [2, 'left']]) {
    const key = `joint:${side}:${part}`;
    if (shown(key)) joints.push({ key, name: `${JOINT[part][0]}, ${JOINT[part][i]}`, color: part === 'head' ? '#48d597' : '#f4c34e' });
  }
  if (joints.length) out.push({ title: 'Суставы', items: joints });
  return out;
}

function row(keys, name, color, { head = false, group = '' } = {}) {
  const on = keys.filter(visible).length;
  const eye = on === keys.length ? 'on' : on ? 'mixed' : '';
  const op = Math.round(Math.min(...keys.map(opacity)) * 100);
  const chevron = head ? `<span class="ob-chev" style="transform:rotate(${bar.open.has(group) ? 90 : 0}deg)">${icons.chevron}</span>` : '';
  return `<div class="ob-row ${head ? 'head' : ''} ${on ? '' : 'off'}">
    <button class="ob-eye ${eye}" data-ob-eye="${keys.join(',')}" title="${on ? 'Скрыть' : 'Показать'}">${icons.eye}</button>
    ${color ? `<i class="dot" style="background:${color}"></i>` : chevron}
    <span class="ob-name" ${head ? `data-ob-open="${group}"` : ''} title="${name}${head ? ` — видно ${on} из ${keys.length}` : ''}">${name}</span>
    <input type="range" min="10" max="100" step="5" value="${op}" style="--v:${op}%" data-ob-op="${keys.join(',')}" title="Непрозрачность ${op}%"></div>`;
}

export function renderObjectBar() {
  if (!bar.el) {
    bar.el = document.createElement('div');
    bar.el.className = 'objbar';
    $('#view3d').appendChild(bar.el);
    bar.el.addEventListener('click', onClick);
    bar.el.addEventListener('input', onInput);
    bar.el.addEventListener('change', () => { bar.dragging = false; render(); }); // флажки и кнопки шагов — по новой прозрачности
    bar.el.addEventListener('pointerdown', (e) => { if (e.target.matches('input[type=range]')) bar.dragging = true; });
    window.addEventListener('pointerup', () => { bar.dragging = false; });
    viewer.onObjects = () => renderObjectBar();
  }
  if (bar.dragging) return; // перерисовка оборвала бы перетаскивание ползунка
  const list = groups();
  bar.el.hidden = !list.length;
  bar.el.classList.toggle('collapsed', bar.collapsed);
  const head = `<div class="ob-title" data-ob-collapse title="${bar.collapsed ? 'Показать' : 'Свернуть'} панель объектов">
    <b>Объекты</b><span class="ob-chev" style="transform:rotate(${bar.collapsed ? 0 : 90}deg)">${icons.chevron}</span></div>`;
  if (bar.collapsed) { bar.el.innerHTML = head; return; }
  bar.el.innerHTML = `${head}<div class="ob-list">${list.map((g) => {
    const keys = g.items.map((i) => i.key);
    const items = bar.open.has(g.title) ? g.items.map((i) => row([i.key], i.name, i.color)).join('') : '';
    return `<div class="ob-group">${row(keys, g.title, '', { head: true, group: g.title })}<div class="ob-sub">${items}</div></div>`;
  }).join('')}</div>`;
}

function onClick(e) {
  const t = e.target.closest('[data-ob-collapse],[data-ob-open],[data-ob-eye]');
  if (!t) return;
  e.stopPropagation();
  const d = t.dataset;
  if ('obCollapse' in d) {
    bar.collapsed = !bar.collapsed;
  } else if (d.obOpen) {
    if (bar.open.has(d.obOpen)) bar.open.delete(d.obOpen); else bar.open.add(d.obOpen);
  } else if (d.obEye) {
    const keys = d.obEye.split(',');
    const on = !keys.every(visible); // все видны — скрыть все; иначе — показать все
    const structures = keys.filter(isStructure);
    for (const k of keys.filter((k) => !isStructure(k))) viewer.setUserHidden(k, !on);
    if (structures.length) return toggleStructures(structures, on); // как флажки шага «КТ»: render и срезы
    app.setCursor(app.cursor); // контуры сканов на срезах — по видимости
  }
  remember();
  renderObjectBar();
}

function onInput(e) {
  const t = e.target.closest('[data-ob-op]');
  if (!t) return;
  const value = Number(t.value) / 100;
  for (const k of t.dataset.obOp.split(',')) {
    state.opacity[k] = value;
    viewer.setOpacity(k, value);
  }
  const follow = [t];
  // ползунки строк группы — вслед за ползунком группы
  const group = t.closest('.ob-group');
  if (group && t.closest('.ob-row.head')) follow.push(...group.querySelectorAll('.ob-sub input[type=range]'));
  for (const r of follow) {
    r.value = t.value;
    r.style.setProperty('--v', `${t.value}%`);
    r.title = `Непрозрачность ${t.value}%`;
  }
}
