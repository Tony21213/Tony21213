// Custom Case Designer: шаги работы, панели и связь видов с сервером.
import { get, mesh, post, run } from './api.js';
import { icons } from './icons.js';
import { SliceView } from './slices.js';
import { Viewer3D } from './viewer3d.js';

const STEPS = [
  { id: 'data', title: 'Данные', icon: 'data' },
  { id: 'align', title: 'Совмещение', icon: 'align' },
  { id: 'structures', title: 'Структуры', icon: 'layers' },
  { id: 'landmarks', title: 'Ориентиры', icon: 'target' },
  { id: 'export', title: 'Экспорт', icon: 'export' },
];
const JAWS = { upper: 'Верхняя', lower: 'Нижняя' };
// Что видно сразу после сегментации: отдельные зубы, кости, каналы, пазухи.
const HIDDEN_BY_DEFAULT = ['upper_teeth', 'lower_teeth', 'skull', 'nasopharynx', 'oropharynx', 'hypopharynx', 'hard_palate'];
const TRANSLUCENT = { mandible: 0.42, maxilla: 0.42, skull: 0.3, maxillary_sinus: 0.55, frontal_sinus: 0.55,
  nasal_cavity: 0.45, pharynx: 0.4, nasopharynx: 0.4, oropharynx: 0.4, hypopharynx: 0.4, soft_palate: 0.7, hard_palate: 0.5 };

const state = {
  step: 'data', ct: null, scans: [], structures: [], visible: new Set(), heat: true, selected: null,
  gizmo: null, stepMm: 0.1, stepDeg: 0.5, modelsDir: null, exportOpts: { bite: 'scan', frame: 'exocad' },
  exported: null, busy: false, groupsOpen: new Set(['Зубы']),
  landmarks: [], planes: [], angles: {}, articulators: [], lmSel: 'Po_R', refPlane: 'frankfurt', refArt: null,
};

const app = {
  cursor: [0, 0, 0],
  window: [400, 3000],
  setCursor(p) {
    this.cursor = p;
    slices.forEach((s) => s.refresh());
  },
  visibleKeys: () => [...state.scans.filter((s) => s.transform).map((s) => s.id), ...state.visible],
  landmarkPoints: () => state.landmarks.filter((l) => l.point).map((l) => ({ ...l, selected: l.key === state.lmSel })),
  placeLandmark: (p) => { if (state.step === 'landmarks') setLandmark(state.lmSel, p, true); },
};

const $ = (sel, root = document) => root.querySelector(sel);
const panel = $('#panel');
const viewer = new Viewer3D($('#view3d'), { onTransformEnd: (id) => evaluateManual(id) });
const slices = [...document.querySelectorAll('.view[data-axis]')].map((el) => new SliceView(el, el.dataset.axis, app));

// ---------- общие помощники ----------
function toast(text, kind = 'error') {
  const t = document.createElement('div');
  t.className = `toast ${kind === 'info' ? 'info' : ''}`;
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), kind === 'info' ? 3500 : 6000);
}

async function busy(title, fn) {
  if (state.busy) return toast('Подождите, идёт другая операция');
  state.busy = true;
  const job = $('#job');
  job.classList.add('on');
  $('.title', job).textContent = title;
  $('.msg', job).textContent = '';
  $('.bar i', job).style.width = '0%';
  const progress = (s) => {
    $('.bar i', job).style.width = `${Math.round((s.progress || 0) * 100)}%`;
    $('.msg', job).textContent = s.message || '';
  };
  try {
    return await fn(progress);
  } catch (e) {
    toast(e.message);
    return null;
  } finally {
    state.busy = false;
    job.classList.remove('on');
    render();
  }
}

// Путь к файлу или папке: системный диалог в окне приложения, поле ввода — в браузере.
async function choose(kind, title) {
  if (window.pywebview?.api) {
    const paths = await window.pywebview.api.choose(kind);
    return paths?.length ? paths : null;
  }
  return new Promise((done) => {
    const back = document.createElement('div');
    back.className = 'modal-back';
    back.innerHTML = `<div class="modal"><h3>${title}</h3><p>Полный путь${kind === 'scan' ? '; несколько — через «;»' : ''}</p>
      <input type="text" spellcheck="false"><div class="row" style="justify-content:flex-end;margin-top:14px">
      <button class="btn ghost" data-x>Отмена</button><button class="btn primary" data-ok>Открыть</button></div></div>`;
    document.body.appendChild(back);
    const input = $('input', back);
    input.focus();
    const close = (value) => { back.remove(); done(value); };
    $('[data-x]', back).onclick = () => close(null);
    $('[data-ok]', back).onclick = () => close(input.value.trim() ? input.value.split(';').map((p) => p.trim()).filter(Boolean) : null);
    input.onkeydown = (e) => { if (e.key === 'Enter') $('[data-ok]', back).click(); if (e.key === 'Escape') close(null); };
  });
}

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const cls = (v, good, fair) => (v <= good ? 'ok' : v <= fair ? 'warn' : 'bad');
const scanById = (id) => state.scans.find((s) => s.id === id);

function updateScan(info) {
  const i = state.scans.findIndex((s) => s.id === info.id);
  if (i >= 0) state.scans[i] = info; else state.scans.push(info);
}

async function showScan(info) {
  if (info.transform) viewer.setTransform(info.id, info.transform);
  const rgb = state.heat && info.registered ? new Uint8Array(await get(`scans/${info.id}/colors`)) : null;
  viewer.setColors(info.id, rgb);
}

// ---------- действия ----------
async function openCt(kind) {
  const paths = await choose(kind, kind === 'ctdir' ? 'Папка DICOM' : 'Файл КТ');
  if (!paths) return;
  await busy('Открываю КТ', async (progress) => {
    state.ct = await run('ct', { path: paths[0] }, progress);
    app.window = state.ct.window;
    app.cursor = [...state.ct.focus];
    for (const s of state.scans) { s.transform = null; s.registered = false; viewer.setVisible(s.id, false); }
    for (const s of state.structures) viewer.remove(s.key);
    state.structures = []; state.visible.clear();
    applyLandmarks(await get('landmarks'));
    await Promise.all(slices.map((s) => s.setCt(state.ct)));
    app.setCursor(app.cursor);
    viewer.setMesh('ct_teeth', await mesh('ct/surface'), { color: '#e9e2d2', opacity: 0.9 });
    viewer.fit('front');
  });
}

async function addScans() {
  const paths = await choose('scan', 'Скан челюсти (STL, PLY, OBJ)');
  if (!paths) return;
  await busy('Загружаю сканы', async () => {
    for (const path of paths) {
      const info = await post('scans', { path });
      updateScan(info);
      viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, null);
      state.selected ??= info.id;
    }
  });
}

async function removeScan(id) {
  await post(`scans/${id}/remove`);
  viewer.remove(id);
  state.scans = state.scans.filter((s) => s.id !== id);
  if (state.selected === id) state.selected = state.scans[0]?.id ?? null;
  render();
}

async function setJaw(id, jaw) {
  updateScan(await post(`scans/${id}/jaw`, { jaw: jaw || null }));
  render();
}

async function register(ids, body = {}) {
  if (!state.ct) return toast('Сначала откройте КТ');
  await busy('Совмещаю по коронкам зубов', async (progress) => {
    for (const id of ids) {
      const info = await run(`scans/${id}/register`, body, (s) => progress({ ...s, message: scanById(id).name }));
      updateScan(info);
      await showScan(info);
    }
    app.setCursor(app.cursor);
    if (!viewer.gizmoTarget) viewer.fit('front');
  });
}

let evalTimer = null;
function evaluateManual(id, delay = 0) {
  clearTimeout(evalTimer);
  evalTimer = setTimeout(async () => {
    const info = await run(`scans/${id}/evaluate`, { transform: viewer.getTransform(id) }).catch((e) => toast(e.message));
    if (!info) return;
    updateScan(info);
    await showScan(info);
    app.setCursor(app.cursor);
    render();
  }, delay);
}

function nudge(axis, sign, rotate) {
  const id = state.selected;
  if (!scanById(id)?.transform) return;
  viewer.nudge(id, axis, sign * (rotate ? state.stepDeg : state.stepMm), rotate);
  evaluateManual(id, 350);
}

async function refine(id) {
  await register([id], { start: viewer.getTransform(id), jaw: scanById(id).jaw });
}

async function resetAuto(id) {
  const s = scanById(id);
  if (!s.auto_transform) return;
  viewer.setTransform(id, s.auto_transform);
  evaluateManual(id);
}

async function accept(id) {
  const r = await post(`scans/${id}/accept`).catch((e) => toast(e.message));
  if (!r) return;
  updateScan(r.scan);
  const m = r.memory || {};
  toast(`Принято. ${r.record.corrected_mm > 0.05 ? `Ваша поправка ${fmt(r.record.corrected_mm)} мм запомнена. ` : ''}` +
    `По этому аппарату КТ принято кейсов: ${m.cases ?? 1}.`, 'info');
  render();
}

function gizmo(mode) {
  state.gizmo = state.gizmo === mode ? null : mode;
  if (state.gizmo && state.selected) viewer.attach(state.selected, state.gizmo); else viewer.detach();
  render();
}

async function segment() {
  if (!state.ct) return toast('Сначала откройте КТ');
  let dir = state.modelsDir;
  if (!dir) {
    const paths = await choose('models', 'Папка моделей сегментации');
    if (!paths) return;
    dir = paths[0];
  }
  await busy('Сегментация КТ', async (progress) => {
    const res = await run('segment', { models_dir: dir }, progress);
    state.modelsDir = dir;
    for (const s of state.structures) viewer.remove(s.key);
    state.structures = res.structures;
    state.visible = new Set(res.structures.map((s) => s.key)
      .filter((k) => !k.startsWith('pulp/') && !HIDDEN_BY_DEFAULT.includes(k)));
    progress({ progress: 1, message: 'Загружаю поверхности' });
    await Promise.all(res.structures.map(async (s) => {
      const m = viewer.setMesh(s.key, await mesh(`structures/${encodeURIComponent(s.key).replace(/%2F/g, '/')}/mesh`),
        { color: s.color, opacity: TRANSLUCENT[s.key] ?? 1, order: TRANSLUCENT[s.key] ? 1 : 0 });
      m.visible = state.visible.has(s.key);
    }));
    viewer.setVisible('ct_teeth', false);
    app.setCursor(app.cursor);
  });
}

function toggleStructures(keys, on) {
  for (const k of keys) {
    if (on) state.visible.add(k); else state.visible.delete(k);
    viewer.setVisible(k, on);
  }
  app.setCursor(app.cursor);
  render();
}

async function doExport() {
  const paths = await choose('out', 'Папка для результата');
  if (!paths) return;
  await busy('Экспорт для exocad', async (progress) => {
    const o = state.exportOpts;
    const ref = o.frame === 'plane' ? `plane:${state.refPlane}` : o.frame === 'articulator' ? `articulator:${state.refArt}` : null;
    state.exported = await run('export', { out_dir: paths[0], bite: o.bite, frame: ref ? 'reference' : o.frame, reference: ref,
      include: [...state.visible] }, progress);
  });
}

function applyLandmarks(info) {
  state.landmarks = info.landmarks;
  state.planes = info.planes;
  state.angles = info.angles;
  state.articulators = info.articulators;
  state.refArt ??= info.articulators[0]?.key;
  for (const l of info.landmarks) {
    if (l.point) viewer.setMarker(l.key, l.point, l.suggested ? '#f5b84b' : '#3ecf8e');
    else viewer.remove(`lm:${l.key}`);
  }
  slices.forEach((v) => v.draw());
}

async function setLandmark(key, point, advance = false) {
  const info = await post('landmarks', { key, point }).catch((e) => toast(e.message));
  if (!info) return;
  applyLandmarks(info);
  if (advance && point) {  // к следующей непоставленной точке
    const next = info.landmarks.find((l) => !l.point);
    if (next) state.lmSel = next.key;
  }
  render();
}

async function suggestLandmarks() {
  const info = await post('landmarks/suggest').catch((e) => toast(e.message));
  if (info) { applyLandmarks(info); render(); }
}

// ---------- панели ----------
function metricsHtml(s) {
  const st = s.stats;
  if (!st || st.mean_mm === undefined) return '';
  const within = Math.round((st.within_0_2_mm ?? 0) * 100);
  return `<div class="metrics">
    <div class="metric"><span>Среднее</span><b class="${cls(st.mean_mm, 0.08, 0.15)}">${fmt(st.mean_mm, 3)}</b></div>
    <div class="metric"><span>90% точек</span><b class="${cls(st.p90_mm, 0.15, 0.25)}">${fmt(st.p90_mm, 3)}</b></div>
    <div class="metric"><span>≤ 0.2 мм</span><b class="${within >= 90 ? 'ok' : within >= 75 ? 'warn' : 'bad'}">${within}%</b></div>
  </div>`;
}

function archHtml(s) {
  if (!s.segments?.length) return '';
  const max = Math.max(0.15, ...s.segments.map((g) => g.shift_mm));
  const bars = s.segments.map((g) => `<div class="${g.shift_mm > 0.1 ? 'warn' : ''}" style="height:${Math.max(8, (g.shift_mm / max) * 100)}%"
    title="${g.where}: ${fmt(g.shift_mm, 3)} мм"></div>`).join('');
  return `<div class="label" style="margin-top:12px">Участки дуги — сдвиг при отдельной подгонке</div>
    <div class="arch">${bars}</div><div class="arch-caption"><span>${s.segments[0].where}</span><span>${s.segments.at(-1).where}</span></div>`;
}

function renderData() {
  const ct = state.ct;
  const ctCard = ct ? `
    <div class="card"><div class="card-head">${icons.ct}<h3>${ct.name}</h3><button class="btn ghost sm" data-a="ct">Заменить</button></div>
      <div class="kv"><span>Размер</span><b>${ct.shape.join(' × ')}</b><span>Воксель</span><b>${ct.spacing.map((v) => fmt(v, 2)).join(' × ')} мм</b>
      <span>Аппарат</span><b>${ct.device || 'не указан'}</b>
      <span>Поправка аппарата</span><b>${ct.learned_edge_shift_mm === null ? 'ещё не выучена' : `${fmt(ct.learned_edge_shift_mm, 3)} мм`}</b></div></div>`
    : `<div class="card"><div class="card-head">${icons.ct}<h3>КТ</h3></div><p class="muted" style="margin:0 0 10px">КЛКТ: папка DICOM, архив, NIfTI, MHA, NRRD.</p>
      <div class="row"><button class="btn primary grow" data-a="ctdir">${icons.folder}Папка DICOM</button><button class="btn grow" data-a="ct">${icons.file}Файл</button></div></div>`;
  const rows = state.scans.map((s) => `
    <div class="scan-row ${s.id === state.selected ? 'sel' : ''}" data-select="${s.id}">
      <i class="dot" style="background:${s.color}"></i><span class="name" title="${s.path}">${s.name}</span>
      <select data-jaw="${s.id}"><option value="">Челюсть: авто</option>${Object.entries(JAWS).map(([k, v]) => `<option value="${k}" ${s.jaw === k ? 'selected' : ''}>${v}</option>`).join('')}</select>
      <button class="btn icon ghost" data-remove="${s.id}" title="Убрать">${icons.trash}</button></div>`).join('');
  return `<h2>Данные кейса</h2><p class="lead">КТ и сканы челюстей. Прикус берётся со сканера.</p>${ctCard}
    <div class="card"><div class="card-head">${icons.tooth}<h3>Сканы челюстей <span class="sub">${state.scans.length || ''}</span></h3>
      <button class="btn sm" data-a="scan">${icons.plus}Добавить</button></div>
      ${rows || '<p class="muted" style="margin:0">STL, PLY или OBJ со сканера — как есть, без ориентации.</p>'}</div>
    <button class="btn primary wide" data-step="align" ${ct && state.scans.length ? '' : 'disabled'}>Дальше: совмещение ${icons.chevron}</button>`;
}

function renderAlign() {
  const sel = scanById(state.selected);
  const cards = state.scans.map((s) => {
    const status = s.accepted ? '<span class="badge ok">принят</span>' : s.registered ? '<span class="badge accent">совмещён</span>' : '<span class="badge">не совмещён</span>';
    const jaw = s.jaw ? `<span class="badge">${JAWS[s.jaw]}</span>` : '';
    const warnings = (s.warnings || []).map((w) => `<div class="warning">${icons.warn}<span>${w}</span></div>`).join('');
    const corrected = s.corrected_mm > 0.05 ? `<div class="muted" style="margin-top:8px">Поправлено вручную: ${fmt(s.corrected_mm)} мм от автоматики</div>` : '';
    return `<div class="card ${s.id === state.selected ? 'sel' : ''}" data-select="${s.id}" style="${s.id === state.selected ? 'border-color:var(--accent)' : ''}">
      <div class="card-head"><i class="dot" style="background:${s.color}"></i><h3>${s.name}</h3>${jaw}${status}</div>
      ${metricsHtml(s)}${archHtml(s)}${warnings}${corrected}
      ${s.registered ? '' : `<button class="btn primary wide" data-register="${s.id}">${icons.play}Совместить</button>`}</div>`;
  }).join('');
  const tools = sel?.registered ? `
    <div class="card"><div class="card-head">${icons.move}<h3>Ручная коррекция <span class="sub">${sel.name}</span></h3></div>
      <div class="seg" style="margin-bottom:10px"><button data-gizmo="translate" class="${state.gizmo === 'translate' ? 'on' : ''}">Перемещение (G)</button>
        <button data-gizmo="rotate" class="${state.gizmo === 'rotate' ? 'on' : ''}">Поворот (R)</button></div>
      <div class="nudge">
        <span></span><span>сдвиг</span><span></span><span></span><span>поворот</span><span></span>
        ${['Л ↔ П', 'Перед ↔ зад', 'Низ ↔ верх'].map((name, axis) => `
          <span>${['x', 'y', 'z'][axis]}</span><button class="btn" data-nudge="${axis},-1,0">−</button><button class="btn" data-nudge="${axis},1,0">+</button>
          <span title="${name}">${['x', 'y', 'z'][axis]}°</span><button class="btn" data-nudge="${axis},-1,1">⟲</button><button class="btn" data-nudge="${axis},1,1">⟳</button>`).join('')}
      </div>
      <p class="muted" style="margin:8px 2px 0;font-size:11.5px">x — вправо/влево пациента, y — вперёд/назад, z — вниз/вверх.</p>
      <div class="row" style="margin-top:10px"><span class="muted">Шаг</span>
        <select data-stepmm>${[0.05, 0.1, 0.25, 0.5, 1].map((v) => `<option ${v === state.stepMm ? 'selected' : ''}>${v}</option>`).join('')}</select><span class="muted">мм</span>
        <select data-stepdeg>${[0.25, 0.5, 1, 2].map((v) => `<option ${v === state.stepDeg ? 'selected' : ''}>${v}</option>`).join('')}</select><span class="muted">°</span></div>
      <div class="row" style="margin-top:12px"><button class="btn grow" data-refine="${sel.id}">${icons.refine}Уточнить</button>
        <button class="btn grow" data-reset="${sel.id}" ${sel.auto_transform ? '' : 'disabled'}>${icons.undo}К автоматике</button></div>
      <button class="btn ok wide" style="margin-top:8px" data-accept="${sel.id}" ${sel.accepted ? 'disabled' : ''}>${icons.check}${sel.accepted ? 'Принято' : 'Принять положение'}</button>
      <p class="muted" style="margin:8px 2px 0;font-size:11.5px">Принятые положения учат программу: следующие совмещения на этом аппарате КТ точнее.</p></div>` : '';
  const pending = state.scans.filter((s) => !s.registered).map((s) => s.id);
  return `<h2>Совмещение</h2><p class="lead">Только по коронкам зубов: десна и нёбо на скане, корни на КТ не участвуют.</p>
    ${pending.length > 1 ? `<button class="btn primary wide" style="margin-bottom:10px" data-registerall>${icons.play}Совместить все</button>` : ''}
    ${cards || '<div class="card muted">Добавьте сканы на шаге «Данные».</div>'}${tools}`;
}

function renderStructures() {
  if (!state.structures.length) {
    return `<h2>Структуры КТ</h2><p class="lead">Зубы с номерами FDI, челюсти, каналы, пазухи, дыхательные пути, импланты.</p>
      <div class="card"><div class="kv"><span>Модели</span><b class="path">${state.modelsDir || 'не выбраны'}</b></div>
      <div class="row" style="margin-top:10px"><button class="btn primary grow" data-a="segment" ${state.ct ? '' : 'disabled'}>${icons.play}Сегментировать</button>
      <button class="btn" data-a="models">${icons.folder}</button></div></div>`;
  }
  const groups = {};
  for (const s of state.structures) (groups[s.group] ??= []).push(s);
  const tree = Object.entries(groups).map(([title, items]) => {
    const on = items.filter((s) => state.visible.has(s.key)).length;
    const open = state.groupsOpen.has(title);
    const check = on === items.length ? 'on' : on ? 'mixed' : '';
    return `<div class="group"><div class="ghead" data-group="${title}"><span class="check ${check}" data-groupcheck="${title}"></span>
      <b>${title}</b><span>${on}/${items.length}</span><span style="transform:rotate(${open ? 90 : 0}deg);display:flex">${icons.chevron.replace('<svg', '<svg width="14" height="14"')}</span></div>
      ${open ? items.map((s) => `<div class="item ${state.visible.has(s.key) ? '' : 'off'}" data-toggle="${s.key}">
        <span class="check ${state.visible.has(s.key) ? 'on' : ''}"></span><i class="dot" style="background:${s.color}"></i>${s.name}</div>`).join('') : ''}</div>`;
  }).join('');
  return `<h2>Структуры КТ</h2><p class="lead">${state.structures.length} структур. Видимые попадут в экспорт.</p>
    <div class="card tree">${tree}</div>
    <button class="btn wide" data-a="segment">${icons.refine}Пересчитать</button>`;
}

function renderExport() {
  const o = state.exportOpts;
  const ready = state.scans.some((s) => s.registered);
  const res = state.exported;
  const result = res ? `<div class="card"><div class="card-head">${icons.check}<h3>Готово</h3></div>
      <div class="path" title="${res.out_dir}">${res.out_dir}</div>
      ${res.bite ? `<div class="muted" style="margin-top:8px">Прикус на КТ отличается от сканов: нижняя челюсть смещена на ${fmt(res.bite.lower_jaw_on_ct_vs_scans_mean_mm)} мм, повёрнута на ${fmt(res.bite.rotation_deg, 1)}°.</div>` : ''}
      ${(res.notes || []).map((n) => `<div class="warning">${icons.warn}<span>${n}</span></div>`).join('')}
      <div class="label">Файлы (${res.files.length})</div><div class="files">${res.files.join('<br>')}</div></div>` : '';
  return `<h2>Экспорт для exocad</h2><p class="lead">STL в координатах сканера — в них exocad открывает сканы, поэтому всё встанет на свои места.</p>
    <div class="card"><div class="label" style="margin-top:0">Прикус</div>
      <div class="seg"><button data-bite="scan" class="${o.bite === 'scan' ? 'on' : ''}">Прикус сканов</button><button data-bite="ct" class="${o.bite === 'ct' ? 'on' : ''}">Как на КТ</button></div>
      <p class="muted" style="margin:8px 2px 0;font-size:12px">${o.bite === 'scan' ? 'Нижняя челюсть, нижние зубы и канал из КТ переезжают к нижнему скану.' : 'Всё стоит как на КТ; сканы переносятся на свои челюсти.'}</p>
      <div class="label">Система координат</div>
      <div class="seg"><button data-frame="exocad" class="${o.frame === 'exocad' ? 'on' : ''}">Сканера</button>
        <button data-frame="plane" class="${o.frame === 'plane' ? 'on' : ''}">Плоскость</button>
        <button data-frame="articulator" class="${o.frame === 'articulator' ? 'on' : ''}">Артикулятор</button>
        <button data-frame="dicom" class="${o.frame === 'dicom' ? 'on' : ''}" ${o.bite === 'scan' ? 'disabled' : ''}>DICOM</button></div>
      ${o.frame === 'plane' ? `<select data-refplane style="width:100%;margin-top:8px">${state.planes.map((p) => `<option value="${p.key}" ${p.key === state.refPlane ? 'selected' : ''} ${p.ready ? '' : 'disabled'}>${p.name}${p.ready ? '' : ' — нет точек'}</option>`).join('')}</select>` : ''}
      ${o.frame === 'articulator' ? `<select data-refart style="width:100%;margin-top:8px">${state.articulators.map((a) => `<option value="${a.key}" ${a.key === state.refArt ? 'selected' : ''}>${a.maker} ${a.name}${a.calibrated ? '' : ' (не откалиброван)'}</option>`).join('')}</select>
        <p class="muted" style="margin:8px 2px 0;font-size:11.5px">По монтажной плоскости артикулятора. Без калибровки по образцу из exocad положение относительно его столика приблизительное.</p>` : ''}
      ${o.frame === 'plane' || o.frame === 'articulator' ? '<p class="muted" style="margin:8px 2px 0;font-size:11.5px">Начало — середина шарнирной оси (мыщелки), Z — вверх по нормали плоскости, X — вправо пациента, Y — вперёд.</p>' : ''}</div>
    <button class="btn primary wide" data-a="export" ${ready ? '' : 'disabled'}>${icons.export}Экспортировать</button>
    ${ready ? '' : '<p class="muted" style="margin:8px 2px">Сначала совместите хотя бы один скан.</p>'}${result}`;
}

function renderLandmarks() {
  const planes = state.planes.map((p) => `<div class="row" style="margin-top:6px">
      <span class="badge ${p.ready ? 'ok' : ''}">${p.ready ? 'готова' : 'нет точек'}</span><span class="grow">${p.name}</span></div>
      ${p.ready ? '' : `<div class="muted" style="font-size:11px;margin:2px 0 0 4px">нужно: ${p.missing.join(', ')}</div>`}`).join('');
  const angles = Object.entries(state.angles).map(([k, v]) => `<span>${k.replace('/', ' / ')}</span><b>${fmt(v, 1)}°</b>`).join('');
  const rows = state.landmarks.map((l) => {
    const st = l.point ? (l.suggested ? 'warn' : 'ok') : '';
    return `<div class="scan-row ${l.key === state.lmSel ? 'sel' : ''}" data-lm="${l.key}" title="${l.hint}">
      <i class="dot" style="background:${st === 'ok' ? 'var(--ok)' : st === 'warn' ? 'var(--warn)' : 'var(--line-2)'}"></i>
      <span class="name">${l.name}${l.suggested ? ' <span class="badge warn">проверьте</span>' : ''}</span>
      ${l.point ? `<button class="btn icon ghost" data-lmgo="${l.key}" title="Показать на срезах">${icons.eye}</button>
      <button class="btn icon ghost" data-lmdel="${l.key}" title="Убрать">${icons.trash}</button>` : ''}</div>`;
  }).join('');
  return `<h2>Ориентиры и плоскости</h2><p class="lead">Точки для Франкфуртской горизонтали, HIP, плоскости Кемпера и шарнирной оси. Выберите точку и дважды щёлкните её место на срезе.</p>
    <div class="row" style="margin-bottom:10px"><button class="btn primary grow" data-lmplace>${icons.target}В перекрестие</button>
      <button class="btn grow" data-lmsuggest ${state.structures.length ? '' : 'disabled'} title="Мыщелки и порионы по сегментации">${icons.refine}Предложить</button></div>
    <div class="card">${rows}</div>
    <div class="card"><div class="card-head">${icons.layers}<h3>Плоскости</h3></div>${planes}
      ${angles ? `<div class="label">Углы между плоскостями</div><div class="kv">${angles}</div>` : ''}</div>`;
}

const RENDER = { data: renderData, align: renderAlign, structures: renderStructures, landmarks: renderLandmarks, export: renderExport };

function render() {
  const done = {
    data: !!state.ct && state.scans.length > 0,
    align: state.scans.length > 0 && state.scans.every((s) => s.registered),
    structures: state.structures.length > 0,
    landmarks: state.planes.some((p) => p.ready),
    export: !!state.exported,
  };
  $('#rail').innerHTML = STEPS.map((s) => `<button class="step ${s.id === state.step ? 'active' : ''} ${done[s.id] ? 'done' : ''}" data-step="${s.id}">
    ${icons[s.icon]}<span>${s.title}</span><i class="dot"></i></button>`).join('');
  panel.innerHTML = RENDER[state.step]();
  const ct = state.ct;
  $('#caseChip').innerHTML = ct ? `${icons.ct.replace('<svg', '<svg width="14" height="14"')}<b>${ct.name}</b>${ct.device ? ` · ${ct.device}` : ''}` +
    `${state.scans.length ? ` · сканов: ${state.scans.length}` : ''}` : 'Кейс не открыт';
  $('#empty3d').innerHTML = ct || state.scans.length ? '' : `${icons.tooth}<span>Откройте КТ и сканы на шаге «Данные»</span>`;
  const legend = $('#legend');
  legend.hidden = !(state.heat && state.scans.some((s) => s.registered));
  legend.innerHTML = '<span><i style="background:#28aa46"></i>≤ 0.1 мм</span><span><i style="background:#e6be1e"></i>≤ 0.2 мм</span>' +
    '<span><i style="background:#d23228"></i>> 0.2 мм</span><span><i style="background:#aaa"></i>десна, нёбо</span>';
  $('#tools3d').innerHTML = [['front', 'Спереди'], ['right', 'Справа'], ['left', 'Слева'], ['top', 'Сверху'], ['bottom', 'Снизу']]
    .map(([v, t]) => `<button class="btn" data-view="${v}" title="${t}" style="width:auto;padding:0 8px;font-size:11.5px">${t}</button>`).join('') +
    `<button class="btn ${state.heat ? 'on' : ''}" data-heat title="Карта отклонений">${icons.heat}</button>`;
}

// ---------- события ----------
document.addEventListener('click', async (e) => {
  const lm = e.target.closest('[data-lm],[data-lmgo],[data-lmdel],[data-lmplace],[data-lmsuggest]');
  if (lm && !lm.disabled) {
    const d = lm.dataset;
    if (d.lmgo) { app.setCursor([...state.landmarks.find((l) => l.key === d.lmgo).point]); return; }
    if (d.lmdel) return setLandmark(d.lmdel, null);
    if ('lmplace' in d) return setLandmark(state.lmSel, [...app.cursor], true);
    if ('lmsuggest' in d) return suggestLandmarks();
    if (d.lm) { state.lmSel = d.lm; return render(); }
  }
  const t = e.target.closest('[data-step],[data-a],[data-select],[data-remove],[data-register],[data-registerall],[data-gizmo],[data-nudge],[data-refine],[data-reset],[data-accept],[data-toggle],[data-groupcheck],[data-group],[data-bite],[data-frame],[data-view],[data-heat]');
  if (!t || t.disabled) return;
  const d = t.dataset;
  if (d.step) { state.step = d.step; return render(); }
  if (d.remove) return removeScan(d.remove);
  if (d.register) return register([d.register]);
  if ('registerall' in d) return register(state.scans.filter((s) => !s.registered).map((s) => s.id));
  if (d.gizmo) return gizmo(d.gizmo);
  if (d.nudge) { const [axis, sign, rot] = d.nudge.split(',').map(Number); return nudge(axis, sign, !!rot); }
  if (d.refine) return refine(d.refine);
  if (d.reset) return resetAuto(d.reset);
  if (d.accept) return accept(d.accept);
  if (d.toggle) return toggleStructures([d.toggle], !state.visible.has(d.toggle));
  if (d.groupcheck) {
    e.stopPropagation();
    const keys = state.structures.filter((s) => s.group === d.groupcheck).map((s) => s.key);
    return toggleStructures(keys, !keys.every((k) => state.visible.has(k)));
  }
  if (d.group) { state.groupsOpen.has(d.group) ? state.groupsOpen.delete(d.group) : state.groupsOpen.add(d.group); return render(); }
  if (d.bite) { state.exportOpts.bite = d.bite; if (d.bite === 'scan') state.exportOpts.frame = 'exocad'; return render(); }
  if (d.frame) {
    if ((d.frame === 'plane' || d.frame === 'articulator') && !state.planes.some((p) => p.ready))
      toast('Сначала поставьте ориентиры (шаг «Ориентиры»): нужна хотя бы одна плоскость и оба мыщелка', 'info');
    state.exportOpts.frame = d.frame;
    return render();
  }
  if (d.view) return viewer.fit(d.view);
  if ('heat' in d) { state.heat = !state.heat; for (const s of state.scans) await showScan(s); return render(); }
  if (d.select && !e.target.closest('select,button')) {
    state.selected = d.select;
    if (state.gizmo) viewer.attach(state.selected, state.gizmo);
    return render();
  }
  const action = { ct: () => openCt('ct'), ctdir: () => openCt('ctdir'), scan: addScans, segment, export: doExport,
    models: async () => { const p = await choose('models', 'Папка моделей сегментации'); if (p) { state.modelsDir = p[0]; render(); } } }[d.a];
  action?.();
});

document.addEventListener('change', (e) => {
  const d = e.target.dataset;
  if (d.jaw) setJaw(d.jaw, e.target.value);
  if ('stepmm' in d) state.stepMm = Number(e.target.value);
  if ('stepdeg' in d) state.stepDeg = Number(e.target.value);
  if ('refplane' in d) state.refPlane = e.target.value;
  if ('refart' in d) state.refArt = e.target.value;
});

document.addEventListener('keydown', (e) => {
  if (e.target.closest('input,select')) return;
  if (state.step !== 'align' || !scanById(state.selected)?.registered) return;
  if (e.key === 'g' || e.key === 'п') gizmo('translate');
  if (e.key === 'r' || e.key === 'к') gizmo('rotate');
  if (e.key === 'Escape' && state.gizmo) gizmo(state.gizmo);
});

// ---------- старт: подхватить уже открытый кейс ----------
(async () => {
  const s = await get('state');
  state.modelsDir = s.models_dir;
  applyLandmarks(s);
  render();
  if (s.ct) {
    state.ct = s.ct;
    app.window = s.ct.window;
    app.cursor = [...s.ct.focus];
    await Promise.all(slices.map((v) => v.setCt(s.ct)));
    viewer.setMesh('ct_teeth', await mesh('ct/surface'), { color: '#e9e2d2', opacity: 0.9 });
  }
  for (const info of s.scans) {
    updateScan(info);
    viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, info.transform);
    await showScan(info);
    state.selected ??= info.id;
  }
  if (s.structures.length) {
    state.structures = s.structures;
    state.visible = new Set(s.structures.map((x) => x.key).filter((k) => !k.startsWith('pulp/') && !HIDDEN_BY_DEFAULT.includes(k)));
    await Promise.all(s.structures.map(async (x) => {
      const m = viewer.setMesh(x.key, await mesh(`structures/${x.key}/mesh`), { color: x.color, opacity: TRANSLUCENT[x.key] ?? 1, order: TRANSLUCENT[x.key] ? 1 : 0 });
      m.visible = state.visible.has(x.key);
    }));
    viewer.setVisible('ct_teeth', false);
  }
  applyLandmarks(s);
  if (s.ct) app.setCursor(app.cursor);
  viewer.fit('front');
  render();
})();
