// Custom Case Designer, первая версия: КТ и сегментация, совмещение сканов с КТ, экспорт в единых координатах.
import { get, mesh, post, run } from './api.js';
import { icons } from './icons.js';
import { SliceView } from './slices.js';
import { Viewer3D } from './viewer3d.js';

const STEPS = [
  { id: 'ct', title: 'КТ', icon: 'ct' },
  { id: 'scans', title: 'Сканы', icon: 'tooth' },
  { id: 'export', title: 'Экспорт', icon: 'export' },
];
// Что видно сразу после сегментации: отдельные зубы, кости, каналы, пазухи.
const HIDDEN_BY_DEFAULT = ['upper_teeth', 'lower_teeth', 'skull', 'nasopharynx', 'oropharynx', 'hypopharynx', 'hard_palate'];
const TRANSLUCENT = { mandible: 0.42, maxilla: 0.42, skull: 0.3, maxillary_sinus: 0.55, frontal_sinus: 0.55,
  nasal_cavity: 0.45, pharynx: 0.4, nasopharynx: 0.4, oropharynx: 0.4, hypopharynx: 0.4, soft_palate: 0.7, hard_palate: 0.5 };

const state = {
  step: 'ct', ct: null, scans: [], structures: [], visible: new Set(), heat: true, selected: null,
  modelsDir: null, frame: 'exocad', exported: null, busy: false, groupsOpen: new Set(['Зубы']),
  models: null, download: null, downloadError: null,
  opacity: {}, // объект → прозрачность, заданная кнопкой (иначе — по умолчанию)
  warnOpen: new Set(), // карточки сканов с раскрытым списком предупреждений
  moving: new Map(), // скан → номер последней ручной поправки, ещё не оценённой сервером
  correcting: null, // { id, start } — режим коррекции положения: манипулятор на срезах, start — положение до него
};
let overlayVersion = 0; // меняется, когда сервер принял новое положение скана: контуры — заново

const app = {
  cursor: [0, 0, 0],
  window: [400, 3000],
  setCursor(p) {
    this.cursor = p;
    slices.forEach((s) => s.refresh());
  },
  visibleKeys: () => [...state.scans.filter((s) => s.transform).map((s) => s.id), ...state.visible],
  landmarkPoints: () => [],
  placeLandmark: () => {},
  overlayVersion: () => overlayVersion,
  // Манипулятор на срезах: выбранный совмещённый скан на шаге «Сканы» и его центр (мм пациента).
  manipTarget() {
    const s = scanById(state.selected);
    if (state.step !== 'scans' || state.correcting?.id !== s?.id || !s?.registered || !viewer.objects.has(s.id)) return null;
    return { id: s.id, centre: viewer.scanCentre(s.id) };
  },
  moveScan(id, M) { moveScan(id, M); },
  settledScan: (id) => !state.moving.has(id),
};

const $ = (sel, root = document) => root.querySelector(sel);
const panel = $('#panel');
const viewer = new Viewer3D($('#view3d'));
const slices = [...document.querySelectorAll('.view[data-axis]')].map((el) => new SliceView(el, el.dataset.axis, app));
window.ccd = { app, slices, viewer, state }; // для проверки интерфейса в браузере

// ---------- общие помощники ----------
function toast(text, kind = 'error') {
  const t = document.createElement('div');
  t.className = `toast ${kind === 'info' || kind === 'learn' ? 'info' : ''}`;
  t.textContent = text;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), kind === 'info' ? 3500 : kind === 'learn' ? 8000 : 6000);
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
const plural = (n, one, few, many) => (n % 10 === 1 && n % 100 !== 11 ? one
  : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14) ? few : many);
const cls = (v, good, fair) => (v <= good ? 'ok' : v <= fair ? 'warn' : 'bad');
const scanById = (id) => state.scans.find((s) => s.id === id);
const JAWS = { upper: 'верхняя', lower: 'нижняя' };

function updateScan(info) {
  const i = state.scans.findIndex((s) => s.id === info.id);
  if (i >= 0) state.scans[i] = info; else state.scans.push(info);
}

async function showScan(info) {
  if (!viewer.objects.has(info.id)) viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, null);
  if (info.transform) viewer.setTransform(info.id, info.transform);
  const rgb = state.heat && info.registered ? new Uint8Array(await get(`scans/${info.id}/colors`)) : null;
  viewer.setColors(info.id, rgb);
}

async function loadStructures(list) {
  for (const s of state.structures) viewer.remove(s.key);
  state.structures = list;
  state.visible = new Set(list.map((s) => s.key).filter((k) => !k.startsWith('pulp/') && !HIDDEN_BY_DEFAULT.includes(k)));
  await Promise.all(list.map(async (s) => {
    const m = viewer.setMesh(s.key, await mesh(`structures/${encodeURIComponent(s.key).replace(/%2F/g, '/')}/mesh`),
      { color: s.color, opacity: TRANSLUCENT[s.key] ?? 1, order: TRANSLUCENT[s.key] ? 1 : 0 });
    m.visible = state.visible.has(s.key);
  }));
  viewer.setVisible('ct_teeth', !list.length);
}

// ---------- действия ----------
async function openCt(kind) {
  const paths = await choose(kind, kind === 'ctdir' ? 'Папка DICOM' : 'Файл КТ');
  if (!paths) return;
  await busy('Открываю КТ', async () => {
    state.ct = await run('ct', { path: paths[0] }, () => {});
    app.window = state.ct.window;
    app.cursor = [...state.ct.focus];
    for (const s of state.scans) { s.transform = null; s.registered = false; s.accepted = false; viewer.setVisible(s.id, false); }
    await loadStructures([]);
    state.exported = null;
    await Promise.all(slices.map((s) => s.setCt(state.ct)));
    app.setCursor(app.cursor);
    viewer.setMesh('ct_teeth', await mesh('ct/surface'), { color: '#e9e2d2' });
    viewer.fit('front');
  });
}

async function segment() {
  const dir = state.modelsDir || state.models?.folder;
  await busy('Сегментация КТ', async (progress) => {
    const res = await run('segment', { models_dir: dir }, progress);
    state.modelsDir = dir;
    progress({ progress: 1, message: 'Загружаю поверхности' });
    await loadStructures(res.structures);
    for (const info of res.scans || []) { updateScan(info); await showScan(info); }  // совмещены заново по зубам
    app.setCursor(app.cursor);
  });
}

// Что сегментировать: части КТ (настройка хранится в программе); с открытым КТ — сразу сегментировать.
function partsDialog() {
  const p = state.parts;
  if (!p) return;
  const chosen = new Set(p.selected);
  const canRun = state.ct && state.models?.ready;
  const back = document.createElement('div');
  back.className = 'modal-back';
  const list = () => p.parts.map((x) => `<div class="item ${chosen.has(x.id) ? '' : 'off'}" data-part="${x.id}">
    <span class="check ${chosen.has(x.id) ? 'on' : ''}"></span>${x.title}</div>`).join('');
  back.innerHTML = `<div class="modal"><h3>Что сегментировать</h3>
    <p>Лишние части не считаются — быстрее. Зубы для совмещения сканов программа находит всегда.</p>
    <div class="card tree" data-list>${list()}</div>
    <div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn ghost" data-x>Отмена</button>
    <button class="btn ${canRun ? '' : 'primary'}" data-save>Сохранить</button>
    ${canRun ? `<button class="btn primary" data-run>${icons.play}Сегментировать</button>` : ''}</div></div>`;
  document.body.appendChild(back);
  const save = async () => {
    state.parts = await post('settings/segment_parts', { parts: [...chosen] });
    back.remove();
    render();
  };
  back.addEventListener('click', async (e) => {
    const item = e.target.closest('[data-part]');
    if (item) {
      if (chosen.has(item.dataset.part)) chosen.delete(item.dataset.part); else chosen.add(item.dataset.part);
      $('[data-list]', back).innerHTML = list();
    } else if (e.target.closest('[data-x]') || e.target === back) back.remove();
    else if (e.target.closest('[data-save]')) await save().catch((err) => toast(err.message));
    else if (e.target.closest('[data-run]')) {
      if (!chosen.size) return toast('Выберите хотя бы одну часть');
      await save().catch((err) => toast(err.message));
      segment();
    }
  });
}

// ---------- модели сегментации: загрузка в фоне, не мешает работе ----------
const mb = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0);
const eta = (s) => (s == null ? '' : s < 60 ? `ещё ${Math.max(1, Math.round(s))} с` : `ещё ${Math.round(s / 60)} мин`);

async function downloadModels() {
  if (state.download) return;
  state.download = { progress: 0, info: {} };
  state.downloadError = null;
  renderModels();
  try {
    state.models = await run('models/download', {}, (job) => { state.download = job; renderModels(); });
    state.modelsDir = state.models.folder;
    toast('Модели сегментации установлены', 'info');
  } catch (e) {
    state.downloadError = e.message;
    state.models = await get('models').catch(() => state.models);
  } finally {
    state.download = null;
    render();
  }
}

function modelsCard() {
  const m = state.models;
  if (!m || (m.ready && !state.download)) return '';
  const d = state.download;
  let body;
  if (d) {
    const i = d.info || {};
    const pct = Math.round((d.progress || 0) * 100);
    body = `<div class="dl-row"><span class="dl-name">${d.message || 'Подключаюсь…'}</span>${i.count > 1 ? `<span class="muted">${i.index}/${i.count}</span>` : ''}</div>
      <div class="dl-bar"><i style="width:${pct}%"></i></div>
      <div class="dl-row muted"><span>${i.total ? `${mb(i.done)} из ${mb(i.total)} МБ${i.speed ? ` · ${mb(i.speed)} МБ/с` : ''}` : ''}</span><span>${pct}%${i.eta_s != null ? ` · ${eta(i.eta_s)}` : ''}</span></div>
      <button class="btn ghost sm" data-a="stopmodels">Остановить</button>`;
  } else {
    const partial = m.left_bytes < m.total_bytes;
    body = `<p class="muted small" style="margin-top:0">Нужны для сегментации КТ. Скачиваются один раз — ${mb(m.left_bytes)} МБ.</p>
      ${state.downloadError?.startsWith('загрузка остановлена') ? `<p class="muted small" style="margin-top:0">Остановлено: скачано ${mb(m.total_bytes - m.left_bytes)} из ${mb(m.total_bytes)} МБ.</p>`
        : state.downloadError ? `<div class="warning">${icons.warn}<span>${state.downloadError}</span></div>` : ''}
      <button class="btn wide" data-a="getmodels">${icons.export.replace('<svg', '<svg style="transform:rotate(180deg)"')}${partial ? 'Продолжить загрузку' : 'Скачать модели'}</button>`;
  }
  return `<div class="card" id="modelsCard"><div class="card-head"><h3>Модели сегментации</h3></div>${body}
    <p class="license">${m.license}</p></div>`;
}

function renderModels() {
  const card = $('#modelsCard');
  if (card && state.step === 'ct') card.outerHTML = modelsCard();
  else if (state.step === 'ct') render();
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
  if (state.ct) registerPending();
}

async function removeScan(id) {
  await post(`scans/${id}/remove`);
  viewer.remove(id);
  state.scans = state.scans.filter((s) => s.id !== id);
  if (state.selected === id) state.selected = state.scans[0]?.id ?? null;
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
    viewer.fit('front');
  });
}

const registerPending = () => register(state.scans.filter((s) => !s.registered).map((s) => s.id));

// Ручная поправка с манипулятора на срезе: M — матрица 4×4 в мм пациента, применяется поверх положения скана.
let evalTimer = null;
function moveScan(id, M) {
  const T = viewer.getTransform(id);
  const R = T.map((_, r) => [0, 1, 2, 3].map((c) => M[r].reduce((sum, m, k) => sum + m * T[k][c], 0)));
  viewer.setTransform(id, R);
  state.moving.set(id, (state.moving.get(id) || 0) + 1);
  evaluateManual(id, 250); // серия нажатий стрелок — одной оценкой
}

let evalChain = Promise.resolve(); // оценки положения — строго по очереди: последнее отправленное положение и остаётся
function evaluateManual(id, delay = 0) {
  clearTimeout(evalTimer);
  evalTimer = setTimeout(() => { evalChain = evalChain.then(() => evaluateNow(id)); }, delay);
}

async function evaluateNow(id) {
  {
    const mark = state.moving.get(id);
    const info = await run(`scans/${id}/evaluate`, { transform: viewer.getTransform(id) }).catch((e) => toast(e.message));
    if (!info) return;
    const later = state.moving.get(id) !== mark; // пока считалось, скан ещё подвинули — положение не трогаем
    updateScan(info);
    if (!later) {
      state.moving.delete(id);
      await showScan(info);
    } else {
      const rgb = state.heat && info.registered ? new Uint8Array(await get(`scans/${id}/colors`)) : null;
      viewer.setColors(id, rgb);
    }
    overlayVersion += 1;
    app.setCursor(app.cursor);
    render();
  }
}

async function refine(id) {
  await register([id], { start: viewer.getTransform(id), jaw: scanById(id).jaw });
}

async function resetAuto(id) {
  const s = scanById(id);
  if (!s.auto_transform) return;
  viewer.setTransform(id, s.auto_transform);
  state.moving.set(id, (state.moving.get(id) || 0) + 1);
  slices.forEach((v) => v.settled(id));
  evaluateManual(id);
}

// Коррекция положения: манипулятор на срезах; «Сохранить» запоминает результат пользователя для обучения.
function startCorrection(id) {
  state.correcting = { id, start: viewer.getTransform(id) };
  slices.forEach((v) => v.draw());
  render();
}

function endCorrection() {
  state.correcting = null;
  slices.forEach((v) => v.draw());
  render();
}

function revertCorrection() {
  const c = state.correcting;
  if (!c) return;
  viewer.setTransform(c.id, c.start);
  state.moving.set(c.id, (state.moving.get(c.id) || 0) + 1);
  slices.forEach((v) => v.settled(c.id));
  evaluateManual(c.id);
}

async function settledPosition(id) { // дождаться, пока сервер примет последнюю поправку
  clearTimeout(evalTimer);
  if (state.moving.has(id)) evalChain = evalChain.then(() => evaluateNow(id));
  await evalChain;
}

async function accept(id) {
  await settledPosition(id);
  const r = await post(`scans/${id}/accept`).catch((e) => toast(e.message));
  if (!r) return;
  updateScan(r.scan);
  const m = r.memory || {};
  const corrected = r.record.corrected_mm > 0.05;
  let text = corrected ? `Поправка ${fmt(r.record.corrected_mm)} мм сохранена.` : 'Положение принято.';
  if (!r.record.learned) text += ' Скан лёг на коронки неуверенно — в обучение этот кейс не взят.';
  else if ((m.usable_for_learning ?? 0) >= r.min_cases) {
    text += ` Учтено в обучении: по этому аппарату КТ ${m.usable_for_learning} кейсов, поправка границы эмали ${fmt(m.edge_shift_mm, 3)} мм — применяется к следующим совмещениям.`;
  } else {
    const left = r.min_cases - (m.usable_for_learning ?? 0);
    text += ` Учтено в обучении: ещё ${left} ${left === 1 ? 'кейс' : 'кейса'} по этому аппарату КТ — и поправка начнёт применяться к следующим совмещениям.`;
  }
  if (state.correcting?.id === id) state.correcting = null;
  slices.forEach((v) => v.draw());
  toast(text, 'learn');
  render();
}

// Прозрачность объекта в 3D: кнопкой — прозрачный ↔ непрозрачный (по умолчанию — как задано для структуры).
const defaultOpacity = (key) => (scanById(key) ? 1 : TRANSLUCENT[key] ?? 1);
const opacityOf = (key) => state.opacity[key] ?? defaultOpacity(key);
function toggleOpacity(keys) {
  const next = keys.every((k) => opacityOf(k) < 1) ? 1 : 0.35;
  for (const k of keys) {
    state.opacity[k] = next;
    viewer.setOpacity(k, next);
  }
  render();
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
  const hasScans = state.scans.some((s) => s.registered);
  const frame = hasScans ? state.frame : 'dicom';
  await busy('Экспорт', async (progress) => {
    state.exported = await run('export', { out_dir: paths[0], bite: frame === 'dicom' ? 'ct' : 'scan', frame,
      include: [...state.visible] }, progress);
  });
}

// ---------- панели ----------
function metricsHtml(s) {
  const st = s.stats;
  if (!st || st.mean_mm === undefined) return '';
  const within = Math.round((st.within_0_2_mm ?? 0) * 100);
  return `<div class="metrics">
    <div class="metric"><span>Среднее, мм</span><b class="${cls(st.mean_mm, 0.08, 0.15)}">${fmt(st.mean_mm, 3)}</b></div>
    <div class="metric"><span>90% точек, мм</span><b class="${cls(st.p90_mm, 0.15, 0.25)}">${fmt(st.p90_mm, 3)}</b></div>
    <div class="metric"><span>≤ 0.2 мм</span><b class="${within >= 90 ? 'ok' : within >= 75 ? 'warn' : 'bad'}">${within}%</b></div>
  </div>`;
}

// Кнопка прозрачности: наполовину залитый кружок — объект прозрачный, залитый — непрозрачный.
function opacityButton(keys, attr) {
  const clear = keys.every((k) => opacityOf(k) < 1);
  return `<button class="opacity-btn ${clear ? 'clear' : ''}" ${attr} title="${clear ? 'Сделать непрозрачным' : 'Сделать прозрачным'}"><i></i></button>`;
}

function renderCt() {
  const ct = state.ct;
  if (!ct) {
    return `<h2>КТ</h2><p class="lead">КЛКТ: папка DICOM, архив или файл NIfTI, MHA, NRRD.</p>
      <button class="btn primary wide" data-a="ctdir">${icons.folder}Открыть папку DICOM</button>
      <button class="btn ghost wide" style="margin-top:6px" data-a="ct">или файл…</button>${modelsCard()}`;
  }
  const info = `<div class="card"><div class="card-head"><h3>${ct.name}</h3><button class="btn ghost sm" data-a="ctdir" title="Открыть другое КТ">Другое…</button></div>
    <div class="kv"><span>Размер</span><b>${ct.shape.join(' × ')}</b><span>Воксель</span><b>${ct.spacing.map((v) => fmt(v, 2)).join(' × ')} мм</b>
    ${ct.device ? `<span>Аппарат</span><b>${ct.device}</b>` : ''}</div></div>`;
  const partsButton = '<button class="btn ghost wide sm" style="margin-top:6px" data-a="parts">Что сегментировать…</button>';
  if (!state.structures.length) {
    const ready = state.models?.ready && state.parts?.selected.length;
    const chosen = (state.parts?.parts || []).filter((p) => state.parts.selected.includes(p.id)).map((p) => p.title);
    return `<h2>КТ</h2>${info}
      <button class="btn primary wide" data-a="segment" ${ready ? '' : 'disabled'}>${icons.play}Сегментировать</button>${partsButton}
      <p class="muted small">${!state.models?.ready ? 'Сначала скачайте модели сегментации.' : chosen.length ? `${chosen.join(', ')}.` : 'Не выбрано, что сегментировать.'}</p>${modelsCard()}`;
  }
  const groups = {};
  for (const s of state.structures) (groups[s.group] ??= []).push(s);
  const tree = Object.entries(groups).map(([title, items]) => {
    const on = items.filter((s) => state.visible.has(s.key)).length;
    const open = state.groupsOpen.has(title);
    const check = on === items.length ? 'on' : on ? 'mixed' : '';
    return `<div class="group"><div class="ghead" data-group="${title}"><span class="check ${check}" data-groupcheck="${title}"></span>
      <b>${title}</b>${opacityButton(items.map((s) => s.key), `data-gopacity="${title}"`)}<span>${on}/${items.length}</span><span style="transform:rotate(${open ? 90 : 0}deg);display:flex">${icons.chevron.replace('<svg', '<svg width="14" height="14"')}</span></div>
      ${open ? items.map((s) => `<div class="item ${state.visible.has(s.key) ? '' : 'off'}" data-toggle="${s.key}">
        <span class="check ${state.visible.has(s.key) ? 'on' : ''}"></span><i class="dot" style="background:${s.color}"></i><span class="grow">${s.name}</span>${opacityButton([s.key], `data-opacity="${s.key}"`)}</div>`).join('') : ''}</div>`;
  }).join('');
  return `<h2>КТ</h2>${info}<div class="label">Структуры — ${state.structures.length}; видимые попадут в экспорт</div>
    <div class="card tree">${tree}</div>${partsButton}`;
}

function renderScans() {
  if (!state.ct) return '<h2>Сканы</h2><p class="lead">Сначала откройте КТ.</p>';
  const sel = scanById(state.selected);
  const cards = state.scans.map((s) => {
    const status = s.accepted ? '<span class="badge ok">принят</span>' : s.registered ? '' : '<span class="badge">не совмещён</span>';
    const jaw = s.jaw ? `<span class="badge">${JAWS[s.jaw]}</span>` : '';
    const list = s.warnings || [];
    const open = state.warnOpen.has(s.id);
    const warnBtn = list.length ? `<button class="warn-btn ${open ? 'on' : ''}" data-warns="${s.id}" title="Предупреждения">${icons.warn}<b>${list.length}</b></button>` : '';
    const warnings = open ? list.map((w) => `<div class="warning">${icons.warn}<span>${w}</span></div>`).join('') : '';
    return `<div class="card ${s.id === state.selected ? 'sel' : ''}" data-select="${s.id}">
      <div class="card-head"><i class="dot" style="background:${s.color}"></i><h3 title="${s.path}">${s.name}</h3>${jaw}${status}
        ${warnBtn}${s.registered ? opacityButton([s.id], `data-opacity="${s.id}"`) : ''}
        <button class="btn icon ghost" data-remove="${s.id}" title="Убрать">${icons.trash}</button></div>
      ${metricsHtml(s)}${warnings}
      ${s.registered ? '' : `<button class="btn wide" data-register="${s.id}">${icons.play}Совместить</button>`}</div>`;
  }).join('');
  const correcting = !!sel && state.correcting?.id === sel.id;
  let tools = '';
  if (sel?.registered && !correcting) {
    tools = `<div class="label">Положение — ${sel.name}</div>
      <div class="row"><button class="btn grow" data-correct="${sel.id}">${icons.move}Скорректировать</button>
        <button class="btn ok grow" data-accept="${sel.id}" ${sel.accepted ? 'disabled' : ''}>${icons.check}${sel.accepted ? 'Принято' : 'Принять'}</button></div>
      <p class="muted small">Не устраивает, как сел скан, — скорректируйте: программа запомнит ваше положение и учтёт его в следующих совмещениях.</p>`;
  } else if (correcting) {
    tools = `<div class="card correcting"><div class="card-head">${icons.move}<h3>Коррекция — ${sel.name}</h3></div>
      <p class="muted small" style="margin-top:0">На срезе: внутри кольца — сдвиг, за кольцо — поворот. Стрелки — точно, Ctrl+←/→ — поворот, Shift — крупнее.</p>
      <div class="row" style="margin-top:8px"><button class="btn grow" data-refine="${sel.id}" title="Подогнать по коронкам от текущего положения">${icons.refine}Уточнить</button>
        <button class="btn icon" data-revert title="Вернуть, как было до коррекции">${icons.undo}</button></div>
      <div class="row" style="margin-top:8px"><button class="btn ghost" data-endcorrect>Отмена</button>
        <button class="btn ok grow" data-accept="${sel.id}">${icons.check}Сохранить поправку</button></div></div>`;
  }
  return `<h2>Сканы</h2><p class="lead">STL, PLY или OBJ как есть со сканера. Совмещение — по коронкам зубов, сразу после загрузки.</p>
    ${cards}<button class="btn ${state.scans.length ? '' : 'primary'} wide" data-a="scan">${icons.plus}Добавить сканы</button>${tools}`;
}

function renderExport() {
  const hasScans = state.scans.some((s) => s.registered);
  const ready = hasScans || state.structures.length;
  const res = state.exported;
  const frames = hasScans ? `<div class="seg"><button data-frame="exocad" class="${state.frame === 'exocad' ? 'on' : ''}">Сканера (exocad)</button>
      <button data-frame="dicom" class="${state.frame === 'dicom' ? 'on' : ''}">КТ (DICOM)</button></div>
      <p class="muted small">${state.frame === 'exocad' ? 'В координатах сканера exocad открывает сканы: структуры КТ встанут к ним, прикус — со сканов.' : 'Всё стоит как на КТ; сканы — на своих челюстях.'}</p>`
    : '<p class="muted small">Сканов нет — структуры КТ в координатах КТ (DICOM).</p>';
  const result = res ? `<div class="card"><div class="card-head">${icons.check}<h3>Готово — ${res.files.length} ${plural(res.files.length, 'файл', 'файла', 'файлов')}</h3></div>
      <div class="path" title="${res.out_dir}">${res.out_dir}</div>
      ${(res.notes || []).map((n) => `<div class="warning">${icons.warn}<span>${n}</span></div>`).join('')}</div>` : '';
  return `<h2>Экспорт</h2><p class="lead">STL всех видимых объектов в единой системе координат и case.json с матрицами.</p>
    <div class="label">Система координат</div>${frames}
    <button class="btn primary wide" data-a="export" ${ready ? '' : 'disabled'}>${icons.export}Экспортировать в папку…</button>
    ${ready ? '' : '<p class="muted small">Сегментируйте КТ или совместите сканы.</p>'}${result}`;
}

const RENDER = { ct: renderCt, scans: renderScans, export: renderExport };

function render() {
  const done = {
    ct: state.structures.length > 0,
    scans: state.scans.length > 0 && state.scans.every((s) => s.registered),
    export: !!state.exported,
  };
  $('#rail').innerHTML = STEPS.map((s) => `<button class="step ${s.id === state.step ? 'active' : ''} ${done[s.id] ? 'done' : ''}" data-step="${s.id}">
    ${icons[s.icon]}<span>${s.title}</span><i class="dot"></i></button>`).join('');
  panel.innerHTML = RENDER[state.step]();
  const ct = state.ct;
  $('#caseChip').textContent = ct ? `${ct.name}${state.scans.length ? ` · сканов: ${state.scans.length}` : ''}` : '';
  $('#empty3d').innerHTML = ct || state.scans.length ? '' : '<span>Откройте КТ</span>';
  const legend = $('#legend');
  legend.hidden = !(state.heat && state.scans.some((s) => s.registered));
  legend.innerHTML = '<span><i style="background:#28aa46"></i>≤ 0.1 мм</span><span><i style="background:#e6be1e"></i>≤ 0.2 мм</span>' +
    '<span><i style="background:#d23228"></i>> 0.2 мм</span><span><i style="background:#aaa"></i>не коронки</span>';
  $('#tools3d').innerHTML = [['front', 'Спереди'], ['right', 'Справа'], ['left', 'Слева'], ['top', 'Сверху']]
    .map(([v, t]) => `<button class="btn" data-view="${v}" style="width:auto;padding:0 8px">${t}</button>`).join('') +
    (state.scans.some((s) => s.registered) ? `<button class="btn ${state.heat ? 'on' : ''}" data-heat title="Карта отклонений скана от КТ">${icons.heat}</button>` : '');
}

// ---------- события ----------
document.addEventListener('click', async (e) => {
  const t = e.target.closest('[data-step],[data-a],[data-select],[data-remove],[data-register],[data-refine],[data-reset],[data-accept],[data-correct],[data-revert],[data-endcorrect],[data-opacity],[data-gopacity],[data-warns],[data-toggle],[data-groupcheck],[data-group],[data-frame],[data-view],[data-heat]');
  if (!t || t.disabled) return;
  const d = t.dataset;
  if (d.step) { state.step = d.step; slices.forEach((v) => v.draw()); return render(); }
  if (d.remove) { e.stopPropagation(); return removeScan(d.remove); }
  if (d.register) return register([d.register]);
  if (d.opacity) { e.stopPropagation(); return toggleOpacity([d.opacity]); }
  if (d.gopacity) {
    e.stopPropagation();
    return toggleOpacity(state.structures.filter((s) => s.group === d.gopacity).map((s) => s.key));
  }
  if (d.warns) {
    e.stopPropagation();
    state.warnOpen.has(d.warns) ? state.warnOpen.delete(d.warns) : state.warnOpen.add(d.warns);
    return render();
  }
  if (d.refine) return refine(d.refine);
  if (d.reset) return resetAuto(d.reset);
  if (d.accept) return accept(d.accept);
  if (d.correct) return startCorrection(d.correct);
  if ('revert' in d) return revertCorrection();
  if ('endcorrect' in d) { revertCorrection(); return endCorrection(); }
  if (d.toggle) return toggleStructures([d.toggle], !state.visible.has(d.toggle));
  if (d.groupcheck) {
    e.stopPropagation();
    const keys = state.structures.filter((s) => s.group === d.groupcheck).map((s) => s.key);
    return toggleStructures(keys, !keys.every((k) => state.visible.has(k)));
  }
  if (d.group) { state.groupsOpen.has(d.group) ? state.groupsOpen.delete(d.group) : state.groupsOpen.add(d.group); return render(); }
  if (d.frame) { state.frame = d.frame; return render(); }
  if (d.view) return viewer.fit(d.view);
  if ('heat' in d) { state.heat = !state.heat; for (const s of state.scans) await showScan(s); return render(); }
  if (d.select && !e.target.closest('button')) {
    if (state.selected !== d.select) state.correcting = null;
    state.selected = d.select;
    slices.forEach((v) => v.draw());
    return render();
  }
  const action = { ct: () => openCt('ct'), ctdir: () => openCt('ctdir'), scan: addScans, segment, export: doExport,
    getmodels: downloadModels, stopmodels: () => post('models/cancel'), parts: partsDialog }[d.a];
  action?.();
});

document.addEventListener('keydown', (e) => {
  if (e.target.closest('input,select')) return;
  SliceView.onKey(e); // стрелки — точная поправка скана в срезе под мышью
});

// ---------- старт: подхватить уже открытый кейс ----------
(async () => {
  const s = await get('state');
  state.modelsDir = s.models_dir;
  state.models = s.models;
  state.parts = s.segment_parts;
  $('#version').textContent = s.version ? `v${s.version}` : '';
  render();
  if (s.ct) {
    state.ct = s.ct;
    app.window = s.ct.window;
    app.cursor = [...s.ct.focus];
    await Promise.all(slices.map((v) => v.setCt(s.ct)));
    viewer.setMesh('ct_teeth', await mesh('ct/surface'), { color: '#e9e2d2' });
  }
  for (const info of s.scans) {
    updateScan(info);
    viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, info.transform);
    await showScan(info);
    state.selected ??= info.id;
  }
  if (s.structures.length) await loadStructures(s.structures);
  if (s.ct) app.setCursor(app.cursor);
  viewer.fit('front');
  render();
})();
