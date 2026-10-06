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
  modelsDir: null, frame: 'exocad', bite: 'scan', exported: null, busy: false, groupsOpen: new Set(['Зубы']),
  models: null, download: null, downloadError: null,
  opacity: {}, // объект → прозрачность, заданная кнопкой (иначе — по умолчанию)
  warnOpen: new Set(), // карточки сканов с раскрытым списком предупреждений
  moving: new Map(), // скан → номер последней ручной поправки, ещё не оценённой сервером
  correcting: null, // { id, start, undo, redo } — режим коррекции: манипулятор на срезах, start — положение до него
  windowName: 'auto', // набор окна КТ ('' — подобрано вручную)
  caseInfo: null, recent: [], // сохранённый кейс и недавние кейсы
  incognito: false, // не показывать имён пациентов и путей (для показа программы)
};
// Ошибки интерфейса — в журнал программы (пути и имена сервер вычищает).
const reportError = (message, stack) => post('log', { message: String(message || ''), stack: String(stack || '') }).catch(() => {});
window.addEventListener('error', (e) => reportError(e.message, e.error?.stack));
window.addEventListener('unhandledrejection', (e) => reportError(e.reason?.message || e.reason, e.reason?.stack));
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
  overlayHeat: () => heatOn() && state.scans.some((x) => x.registered), // контур скана — по цвету отклонения
  // Окно КТ (яркость и контраст): наборы по уровням плотности этого снимка и ручная настройка правой кнопкой.
  windowPresets: () => [...windowPresets(), ...(state.windowName ? [] : [{ id: '', title: 'Вручную' }])],
  windowName: () => state.windowName,
  setWindowPreset(id) {
    const p = windowPresets().find((x) => x.id === id);
    if (p) this.setWindow(p.window, id);
  },
  setWindow(w, name = '') {
    this.window = w;
    state.windowName = name;
    slices.forEach((v) => v.refreshSoon());
  },
  toggleMax(el) {
    const main = $('.main');
    const on = !el.classList.contains('max');
    main.querySelectorAll('.view').forEach((v) => v.classList.remove('max'));
    el.classList.toggle('max', on);
    main.classList.toggle('one-max', on);
  },
  // Манипулятор на срезах: выбранный совмещённый скан на шаге «Сканы» и его центр (мм пациента).
  manipTarget() {
    const s = scanById(state.selected);
    if (state.step !== 'scans' || state.correcting?.id !== s?.id || !s?.registered || !viewer.objects.has(s.id)) return null;
    return { id: s.id, centre: viewer.scanCentre(s.id) }; // центр всего скана
  },
  moveScan(id, M) { moveScan(id, M); },
  settledScan: (id) => !state.moving.has(id),
};

const $ = (sel, root = document) => root.querySelector(sel);

function windowPresets() {
  const ct = state.ct;
  if (!ct?.levels) return [];
  const { hard, dense, min } = ct.levels;
  const gap = Math.max(dense - hard, 1);
  return [
    { id: 'auto', title: 'Авто', window: ct.window },
    { id: 'teeth', title: 'Зубы', window: [dense + gap * 0.2, gap * 2.2] },
    { id: 'bone', title: 'Кость', window: [(hard + dense) / 2, gap * 3] },
    { id: 'soft', title: 'Мягкие ткани', window: [hard - (hard - min) * 0.3, Math.max(hard - min, 1) * 1.1] },
  ];
}
const panel = $('#panel');
const viewer = new Viewer3D($('#view3d'));
const slices = [...document.querySelectorAll('.view[data-axis]')].map((el) => new SliceView(el, el.dataset.axis, app));
window.ccd = { app, slices, viewer, state }; // для проверки интерфейса в браузере
$('#view3d').addEventListener('dblclick', () => app.toggleMax($('#view3d'))); // развернуть 3D (Esc — обратно)

// ---------- общие помощники ----------
function toast(text, kind = 'error') {
  const t = document.createElement('div');
  t.className = `toast ${kind === 'info' || kind === 'learn' ? 'info' : ''}`;
  t.textContent = hide(text);
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
    $('.msg', job).textContent = hide(s.message || '');
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
  const paths = await choosePaths(kind, title);
  if (paths) chosen.push(...paths); // в инкогнито скрываются и они (архив, который не открылся, и т. п.)
  return paths;
}

async function choosePaths(kind, title) {
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
const JAWS = { upper: 'верхняя', lower: 'нижняя', bite: 'прикус' };
const NEXT_JAW = { upper: 'lower', lower: 'bite', bite: 'upper' }; // значок челюсти у скана — по кругу
const heatOn = (info) => state.heat && !!state.ct && (!info || info.registered); // отклонение от КТ — только с КТ
const shownRecent = () => state.recent.filter((r) => r.exists);

// ---------- инкогнито: показ программы без данных пациента ----------
// Имена пациентов бывают в путях и именах файлов (папка КТ, сканы из exocad, файл кейса, проект exocad), а из
// них — в названиях сканов и в сообщениях. В инкогнито на экране вместо них метки «КТ», «Скан 1», «Кейс»,
// вместо остальных путей — «…». Сами данные не меняются. Так же, как в журнале ошибок (errorlog.py).
const WIN_PATH = /(?:[A-Za-z]:[\\/]|\\\\)[^:;"'<>|\r\n\t*?«»]*/g; // в именах файлов Windows нет «:» — путь до неё
const OWN_FOLDERS = /^CustomCaseDesigner$/i; // папки программы — не личные
const chosen = []; // пути, выбранные в этом сеансе
const scanLabel = (s) => (state.incognito ? `Скан ${state.scans.findIndex((x) => x.id === s.id) + 1}` : s.name);

function secrets() {
  const out = [];
  const add = (value, label) => {
    if (!value || String(value).length < 3) return; // короче — не имя, а заменялось бы в любом тексте
    out.push([String(value), label]);
    for (const part of String(value).split(/[\\/]/)) { // папки и имя файла по отдельности: «Иванов И.И», «Иванов-upperjaw.stl»
      const stem = part.replace(/\.[^.]*$/, '');
      if (stem.length >= 3 && !/^[A-Za-z]:$/.test(part) && !OWN_FOLDERS.test(stem)) out.push([part, label], [stem, label]);
    }
  };
  for (const s of state.scans) { add(s.path, scanLabel(s)); add(s.name, scanLabel(s)); }
  add(state.ct?.path, 'КТ');
  add(state.caseInfo?.path, 'Кейс');
  shownRecent().forEach((r, i) => add(r.path, `Кейс ${i + 1}`));
  add(state.exported?.out_dir, 'папка экспорта');
  for (const p of chosen) add(p, 'файл');
  return out.sort((a, b) => b[0].length - a[0].length); // длинные — первыми: путь целиком, потом его части
}

// Текст для экрана: в инкогнито — без имён и путей.
function hide(text) {
  if (!state.incognito || text === null || text === undefined) return text;
  let t = String(text);
  const list = secrets();
  if (list.length) { // за один проход: метки («папка экспорта») не заменяются ещё раз
    const labels = new Map();
    for (const [value, label] of list) if (!labels.has(value)) labels.set(value, label);
    const any = new RegExp([...labels.keys()].map((v) => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g');
    t = t.replace(any, (m) => labels.get(m));
  }
  return t.replace(WIN_PATH, (p) => { // незнакомый путь; после имени файла с расширением — обычный текст
    const tail = p.slice(Math.max(p.lastIndexOf('\\'), p.lastIndexOf('/')) + 1);
    const f = /^[^\s\\/]*\.\w{1,8}(?=\s|$)/.exec(tail);
    return f ? `…${tail.slice(f[0].length)}` : '…'; // без расширения не понять, где кончается путь
  });
}

async function toggleIncognito() {
  const r = await post('settings/incognito', { on: !state.incognito }).catch((e) => toast(e.message));
  if (!r) return;
  state.incognito = r.incognito;
  render();
}

function updateScan(info) {
  const i = state.scans.findIndex((s) => s.id === info.id);
  if (i >= 0) state.scans[i] = info; else state.scans.push(info);
}

async function showScan(info) {
  if (!viewer.objects.has(info.id)) viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, null);
  if (info.transform) viewer.setTransform(info.id, info.transform);
  const rgb = heatOn(info) ? new Uint8Array(await get(`scans/${info.id}/colors`)) : null;
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
const SERIES_MIN = 20; // серии короче — скауты и отдельные снимки, их не предлагаем

// Какую серию открыть, если в папке или архиве их несколько; undefined — отмена.
function seriesDialog(list) {
  return new Promise((done) => {
    let pick = list[0].id;
    const back = document.createElement('div');
    back.className = 'modal-back';
    const rows = () => list.map((x) => `<div class="item ${x.id === pick ? '' : 'off'}" data-series="${x.id}">
      <span class="check radio ${x.id === pick ? 'on' : ''}"></span><span class="grow">${hide(x.description) || 'без описания'}</span>
      <span class="muted small">${x.modality || ''} ${x.size ? x.size.join(' × ') : `${x.files} ${plural(x.files, 'файл', 'файла', 'файлов')}`}</span></div>`).join('');
    back.innerHTML = `<div class="modal"><h3>Несколько серий</h3><p>Выберите, какую открыть. Первая — самая длинная.</p>
      <div class="card tree series" data-list>${rows()}</div>
      <div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn ghost" data-x>Отмена</button>
      <button class="btn primary" data-ok>Открыть</button></div></div>`;
    document.body.appendChild(back);
    const close = (value) => { back.remove(); document.removeEventListener('keydown', key, true); done(value); };
    const key = (e) => { if (e.key === 'Escape') close(undefined); if (e.key === 'Enter') close(pick); };
    document.addEventListener('keydown', key, true);
    back.addEventListener('click', (e) => {
      const item = e.target.closest('[data-series]');
      if (item) { pick = item.dataset.series; $('[data-list]', back).innerHTML = rows(); }
      else if (e.target.closest('[data-ok]')) close(pick);
      else if (e.target.closest('[data-x]') || e.target === back) close(undefined);
    });
    back.addEventListener('dblclick', (e) => { if (e.target.closest('[data-series]')) close(pick); });
  });
}

async function openCt(kind) {
  const paths = await choose(kind, kind === 'ctdir' ? 'Папка DICOM' : 'Файл КТ или архив');
  if (!paths) return;
  const all = await busy('Читаю КТ', () => post('ct/series', { path: paths[0] }));
  if (!all) return;
  const list = all.filter((x) => Math.max(x.files, x.size?.[2] || 0) >= SERIES_MIN);
  const series = list.length > 1 ? await seriesDialog(list) : list[0]?.id ?? null;
  if (series === undefined) return;
  await busy('Открываю КТ', async () => {
    state.ct = await run('ct', { path: paths[0], series }, () => {});
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
  if (state.ct && state.scans.length) registerPending(); // сканы, добавленные без КТ, — на КТ
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
        : state.downloadError ? `<div class="warning">${icons.warn}<span>${hide(state.downloadError)}</span></div>` : ''}
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
  registerPending(); // без КТ — в координатах сканера
}

async function removeScan(id) {
  await post(`scans/${id}/remove`);
  viewer.remove(id);
  state.scans = state.scans.filter((s) => s.id !== id);
  if (state.selected === id) state.selected = state.scans[0]?.id ?? null;
  await refreshScans();
  render();
}

// Сканы прикуса стоят на сканах челюстей, и нижний скан может встать в прикус по ним: после изменения
// положения челюсти — свежие положения всех сканов (кроме тех, что сейчас двигают).
async function refreshScans(force = false) {
  const wasOn = state.biteView?.on;
  const fresh = await get('state');
  state.biteView = fresh.bite_view; // есть ли прикус сканов — после каждого изменения челюстей
  if (!force && !state.scans.some((s) => s.role === 'bite') && !wasOn) return;
  for (const info of fresh.scans) {
    if (state.moving.has(info.id)) continue;
    updateScan(info);
    await showScan(info);
  }
  if ((wasOn || state.biteView?.on) && !force) await reloadStructureMeshes(); // челюсть могла переехать — структуры за ней
}

// Сетки структур заново с сервера (как показаны: с прикусом или как на КТ), видимость и прозрачность — прежние.
async function reloadStructureMeshes() {
  await Promise.all(state.structures.map(async (s) => {
    const old = viewer.objects.get(s.key);
    const opacity = old ? old.material.opacity : (TRANSLUCENT[s.key] ?? 1);
    const visible = old ? old.visible : state.visible.has(s.key);
    const m = viewer.setMesh(s.key, await mesh(`structures/${encodeURIComponent(s.key).replace(/%2F/g, '/')}/mesh`),
      { color: s.color, opacity, order: opacity < 1 ? 1 : 0 });
    m.visible = visible;
  }));
}

// «Прикус»: нижняя челюсть со всеми её структурами — в прикусе сканов (врача); выключено — как на КТ.
async function toggleBite() {
  const r = await post('bite_view', { on: !state.biteView?.on }).catch((e) => toast(e.message));
  if (!r) return;
  state.biteView = r;
  await busy(r.on ? 'Прикус сканов' : 'Как на КТ', async () => {
    await reloadStructureMeshes();
    await refreshScans(true);
    overlayVersion += 1;
    app.setCursor(app.cursor);
  });
}

async function register(ids, body = {}) {
  await busy(state.ct ? 'Совмещаю по коронкам зубов' : 'Расставляю сканы', async (progress) => {
    const failed = [];
    for (const id of ids) {
      try {
        const info = await run(`scans/${id}/register`, body, (s) => progress({ ...s, message: scanLabel(scanById(id)) }));
        updateScan(info);
        await showScan(info);
      } catch (e) {
        failed.push(e.message);
      }
    }
    await refreshScans(!state.ct); // без КТ подсказки у сканов зависят друг от друга — обновить все
    app.setCursor(app.cursor);
    viewer.fit('front');
    if (failed.length) throw new Error(failed.join(' · '));
  });
}

// Сначала сканы челюстей, потом сканы прикуса: они ставятся на челюсти.
const registerPending = () => register(state.scans.filter((s) => !s.registered)
  .sort((a, b) => (a.role === 'bite') - (b.role === 'bite')).map((s) => s.id));

// Ручная поправка с манипулятора на срезе: M — матрица 4×4 в мм пациента, применяется поверх положения скана.
let evalTimer = null;
function remember(id) { // перед изменением положения в режиме коррекции — для Ctrl+Z
  const c = state.correcting;
  if (c?.id !== id) return;
  c.undo.push(viewer.getTransform(id));
  c.redo = [];
}

function moveScan(id, M) {
  remember(id);
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
      await refreshScans();
    } else {
      const rgb = heatOn(info) ? new Uint8Array(await get(`scans/${id}/colors`)) : null;
      viewer.setColors(id, rgb);
    }
    overlayVersion += 1;
    app.setCursor(app.cursor);
    render();
  }
}

async function refine(id) {
  remember(id);
  await register([id], { start: viewer.getTransform(id), jaw: scanById(id).jaw });
}

async function resetAuto(id) {
  const s = scanById(id);
  if (!s.auto_transform) return;
  remember(id);
  viewer.setTransform(id, s.auto_transform);
  state.moving.set(id, (state.moving.get(id) || 0) + 1);
  slices.forEach((v) => v.settled(id));
  evaluateManual(id);
}

// Коррекция положения: манипулятор на срезах; «Сохранить» запоминает результат пользователя для обучения.
async function startCorrection(id) {
  if (state.biteView?.on && scanById(id)?.jaw === 'lower') await toggleBite(); // коррекция — по КТ
  state.correcting = { id, start: viewer.getTransform(id), undo: [], redo: [] };
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

// Ctrl+Z / Ctrl+Y в режиме коррекции: шаг назад и вперёд по положениям скана.
function stepHistory(back) {
  const c = state.correcting;
  if (!c) return;
  const from = back ? c.undo : c.redo, to = back ? c.redo : c.undo;
  if (!from.length) return;
  to.push(viewer.getTransform(c.id));
  viewer.setTransform(c.id, from.pop());
  state.moving.set(c.id, (state.moving.get(c.id) || 0) + 1);
  slices.forEach((v) => v.settled(c.id));
  evaluateManual(c.id);
  render();
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

// Прикус выгрузки: выбирается, когда прикус сканов отличается от прикуса на КТ (как кнопка «Прикус» в 3D).
const biteChoice = () => state.scans.some((s) => s.registered) && !!state.biteView?.available;
const exportBite = () => (biteChoice() ? state.bite : state.frame === 'dicom' ? 'ct' : 'scan');
const EXPORT_HINTS = {
  'exocad scan': 'Сканы — как их открывает exocad; структуры КТ — к скану своей челюсти: прикус со сканов (врача).',
  'exocad ct': 'Верхний скан — как его открывает exocad; нижний скан и все структуры — как на КТ относительно него.',
  'dicom scan': 'Верхняя челюсть — как на КТ; нижняя со своими структурами — в прикусе сканов (как в 3D с кнопкой «Прикус»).',
  'dicom ct': 'Всё стоит как на КТ; сканы — на своих челюстях.',
};

async function doExport() {
  const paths = await choose('out', 'Папка для результата');
  if (!paths) return;
  const hasScans = state.scans.some((s) => s.registered);
  const frame = !state.ct ? 'exocad' : hasScans ? state.frame : 'dicom';
  const bite = state.ct ? exportBite() : 'scan';
  await busy('Экспорт', async (progress) => {
    state.exported = await run('export', { out_dir: paths[0], bite, frame, include: [...state.visible] }, progress);
    state.exported.biteChosen = biteChoice() ? bite : null;
  });
}

// ---------- кейс: сохранить и открыть ----------
async function saveCase(as = false) {
  if (!state.ct && !state.scans.length) return toast('Нечего сохранять: откройте КТ или добавьте сканы');
  let path = as ? null : state.caseInfo?.path;
  if (!path) {
    const paths = await choose('save', 'Куда сохранить кейс (.ccdcase)');
    if (!paths) return;
    path = paths[0];
  }
  for (const s of state.scans) if (state.moving.has(s.id)) await settledPosition(s.id); // последнее положение — в файл
  const r = await post('case/save', { path }).catch((e) => toast(e.message));
  if (!r) return;
  state.caseInfo = { path: r.path, name: r.name.replace(/\.ccdcase$/i, '') };
  toast(`Кейс сохранён: ${r.name}`, 'info');
  render();
}

async function openCase(path) {
  if (!path) {
    const paths = await choose('case', 'Кейс (.ccdcase)');
    if (!paths) return;
    path = paths[0];
  }
  const opened = await busy('Открываю кейс', async (progress) => {
    try {
      return await run('case/open', { path }, progress);
    } catch (e) {
      if (!/КТ этого кейса не найден/.test(e.message)) throw e;
      toast(`${e.message}. Укажите папку КТ.`, 'info');
      const ct = await choose('ctdir', 'Где теперь КТ этого кейса');
      if (!ct) return null;
      return run('case/open', { path, ct_path: ct[0] }, progress);
    }
  });
  if (opened) location.reload(); // интерфейс собирается заново по открытому кейсу
}

// ---------- подсказка: мышь и клавиши ----------
const KEYS = [
  ['Срезы', [['Колесо', 'приблизить'], ['Колесо над ползунком', 'листать срезы (Shift — по 5)'], ['Левая кнопка', 'перекрестие'],
    ['Средняя кнопка', 'сдвинуть изображение'], ['Правая кнопка', 'яркость и контраст КТ'], ['Двойной щелчок', 'развернуть вид · Esc — обратно']]],
  ['Коррекция скана', [['Кольцо', 'повернуть'], ['Центр', 'сдвинуть'], ['← → ↑ ↓', 'сдвиг 0,05 мм (Shift — 0,25)'],
    ['Ctrl + ← →', 'поворот 0,1° (Shift — 0,5°)'], ['Ctrl+Z / Ctrl+Y', 'отменить / вернуть']]],
  ['Кейс', [['Ctrl+S', 'сохранить (Shift — как…)'], ['Ctrl+O', 'открыть'], ['?', 'эта подсказка']]],
];

function helpDialog() {
  if ($('.modal-back.help')) return;
  const back = document.createElement('div');
  back.className = 'modal-back help';
  back.innerHTML = `<div class="modal"><h3>Мышь и клавиши</h3>${KEYS.map(([group, rows]) => `<div class="label">${group}</div>
    <div class="keys">${rows.map(([k, v]) => `<kbd>${k}</kbd><span>${v}</span>`).join('')}</div>`).join('')}
    <div class="row" style="justify-content:space-between;margin-top:14px">
    <button class="btn ghost sm" data-log title="Ошибки программы, без имён пациентов и путей">${icons.folder}Журнал ошибок</button>
    <button class="btn primary" data-x>Понятно</button></div></div>`;
  document.body.appendChild(back);
  back.addEventListener('click', (e) => {
    if (e.target.closest('[data-log]')) post('log/open').catch((err) => toast(err.message));
    else if (e.target === back || e.target.closest('[data-x]')) back.remove();
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
    const recent = shownRecent();
    const recentHtml = recent.length ? `<div class="label">Недавние кейсы</div><div class="card recent">${recent.map((r, i) =>
      `<div class="item" data-recent="${i}" ${state.incognito ? '' : `title="${r.path}"`}>${icons.folder}<span>${state.incognito ? `Кейс ${i + 1}` : r.name}</span></div>`).join('')}</div>` : '';
    return `${recentHtml}<h2>КТ</h2><p class="lead">КЛКТ: папка DICOM, архив (zip, 7z, rar, tar) или файл NIfTI, MHA, NRRD.</p>
      <button class="btn primary wide" data-a="ctdir">${icons.folder}Открыть папку DICOM</button>
      <button class="btn ghost wide" style="margin-top:6px" data-a="ct">или архив, файл…</button>
      <button class="btn ghost wide" style="margin-top:6px" data-a="scansonly">Без КТ — только сканы</button>${modelsCard()}`;
  }
  const info = `<div class="card"><div class="card-head"><h3>${state.incognito ? 'КТ пациента' : ct.name}</h3><button class="btn ghost sm" data-a="ctdir" title="Открыть другое КТ">Другое…</button></div>
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
  const ct = !!state.ct;
  const sel = scanById(state.selected);
  const cards = state.scans.map((s) => {
    const status = s.accepted ? '<span class="badge ok">принят</span>' : s.registered ? ''
      : `<span class="badge">${ct ? 'не совмещён' : 'не поставлен'}</span>`;
    const jaw = s.jaw || !ct ? `<button class="badge ${s.jaw ? '' : 'warn'}" data-jaw="${s.id}"
      title="Челюсть скана — щёлкните, чтобы сменить: верхняя → нижняя → прикус">${s.jaw ? JAWS[s.jaw] : 'челюсть?'}</button>` : '';
    const list = s.warnings || [];
    const open = state.warnOpen.has(s.id);
    const warnBtn = list.length ? `<button class="warn-btn ${open ? 'on' : ''}" data-warns="${s.id}" title="Предупреждения">${icons.warn}<b>${list.length}</b></button>` : '';
    const warnings = open ? list.map((w) => `<div class="warning">${icons.warn}<span>${hide(w)}</span></div>`).join('') : '';
    return `<div class="card ${s.id === state.selected ? 'sel' : ''}" data-select="${s.id}">
      <div class="card-head"><i class="dot" style="background:${s.color}"></i><h3 ${state.incognito ? '' : `title="${s.path}"`}>${scanLabel(s)}</h3>${jaw}${status}
        ${warnBtn}${s.registered ? opacityButton([s.id], `data-opacity="${s.id}"`) : ''}
        <button class="btn icon ghost" data-remove="${s.id}" title="Убрать">${icons.trash}</button></div>
      ${metricsHtml(s)}${warnings}
      ${s.registered ? '' : `<button class="btn wide" data-register="${s.id}">${icons.play}${ct ? 'Совместить' : 'Поставить'}</button>`}</div>`;
  }).join('');
  const correcting = !!sel && state.correcting?.id === sel.id;
  let tools = '';
  if (sel?.role === 'bite') {
    tools = '<p class="muted small">Скан прикуса стоит на сканах челюстей и двигается вместе с ними; с КТ не совмещается.</p>';
  } else if (!ct) {
    tools = ''; // без КТ корректировать и принимать не по чему
  } else if (sel?.registered && !correcting) {
    tools = `<div class="label">Положение — ${scanLabel(sel)}</div>
      <div class="row"><button class="btn grow" data-correct="${sel.id}">${icons.move}Скорректировать</button>
        <button class="btn ok grow" data-accept="${sel.id}" ${sel.accepted ? 'disabled' : ''}>${icons.check}${sel.accepted ? 'Принято' : 'Принять'}</button></div>
      <p class="muted small">Не устраивает, как сел скан, — скорректируйте: программа запомнит ваше положение и учтёт его в следующих совмещениях.</p>`;
  } else if (correcting) {
    tools = `<div class="card correcting"><div class="card-head">${icons.move}<h3>Коррекция — ${scanLabel(sel)}</h3></div>
      <p class="muted small" style="margin-top:0">На срезе: внутри кольца — сдвиг, за кольцо — поворот. Стрелки — точно, Ctrl+←/→ — поворот, Shift — крупнее.</p>
      <div class="row" style="margin-top:8px"><button class="btn grow" data-refine="${sel.id}" title="Подогнать по коронкам от текущего положения">${icons.refine}Уточнить</button>
        <button class="btn icon" data-hist="back" title="Шаг назад (Ctrl+Z)" ${state.correcting.undo.length ? '' : 'disabled'}>${icons.undo}</button>
        <button class="btn icon" data-hist="fwd" title="Шаг вперёд (Ctrl+Y)" ${state.correcting.redo.length ? '' : 'disabled'}><span style="display:flex;transform:scaleX(-1)">${icons.undo}</span></button></div>
      <div class="row" style="margin-top:8px"><button class="btn ghost" data-endcorrect>Отмена</button>
        <button class="btn ok grow" data-accept="${sel.id}">${icons.check}Сохранить поправку</button></div></div>`;
  }
  const lead = ct ? 'STL, PLY или OBJ как есть со сканера. Совмещение — по коронкам зубов, сразу после загрузки; сканы прикуса (bite, TotalJaw) ставятся на сканы челюстей и задают прикус.'
    : 'Без КТ: STL, PLY или OBJ как есть со сканера — сканы стоят в его координатах, сканы прикуса (bite, TotalJaw) ставят нижний скан в прикус. Откройте КТ — сканы совместятся с ним.';
  return `<h2>Сканы</h2><p class="lead">${lead}</p>
    ${cards}<button class="btn ${state.scans.length ? '' : 'primary'} wide" data-a="scan">${icons.plus}Добавить сканы</button>${tools}`;
}

function renderExport() {
  const hasScans = state.scans.some((s) => s.registered);
  const ready = hasScans || state.structures.length;
  const res = state.exported;
  const bite = exportBite();
  const frames = !state.ct ? '<p class="muted small">Без КТ — сканы в координатах сканера, как их открывает exocad; нижний — в прикусе по сканам прикуса.</p>'
    : hasScans ? `<div class="seg"><button data-frame="exocad" class="${state.frame === 'exocad' ? 'on' : ''}">Сканера (exocad)</button>
      <button data-frame="dicom" class="${state.frame === 'dicom' ? 'on' : ''}">КТ (DICOM)</button></div>
      ${biteChoice() ? `<div class="label">Прикус</div><div class="seg"><button data-exbite="scan" class="${bite === 'scan' ? 'on' : ''}">Сканов (врача)</button>
      <button data-exbite="ct" class="${bite === 'ct' ? 'on' : ''}">Как на КТ</button></div>` : ''}
      <p class="muted small">${EXPORT_HINTS[`${state.frame} ${bite}`]}</p>`
    : '<p class="muted small">Сканов нет — структуры КТ в координатах КТ (DICOM).</p>';
  const resultBite = res?.biteChosen ? ` · прикус ${res.biteChosen === 'scan' ? 'сканов' : 'КТ'}` : '';
  const result = res ? `<div class="card"><div class="card-head">${icons.check}<h3>Готово — ${res.files.length} ${plural(res.files.length, 'файл', 'файла', 'файлов')}${resultBite}</h3></div>
      <div class="path" ${state.incognito ? '' : `title="${res.out_dir}"`}>${hide(res.out_dir)}</div>
      <button class="btn ghost wide sm" style="margin-top:8px" data-a="openout">${icons.folder}Открыть папку</button>
      ${(res.notes || []).map((n) => `<div class="warning">${icons.warn}<span>${hide(n)}</span></div>`).join('')}</div>` : '';
  return `<h2>Экспорт</h2><p class="lead">STL всех видимых объектов в единой системе координат и case.json с матрицами.</p>
    <div class="label">Система координат</div>${frames}
    <button class="btn primary wide" data-a="export" ${ready ? '' : 'disabled'}>${icons.export}Экспортировать в папку…</button>
    ${ready ? '' : `<p class="muted small">${state.ct ? 'Сегментируйте КТ или совместите сканы.' : 'Добавьте сканы.'}</p>`}${result}`;
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
  const caseName = state.incognito ? 'Кейс' : state.caseInfo?.name || ct?.name || 'Без КТ';
  $('#caseChip').textContent = ct || state.scans.length ? `${caseName}${state.scans.length ? ` · сканов: ${state.scans.length}` : ''}` : '';
  $('#caseActions').innerHTML = `<button class="btn ghost sm" data-case="open" title="Открыть кейс (Ctrl+O)">${icons.folder}Открыть</button>
    <button class="btn ghost sm" data-case="save" title="Сохранить кейс (Ctrl+S)" ${ct || state.scans.length ? '' : 'disabled'}>${icons.check}Сохранить</button>
    <button class="btn ghost sm incognito-btn ${state.incognito ? 'on' : ''}" data-a="incognito" title="${state.incognito
      ? 'Инкогнито: имена пациентов и пути скрыты. Нажмите, чтобы показать'
      : 'Инкогнито: скрыть имена пациентов и пути — для показа программы. Системные окна выбора файлов и проводник показывают их как есть'}">${icons.incognito}${state.incognito ? 'Инкогнито' : ''}</button>
    <button class="btn ghost sm help-btn" data-a="help" title="Мышь и клавиши (?)">?</button>`;
  $('#empty3d').innerHTML = ct || state.scans.length ? '' : '<span>Откройте КТ или добавьте сканы</span>';
  const legend = $('#legend');
  legend.hidden = !(heatOn() && state.scans.some((s) => s.registered));
  legend.innerHTML = '<span><i style="background:#28aa46"></i>≤ 0.1 мм</span><span><i style="background:#e6be1e"></i>≤ 0.2 мм</span>' +
    '<span><i style="background:#d23228"></i>> 0.2 мм</span><span><i style="background:#aaa"></i>не коронки</span>';
  $('#tools3d').innerHTML = [['front', 'Спереди'], ['right', 'Справа'], ['left', 'Слева'], ['top', 'Сверху']]
    .map(([v, t]) => `<button class="btn" data-view="${v}" style="width:auto;padding:0 8px">${t}</button>`).join('') +
    (state.biteView?.available ? `<button class="btn ${state.biteView.on ? 'on' : ''}" data-bite style="width:auto;padding:0 8px"
      title="Прикус сканов: нижняя челюсть со всеми её структурами — в прикусе со сканов (врача), а не как на КТ (на КТ отличается на ${fmt(state.biteView.shift_mm, 1)} мм)">Прикус</button>` : '') +
    (ct && state.scans.some((s) => s.registered) ? `<button class="btn ${state.heat ? 'on' : ''}" data-heat title="Карта отклонений скана от КТ">${icons.heat}</button>` : '');
}

// ---------- события ----------
document.addEventListener('click', async (e) => {
  const t = e.target.closest('[data-step],[data-a],[data-select],[data-remove],[data-register],[data-refine],[data-reset],[data-accept],[data-correct],[data-revert],[data-hist],[data-endcorrect],[data-recent],[data-case],[data-opacity],[data-gopacity],[data-warns],[data-toggle],[data-groupcheck],[data-group],[data-frame],[data-exbite],[data-view],[data-heat],[data-bite],[data-jaw]');
  if (!t || t.disabled) return;
  const d = t.dataset;
  if (d.step) { state.step = d.step; slices.forEach((v) => v.draw()); return render(); }
  if (d.remove) { e.stopPropagation(); return removeScan(d.remove); }
  if (d.register) return register([d.register]);
  if (d.jaw) { e.stopPropagation(); return register([d.jaw], { jaw: NEXT_JAW[scanById(d.jaw)?.jaw] || 'upper' }); }
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
  if (d.hist) return stepHistory(d.hist === 'back');
  if (d.recent) return openCase(shownRecent()[Number(d.recent)]?.path);
  if (d.case === 'open') return openCase();
  if (d.case === 'save') return saveCase();
  if ('endcorrect' in d) { revertCorrection(); return endCorrection(); }
  if (d.toggle) return toggleStructures([d.toggle], !state.visible.has(d.toggle));
  if (d.groupcheck) {
    e.stopPropagation();
    const keys = state.structures.filter((s) => s.group === d.groupcheck).map((s) => s.key);
    return toggleStructures(keys, !keys.every((k) => state.visible.has(k)));
  }
  if (d.group) { state.groupsOpen.has(d.group) ? state.groupsOpen.delete(d.group) : state.groupsOpen.add(d.group); return render(); }
  if (d.frame) { state.frame = d.frame; return render(); }
  if (d.exbite) { state.bite = d.exbite; return render(); }
  if (d.view) return viewer.fit(d.view);
  if ('bite' in d) return toggleBite();
  if ('heat' in d) { state.heat = !state.heat; for (const s of state.scans) await showScan(s); return render(); }
  if (d.select && !e.target.closest('button')) {
    if (state.selected !== d.select) state.correcting = null;
    state.selected = d.select;
    slices.forEach((v) => v.draw());
    return render();
  }
  const action = { ct: () => openCt('ct'), ctdir: () => openCt('ctdir'), scan: addScans, segment, export: doExport,
    getmodels: downloadModels, stopmodels: () => post('models/cancel'), parts: partsDialog,
    openout: () => post('export/open').catch((err) => toast(err.message)), help: helpDialog, incognito: toggleIncognito,
    scansonly: () => { state.step = 'scans'; render(); } }[d.a];
  action?.();
});

document.addEventListener('keydown', (e) => {
  const ctrl = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (ctrl && (key === 's' || key === 'ы')) { e.preventDefault(); return saveCase(e.shiftKey); }
  if (ctrl && (key === 'o' || key === 'щ')) { e.preventDefault(); return openCase(); }
  if (e.target.closest('input,select')) return;
  if (ctrl && state.correcting && (key === 'z' || key === 'я' || key === 'y' || key === 'н')) {
    e.preventDefault();
    return stepHistory((key === 'z' || key === 'я') && !e.shiftKey);
  }
  if (e.key === 'Escape' && $('.modal-back.help')) return $('.modal-back.help').remove();
  if (e.key === '?' || (e.key === ',' && e.shiftKey && e.code === 'Slash')) return helpDialog();
  if (e.key === 'Escape' && $('.main').classList.contains('one-max')) return app.toggleMax($('.view.max'));
  SliceView.onKey(e); // стрелки — точная поправка скана в срезе под мышью
});

// ---------- старт: подхватить уже открытый кейс ----------
(async () => {
  const s = await get('state');
  state.modelsDir = s.models_dir;
  state.models = s.models;
  state.parts = s.segment_parts;
  state.biteView = s.bite_view;
  state.caseInfo = s.case;
  state.recent = s.recent || [];
  state.incognito = !!s.incognito;
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
