// KStom Case Designer — общее для шагов: состояние, окна КТ и 3D, помощники, инкогнито, render.
import { get, mesh, post } from './api.js';
import { icons } from './icons.js';
import { SliceView } from './slices.js';
import { Viewer3D } from './viewer3d.js';
import { moveScan } from './steps/scans.js';

// KStom Case Designer, первая версия: КТ и сегментация, совмещение сканов с КТ, экспорт в единых координатах.

export const STEPS = [
  { id: 'ct', title: 'КТ', icon: 'ct' },
  { id: 'scans', title: 'Сканы', icon: 'tooth' },
  { id: 'aesthetics', title: 'Эстетика', icon: 'eye' },
  { id: 'articulation', title: 'Артикулятор', icon: 'move' },
  { id: 'export', title: 'Экспорт', icon: 'export' },
];
// Что видно сразу после сегментации: отдельные зубы, кости, каналы, пазухи.
export const HIDDEN_BY_DEFAULT = ['upper_teeth', 'lower_teeth', 'skull', 'nasopharynx', 'oropharynx', 'hypopharynx', 'hard_palate'];
export const TRANSLUCENT = { mandible: 0.42, maxilla: 0.42, skull: 0.3, maxillary_sinus: 0.55, frontal_sinus: 0.55,
  nasal_cavity: 0.45, pharynx: 0.4, nasopharynx: 0.4, oropharynx: 0.4, hypopharynx: 0.4, soft_palate: 0.7, hard_palate: 0.5 };

export const state = {
  scanColors: true, // цвет скана со сканера в окне 3D (переключатель «Цвет»)
  step: 'ct', ct: null, scans: [], structures: [], visible: new Set(), heat: true, selected: null,
  modelsDir: null, frame: 'exocad', bite: 'scan', exported: null, busy: false, groupsOpen: new Set(['Зубы']),
  exocadProjectPath: null, exocadArticulatorResult: null,
  aesthetics: null, // шаг «Эстетика»: фото, пары, камеры (GET aesthetics)
  models: null, download: null, downloadError: null, articulation: null, articulationAnalysis: null,
  articulationContacts: [], articulationGuides: null, articulationCondyles: null,
  articulationPlanes: null, mountPlaneDefault: 'aesthetic_hinge',
  jointParts: null, jointsVisible: false, landmarks: [],
  condyleEditing: null, // right / left: точка ставится на кости в 3D и двигается стрелками
  opacity: {}, // объект → прозрачность, заданная кнопкой (иначе — по умолчанию)
  warnOpen: new Set(), // карточки сканов с раскрытым списком предупреждений
  moving: new Map(), // скан → номер последней ручной поправки, ещё не оценённой сервером
  correcting: null, // { id, start, undo, redo } — режим коррекции: манипулятор на срезах, start — положение до него
  windowName: 'auto', // набор окна КТ ('' — подобрано вручную)
  caseInfo: null, recent: [], // сохранённый кейс и недавние кейсы
  incognito: false, // не показывать имён пациентов и путей (для показа программы)
};
// Ошибки интерфейса — в журнал программы (пути и имена сервер вычищает).
export const reportError = (message, stack) => post('log', { message: String(message || ''), stack: String(stack || '') }).catch(() => {});
window.addEventListener('error', (e) => reportError(e.message, e.error?.stack));
window.addEventListener('unhandledrejection', (e) => reportError(e.reason?.message || e.reason, e.reason?.stack));
export let overlayVersion = 0; // меняется, когда сервер принял новое положение скана: контуры — заново
export function bumpOverlay() { overlayVersion += 1; } // из шагов: импорт присваивать нельзя

export const app = {
  cursor: [0, 0, 0],
  window: [400, 3000],
  setCursor(p) {
    this.cursor = p;
    slices.forEach((s) => s.refresh());
  },
  visibleKeys: () => [...state.scans.filter((s) => s.transform && !viewer.userHidden.has(s.id)).map((s) => s.id), ...state.visible],
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

export const $ = (sel, root = document) => root.querySelector(sel);

export function windowPresets() {
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
export const panel = $('#panel');
export const viewer = new Viewer3D($('#view3d'));
export const slices = [...document.querySelectorAll('.view[data-axis]')].map((el) => new SliceView(el, el.dataset.axis, app));
window.ccd = { app, slices, viewer, state }; // для проверки интерфейса в браузере
$('#view3d').addEventListener('dblclick', (e) => {
  if (!state.condyleEditing && !e.target.closest('.objbar')) app.toggleMax($('#view3d'));
}); // развернуть 3D (Esc — обратно)

// ---------- общие помощники ----------
export function toast(text, kind = 'error') {
  const t = document.createElement('div');
  t.className = `toast ${kind === 'info' || kind === 'learn' ? 'info' : ''}`;
  t.textContent = hide(text);
  document.body.appendChild(t);
  setTimeout(() => t.remove(), kind === 'info' ? 3500 : kind === 'learn' ? 8000 : 6000);
}

export async function busy(title, fn) {
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
export async function choose(kind, title) {
  const paths = await choosePaths(kind, title);
  if (paths) chosen.push(...paths); // в инкогнито скрываются и они (архив, который не открылся, и т. п.)
  return paths;
}

export async function choosePaths(kind, title) {
  if (typeof window.pywebview?.api?.choose === 'function') {
    try {
      const paths = await window.pywebview.api.choose(kind);
      return paths?.length ? paths : null;
    } catch (err) {
      reportError(err.message, err.stack);
      toast('Не удалось открыть диалог. Укажите путь в появившемся поле.');
    }
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

export const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
export const plural = (n, one, few, many) => (n % 10 === 1 && n % 100 !== 11 ? one
  : n % 10 >= 2 && n % 10 <= 4 && (n % 100 < 12 || n % 100 > 14) ? few : many);
export const cls = (v, good, fair) => (v <= good ? 'ok' : v <= fair ? 'warn' : 'bad');
export const scanById = (id) => state.scans.find((s) => s.id === id);
export const JAWS = { upper: 'верхняя', lower: 'нижняя', bite: 'прикус' };
export const NEXT_JAW = { upper: 'lower', lower: 'bite', bite: 'upper' }; // значок челюсти у скана — по кругу
export const heatOn = (info) => state.heat && !!state.ct && (!info || info.registered); // отклонение от КТ — только с КТ
export const shownRecent = () => state.recent.filter((r) => r.exists);

// ---------- инкогнито: показ программы без данных пациента ----------
// Имена пациентов бывают в путях и именах файлов (папка КТ, сканы из exocad, файл кейса, проект exocad), а из
// них — в названиях сканов и в сообщениях. В инкогнито на экране вместо них метки «КТ», «Скан 1», «Кейс»,
// вместо остальных путей — «…». Сами данные не меняются. Так же, как в журнале ошибок (errorlog.py).
export const WIN_PATH = /(?:[A-Za-z]:[\\/]|\\\\)[^:;"'<>|\r\n\t*?«»]*/g; // в именах файлов Windows нет «:» — путь до неё
export const OWN_FOLDERS = /^(?:KStomCaseDesigner|CustomCaseDesigner)$/i; // папки программы — не личные
export const chosen = []; // пути, выбранные в этом сеансе
export const scanLabel = (s) => (state.incognito ? `Скан ${state.scans.findIndex((x) => x.id === s.id) + 1}` : s.name);

export function secrets() {
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
export function hide(text) {
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

export function html(text) {
  return String(text ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

export async function toggleIncognito() {
  const r = await post('settings/incognito', { on: !state.incognito }).catch((e) => toast(e.message));
  if (!r) return;
  state.incognito = r.incognito;
  render();
}

export function updateScan(info) {
  const i = state.scans.findIndex((s) => s.id === info.id);
  if (i >= 0) state.scans[i] = info; else state.scans.push(info);
}

// Цвета скана в окне 3D: карта отклонений, если включена; иначе — цвет со сканера (переключатель «Цвет»), иначе ровный.
export async function scanRgb(info) {
  if (heatOn(info)) return new Uint8Array(await get(`scans/${info.id}/colors`));
  if (state.scanColors && info.has_colors) return new Uint8Array(await get(`scans/${info.id}/texture`));
  return null;
}
export async function showScan(info) {
  if (!viewer.objects.has(info.id)) viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, null);
  if (info.transform) viewer.setTransform(info.id, info.transform);
  viewer.setColors(info.id, await scanRgb(info));
}

export async function loadStructures(list) {
  for (const s of state.structures) viewer.remove(s.key);
  state.structures = list;
  state.visible = new Set(list.map((s) => s.key).filter((k) => !k.startsWith('pulp/') && !HIDDEN_BY_DEFAULT.includes(k)));
  await Promise.all(list.map(async (s) => {
    viewer.setMesh(s.key, await mesh(`structures/${encodeURIComponent(s.key).replace(/%2F/g, '/')}/mesh`),
      { color: s.color, opacity: TRANSLUCENT[s.key] ?? 1, order: TRANSLUCENT[s.key] ? 1 : 0 });
    viewer.setVisible(s.key, state.visible.has(s.key));
  }));
  viewer.setVisible('ct_teeth', !list.length);
}

export const defaultOpacity = (key) => (scanById(key) ? 1 : TRANSLUCENT[key] ?? 1);
export const opacityOf = (key) => state.opacity[key] ?? defaultOpacity(key);
export function toggleOpacity(keys) {
  const next = keys.every((k) => opacityOf(k) < 1) ? 1 : 0.35;
  for (const k of keys) {
    state.opacity[k] = next;
    viewer.setOpacity(k, next);
  }
  render();
}

export function toggleStructures(keys, on) {
  for (const k of keys) {
    if (on) state.visible.add(k); else state.visible.delete(k);
    viewer.setVisible(k, on);
  }
  app.setCursor(app.cursor);
  render();
}

// Прикус выгрузки: выбирается, когда прикус сканов отличается от прикуса на КТ (как кнопка «Прикус» в 3D).
// ---------- панели ----------
export function metricsHtml(s) {
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
export function opacityButton(keys, attr) {
  const clear = keys.every((k) => opacityOf(k) < 1);
  return `<button class="opacity-btn ${clear ? 'clear' : ''}" ${attr} title="${clear ? 'Сделать непрозрачным' : 'Сделать прозрачным'}"><i></i></button>`;
}

export const RENDER = {}; // панели шагов по id шага — заполняет app.js

export function render() {
  const done = {
    ct: state.structures.length > 0,
    scans: state.scans.length > 0 && state.scans.every((s) => s.registered),
    aesthetics: !!state.aesthetics?.mount_kind,
    articulation: !!state.articulation?.mounting,
    export: !!state.exported,
  };
  $('#rail').innerHTML = STEPS.map((s) => `<button class="step ${s.id === state.step ? 'active' : ''} ${done[s.id] ? 'done' : ''}" data-step="${s.id}">
    ${icons[s.icon]}<span>${s.title}</span><i class="dot"></i></button>`).join('');
  $('.main').classList.toggle('photo-mode', state.step === 'aesthetics'); // фото — вместо срезов КТ
  panel.innerHTML = RENDER[state.step]();
  RENDER.objectBar?.(); // панель объектов окна 3D — на всех шагах
  const ct = state.ct;
  const caseName = state.incognito ? 'Кейс' : state.caseInfo?.name || ct?.name || 'Без КТ';
  $('#caseChip').textContent = ct || state.scans.length ? `${caseName}${state.scans.length ? ` · сканов: ${state.scans.length}` : ''}` : '';
  $('#caseActions').innerHTML = `<button class="btn ghost sm" data-case="open" title="Открыть кейс (Ctrl+O)">${icons.folder}Открыть</button>
    <button class="btn ghost sm" data-case="save" title="Сохранить кейс (Ctrl+S)" ${ct || state.scans.length ? '' : 'disabled'}>${icons.check}Сохранить</button>
    <button class="btn ghost sm" data-a="settings" title="Настройки программы">Настройки</button>
    <button class="btn ghost sm incognito-btn ${state.incognito ? 'on' : ''}" data-a="incognito" title="${state.incognito
      ? 'Инкогнито: имена пациентов и пути скрыты. Нажмите, чтобы показать'
      : 'Инкогнито: скрыть имена пациентов и пути — для показа программы. Системные окна выбора файлов и проводник показывают их как есть'}">${icons.incognito}${state.incognito ? 'Инкогнито' : ''}</button>
    <button class="btn ghost sm help-btn" data-a="help" title="Мышь и клавиши (?)">?</button>`;
  $('#empty3d').innerHTML = ct || state.scans.length ? '' : '<span>Откройте КТ или добавьте сканы</span>';
  $('#view3d').classList.toggle('picking', state.step === 'articulation' && !!state.condyleEditing);
  const legend = $('#legend');
  legend.hidden = !(heatOn() && state.scans.some((s) => s.registered));
  legend.innerHTML = '<span><i style="background:#28aa46"></i>≤ 0.1 мм</span><span><i style="background:#e6be1e"></i>≤ 0.2 мм</span>' +
    '<span><i style="background:#d23228"></i>> 0.2 мм</span><span><i style="background:#aaa"></i>не коронки</span>';
  $('#tools3d').innerHTML = [['front', 'Спереди'], ['right', 'Справа'], ['left', 'Слева'], ['top', 'Сверху']]
    .map(([v, t]) => `<button class="btn" data-view="${v}" style="width:auto;padding:0 8px">${t}</button>`).join('') +
    (state.biteView?.available ? `<button class="btn ${state.biteView.on ? 'on' : ''}" data-bite style="width:auto;padding:0 8px"
      title="Прикус сканов: нижняя челюсть со всеми её структурами — в прикусе со сканов (врача), а не как на КТ (на КТ отличается на ${fmt(state.biteView.shift_mm, 1)} мм)">Прикус</button>` : '') +
    (state.scans.some((s) => s.has_colors) ? `<button class="btn ${state.scanColors ? 'on' : ''}" data-scancolor style="width:auto;padding:0 8px"
      title="Цвет скана со сканера (карта отклонений, если включена, — поверх)">Цвет</button>` : '') +
    (ct && state.scans.some((s) => s.registered) ? `<button class="btn ${state.heat ? 'on' : ''}" data-heat title="Карта отклонений скана от КТ">${icons.heat}</button>` : '');
}
