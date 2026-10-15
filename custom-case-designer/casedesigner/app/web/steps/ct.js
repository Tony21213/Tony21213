// Шаг «КТ»: открыть КТ, серии, сегментация и модели, панель.
import { $, app, busy, choose, chosen, fmt, hide, loadStructures, opacityButton, plural, render, showScan, shownRecent, slices, state, toast, updateScan, viewer } from '../core.js';
import { get, mesh, post, run } from '../api.js';
import { icons } from '../icons.js';
import { registerPending } from './scans.js';

// ---------- действия ----------
export const SERIES_MIN = 20; // серии короче — скауты и отдельные снимки, их не предлагаем

// Какую серию открыть, если в папке или архиве их несколько; undefined — отмена.
export function seriesDialog(list) {
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

export async function openCt(kind) {
  const paths = await choose(kind, kind === 'ctdir' ? 'Папка DICOM' : 'Файл КТ или архив');
  if (!paths) return;
  const all = await busy('Читаю КТ', () => run('ct/series', { path: paths[0] }, () => {}));
  if (!all) return;
  const list = all.filter((x) => Math.max(x.files, x.size?.[2] || 0) >= SERIES_MIN);
  const series = list.length > 1 ? await seriesDialog(list) : list[0]?.id ?? null;
  if (series === undefined) return;
  await busy('Открываю КТ', async () => {
    state.ct = await run('ct', { path: paths[0], series }, () => {});
    state.landmarks = (await get('landmarks')).landmarks || [];
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

export async function segment() {
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
export function partsDialog() {
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
export const mb = (b) => (b / 1048576).toFixed(b < 10485760 ? 1 : 0);
export const eta = (s) => (s == null ? '' : s < 60 ? `ещё ${Math.max(1, Math.round(s))} с` : `ещё ${Math.round(s / 60)} мин`);

export async function downloadModels() {
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

export function modelsCard() {
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

export function renderModels() {
  const card = $('#modelsCard');
  if (card && state.step === 'ct') card.outerHTML = modelsCard();
  else if (state.step === 'ct') render();
}

// ---------- входной контроль кейса (casedesigner/intake.py) ----------
export async function intakeCheck() {
  await busy('Входной контроль', async (progress) => {
    state.intake = await run('intake', {}, progress);
  });
}

export async function intakeSave() {
  const paths = await choose('out', 'Папка для отчёта входного контроля');
  if (!paths) return;
  const r = await post('intake/save', { out_dir: paths[0] }).catch((e) => toast(e.message));
  if (!r) return;
  state.intakeSaved = r;
  render();
}

export function intakeCard() {
  if (!state.ct && !state.scans.length) return '';
  const res = state.intake;
  const head = `<div class="card intake"><div class="card-head">${icons.check}<h3>Входной контроль</h3></div>`;
  if (!res) {
    return `${head}<p class="muted small" style="margin-top:0">Чего не хватает в кейсе: поле и воксель КТ, дыры в сканах,
      скан прикуса, фото — со снимками, чтобы сразу отправить клинике.</p>
      <button class="btn wide" data-a="intake">${icons.check}Проверить кейс</button></div>`;
  }
  const missing = res.findings.filter((f) => f.level !== 'ok');
  const mark = { error: 'нет / плохо', warn: 'проверить' };
  const list = missing.length ? missing.map((f) => `<div class="warning ${f.level}">${icons.warn}<span><b>${hide(f.section)} · ${hide(f.title)}</b>: ${hide(f.text)}
      <i class="muted"> — ${mark[f.level]}${f.image ? ', снимок в отчёте' : ''}</i></span></div>`).join('')
    : '<p class="muted small">Всего хватает: КТ, сканы, прикус и фото в порядке.</p>';
  const saved = state.intakeSaved ? `<div class="path" ${state.incognito ? '' : `title="${state.intakeSaved.out_dir}"`}>${hide(state.intakeSaved.out_dir)}</div>
      <button class="btn ghost wide sm" data-a="intakeopen">Открыть отчёт</button>` : '';
  return `${head}${list}
    <div class="row" style="margin-top:8px"><button class="btn ghost" data-a="intake">Проверить ещё раз</button>
      <button class="btn grow" data-a="intakesave">${icons.folder}Сохранить отчёт…</button></div>${saved}</div>`;
}

export function renderCt() {
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
  const jointNotes = (state.landmarks || []).filter((p) => ['Co_R', 'Co_L'].includes(p.key) && p.note)
    .map((p) => `<div class="warning landmark-note"><i class="dot" style="background:${p.key === 'Co_R' ? '#ff6b6b' : '#6ba8ff'}"></i><span>${hide(p.name)}: ${hide(p.note)}</span></div>`).join('');
  if (!state.structures.length) {
    const ready = state.models?.ready && state.parts?.selected.length;
    const chosen = (state.parts?.parts || []).filter((p) => state.parts.selected.includes(p.id)).map((p) => p.title);
    return `<h2>КТ</h2>${info}${jointNotes}
      <button class="btn primary wide" data-a="segment" ${ready ? '' : 'disabled'}>${icons.play}Сегментировать</button>${partsButton}
      <p class="muted small">${!state.models?.ready ? 'Сначала скачайте модели сегментации.' : chosen.length ? `${chosen.join(', ')}.` : 'Не выбрано, что сегментировать.'}</p>${modelsCard()}${intakeCard()}`;
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
  return `<h2>КТ</h2>${info}${jointNotes}<div class="label">Структуры — ${state.structures.length}; видимые попадут в экспорт</div>
    <div class="card tree">${tree}</div>${partsButton}${intakeCard()}`;
}
