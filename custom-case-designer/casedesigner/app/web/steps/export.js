// Шаг «Экспорт»: выгрузка в папку и для exocad с артикулятором, панель.
import { $, busy, choose, fmt, gumAvailable, hide, html, plural, render, showScan, state, toast, viewer } from '../core.js';
import { get, mesh, run } from '../api.js';
import { icons } from '../icons.js';
import { condyleReason } from './articulation.js';

export const biteChoice = () => state.scans.some((s) => s.registered) && !!state.biteView?.available;
export const exportBite = () => (biteChoice() ? state.bite : state.frame === 'dicom' ? 'ct' : 'scan');
export const EXPORT_HINTS = {
  'exocad scan': 'Сканы — как их открывает exocad; структуры КТ — к скану своей челюсти: прикус со сканов (врача).',
  'exocad ct': 'Верхний скан — как его открывает exocad; нижний скан и все структуры — как на КТ относительно него.',
  'dicom scan': 'Верхняя челюсть — как на КТ; нижняя со своими структурами — в прикусе сканов (как в 3D с кнопкой «Прикус»).',
  'dicom ct': 'Всё стоит как на КТ; сканы — на своих челюстях.',
};

export async function doExport() {
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

export async function chooseExocadProject() {
  const paths = await choose('ctdir', 'Папка проекта exocad');
  if (!paths) return;
  state.exocadProjectPath = paths[0];
  state.exocadArticulatorResult = null;
  render();
}

// Снимки окна 3D для паспорта кейса (reports.passport): спереди и справа, без подписей с именем.
export async function captureScreens() {
  // кадр рисуется сразу: в скрытом или свёрнутом окне requestAnimationFrame не срабатывает
  const draw = () => { viewer.controls.update(); viewer.clip?.(); viewer.renderer.render(viewer.scene, viewer.camera); };
  const shots = [];
  for (const [view, caption] of [['front', 'Модели спереди'], ['right', 'Справа: шарнир и плоскость монтажа']]) {
    viewer.fit(view);
    draw();
    shots.push({ caption, png: viewer.renderer.domElement.toDataURL('image/png') });
  }
  viewer.fit('front');
  draw();
  return shots;
}

export async function doExportExocadArticulator() {
  if (!state.exocadProjectPath) return toast('Выберите папку проекта exocad');
  const c = state.articulationCondyles;
  if (!c?.available || !c.ready || !c.applied) return toast('Сначала подготовьте и примените монтаж по мыщелкам');
  const screens = await captureScreens();
  await busy('Выгрузка для exocad с артикулятором', async (progress) => {
    state.exocadArticulatorResult = await run('export/exocad-articulator',
      { project: state.exocadProjectPath, out: null, screens }, progress);
  });
}

export async function prepareExport() {
  try { state.articulationCondyles = await get('articulation/condyles'); }
  catch (e) { toast(e.message); }
  render();
  // корни зубов из КТ (roots.py) и толщина десны (gingiva.py) — считаются секунды: панель уже показана
  const jobs = [];
  if (state.structures.some((s) => s.key.startsWith('teeth/tooth_'))) {
    jobs.push(get('export/roots').then((r) => { state.roots = r; }).catch((e) => toast(e.message)));
  } else state.roots = null;
  if (gumAvailable()) jobs.push(get('gingiva').then((g) => { state.gingiva = g; }).catch((e) => toast(e.message)));
  if (jobs.length) { await Promise.all(jobs); if (state.step === 'export') render(); }
}

// ---------- толщина десны над костью и корнями: карта на сканах вместо карты отклонений ----------
export async function toggleGum() {
  await busy('Толщина десны', async () => {
    state.gumMap = !state.gumMap;
    if (state.gumMap && !state.gingiva) state.gingiva = await get('gingiva');
    for (const s of state.scans) await showScan(s);
  });
}

function gumCard() {
  const g = state.gingiva;
  const jaws = Object.entries(g?.jaws || {});
  if (!jaws.length) return '';
  const name = { upper: 'Верхняя', lower: 'Нижняя' };
  const rows = jaws.map(([jaw, s]) => {
    const thin = s.teeth.filter((t) => t.thin).map((t) => t.fdi);
    const unclear = s.teeth.filter((t) => t.thin === null).map((t) => t.fdi);
    return `<div class="aes-row"><span>${name[jaw]}: медиана</span><b>${fmt(s.median_mm, 1)} мм</b></div>
      <div class="aes-row"><span>тоньше 1 мм</span><b>${Math.round((s.shares['< 1 мм — тонкая'] || 0) * 100)}% десны</b></div>
      ${thin.length ? `<p class="small roots-list" title="Медиана толщины десны над корнем меньше 1 мм">Тонкий биотип: ${thin.join(', ')}</p>` : ''}
      ${unclear.length ? `<p class="small roots-list" title="Скан лёг глубже корня — проверьте совмещение и сегментацию">Биотип не определён: ${unclear.join(', ')}</p>` : ''}
      ${s.notes.map((n) => `<div class="warning">${icons.warn}<span>${html(n)}</span></div>`).join('')}`;
  }).join('');
  return `<div class="card gum-card"><div class="card-head">${icons.tooth}<h3>Толщина десны</h3></div>
    <p class="muted small" style="margin-top:0">Над костью и корнями по КТ — для промежуточных частей мостов, коронок на имплантах,
      биотипа. Выгружается цветным PLY рядом с картой отклонений (…_gingiva.ply), в копии для exocad — в KStomCaseDesigner.</p>
    ${rows}<button class="btn ghost wide sm" data-gum>${state.gumMap ? 'Скрыть карту десны' : 'Показать карту десны в 3D'}</button></div>`;
}

// ---------- корни зубов из КТ: показать в 3D (корни и оси обеих челюстей), сканы — полупрозрачные ----------
export const ROOT_KEYS = ['roots:upper', 'roots:lower', 'axes:upper', 'axes:lower'];
export async function toggleRoots() {
  if (state.rootsShown) {
    ROOT_KEYS.forEach((k) => viewer.remove(k));
    state.rootsShown = false;
    return render();
  }
  await busy('Корни из КТ', async () => {
    for (const key of ROOT_KEYS) {
      const [part, jaw] = key.split(':');
      try {
        const m = await mesh(`export/roots?jaw=${jaw}&part=${part}`);
        viewer.setMesh(key, m, { color: part === 'roots' ? '#f1d9a6' : '#ff8a3d', order: 3 });
      } catch { /* корней этой челюсти нет */ }
    }
    for (const s of state.scans.filter((x) => x.registered)) { // корни — под коронками: сканы прозрачнее
      state.opacity[s.id] = Math.min(state.opacity[s.id] ?? 1, 0.45);
      viewer.setOpacity(s.id, state.opacity[s.id]);
    }
    state.rootsShown = true;
  });
}

function rootsCard() {
  const r = state.roots;
  if (!r || (!r.teeth.length && !r.notes.length)) return '';
  const list = r.teeth.map((t) => `${t.fdi} — ${fmt(t.root_mm, 1)}`).join(' · ');
  return `<div class="card roots-card"><div class="card-head">${icons.tooth}<h3>Корни из КТ</h3><span class="badge">${r.teeth.length}</span></div>
    <p class="muted small" style="margin-top:0">Целый зуб в exocad: коронка — со скана, корень и ось зуба — с КТ. Выгружаются
      вместе с костями (roots_…, tooth_axes_…) и двигаются с челюстями.</p>
    ${list ? `<p class="small roots-list" title="Номер зуба — длина корня вдоль оси, мм">${list}</p>` : ''}
    ${r.notes.map((n) => `<div class="warning">${icons.warn}<span>${html(n)}</span></div>`).join('')}
    ${r.teeth.length ? `<button class="btn ghost wide sm" data-a="roots">${state.rootsShown ? 'Скрыть корни в 3D' : 'Показать корни в 3D'}</button>` : ''}</div>`;
}

export function renderExport() {
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
  const c = state.articulationCondyles;
  const condylesReady = !!c?.available && !!c.ready && !!c.applied;
  const movementCount = state.articulation?.movements?.length || 0;
  const readiness = (kind, text) => `<div class="exocad-export-readiness ${kind}"><i></i><span>${html(text)}</span></div>`;
  const projectStatus = state.exocadProjectPath ? 'Папка проекта выбрана.' : 'Выберите папку проекта exocad.';
  const condyleStatus = !c?.available ? 'Откройте КТ, чтобы расставить мыщелки.'
    : !c.ready ? `Точки мыщелков не готовы: ${condyleReason(c)}.`
      : !c.applied ? 'Выполните монтаж моделей по точкам мыщелков.' : 'Мыщелки готовы, монтаж применён.';
  const motionStatus = movementCount ? `Движения подготовлены: ${movementCount}.`
    : 'Движения ещё не построены. Выгрузка будет без траекторий.';
  const exportResult = state.exocadArticulatorResult;
  const exportRoles = Object.entries(exportResult?.files || {}).map(([role, files]) => {
    const count = Array.isArray(files) ? files.length : Number.isFinite(files) ? files : 1;
    const title = role === 'эстетический вид' ? 'Вид Smile Design' : role;
    return `<div class="exocad-export-role"><b>${html(title)}</b><span>${count} ${plural(count, 'файл', 'файла', 'файлов')}</span></div>`;
  }).join('');
  const exportNotes = (exportResult?.notes || []).map((n) => `<div class="warning">${icons.warn}<span>${html(hide(n))}</span></div>`).join('');
  // модульный артикулятор (docs/handoff/modular-articulator.md): значения пациента — в проекте, артикулятор — общий
  const pp = exportResult?.project_parameters || {};
  const sides = (key, unit) => (`${key}Right` in pp || `${key}Left` in pp
    ? `справа ${fmt(pp[`${key}Right`], 1)}${unit} · слева ${fmt(pp[`${key}Left`], 1)}${unit}` : null);
  const ins = exportResult?.inserts;
  const insertText = ins ? ['right', 'left'].filter((s) => ins[s]).map((s) => `${s === 'right' ? 'справа' : 'слева'} «${html(ins[s].insert)}», ССП ${fmt(ins[s].tilt_deg, 1)}°`).join(' · ')
    : 'стандартные (путей мыщелков по КТ нет)';
  const projectRows = [['ССП', sides('TiltCondylarGuide', '°')], ['Беннетт', sides('BennettAngle', '°')],
    ['Боковой сдвиг', sides('ImmediateSideshift', ' мм')], ['Резцовый столик', 'TiltFrontplate' in pp ? `${fmt(pp.TiltFrontplate, 0)}°` : null],
    ['Межмыщелковое', 'IntercondylarDistance' in pp ? `${fmt(pp.IntercondylarDistance, 1)} мм` : `${fmt(exportResult?.icd_mm, 1)} мм — в проект не записано`],
    ['Вставки ССП по пути мыщелка', insertText]].filter(([, v]) => v);
  const installed = exportResult && exportResult.library_installed === exportResult.library_version;
  const library = !exportResult ? '' : installed
    ? `<div class="exocad-export-readiness ok"><i></i><span>Общий артикулятор установлен в exocad (версия ${exportResult.library_version}).</span></div>`
    : `<div class="exocad-export-copy">Один раз (не на каждый кейс): скопируйте папку из <b>KStomCaseDesigner/articulator</b>
      в <b>library/articulator</b> exocad — это общий артикулятор${exportResult.library_installed ? ` (в exocad сейчас версия ${exportResult.library_installed}, нужна ${exportResult.library_version})` : ''}.</div>`;
  const exocadResult = exportResult ? `<div class="exocad-export-summary">${exportRoles}</div>
    <div class="label">В проекте exocad</div><div class="kv exocad-project-values">${projectRows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join('')}</div>
    ${library}${exportResult.files?.['паспорт кейса'] ? '<button class="btn ghost wide sm" data-a="openpassport">Открыть паспорт кейса</button>' : ''}${exportNotes}` : '';
  const exocadExport = `<div class="card exocad-export"><div class="card-head">${icons.export}<h3>Для exocad с артикулятором</h3></div>
    <p class="muted small">Создаёт копию проекта с моделями, движениями, костями и настройками артикулятора.
      При эстетическом монтаже добавляется роль «Вид Smile Design».</p>
    ${readiness(state.exocadProjectPath ? 'ok' : '', projectStatus)}
    ${readiness(condylesReady ? 'ok' : 'warn', condyleStatus)}
    ${readiness(movementCount ? 'ok' : '', motionStatus)}
    ${state.exocadProjectPath ? '<button class="btn ghost wide sm" data-a="exocad-project">Выбрать другой проект…</button>'
      : '<button class="btn wide" data-a="exocad-project">Выбрать папку проекта…</button>'}
    <button class="btn primary wide" style="margin-top:6px" data-a="exocad-articulator" ${state.exocadProjectPath && condylesReady ? '' : 'disabled'}>
      ${icons.export}Создать копию для exocad</button>${exocadResult}</div>`;
  return `<h2>Экспорт</h2><p class="lead">STL всех видимых объектов в единой системе координат и case.json с матрицами.</p>
    <div class="label">Система координат</div>${frames}
    <button class="btn primary wide" data-a="export" ${ready ? '' : 'disabled'}>${icons.export}Экспортировать в папку…</button>
    ${ready ? '' : `<p class="muted small">${state.ct ? 'Сегментируйте КТ или совместите сканы.' : 'Добавьте сканы.'}</p>`}${result}${exocadExport}${rootsCard()}${gumCard()}`;
}
