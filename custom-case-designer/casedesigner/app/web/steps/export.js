// Шаг «Экспорт»: выгрузка в папку и для exocad с артикулятором, панель.
import { $, busy, choose, hide, html, plural, render, state, toast } from '../core.js';
import { get, run } from '../api.js';
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

export async function doExportExocadArticulator() {
  if (!state.exocadProjectPath) return toast('Выберите папку проекта exocad');
  const c = state.articulationCondyles;
  if (!c?.available || !c.ready || !c.applied) return toast('Сначала подготовьте и примените монтаж по мыщелкам');
  await busy('Выгрузка для exocad с артикулятором', async (progress) => {
    state.exocadArticulatorResult = await run('export/exocad-articulator',
      { project: state.exocadProjectPath, out: null }, progress);
  });
}

export async function prepareExport() {
  try { state.articulationCondyles = await get('articulation/condyles'); }
  catch (e) { toast(e.message); }
  render();
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
  const exocadResult = exportResult ? `<div class="exocad-export-summary">${exportRoles}</div>
    <div class="exocad-export-copy">Скопируйте папку артикулятора из <b>KStomCaseDesigner/articulator</b>
      в <b>library/articulator</b> exocad. Файлы программы exocad приложение не меняет.</div>${exportNotes}` : '';
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
    ${ready ? '' : `<p class="muted small">${state.ct ? 'Сегментируйте КТ или совместите сканы.' : 'Добавьте сканы.'}</p>`}${result}${exocadExport}`;
}
