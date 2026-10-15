// Кейс и программа: настройки, сохранить и открыть кейс, подсказка по мыши и клавишам.
import { $, busy, choose, hide, render, state, toast } from '../core.js';
import { get, post, run } from '../api.js';
import { icons } from '../icons.js';
import { settledPosition } from './scans.js';

export async function settingsDialog() {
  state.articulationPlanes = await get('articulation/planes').catch((e) => { toast(e.message); return null; });
  const planes = state.articulationPlanes?.planes;
  if (!planes) return;
  const back = document.createElement('div');
  back.className = 'modal-back';
  back.innerHTML = `<div class="modal"><h3>Настройки</h3><label class="label" for="defaultMountPlane">Плоскость монтажа по умолчанию</label>
    <select id="defaultMountPlane">${Object.entries(planes).map(([key, p]) => `<option value="${key}" ${key === state.mountPlaneDefault ? 'selected' : ''}>${hide(p.name)}</option>`).join('')}</select>
    <p class="muted small">Если для кейса не хватает точек или костей лица, программа последовательно предложит доступную плоскость по цепочке замен и сообщит причину.</p>
    <div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn primary" data-x>Готово</button></div></div>`;
  document.body.appendChild(back);
  $('#defaultMountPlane', back).addEventListener('change', async (e) => {
    try {
      const saved = await post('settings/mount_plane', { plane: e.target.value });
      state.mountPlaneDefault = saved.mount_plane;
      state.articulationPlanes.default = saved.mount_plane;
      if (!state.articulationPlanes.current && !state.articulationPlanes.selected)
        state.articulationPlanes.selected = saved.mount_plane;
    } catch (err) { toast(err.message); }
  });
  back.addEventListener('click', (e) => { if (e.target.closest('[data-x]') || e.target === back) back.remove(); });
}

// ---------- кейс: сохранить и открыть ----------
export async function saveCase(as = false) {
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

export async function openCase(path) {
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
export const KEYS = [
  ['Окно 3D', [['Правая кнопка', 'вращать'], ['Левая + правая', 'сдвинуть (и нажатое колесо)'],
    ['Щелчок колесом', 'центр вращения — в точку'], ['Колесо', 'приблизить к курсору'],
    ['Двойной щелчок', 'развернуть окно · Esc — обратно']]],
  ['Срезы', [['Колесо', 'приблизить'], ['Колесо над ползунком', 'листать срезы (Shift — по 5)'], ['Левая кнопка', 'перекрестие'],
    ['Средняя кнопка', 'сдвинуть изображение'], ['Правая кнопка', 'яркость и контраст КТ'], ['Двойной щелчок', 'развернуть вид · Esc — обратно']]],
  ['Коррекция скана', [['Кольцо', 'повернуть'], ['Центр', 'сдвинуть'], ['← → ↑ ↓', 'сдвиг 0,05 мм (Shift — 0,25)'],
    ['Ctrl + ← →', 'поворот 0,1° (Shift — 0,5°)'], ['Ctrl+Z / Ctrl+Y', 'отменить / вернуть']]],
  ['Кейс', [['Ctrl+S', 'сохранить (Shift — как…)'], ['Ctrl+O', 'открыть'], ['?', 'эта подсказка']]],
];

export function helpDialog() {
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
