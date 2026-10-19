// Шаг «Ориентиры»: точки КТ (поиск ALI-CBCT, постановка щелчком по срезу, подтверждение) и цефалометрия — анализы
// Steiner и Ricketts по точкам кейса, отчёт (сервер — app/steps/landmarks.py, cephalometry.py).
import { busy, choose, fmt, hide, html, render, slices, state, toast, viewer, app } from '../core.js';
import { get, post, run } from '../api.js';
import { icons } from '../icons.js';

// Группы точек по порядку постановки: «Поставить» после щелчка переходит к следующей пустой точке группы.
const GROUPS = [
  ['Основание черепа', ['N', 'S', 'Ba', 'Po_R', 'Po_L', 'Or_R', 'Or_L']],
  ['Верхняя челюсть', ['ANS', 'PNS', 'A']],
  ['Нижняя челюсть', ['B', 'Pog', 'Gn', 'Me', 'Go_R', 'Go_L']],
  ['Резцы', ['U1_tip', 'U1_apex', 'L1_tip', 'L1_apex']],
  ['Суставы и артикулятор', ['Co_R', 'Co_L', 'HN_R', 'HN_L', 'IP', 'Tr_R', 'Tr_L', 'Al_R', 'Al_L']],
];
const SOURCE = { user: 'врач', 'ali-cbct': 'ALI-CBCT', mandible: 'по нижней челюсти', segmentation: 'по сегментации' };
const COLOR = { none: '#5b6170', suggested: '#f5b84b', user: '#3ecf8e' };

const lm = { sel: null, placing: null, tab: 'steiner', saved: null, models: null };

const points = () => state.landmarkInfo?.landmarks || [];
const byKey = (key) => points().find((p) => p.key === key);

export async function prepareLandmarks() {
  await refresh();
  if (!lm.models) get('landmarks/models').then((m) => { lm.models = m; render(); }).catch(() => {});
}

async function refresh(info) {
  try {
    state.landmarkInfo = info || await get('landmarks');
    state.landmarks = state.landmarkInfo.landmarks;
    state.ceph = await get('cephalometry');
  } catch (e) { toast(e.message); }
  slices.forEach((v) => v.draw());
  drawMarkers();
  render();
}

// ---------- срезы и 3D ----------
export function landmarkPoints() {
  if (state.step !== 'landmarks') return [];
  return points().filter((p) => p.point).map((p) => ({ key: p.key, point: p.point, suggested: p.suggested, selected: p.key === lm.sel }));
}

// Щелчок левой по срезу в режиме «Поставить»: точка — сюда, дальше — следующая пустая точка группы.
export function placeLandmark(point) {
  if (state.step !== 'landmarks' || !lm.placing) return false;
  const key = lm.placing;
  post('landmarks', { key, point }).then((info) => {
    lm.sel = key;
    const group = GROUPS.find(([, keys]) => keys.includes(key))?.[1] || [];
    lm.placing = group.slice(group.indexOf(key) + 1).find((k) => !info.landmarks.find((p) => p.key === k)?.point) || null;
    return refresh(info);
  }).catch((e) => toast(e.message));
  return true;
}

function drawMarkers() {
  for (const key of [...viewer.objects.keys()]) if (key.startsWith('lm:ld:')) viewer.remove(key);
  if (state.step !== 'landmarks') return;
  for (const p of points()) {
    if (p.point) viewer.setMarker(`ld:${p.key}`, p.point, p.key === lm.sel ? '#ffffff' : p.suggested ? COLOR.suggested : COLOR.user);
  }
}

export function endLandmarks() {
  lm.placing = null;
  for (const key of [...viewer.objects.keys()]) if (key.startsWith('lm:ld:')) viewer.remove(key);
  slices.forEach((v) => v.draw());
}

// ---------- действия ----------
export async function landmarkAction(action, key) {
  const p = key ? byKey(key) : null;
  if (action === 'go') {
    lm.sel = key;
    if (p?.point) app.setCursor([...p.point]);
    drawMarkers();
    return render();
  }
  if (action === 'place') {
    lm.placing = lm.placing === key ? null : key;
    lm.sel = key;
    return render();
  }
  if (action === 'confirm' && p?.point) return refresh(await post('landmarks', { key, point: p.point }).catch((e) => toast(e.message)));
  if (action === 'delete') return refresh(await post('landmarks', { key, point: null }).catch((e) => toast(e.message)));
  if (action === 'tab') { lm.tab = key; return render(); }
  if (action === 'auto') {
    return busy('Ищу ориентиры (ALI-CBCT)', async (progress) => {
      const info = await run('landmarks/auto', {}, progress);
      await refresh(info);
    });
  }
  if (action === 'suggest') return refresh(await post('landmarks/suggest').catch((e) => toast(e.message)));
  if (action === 'models') {
    return busy('Загрузка моделей ориентиров', async (progress) => {
      lm.models = await run('landmarks/models/download', {}, progress);
    });
  }
  if (action === 'cephsave') {
    const paths = await choose('out', 'Папка для отчёта цефалометрии');
    if (!paths) return;
    lm.saved = await post('cephalometry/save', { out_dir: paths[0] }).catch((e) => toast(e.message));
    return render();
  }
  if (action === 'cephopen') return post('cephalometry/open').catch((e) => toast(e.message));
}

// ---------- панель ----------
function row(key) {
  const p = byKey(key);
  if (!p) return '';
  const status = !p.point ? 'none' : p.suggested ? 'suggested' : 'user';
  const src = p.point ? (p.suggested ? `${SOURCE[p.source] || 'предложено'} · проверьте` : SOURCE[p.source] || 'врач') : 'нет';
  const placing = lm.placing === key;
  return `<div class="lm-row ${key === lm.sel ? 'sel' : ''} ${placing ? 'placing' : ''}">
    <i class="dot" style="background:${COLOR[status]}"></i>
    <button class="lm-name" data-lm="go" data-key="${key}" title="${html(p.hint)}${p.point ? ' — перейти на срезах' : ''}">${html(p.name)}</button>
    <span class="lm-src">${src}</span>
    <button class="btn icon ghost lm-btn ${placing ? 'on' : ''}" data-lm="place" data-key="${key}" title="${placing ? 'Отменить постановку' : 'Поставить: щелчок левой по срезу'}">${icons.target}</button>
    ${p.suggested ? `<button class="btn icon ghost lm-btn" data-lm="confirm" data-key="${key}" title="Подтвердить">${icons.check}</button>` : ''}
    ${p.point ? `<button class="btn icon ghost lm-btn" data-lm="delete" data-key="${key}" title="Убрать точку">${icons.trash}</button>` : ''}
    ${p.note ? `<div class="lm-note">${hide(p.note)}</div>` : ''}</div>`;
}

function cephCard() {
  const c = state.ceph;
  if (!c) return '';
  const head = `<div class="card ceph-card"><div class="card-head">${icons.data}<h3>Цефалометрия</h3></div>`;
  if (c.error) {
    const why = c.error.replace(/^срединная плоскость:\s*/, '');
    return `${head}<p class="muted small" style="margin-top:0">Нет срединной плоскости: ${html(why)} — например N, S, ANS, PNS
      или пары Po, Or.</p></div>`;
  }
  const tabs = `<div class="seg" style="margin-bottom:8px">${Object.entries(c.analyses).map(([k, a]) => `<button class="${k === lm.tab ? 'on' : ''}" data-lm="tab" data-key="${k}">${html(a.name)}</button>`).join('')}</div>`;
  const a = c.analyses[lm.tab] || Object.values(c.analyses)[0];
  const rows = a.results.map((r) => {
    const level = Math.abs(r.z) <= 1 ? 'ok' : Math.abs(r.z) <= 2 ? 'warn' : 'bad';
    return `<div class="ceph-row"><span class="ceph-name">${html(r.name)}</span><b class="${level}">${fmt(r.value, 1)}${r.unit}</b>
      <span class="ceph-norm">${fmt(r.norm, 1)} ± ${fmt(r.sd, 1)}</span><span class="ceph-text ${level}">${html(r.text)}</span></div>`;
  }).join('');
  const skipped = a.skipped.map((s) => `<div class="ceph-skip">${html(s.name)}: ${html(s.reason || `нет точек ${s.missing.join(', ')}`)}</div>`).join('');
  const saved = lm.saved ? `<button class="btn ghost wide sm" data-lm="cephopen">Открыть отчёт</button>` : '';
  return `${head}${tabs}${rows || '<p class="muted small">Нет ни одного измерения — поставьте точки.</p>'}
    ${skipped ? `<div class="label">Не посчитано</div>${skipped}` : ''}
    ${a.note ? `<p class="muted small">${html(a.note)}</p>` : ''}
    <p class="muted small">В проекции на срединную плоскость, как на боковой ТРГ.</p>
    <button class="btn wide" data-lm="cephsave">${icons.folder}Сохранить отчёт (PDF, PNG)…</button>${saved}</div>`;
}

export function renderLandmarks() {
  queueMicrotask(drawMarkers);
  const lead = `<h2>Ориентиры</h2><p class="lead">Точки КТ для плоскостей монтажа и цефалометрии. Найдите их нейросетью или поставьте
    щелчком по срезу; найденные программой — жёлтые, проверьте и подтвердите.</p>`;
  if (!state.ct) return `${lead}<p class="muted small">Откройте КТ.</p>`;
  if (!state.landmarkInfo) return `${lead}<p class="muted small">Загружаю…</p>`;
  const m = lm.models;
  const auto = m && !m.usable
    ? `<button class="btn wide" data-lm="models">${icons.plus}Скачать модели ориентиров</button>`
    : `<button class="btn primary wide" data-lm="auto">${icons.play}Найти ориентиры (ALI-CBCT)</button>`;
  const suggest = state.structures.some((s) => s.key === 'mandible')
    ? '<button class="btn ghost wide sm" style="margin-top:6px" data-lm="suggest">Мыщелки и порионы — по сегментации</button>' : '';
  const placing = lm.placing ? `<div class="warning">${icons.target}<span>Ставлю: <b>${html(byKey(lm.placing)?.name || lm.placing)}</b> — щелчок левой по срезу (Esc — отмена).</span></div>` : '';
  const groups = GROUPS.map(([title, keys]) => {
    const set = keys.filter((k) => byKey(k)?.point).length;
    return `<div class="label">${title} <span class="muted">${set}/${keys.length}</span></div><div class="lm-list">${keys.map(row).join('')}</div>`;
  }).join('');
  return `${lead}${auto}${suggest}${placing}${cephCard()}${groups}`;
}

export const landmarkPlacing = () => !!lm.placing;
export function cancelPlacing() { lm.placing = null; render(); }
