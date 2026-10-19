// Шаг «Артикулятор»: монтаж, плоскость, движения, мыщелки, суставы, панель.
import { $, app, bumpOverlay, busy, fmt, hide, render, state, toast, viewer } from '../core.js';
import { get, mesh, post, run } from '../api.js';
import { icons } from '../icons.js';
import { refreshScans, reloadStructureMeshes } from './scans.js';

export let viewPointerDown = null;
export function rememberPointerDown(e) { viewPointerDown = [e.clientX, e.clientY]; } // окно 3D: где нажали — слушатель в app.js
export async function prepareArticulation() {
  try {
    if (state.biteView?.available && !state.biteView.on) {
      state.biteView = await post('bite_view', { on: true });
      await reloadStructureMeshes();
      await refreshScans(true);
      bumpOverlay();
    }
  } catch (e) { toast(e.message); }
  try {
    state.articulation = await post('articulation/prepare');
  } catch (e) { toast(e.message); }
  try {
    state.articulationPlanes = await get('articulation/planes');
    state.articulationCondyles = await get('articulation/condyles');
    state.articulationGuides = await get('articulation/guides');
    state.landmarks = (await get('landmarks')).landmarks || [];
    state.jointParts = await get('articulation/joint');
    drawArticulationGuides();
    await drawJointParts();
  } catch (e) { toast(e.message); }
  await loadRehab();
  render();
}

export async function articulationAction(kind, body = {}, title = 'Артикулятор') {
  await busy(title, async (progress) => {
    state.articulation = await run(`articulation/${kind}`, body, progress);
    state.articulationPlanes = await get('articulation/planes');
    state.articulationCondyles = await get('articulation/condyles');
    state.articulationGuides = await get('articulation/guides');
    drawArticulationGuides();
    await refreshJointParts();
    await loadRehab(); // система монтажа сменилась — подъём и шаблон заново
  });
}

export async function selectMountPlane(plane) {
  try {
    state.articulationPlanes = await post('articulation/planes', { plane });
    render();
  } catch (e) { toast(e.message); }
}

export async function remountPlane() {
  await busy('Перемонтаж по выбранной плоскости', async (progress) => {
    state.articulation = await run('articulation/mount', { method: 'ct' }, progress);
    state.articulation = await run('articulation/generate', { travel: 6.0 }, progress);
    state.articulationPlanes = await get('articulation/planes');
    state.articulationCondyles = await get('articulation/condyles');
    state.articulationGuides = await get('articulation/guides');
    drawArticulationGuides();
    await refreshJointParts();
  });
}

export async function mountWithPlane(method) {
  if (method === 'ct' && state.articulation?.mounting) return remountPlane();
  return articulationAction('mount', { method }, 'Гипсовка моделей');
}

export function clearArticulationGuides() {
  for (const key of [...viewer.objects.keys()].filter((k) => String(k).startsWith('art:') || String(k).startsWith('lm:art:')))
    viewer.remove(key);
}

export async function drawJointParts() {
  const keys = [...viewer.objects.keys()].filter((k) => String(k).startsWith('joint:'));
  if (!state.jointsVisible || !state.jointParts) {
    keys.forEach((key) => viewer.setVisible(key, false));
    return;
  }
  const requests = [];
  for (const side of ['right', 'left']) for (const part of ['head', 'slope']) {
    if (!state.jointParts[side]?.[part]) continue;
    const key = `joint:${side}:${part}`;
    if (viewer.objects.has(key)) { viewer.setVisible(key, true); continue; }
    requests.push(mesh(`articulation/joint?side=${side}&part=${part}`).then((data) => {
      if (state.jointsVisible) viewer.setMesh(key, data,
        { color: part === 'head' ? '#48d597' : '#f4c34e', opacity: 0.94, order: 3 });
    }).catch((e) => toast(e.message)));
  }
  await Promise.all(requests);
}

export async function refreshJointParts() {
  for (const key of [...viewer.objects.keys()].filter((k) => String(k).startsWith('joint:'))) viewer.remove(key);
  state.jointParts = await get('articulation/joint');
  await drawJointParts();
}

export async function toggleJoints() {
  state.jointsVisible = !state.jointsVisible;
  render();
  await drawJointParts();
}

export function condyleSpace() {
  if (state.condyleEditing && !state.articulationCondyles?.segmentation.mandible) return 'ct';
  return state.biteView?.on ? 'view' : 'ct';
}

export function endCondyleEdit() {
  state.condyleEditing = null;
  viewer.remove('ct_bone');
  drawArticulationGuides();
}

export function drawArticulationGuides() {
  clearArticulationGuides();
  const g = state.articulationGuides;
  const current = state.articulationCondyles?.points || {};
  const p = g?.points || {};
  const space = condyleSpace();
  const right = current.right?.[space] || (!state.articulationCondyles?.available ? p.condyle_right : null);
  const left = current.left?.[space] || (!state.articulationCondyles?.available ? p.condyle_left : null);
  if (right) viewer.setMarker('art:condyle_right', right, '#ff6b6b');
  if (left) viewer.setMarker('art:condyle_left', left, '#6ba8ff');
  if (p.incisal) viewer.setMarker('art:incisal', p.incisal, '#f5d76e');
  if (right && left) viewer.setPath('art:hinge', [right, left], '#ffffff');
  if ((space === 'ct' && state.articulationCondyles?.bite === 'scan') || !g?.available
      || (state.articulationCondyles?.available && !state.articulationCondyles.applied)) return;
  const colors = { condyle_right: '#ff6b6b', condyle_left: '#6ba8ff', incisal: '#f5d76e' };
  for (const path of g.paths || []) for (const key of Object.keys(colors))
    viewer.setPath(`art:path:${path.id}:${key}`, path[key], colors[key]);
}

export async function updateCondyles(path = '', body = {}) {
  try {
    state.articulationCondyles = await post(`articulation/condyles${path}`, body);
    for (const [side, key] of [['right', 'Co_R'], ['left', 'Co_L']]) {
      const point = state.articulationCondyles.points?.[side];
      const landmark = (state.landmarks || []).find((item) => item.key === key);
      if (landmark && point) Object.assign(landmark, { point: point.ct, source: point.source, note: point.note });
    }
    drawArticulationGuides();
    if (state.jointsVisible) await refreshJointParts();
    render();
    return true;
  } catch (e) {
    toast(e.message);
    return false;
  }
}

export async function editCondyle(side) {
  if (state.condyleEditing === side) {
    endCondyleEdit();
    return render();
  }
  try {
    const target = state.articulationCondyles?.segmentation.mandible ? 'mandible' : 'ct_bone';
    if (target === 'ct_bone' && !viewer.objects.has(target)) {
      const surface = await mesh('ct/surface?part=bone');
      if (state.step !== 'articulation') return;
      viewer.setMesh(target, surface, { color: '#cfc5b5' });
    }
    viewer.setVisible(target, true);
    if (target === 'mandible') state.visible.add('mandible');
    state.condyleEditing = side;
    drawArticulationGuides();
  } catch (e) { toast(e.message); }
  render();
}

export async function handleCondylePick(e) {
  const side = state.condyleEditing;
  const started = viewPointerDown;
  viewPointerDown = null;
  if (!side || e.button !== 0 || state.step !== 'articulation' || !started) return;
  if (e.target !== viewer.renderer.domElement) return;
  if (Math.hypot(e.clientX - started[0], e.clientY - started[1]) > 4) return;
  const target = state.articulationCondyles?.segmentation.mandible ? 'mandible' : 'ct_bone';
  const point = viewer.pick(target, e.clientX, e.clientY);
  if (!point) return toast(target === 'mandible' ? 'Щёлкните по кости нижней челюсти' : 'Щёлкните по поверхности КТ');
  await updateCondyles('', { side, point, space: condyleSpace() });
}

// Нажатия стрелок — по очереди: каждый шаг считается от точки, которую сервер вернул на предыдущий
// (иначе при автоповторе клавиши два запроса считаются от одной точки и шаг теряется).
export let condyleNudges = Promise.resolve();

export function nudgeCondyle(key, precise = false, depth = false) {
  const side = state.condyleEditing;
  if (!side) return condyleNudges;
  condyleNudges = condyleNudges.then(async () => {
    const item = state.articulationCondyles?.points?.[side];
    const space = condyleSpace();
    if (state.condyleEditing !== side || !item?.[space]) return;
    const axes = state.articulationGuides?.[space === 'ct' ? 'ct_axes' : 'axes'] || {
      right: [-1, 0, 0], forward: [0, -1, 0], up: [0, 0, 1],
    };
    let axis, sign;
    if (key === 'ArrowLeft' || key === 'ArrowRight') {
      axis = axes.right; sign = key === 'ArrowRight' ? 1 : -1;
    } else {
      axis = depth ? axes.forward : axes.up;
      sign = key === 'ArrowUp' ? 1 : -1;
    }
    const amount = precise ? 0.1 : 0.5;
    const point = item[space].map((v, i) => v + sign * amount * axis[i]);
    await updateCondyles('', { side, point, space });
  });
  return condyleNudges;
}

export async function remountCondyles() {
  await busy('Перемонтаж по мыщелкам', async (progress) => {
    state.articulation = await run('articulation/mount', { method: 'ct' }, progress);
    state.articulation = await run('articulation/generate', { travel: 6.0 }, progress);
    state.articulationCondyles = await get('articulation/condyles');
    state.articulationGuides = await get('articulation/guides');
    drawArticulationGuides();
  });
}

export const matMul = (a, b) => {
  const out = Array.from({ length: 4 }, () => Array(4).fill(0));
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++)
    for (let k = 0; k < 4; k++) out[i][j] += a[i][k] * b[k][j];
  return out;
};

export async function playArticulation(id) {
  try {
    const motion = await get(`articulation/motion/${id}`);
    const lower = state.scans.find((s) => s.jaw === 'lower' && s.role !== 'bite' && s.transform);
    if (!lower) throw new Error('нижний скан не поставлен');
    const raise = raiseMatrix(); // поднятый прикус: движения — от поднятого положения (как с поднятым штифтом)
    const base = matMul(raise, lower.transform);
    const lowerStructures = state.structures.filter((s) => s.jaw === 'lower');
    const existing = new Map(lowerStructures.map((s) => [s.key, viewer.objects.get(s.key)?.visible]));
    for (const frame of motion.frames) {
      viewer.setTransform(lower.id, matMul(frame, base));
      for (const s of lowerStructures) viewer.setObjectTransform(s.key, matMul(frame, raise));
      await new Promise((ok) => setTimeout(ok, 28));
    }
    applyRaise();
    for (const [key, visible] of existing) if (visible !== undefined) viewer.setVisible(key, visible);
  } catch (e) { toast(e.message); }
}

// ---------- полная реабилитация (rehab.py): подъём прикуса по оси пациента и шаблон окклюзионной плоскости ----------
const IDENTITY4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const TEMPLATE_KEYS = ['occl:surface', 'occl:curves'];
let templateKey = '';

// Нижняя челюсть в окне 3D: с прикусом сканов — одна матрица, как на КТ — другая (сервер считает обе).
export function raiseMatrix() {
  const r = state.rehab;
  if (state.step !== 'articulation' || !r?.raise_mm) return IDENTITY4;
  return state.biteView?.on ? r.display_bite : r.display_ct;
}

export function applyRaise(M = raiseMatrix()) {
  const lower = state.scans.find((s) => s.jaw === 'lower' && s.role !== 'bite' && s.transform);
  if (lower) viewer.setTransform(lower.id, matMul(M, lower.transform));
  for (const s of state.structures.filter((x) => x.jaw === 'lower')) viewer.setObjectTransform(s.key, M);
  for (const key of ['roots:lower', 'axes:lower']) viewer.setObjectTransform(key, M);
}

async function drawTemplate() {
  const t = state.rehab?.template_info;
  const key = t && state.step === 'articulation' ? `${t.radius}:${t.center.join(',')}` : '';
  if (key === templateKey) return;
  templateKey = key;
  TEMPLATE_KEYS.forEach((k) => viewer.remove(k));
  if (!key) return;
  try {
    viewer.setMesh('occl:surface', await mesh('articulation/template/surface'), { color: '#7fd1ff', opacity: 0.35, order: 4 });
    viewer.setMesh('occl:curves', await mesh('articulation/template/curves'), { color: '#ffcf5a', order: 5 });
  } catch (e) { toast(e.message); }
}

export function endRehab() {
  applyRaise(IDENTITY4);
  templateKey = '';
  TEMPLATE_KEYS.forEach((k) => viewer.remove(k));
}

export async function loadRehab() {
  try { state.rehab = await get('articulation/rehab'); } catch { state.rehab = null; }
}

export async function rehabAction(kind, value) {
  const r = state.rehab || {};
  const body = kind === 'raise' ? { raise_mm: Number(value) }
    : kind === 'template' ? { template: !!value }
      : { radius: value === 'reset' ? r.monson_mm : Math.max(60, Math.min(160, (r.radius || r.monson_mm) + Number(value))) };
  try { state.rehab = await post('articulation/rehab', body); } catch (e) { toast(e.message); }
  render();
}

function rehabCard(mounted) {
  const r = state.rehab;
  if (!mounted || !r?.available) return '';
  queueMicrotask(() => { applyRaise(); drawTemplate(); });
  const m = r.report?.molars || {};
  const raiseText = r.raise_mm
    ? `Поворот ${fmt(r.angle_deg, 1)}° вокруг оси мыщелков пациента${m.right != null ? `; на молярах справа ${fmt(m.right, 1)} мм, слева ${fmt(m.left, 1)} мм` : ''}.`
    : 'Нижняя челюсть поворачивается вокруг настоящей оси мыщелков; в exocad — в новом положении.';
  const t = r.template_info;
  const norm = (x) => `${fmt(x.deg, 1)}° <span class="muted">(норма ${fmt(x.norm[0], 1)} ± ${fmt(x.norm[1], 1)})</span>`;
  const tpl = t ? `<div class="aes-row"><span>Сфера Монсона</span><b>${fmt(t.radius, 1)} мм</b></div>
      <div class="row" style="gap:4px;margin:2px 0 6px"><button class="btn ghost sm" data-rehab-radius="-5">−5 мм</button>
        <button class="btn ghost sm" data-rehab-radius="5">+5 мм</button><button class="btn ghost sm" data-rehab-radius="reset" title="4 дюйма">4″</button></div>
      <div class="aes-row"><span>Кривая Шпее, глубина</span><b>${fmt(t.spee_depth_mm, 1)} мм</b></div>
      ${t.tilt?.fh ? `<div class="aes-row"><span>Наклон к Франкфурту</span><b>${norm(t.tilt.fh)}</b></div>` : ''}
      ${t.tilt?.sn ? `<div class="aes-row"><span>Наклон к SN</span><b>${norm(t.tilt.sn)}</b></div>` : ''}
      ${!t.tilt?.fh && !t.tilt?.sn ? '<p class="muted small">Наклон к FH и SN — после точек в шаге «Ориентиры».</p>' : ''}` : '';
  return `<div class="card rehab-card"><div class="card-head">${icons.move}<h3>Высота прикуса и окклюзионная плоскость</h3></div>
    <div class="aes-row"><span>Подъём по резцам</span><b>${fmt(r.raise_mm, 1)} мм</b></div>
    <input type="range" class="rehab-range" min="0" max="10" step="0.5" value="${r.raise_mm}" data-rehab-raise>
    <p class="muted small">${raiseText}</p>
    <label class="joint-toggle"><input type="checkbox" data-rehab-template ${r.template ? 'checked' : ''}><span>Шаблон окклюзионной плоскости</span>
      <small>сфера через мыщелки и резцы: кривые Шпее и Уилсона</small></label>${tpl}</div>`;
}

export const CONDYLE_SIDES = {
  right: { title: 'Правый', color: '#ff6b6b' },
  left: { title: 'Левый', color: '#6ba8ff' },
};

export function condyleReason(c) {
  if (!c) return 'точки ещё не прочитаны';
  const missing = Object.entries(c.points || {}).filter(([, p]) => !p.ct).map(([side]) => CONDYLE_SIDES[side].title.toLowerCase());
  const errors = (c.checks || []).filter((x) => x.level === 'error').map((x) => x.text);
  return [...(missing.length ? [`нет точки: ${missing.join(', ')}`] : []), ...errors].join('; ') || 'точки готовы';
}

export function renderCondylePanel(scansReady) {
  const c = state.articulationCondyles;
  if (!c?.available) return '';
  const rows = Object.entries(CONDYLE_SIDES).map(([side, meta]) => {
    const p = c.points?.[side] || {};
    const source = p.source_label || 'нет точки';
    const verify = p.ct && !p.confirmed ? ' · проверьте' : '';
    return `<div class="condyle-side ${state.condyleEditing === side ? 'editing' : ''}">
      <div class="condyle-name"><i style="background:${meta.color}"></i><b>${meta.title}</b>
        <em class="${p.confirmed ? 'ok' : ''}">${p.confirmed ? 'подтверждён' : 'не подтверждён'}</em></div>
      <div class="condyle-source">${hide(source + verify)}</div>
      ${p.note ? `<div class="condyle-note">${hide(p.note)}</div>` : ''}
      <div class="condyle-actions">
        <button class="btn sm" data-condyle-confirm="${side}" ${p.ct && !p.confirmed ? '' : 'disabled'}>${icons.check}Подтвердить</button>
        <button class="btn sm ${state.condyleEditing === side ? 'primary' : ''}" data-condyle-edit="${side}">${icons.target}Поправить</button>
        <button class="btn ghost sm" data-condyle-reset="${side}" ${p.ct || c.segmentation.mandible ? '' : 'disabled'}>${icons.undo}Сбросить</button>
      </div></div>`;
  }).join('');
  const checks = (c.checks || []).map((x) => `<div class="condyle-check ${x.level}"><i></i><span>${hide(x.text)}</span></div>`).join('');
  const missing = Object.entries(c.points || {}).filter(([, p]) => !p.ct).map(([s]) => CONDYLE_SIDES[s].title.toLowerCase());
  const hints = [
    !c.segmentation.mandible ? 'Нижняя челюсть не сегментирована: «Поправить» покажет кость по плотности КТ. Выберите верхушку головки мыщелка; точки и кость показаны как на КТ.' : '',
    missing.length === 1 ? `Не найдена ${missing[0]} точка — поставьте её вручную.` : '',
    !c.segmentation.skull ? 'Череп не сегментирован: наклон суставных путей останется по умолчанию.' : '',
  ].filter(Boolean).map((x) => `<div class="condyle-hint">${icons.warn}<span>${hide(x)}</span></div>`).join('');
  const apply = c.applied
    ? `<div class="condyle-applied">${icons.check}<span>Точки применены к текущему монтажу</span></div>`
    : `<button class="btn primary wide" data-condyle-remount="1" ${c.ready && scansReady ? '' : 'disabled'}>${icons.move}Перемонтировать</button>
       <p class="muted small">${hide(c.ready ? (scansReady ? 'После перемонтажа движения будут рассчитаны заново.' : 'Сначала поставьте оба скана.') : condyleReason(c))}</p>`;
  const editHint = state.condyleEditing
    ? `<div class="condyle-edit-hint">Щёлкните по ${c.segmentation.mandible ? 'нижней челюсти' : 'головке мыщелка на кости КТ'}. ←/→ — вправо/влево, ↑/↓ — вверх/вниз, Ctrl+↑/↓ — вперёд/назад по осям монтажа; шаг 0,5 мм, с Shift — 0,1 мм.</div>` : '';
  return `<div class="card condyles"><div class="card-head">${icons.target}<h3>Мыщелки</h3>
      <span class="badge ${c.ready ? 'ok' : 'warn'}">${c.icd_mm == null ? 'расстояние —' : `${fmt(c.icd_mm, 1)} мм`}</span></div>
    <p class="muted small">${condyleSpace() === 'ct' ? 'Точки показаны в координатах КТ.' : 'Модели показаны в прикусе сканов — точки ставятся в нём'}</p>
    ${rows}${editHint}<div class="condyle-checks">${checks}</div>${hints}${apply}</div>`;
}

export function renderArticulation() {
  const a = state.articulation || {};
  const scans = a.scans || {};
  const ready = !!(scans.upper && scans.lower);
  const mounted = !!a.mounting;
  const moves = a.movements || [];
  const analysis = state.articulationAnalysis;
  const guides = state.articulationGuides;
  const disabled = ready ? '' : 'disabled';
  const c = state.articulationCondyles;
  const mountPlanes = state.articulationPlanes;
  const selectedPlane = mountPlanes?.selected || mountPlanes?.default || state.mountPlaneDefault;
  const selectedInfo = mountPlanes?.planes?.[selectedPlane];
  const ctDisabled = ready && c?.ready ? '' : 'disabled';
  const planeOptions = state.ct && mountPlanes?.planes ? `<div class="label">Плоскость монтажа по КТ</div>
    <select class="mount-plane-select" data-mount-plane ${ready && c?.ready ? '' : 'disabled'}>
      ${Object.entries(mountPlanes.planes).map(([key, p]) => `<option value="${key}" ${key === selectedPlane ? 'selected' : ''}>${hide(p.name)}${p.available && mountPlanes.tilt?.[key] ? ` · ${mountPlanes.tilt[key].pitch_deg}° / ${mountPlanes.tilt[key].roll_deg}°` : ''}${p.available ? '' : ` — не хватает: ${hide(p.missing.join(', '))}`}</option>`).join('')}
    </select>${selectedInfo && !selectedInfo.available ? `<p class="muted small">Недоступна: ${hide(selectedInfo.missing.join(', '))}.</p>` : ''}` : '';
  const selectedIsCurrent = !!a.mounting && a.mounting.plane?.key === selectedPlane;
  const mountButtons = '<div class="label">Гипсовка</div><div class="seg">'
    + (state.ct ? `<button data-art-mount="ct" ${ctDisabled}>${a.mounting && !selectedIsCurrent ? 'Перемонтировать' : 'По точкам КТ'}</button>` : '')
    + `<button data-art-mount="average" ${disabled}>Средний артикулятор</button></div>`
    + (state.ct && !c?.ready ? `<p class="muted small">Монтаж по КТ недоступен: ${hide(condyleReason(c))}.</p>` : '');
  const movementCards = moves.length ? '<div class="label">Движения</div>' + moves.map((m) =>
    '<div class="card"><div class="card-head">' + icons.move + '<h3>' + hide(m.name) + '</h3>'
    + '<span class="muted small">' + m.frames + ' кадров</span></div>'
    + '<button class="btn wide" data-art-play="' + m.id + '">' + icons.play + 'Проиграть в 3D</button></div>').join('') : '';
  const settings = a.settings?.values;
  const planeName = mountPlanes?.planes?.[a.mounting?.plane?.key]?.name || 'окклюзионной плоскости';
  const settingsText = settings ? '<p class="muted small">ССП от ' + hide(planeName) + ': ' + fmt(settings.sagittal_right_deg, 1) + '° / '
    + fmt(settings.sagittal_left_deg, 1) + '° · Беннетт: ' + fmt(settings.bennett_right_deg, 1) + '° / '
    + fmt(settings.bennett_left_deg, 1) + '°</p>' : '';
  const jointAvailable = Object.values(state.jointParts || {}).some((parts) => parts.head || parts.slope);
  const jointHint = !jointAvailable ? `<p class="muted small">${!state.articulationCondyles?.segmentation.mandible && !state.articulationCondyles?.segmentation.skull
    ? 'Чтобы показать головки и скаты, сегментируйте нижнюю челюсть и череп.'
    : !state.articulationCondyles?.segmentation.mandible ? 'Для головок нужна сегментация нижней челюсти.'
      : !state.articulationCondyles?.segmentation.skull ? 'Для скатов нужна сегментация черепа.' : 'Не удалось выделить суставные поверхности.'}</p>` : '';
  const jointsControl = `<label class="joint-toggle ${jointAvailable ? '' : 'disabled'}"><input type="checkbox" data-joints ${state.jointsVisible ? 'checked' : ''} ${jointAvailable ? '' : 'disabled'}><span>Суставы</span><small>головки зелёные · скаты жёлтые</small></label>${jointHint}`;
  const mountNotes = (a.notes || []).filter((n) => n.includes('ССП') && n.includes('вне 5–75°'))
    .map((n) => `<div class="warning mount-warning">${icons.warn}<span>${hide(n)}</span></div>`).join('');
  const analysisSummary = analysis && (analysis.notes?.join(' ') || (analysis.recordings?.length
    ? 'Проанализировано движений: ' + analysis.recordings.length : 'Анализ завершён.'));
  const contactText = state.articulationContacts.length ? state.articulationContacts.map((c) => {
    const sectors = Object.keys(c.sectors || {}).join(', ') || 'нет выраженных контактов';
    return '<p class="muted small"><b>' + hide(c.name) + '</b>: ' + c.pairs + ' пар · ' + hide(sectors) + '</p>';
  }).join('') : '';
  const basisText = guides?.available ? '<p class="muted small">Основа динамики: ' + hide(guides.basis) + '</p>' : '';
  const analysisText = analysis ? '<div class="card"><div class="card-head">' + icons.check
    + '<h3>Проверка движений</h3></div><p class="muted small">' + hide(analysisSummary) + '</p>'
    + (analysis.worst_incisal_mm != null ? '<p class="muted small">Максимальное отклонение резцов: '
      + fmt(analysis.worst_incisal_mm, 2) + ' мм</p>' : '') + contactText + '</div>' : '';
  let body = '';
  if (!ready) {
    body = '<div class="card"><p>Нужны поставленные верхний и нижний сканы.</p>'
      + '<button class="btn wide" data-step="scans">Перейти к сканам</button></div>';
  } else {
    const mountingName = mounted ? `Монтаж: ${hide(a.mounting.name)}` : 'Монтаж ещё не выполнен';
    body = planeOptions + mountButtons
      + `<div class="card"><div class="card-head">${icons.align}<h3>${mountingName}</h3></div>`
      + `<p class="muted small">${hide(a.mounting?.source || 'Положение моделей будет показано в 3D.')}</p>${basisText}${settingsText}${mountNotes}`
      + `<button class="btn ${mounted ? '' : 'primary'} wide" data-art-generate="1" ${mounted ? '' : 'disabled'}>${icons.play}Рассчитать движения</button>`
      + `<button class="btn wide" data-art-analysis="1" ${moves.length ? '' : 'disabled'}>${icons.heat}Проверить контакты и траектории</button></div>`
      + movementCards + analysisText;
  }
  return '<h2>Артикулятор</h2><p class="lead">Здесь выполняется предварительная гипсовка, проигрываются движения и проверяются контакты до экспорта в exocad.</p>'
    + jointsControl + renderCondylePanel(ready) + body + (ready ? rehabCard(mounted) : '');
}
