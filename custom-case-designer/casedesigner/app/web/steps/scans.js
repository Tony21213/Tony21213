// Шаг «Сканы»: добавить, совместить, поправить, принять, прикус сканов, панель.
import { $, JAWS, TRANSLUCENT, app, bumpOverlay, busy, choose, fmt, heatOn, hide, metricsHtml, opacityButton, render, scanById, scanLabel, scanRgb, showScan, slices, state, toast, updateScan, viewer } from '../core.js';
import { get, mesh, post, run } from '../api.js';
import { icons } from '../icons.js';
import { drawArticulationGuides, refreshJointParts } from './articulation.js';
import { intakeCard } from './ct.js';

export async function addScans() {
  const paths = await choose('scan', 'Скан челюсти (STL, PLY, OBJ)');
  if (!paths) return;
  await busy('Загружаю сканы', async () => {
    // File parsing and mesh transfer are independent; start them together so
    // selecting upper, lower and bite scans does not wait serially.
    const infos = await Promise.all(paths.map((path) => post('scans', { path })));
    infos.forEach(updateScan);
    await Promise.all(infos.map(async (info) => {
      viewer.setScan(info.id, await mesh(`scans/${info.id}/mesh`), info.color, null);
      state.selected ??= info.id;
    }));
  });
  registerPending(); // без КТ — в координатах сканера
}

export async function removeScan(id) {
  await post(`scans/${id}/remove`);
  viewer.remove(id);
  state.scans = state.scans.filter((s) => s.id !== id);
  if (state.selected === id) state.selected = state.scans[0]?.id ?? null;
  await refreshScans();
  render();
}

// Сканы прикуса стоят на сканах челюстей, и нижний скан может встать в прикус по ним: после изменения
// положения челюсти — свежие положения всех сканов (кроме тех, что сейчас двигают).
export async function refreshScans(force = false) {
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
export async function reloadStructureMeshes() {
  await Promise.all(state.structures.map(async (s) => {
    const old = viewer.objects.get(s.key);
    const opacity = old ? old.material.opacity : (TRANSLUCENT[s.key] ?? 1);
    const visible = old ? old.visible : state.visible.has(s.key);
    viewer.setMesh(s.key, await mesh(`structures/${encodeURIComponent(s.key).replace(/%2F/g, '/')}/mesh`),
      { color: s.color, opacity, order: opacity < 1 ? 1 : 0 });
    viewer.setVisible(s.key, visible);
  }));
}

// «Прикус»: нижняя челюсть со всеми её структурами — в прикусе сканов (врача); выключено — как на КТ.
export async function toggleBite() {
  const r = await post('bite_view', { on: !state.biteView?.on }).catch((e) => toast(e.message));
  if (!r) return;
  state.biteView = r;
  await busy(r.on ? 'Прикус сканов' : 'Как на КТ', async () => {
    await reloadStructureMeshes();
    await refreshScans(true);
    if (state.jointsVisible) await refreshJointParts();
    bumpOverlay();
    app.setCursor(app.cursor);
    if (state.step === 'articulation') drawArticulationGuides();
  });
}

export async function register(ids, body = {}) {
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
export const registerPending = () => register(state.scans.filter((s) => !s.registered)
  .sort((a, b) => (a.role === 'bite') - (b.role === 'bite')).map((s) => s.id));

// Ручная поправка с манипулятора на срезе: M — матрица 4×4 в мм пациента, применяется поверх положения скана.
export let evalTimer = null;
export function remember(id) { // перед изменением положения в режиме коррекции — для Ctrl+Z
  const c = state.correcting;
  if (c?.id !== id) return;
  c.undo.push(viewer.getTransform(id));
  c.redo = [];
}

export function moveScan(id, M) {
  remember(id);
  const T = viewer.getTransform(id);
  const R = T.map((_, r) => [0, 1, 2, 3].map((c) => M[r].reduce((sum, m, k) => sum + m * T[k][c], 0)));
  viewer.setTransform(id, R);
  state.moving.set(id, (state.moving.get(id) || 0) + 1);
  evaluateManual(id, 250); // серия нажатий стрелок — одной оценкой
}

export let evalChain = Promise.resolve(); // оценки положения — строго по очереди: последнее отправленное положение и остаётся
export function evaluateManual(id, delay = 0) {
  clearTimeout(evalTimer);
  evalTimer = setTimeout(() => { evalChain = evalChain.then(() => evaluateNow(id)); }, delay);
}

export async function evaluateNow(id) {
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
      viewer.setColors(id, await scanRgb(info));
    }
    bumpOverlay();
    app.setCursor(app.cursor);
    render();
  }
}

export async function refine(id) {
  remember(id);
  await register([id], { start: viewer.getTransform(id), jaw: scanById(id).jaw });
}

export async function resetAuto(id) {
  const s = scanById(id);
  if (!s.auto_transform) return;
  remember(id);
  viewer.setTransform(id, s.auto_transform);
  state.moving.set(id, (state.moving.get(id) || 0) + 1);
  slices.forEach((v) => v.settled(id));
  evaluateManual(id);
}

// Коррекция положения: манипулятор на срезах; «Сохранить» запоминает результат пользователя для обучения.
export async function startCorrection(id) {
  if (state.biteView?.on && scanById(id)?.jaw === 'lower') await toggleBite(); // коррекция — по КТ
  state.correcting = { id, start: viewer.getTransform(id), undo: [], redo: [] };
  slices.forEach((v) => v.draw());
  render();
}

export function endCorrection() {
  state.correcting = null;
  slices.forEach((v) => v.draw());
  render();
}

export function revertCorrection() {
  const c = state.correcting;
  if (!c) return;
  viewer.setTransform(c.id, c.start);
  state.moving.set(c.id, (state.moving.get(c.id) || 0) + 1);
  slices.forEach((v) => v.settled(c.id));
  evaluateManual(c.id);
}

// Ctrl+Z / Ctrl+Y в режиме коррекции: шаг назад и вперёд по положениям скана.
export function stepHistory(back) {
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

export async function settledPosition(id) { // дождаться, пока сервер примет последнюю поправку
  clearTimeout(evalTimer);
  if (state.moving.has(id)) evalChain = evalChain.then(() => evaluateNow(id));
  await evalChain;
}

export async function accept(id) {
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
// ---------- «Исправить прикус» (bite.correct_bite; docs/handoff/bite-correction.md) ----------
export async function biteCorrect() {
  state.biteError = null;
  await busy('Исправляю прикус', async (progress) => {
    try {
      const r = await run('bite/correct', {}, progress);
      state.biteProposal = r;
      state.biteView = r.bite_view;
    } catch (e) {
      state.biteError = e.message; // отказ ядра (нет опоры, сверх предела) — в карточке, а не всплывающим окном
    }
  });
}

export async function biteDecide(kind) { // accept · reject · reset
  const r = await post(`bite/${kind}`).catch((e) => toast(e.message));
  if (!r) return;
  state.biteView = r;
  state.biteProposal = null;
  state.biteError = null;
  if (kind === 'reject') return render();
  await busy(kind === 'accept' ? 'Прикус исправлен' : 'Прикус без поправки', async () => {
    await reloadStructureMeshes();
    await refreshScans(true);
    if (state.jointsVisible) await refreshJointParts();
    bumpOverlay();
    app.setCursor(app.cursor);
  });
}

function biteCard() {
  const bv = state.biteView;
  if (!bv?.available || !state.ct) return '';
  const p = state.biteProposal;
  if (p) {
    const r = p.report;
    const contacts = (c) => `${c.points} (справа ${c.sectors.right}, спереди ${c.sectors.front}, слева ${c.sectors.left})`;
    const notes = (p.notes || []).map((n) => `<div class="warning">${icons.warn}<span>${hide(n)}</span></div>`).join('');
    return `<div class="card bite-fix"><div class="card-head">${icons.refine}<h3>Исправление прикуса</h3></div>
      <div class="kv"><span>Резцы</span><b>${fmt(r.incisal_mm)} мм</b><span>Моляры П / Л</span><b>${fmt(r.molar_right_mm)} / ${fmt(r.molar_left_mm)} мм</b>
        <span>Поворот</span><b>${fmt(r.turn_deg)}°</b><span>Контакты до</span><b>${contacts(r.before)}</b>
        <span>Контакты после</span><b>${contacts(r.after)}</b><span>Мыщелки</span><b>${hide(r.condyles)}</b></div>${notes}
      <div class="row" style="margin-top:8px"><button class="btn ghost" data-a="bitereject">Отменить</button>
        <button class="btn ok grow" data-a="biteaccept">${icons.check}Принять</button></div></div>`;
  }
  const refused = state.biteError ? `<div class="warning">${icons.warn}<span>Прикус не исправлен: ${hide(state.biteError)}</span></div>` : '';
  const fixed = bv.fixed ? `<p class="muted small">Прикус исправлен: нижняя челюсть сдвинута в среднем на ${fmt(bv.fix_mm)} мм.</p>
    <button class="btn ghost wide sm" data-a="bitereset">Вернуть прикус сканера</button>` : '';
  return `<div class="card bite-fix"><div class="card-head">${icons.refine}<h3>Прикус сканов</h3></div>
    <p class="muted small" style="margin-top:0">Нижняя челюсть садится в фиссурно-бугорковый контакт от прикуса сканера: исправляются
      ошибки скана (до 1,5 мм), а не ищется другой прикус.</p>
    <button class="btn wide" data-a="bitefix">${icons.refine}${bv.fixed ? 'Исправить ещё раз' : 'Исправить прикус'}</button>${refused}${fixed}</div>`;
}

export function renderScans() {
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
    ${cards}<button class="btn ${state.scans.length ? '' : 'primary'} wide" data-a="scan">${icons.plus}Добавить сканы</button>${tools}${biteCard()}${ct ? '' : intakeCard()}`;
}
