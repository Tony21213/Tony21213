// KStom Case Designer — вход: события интерфейса и старт; шаги — steps/<шаг>.js, общее — core.js.
import { $, NEXT_JAW, RENDER, app, busy, loadStructures, render, scanById, showScan, shownRecent, slices, state, toast, toggleIncognito, toggleOpacity, toggleStructures, updateScan, viewer } from './core.js';
import { get, mesh, post, run } from './api.js';
import { SliceView } from './slices.js';
import { articulationAction, editCondyle, endCondyleEdit, endRehab, handleCondylePick, mountWithPlane, nudgeCondyle, playArticulation, prepareArticulation, rehabAction, rememberPointerDown, remountCondyles, renderArticulation, selectMountPlane, toggleJoints, updateCondyles } from './steps/articulation.js';
import { helpDialog, openCase, saveCase, settingsDialog } from './steps/case.js';
import { downloadModels, intakeCheck, intakeSave, openCt, partsDialog, renderCt, segment } from './steps/ct.js';
import { chooseExocadProject, doExport, doExportExocadArticulator, prepareExport, renderExport, toggleGum, toggleRoots } from './steps/export.js';
import { aestheticAction, aestheticPick, aestheticPointerDown, endAesthetics, prepareAesthetics, renderAesthetics } from './steps/aesthetics.js';
import { renderObjectBar } from './objectbar.js';
import { cancelPlacing, endLandmarks, landmarkAction, landmarkPlacing, landmarkPoints, placeLandmark, prepareLandmarks, renderLandmarks } from './steps/landmarks.js';
import { accept, addScans, biteCorrect, biteDecide, endCorrection, refine, register, removeScan, renderScans, resetAuto, revertCorrection, startCorrection, stepHistory, toggleBite } from './steps/scans.js';

// панели шагов и мышь в окне 3D — здесь: шаги исполняются при загрузке раньше core.js (циклические импорты)
RENDER.ct = renderCt;
RENDER.scans = renderScans;
RENDER.objectBar = renderObjectBar; // панель объектов окна 3D
RENDER.landmarks = renderLandmarks;
app.landmarkPoints = landmarkPoints; // точки на срезах и постановка щелчком (шаг «Ориентиры»)
app.placeLandmark = placeLandmark;
RENDER.aesthetics = renderAesthetics;
RENDER.articulation = renderArticulation;
RENDER.export = renderExport;
$('#view3d').addEventListener('pointerdown', rememberPointerDown);
$('#view3d').addEventListener('pointerup', handleCondylePick);
$('#view3d').addEventListener('pointerdown', aestheticPointerDown); // «Эстетика»: точка пары на скане
$('#view3d').addEventListener('pointerup', aestheticPick);

// ---------- события ----------
document.addEventListener('click', async (e) => {
  const t = e.target.closest('[data-step],[data-a],[data-select],[data-remove],[data-register],[data-refine],[data-reset],[data-accept],[data-correct],[data-revert],[data-hist],[data-endcorrect],[data-recent],[data-case],[data-opacity],[data-gopacity],[data-warns],[data-toggle],[data-groupcheck],[data-group],[data-frame],[data-exbite],[data-view],[data-heat],[data-scancolor],[data-bite],[data-jaw],[data-art-mount],[data-art-generate],[data-art-analysis],[data-art-play],[data-condyle-confirm],[data-condyle-reset],[data-condyle-edit],[data-condyle-remount],[data-joints],[data-aes],[data-gum],[data-lm],[data-rehab-radius]');
  if (!t || t.disabled) return;
  const d = t.dataset;
  if (d.step) {
    state.step = d.step;
    endCondyleEdit();
    endAesthetics();
    endLandmarks();
    endRehab(); // поднятая нижняя челюсть и шаблон — только в шаге «Артикулятор»
    slices.forEach((v) => v.draw());
    return d.step === 'articulation' ? prepareArticulation() : d.step === 'export' ? prepareExport()
      : d.step === 'aesthetics' ? prepareAesthetics() : d.step === 'landmarks' ? prepareLandmarks() : render();
  }
  if (d.aes) return aestheticAction(d.aes, d.arg);
  if (d.lm) return landmarkAction(d.lm, d.key);
  if (d.rehabRadius) return rehabAction('radius', d.rehabRadius);
  if (d.condyleConfirm) return updateCondyles('/confirm', { side: d.condyleConfirm });
  if (d.condyleReset) return updateCondyles('/reset', { side: d.condyleReset });
  if (d.condyleEdit) return editCondyle(d.condyleEdit);
  if ('condyleRemount' in d) return remountCondyles();
  if ('joints' in d) return toggleJoints();
  if (d.artMount) return mountWithPlane(d.artMount);
  if (d.artGenerate) return articulationAction('generate', { travel: 6.0 }, 'Расчёт движений');
  if (d.artAnalysis) return busy('Анализ контактов и движений', async (progress) => {
    const report = await run('articulation/analysis', {}, progress);
    state.articulationAnalysis = report.analysis;
    state.articulationContacts = report.contacts || [];
  });
  if (d.artPlay) return playArticulation(d.artPlay);
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
  if ('scancolor' in d) { state.scanColors = !state.scanColors; for (const s of state.scans) await showScan(s); return render(); }
  if ('gum' in d) return toggleGum();
  if (d.select && !e.target.closest('button')) {
    if (state.selected !== d.select) state.correcting = null;
    state.selected = d.select;
    slices.forEach((v) => v.draw());
    return render();
  }
  const action = { ct: () => openCt('ct'), ctdir: () => openCt('ctdir'), scan: addScans, segment, export: doExport,
    'exocad-project': chooseExocadProject, 'exocad-articulator': doExportExocadArticulator,
    getmodels: downloadModels, stopmodels: () => post('models/cancel'), parts: partsDialog,
    openout: () => post('export/open').catch((err) => toast(err.message)), help: helpDialog, incognito: toggleIncognito,
    settings: settingsDialog, intake: intakeCheck, intakesave: intakeSave, roots: toggleRoots,
    intakeopen: () => post('intake/open').catch((err) => toast(err.message)), openpassport: () => post('export/passport/open').catch((err) => toast(err.message)), bitefix: biteCorrect, biteaccept: () => biteDecide('accept'),
    bitereject: () => biteDecide('reject'), bitereset: () => biteDecide('reset'),
    scansonly: () => { state.step = 'scans'; render(); } }[d.a];
  action?.();
});

document.addEventListener('change', (e) => {
  const select = e.target.closest('[data-mount-plane]');
  if (select) selectMountPlane(select.value);
  if (e.target.matches('[data-rehab-raise]')) rehabAction('raise', e.target.value); // подъём прикуса по оси пациента
  if (e.target.matches('[data-rehab-template]')) rehabAction('template', e.target.checked);
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
  if (e.key === 'Escape' && state.condyleEditing) {
    endCondyleEdit();
    return render();
  }
  if (e.key === 'Escape' && state.step === 'landmarks' && landmarkPlacing()) return cancelPlacing();
  if (e.key === 'Escape' && state.step === 'aesthetics') {
    endAesthetics();
    return render();
  }
  if (state.step === 'articulation' && state.condyleEditing
      && ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
    e.preventDefault();
    return nudgeCondyle(e.key, e.shiftKey, e.ctrlKey || e.metaKey);
  }
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
  state.articulation = s.articulation;
  state.articulationGuides = s.articulation_guides;
  state.landmarks = s.landmarks || [];
  state.caseInfo = s.case;
  state.recent = s.recent || [];
  state.incognito = !!s.incognito;
  state.mountPlaneDefault = s.mount_plane_default || 'aesthetic_hinge';
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
