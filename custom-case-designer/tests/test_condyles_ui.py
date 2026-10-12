"""Проверка координат интерактивной поправки без графического контекста."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_pick_and_arrows_use_displayed_surface_space():
    node = shutil.which("node")
    if node is None:
        pytest.skip("нужен Node.js для проверки обработчиков интерфейса")
    app = Path(__file__).parents[1] / "casedesigner/app/web/app.js"
    source = app.read_text(encoding="utf-8")
    handlers = source[source.index("function clearArticulationGuides()") : source.index("const matMul =")]
    script = r"""
const assert = require('node:assert/strict');
const state = {step: 'articulation', biteView: {on: true}, articulationGuides: {
  axes: {right: [-1,0,0], forward: [0,-1,0], up: [0,0,1]},
  ct_axes: {right: [-1,0,0], forward: [0,0,-1], up: [0,-1,0]},
}, visible: new Set(), articulationCondyles: {available: true, segmentation: {mandible: false},
  points: {right: {ct: [-49,10,0], view: [-49,20,30]}}}};
const viewer = {objects: new Map(), renderer: {domElement: {}}, markers: {},
  setMesh(k, m) {this.objects.set(k, m)}, setVisible() {},
  setMarker(k, p) {this.markers[k] = p}, paths: [], setPath(k) {this.paths.push(k)},
  remove(k) {this.objects.delete(k)}, pick(k) {picked = k; return [-48,11,2]},
};
const requests = [];
let picked, viewPointerDown;
const mesh = async url => {assert.equal(url, 'ct/surface?part=bone'); return {bone: true}};
const render = () => {};
const toast = message => {throw new Error(message)};
const post = async (url, body) => {requests.push(body); return state.articulationCondyles};
""" + handlers + r"""
(async () => {
  await editCondyle('right');
  assert(viewer.objects.has('ct_bone'));
  assert.deepEqual(viewer.markers['art:condyle_right'], [-49,10,0]);
  viewPointerDown = [10,20];
  await handleCondylePick({button: 0, clientX: 10, clientY: 20, target: viewer.renderer.domElement});
  assert.equal(picked, 'ct_bone');
  assert.deepEqual(requests.at(-1), {side:'right', point:[-48,11,2], space:'ct'});
  await nudgeCondyle('ArrowUp');
  assert.deepEqual(requests.at(-1), {side:'right', point:[-49,9.5,0], space:'ct'});
  endCondyleEdit();
  assert(!viewer.objects.has('ct_bone'));
  assert.deepEqual(viewer.markers['art:condyle_right'], [-49,20,30]);
  state.articulationCondyles.segmentation.mandible = true;
  viewer.objects.set('mandible', {});
  await editCondyle('right');
  viewPointerDown = [10,20];
  await handleCondylePick({button:0, clientX:10, clientY:20, target:viewer.renderer.domElement});
  assert.equal(picked, 'mandible');
  assert.equal(requests.at(-1).space, 'view');
  await nudgeCondyle('ArrowUp', true);
  assert.deepEqual(requests.at(-1), {side:'right', point:[-49,20,30.1], space:'view'});
  state.biteView.on = false;
  await nudgeCondyle('ArrowUp');
  assert.equal(requests.at(-1).space, 'ct');
  state.articulationGuides.available = true;
  state.articulationGuides.paths = [{id:'test', condyle_right:[], condyle_left:[], incisal:[]}];
  state.articulationCondyles.applied = true;
  state.articulationCondyles.bite = 'ct';
  drawArticulationGuides();
  assert(viewer.paths.includes('art:path:test:condyle_right'));
  viewer.paths = [];
  state.articulationCondyles.bite = 'scan';
  drawArticulationGuides();
  assert(!viewer.paths.includes('art:path:test:condyle_right'));
  state.condyleEditing = null;
  state.articulationCondyles = {available:false};
  drawArticulationGuides();
  assert(viewer.paths.includes('art:path:test:condyle_right'));
})().catch(error => {console.error(error); process.exitCode = 1});
"""
    result = subprocess.run([node, "-"], input=script, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
