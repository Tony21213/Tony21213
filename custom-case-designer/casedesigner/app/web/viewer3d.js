// 3D-вид: структуры из КТ, сканы и ручная коррекция положения скана гизмо.
// Всё в мм пациента (DICOM, LPS): x — к левой стороне пациента, y — назад, z — вверх.
import * as THREE from './vendor/three/three.module.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { TransformControls } from './vendor/three/TransformControls.js';

const VIEWS = {
  front: [0, -1, 0.12],   // спереди: камера перед лицом (−y)
  right: [-1, 0, 0.08],   // справа: со стороны правой щеки пациента (−x)
  left: [1, 0, 0.08],
  top: [0, -0.02, 1],     // сверху
  bottom: [0, -0.02, -1], // снизу — жевательные поверхности верхних зубов
};

// Грани замкнутой поверхности — лицевой стороной наружу (по знаку объёма); иначе как есть.
function outward(v, f) {
  let vol = 0;
  for (let k = 0; k < f.length; k += 3) {
    const a = 3 * f[k], b = 3 * f[k + 1], c = 3 * f[k + 2];
    vol += v[a] * (v[b + 1] * v[c + 2] - v[b + 2] * v[c + 1]) - v[a + 1] * (v[b] * v[c + 2] - v[b + 2] * v[c])
      + v[a + 2] * (v[b] * v[c + 1] - v[b + 1] * v[c]);
  }
  if (vol >= 0) return f;
  const flipped = f.slice();
  for (let k = 0; k < f.length; k += 3) { flipped[k + 1] = f[k + 2]; flipped[k + 2] = f[k + 1]; }
  return flipped;
}

export class Viewer3D {
  constructor(el, { onTransformEnd } = {}) {
    this.el = el;
    this.onTransformEnd = onTransformEnd;
    this.objects = new Map();
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 1, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(0, -300, 40);
    this.scene.add(new THREE.HemisphereLight(0xe6edff, 0x1a1e2a, 1.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-0.4, -0.6, 1);
    this.camera.add(key);
    const rim = new THREE.DirectionalLight(0x9fb8ff, 0.6);
    rim.position.set(0.8, 0.5, -0.4);
    this.camera.add(rim);
    this.scene.add(this.camera);

    // Левая кнопка — вращение, колесо — масштаб к точке под курсором, нажатое колесо или правая — сдвиг.
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.rotateSpeed = 0.8;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.PAN };
    this.radius = 100; // размер сцены, мм (fit)

    this.gizmo = new TransformControls(this.camera, this.renderer.domElement);
    this.gizmo.setSpace('world');
    this.gizmo.setSize(0.9);
    this.gizmo.addEventListener('dragging-changed', (e) => {
      this.controls.enabled = !e.value;
      if (!e.value && this.gizmoTarget) this.onTransformEnd?.(this.gizmoTarget);
    });
    this.scene.add(this.gizmo.getHelper());

    new ResizeObserver(() => this.resize()).observe(el);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.clip();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    loop();
  }

  // Ближняя и дальняя плоскости — по расстоянию до цели: при приближении колесом модель не срезается.
  clip() {
    const d = this.camera.position.distanceTo(this.controls.target);
    const near = Math.max(0.02, d / 200), far = Math.max(d * 20, d + 4 * this.radius);
    if (Math.abs(near - this.camera.near) > 1e-3 * near || Math.abs(far - this.camera.far) > 1e-3 * far) {
      this.camera.near = near;
      this.camera.far = far;
      this.camera.updateProjectionMatrix();
    }
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // Полупрозрачное рисуется только лицевой стороной: без записи глубины двусторонняя поверхность
  // показывает изнанку и дальние слои поверх ближних — изображение «ломается» полосами.
  material(color, opacity = 1) {
    return new THREE.MeshStandardMaterial({
      color, roughness: 0.55, metalness: 0.04, transparent: opacity < 1, opacity,
      depthWrite: opacity >= 1, side: opacity < 1 ? THREE.FrontSide : THREE.DoubleSide,
    });
  }

  // Сетка структуры из КТ (координаты уже в мм пациента).
  setMesh(key, { vertices, faces }, { color = '#d8d0c0', opacity = 1, order = 0 } = {}) {
    this.remove(key);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    g.setIndex(new THREE.BufferAttribute(outward(vertices, faces), 1)); // наружу: прозрачность включается кнопкой
    g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, this.material(color, opacity));
    mesh.renderOrder = order;
    this.scene.add(mesh);
    this.objects.set(key, mesh);
    return mesh;
  }

  // Скан: геометрия сдвинута к своему центру, чтобы гизмо вращало скан вокруг него.
  setScan(id, { vertices, faces }, color, transform) {
    this.remove(id);
    const c = new THREE.Vector3();
    for (let i = 0; i < vertices.length; i += 3) c.add(new THREE.Vector3(vertices[i], vertices[i + 1], vertices[i + 2]));
    c.divideScalar(vertices.length / 3);
    const centred = new Float32Array(vertices.length);
    for (let i = 0; i < vertices.length; i += 3) {
      centred[i] = vertices[i] - c.x; centred[i + 1] = vertices[i + 1] - c.y; centred[i + 2] = vertices[i + 2] - c.z;
    }
    const mesh = this.setMesh(id, { vertices: centred, faces }, { color, order: 2 });
    mesh.userData = { centre: c, color, scan: true };
    mesh.visible = !!transform;
    if (transform) this.setTransform(id, transform);
    return mesh;
  }

  // transform — матрица 4×4 «скан → КТ» по строкам.
  setTransform(id, transform) {
    const mesh = this.objects.get(id);
    if (!mesh) return;
    const T = new THREE.Matrix4().set(...transform.flat());
    const M = T.multiply(new THREE.Matrix4().makeTranslation(mesh.userData.centre));
    M.decompose(mesh.position, mesh.quaternion, mesh.scale);
    mesh.visible = true;
  }

  // Центр скана в мм пациента (вокруг него вращает манипулятор на срезах).
  scanCentre(id) {
    const mesh = this.objects.get(id);
    return mesh ? [mesh.position.x, mesh.position.y, mesh.position.z] : null;
  }

  getTransform(id) {
    const mesh = this.objects.get(id);
    mesh.updateMatrix();
    const c = mesh.userData.centre;
    const T = mesh.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
    const e = T.clone().transpose().elements;
    return [0, 1, 2, 3].map((r) => Array.from(e.slice(r * 4, r * 4 + 4)));
  }

  // Карта отклонений по вершинам (RGB) или ровный цвет скана.
  setColors(id, rgb) {
    const mesh = this.objects.get(id);
    if (!mesh) return;
    if (rgb && rgb.length) {
      const colors = new Float32Array(rgb.length);
      for (let i = 0; i < rgb.length; i++) colors[i] = (rgb[i] / 255) ** 2.2;
      mesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      mesh.material.vertexColors = true;
      mesh.material.color.set('#ffffff');
    } else {
      mesh.material.vertexColors = false;
      mesh.material.color.set(mesh.userData.color);
    }
    mesh.material.needsUpdate = true;
  }

  // Ориентир: маленькая сфера поверх всего.
  setMarker(key, point, color) {
    const id = `lm:${key}`;
    let m = this.objects.get(id);
    if (!m) {
      m = new THREE.Mesh(new THREE.SphereGeometry(1.2, 20, 14),
        new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.95 }));
      m.renderOrder = 10;
      this.scene.add(m);
      this.objects.set(id, m);
    }
    m.material.color.set(color);
    m.position.set(...point);
  }

  setVisible(key, on) {
    const mesh = this.objects.get(key);
    if (mesh) mesh.visible = on;
  }

  setOpacity(key, opacity) {
    const mesh = this.objects.get(key);
    if (!mesh) return;
    mesh.material.opacity = opacity;
    mesh.material.transparent = opacity < 1;
    mesh.material.depthWrite = opacity >= 1;
    mesh.material.side = opacity < 1 ? THREE.FrontSide : THREE.DoubleSide;
    mesh.material.needsUpdate = true;
  }

  remove(key) {
    const mesh = this.objects.get(key);
    if (!mesh) return;
    if (this.gizmoTarget === key) this.detach();
    this.scene.remove(mesh);
    mesh.geometry.dispose();
    mesh.material.dispose();
    this.objects.delete(key);
  }

  attach(id, mode) {
    const mesh = this.objects.get(id);
    if (!mesh) return;
    this.gizmoTarget = id;
    this.gizmo.setMode(mode);
    this.gizmo.attach(mesh);
  }

  detach() {
    this.gizmoTarget = null;
    this.gizmo.detach();
  }

  // Точный сдвиг (мм) или поворот (°) скана вокруг его центра по оси пациента.
  nudge(id, axis, amount, rotate = false) {
    const mesh = this.objects.get(id);
    if (!mesh) return;
    const a = new THREE.Vector3().setComponent(axis, 1);
    if (rotate) mesh.rotateOnWorldAxis(a, THREE.MathUtils.degToRad(amount));
    else mesh.position.addScaledVector(a, amount);
  }

  fit(direction = 'front') {
    const box = new THREE.Box3();
    for (const mesh of this.objects.values()) if (mesh.visible) box.expandByObject(mesh);
    if (box.isEmpty()) return;
    const centre = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    const dir = new THREE.Vector3(...VIEWS[direction]).normalize();
    const dist = size / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2))) * 1.05;
    this.radius = size / 2;
    this.camera.position.copy(centre).addScaledVector(dir, dist);
    this.controls.target.copy(centre);
    this.clip();
    this.controls.update();
  }
}
