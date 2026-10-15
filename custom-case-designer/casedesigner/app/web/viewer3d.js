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
    // Видимость: wanted — что показывает шаг программы, userHidden — что скрыл врач в панели объектов (на всех шагах);
    // на экране — пересечение. userOpacity — прозрачность из панели: переживает пересоздание сетки.
    this.wanted = new Map();
    this.userHidden = new Set();
    this.userOpacity = new Map();
    this.onObjects = null; // панель объектов: сетки добавились, убрались или сменили видимость
    this.raycaster = new THREE.Raycaster();
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

    // Мышь как в exocad: правая — вращение, левая + правая — сдвиг, щелчок колесом — центр вращения в точку под
    // курсором (нажатое колесо с движением — тоже сдвиг), колесо — масштаб к точке под курсором. Левая одна — для
    // точек и манипулятора. Вращение и масштаб — OrbitControls, сдвиг и центр — navigation().
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false; // без инерции: отпустили кнопку — вид стоит (решение владельца 15.10)
    this.controls.rotateSpeed = 0.8;
    this.controls.screenSpacePanning = true;
    this.controls.zoomToCursor = true;
    this.controls.mouseButtons = { LEFT: null, MIDDLE: null, RIGHT: THREE.MOUSE.ROTATE };
    this.radius = 100; // размер сцены, мм (fit)
    this.navigation(el);

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

  // Сдвиг (левая + правая или нажатое колесо) и центр вращения (щелчок колесом). Слушаем на окне в фазе захвата —
  // раньше OrbitControls: пока идёт сдвиг, его движения мыши до OrbitControls не доходят (иначе вращал бы).
  navigation(el) {
    let nav = null; // { x, y, mode: 'pan' | 'middle', moved }
    const onCanvas = (e) => e.target === this.renderer.domElement;
    el.addEventListener('pointerdown', (e) => {
      if (!onCanvas(e) || e.pointerType !== 'mouse' || this.gizmo.dragging) return;
      if (e.button === 1) {
        e.preventDefault(); // без автопрокрутки браузера
        nav = { x: e.clientX, y: e.clientY, mode: 'middle', moved: false };
      }
    }, true);
    el.addEventListener('pointermove', (e) => {
      if (e.pointerType !== 'mouse' || this.gizmo.dragging) return;
      const chord = (e.buttons & 3) === 3; // левая и правая вместе
      if (chord && nav?.mode !== 'pan') nav = { x: e.clientX, y: e.clientY, mode: 'pan', moved: false };
      if (!nav) return;
      e.stopPropagation();
      if (nav.mode === 'pan' && !chord) { // отпустили одну из кнопок
        if (e.buttons === 2) { // осталась правая — дальше вращение, без скачка
          this.controls._rotateStart?.set(e.clientX, e.clientY);
          nav = null;
        } else {
          nav.x = e.clientX; // осталась левая — ждём: нажмут правую снова — сдвиг отсюда
          nav.y = e.clientY;
        }
        return;
      }
      const dx = e.clientX - nav.x, dy = e.clientY - nav.y;
      if (!nav.moved && Math.hypot(dx, dy) < 3) return;
      nav.moved = true;
      nav.x = e.clientX;
      nav.y = e.clientY;
      this.pan(dx, dy);
    }, true);
    el.addEventListener('pointerup', (e) => {
      if (nav?.mode === 'middle' && e.button === 1 && !nav.moved && onCanvas(e)) this.centerAt(e.clientX, e.clientY);
      if (!e.buttons) nav = null;
    }, true);
    el.addEventListener('auxclick', (e) => { if (e.button === 1) e.preventDefault(); });
  }

  // Сдвиг вида на (dx, dy) пикселей экрана: камера и центр вращения вместе, точка под курсором идёт за мышью.
  pan(dx, dy) {
    const h = this.renderer.domElement.clientHeight || 1;
    const dist = this.camera.position.distanceTo(this.controls.target) * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
    const move = right.multiplyScalar(-2 * dx * dist / h).addScaledVector(up, 2 * dy * dist / h);
    this.camera.position.add(move);
    this.controls.target.add(move);
  }

  // Видимая поверхность под курсором (координаты сцены) или null.
  hitAt(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const mouse = new THREE.Vector2(2 * (clientX - rect.left) / rect.width - 1, 1 - 2 * (clientY - rect.top) / rect.height);
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(mouse, this.camera);
    const meshes = [...this.objects.values()].filter((m) => m.visible && m.material?.opacity > 0.05);
    return this.raycaster.intersectObjects(meshes, false)[0]?.point ?? null;
  }

  // Щелчок колесом: точка под курсором — сразу в центр окна и центр вращения (без переезда), расстояние то же.
  centerAt(clientX, clientY) {
    const hit = this.hitAt(clientX, clientY);
    if (!hit) return false;
    const shift = hit.clone().sub(this.controls.target);
    this.camera.position.add(shift);
    this.controls.target.add(shift);
    return true;
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
    const mesh = new THREE.Mesh(g, this.material(color, this.userOpacity.get(key) ?? opacity));
    mesh.renderOrder = order;
    this.scene.add(mesh);
    this.objects.set(key, mesh);
    this.show(key, mesh, true);
    return mesh;
  }

  // Видимость, которую задаёт программа; скрытое врачом в панели объектов остаётся скрытым.
  show(key, mesh, on) {
    this.wanted.set(key, on);
    mesh.visible = on && !this.userHidden.has(key);
    this.objectsChanged();
  }

  objectsChanged() {
    clearTimeout(this.objectsTimer);
    this.objectsTimer = setTimeout(() => this.onObjects?.(), 30);
  }

  // Панель объектов: скрыть или показать объект на всех шагах.
  setUserHidden(key, hidden) {
    if (hidden) this.userHidden.add(key); else this.userHidden.delete(key);
    const mesh = this.objects.get(key);
    if (mesh) mesh.visible = (this.wanted.get(key) ?? true) && !hidden;
    this.objectsChanged();
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
    this.show(id, mesh, !!transform);
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
    if (!this.wanted.get(id)) this.show(id, mesh, true);
  }

  // Матрица движения для структуры, чья геометрия уже лежит в координатах КТ.
  setObjectTransform(key, transform) {
    const mesh = this.objects.get(key);
    if (!mesh) return;
    const M = new THREE.Matrix4().set(...transform.flat());
    M.decompose(mesh.position, mesh.quaternion, mesh.scale);
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

  setPath(key, points, color = '#f5b84b') {
    this.remove(key);
    if (!points || points.length < 2) return;
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(...p)));
    const line = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }));
    line.renderOrder = 9;
    this.scene.add(line);
    this.objects.set(key, line);
  }

  // Точка на видимой сетке в мировых координатах текущего 3D-вида.
  pick(key, clientX, clientY) {
    const object = this.objects.get(key);
    if (!object?.visible) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const mouse = new THREE.Vector2(
      2 * (clientX - rect.left) / rect.width - 1,
      1 - 2 * (clientY - rect.top) / rect.height,
    );
    this.raycaster.setFromCamera(mouse, this.camera);
    const hit = this.raycaster.intersectObject(object, true)[0];
    return hit ? hit.point.toArray() : null;
  }

  setVisible(key, on) {
    const mesh = this.objects.get(key);
    if (mesh) this.show(key, mesh, on);
  }

  setOpacity(key, opacity) {
    this.userOpacity.set(key, opacity);
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
    this.wanted.delete(key);
    this.objectsChanged();
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
