// Связь с локальным сервером приложения.

export async function get(path) {
  const r = await fetch(`/api/${path}`);
  return answer(r);
}

export async function post(path, body = {}) {
  const r = await fetch(`/api/${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  return answer(r);
}

async function answer(r) {
  const type = r.headers.get('Content-Type') || '';
  if (type.includes('application/json')) {
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || `ошибка ${r.status}`);
    return data;
  }
  if (!r.ok) throw new Error(`ошибка ${r.status}`);
  return r.arrayBuffer();
}

// Долгая операция: сервер отвечает номером задачи, ждём её, показывая прогресс.
export async function run(path, body, onProgress) {
  const { job } = await post(path, body);
  for (;;) {
    await new Promise((ok) => setTimeout(ok, 250));
    const state = await get(`jobs/${job}`);
    onProgress?.(state);
    if (state.status === 'done') return state.result;
    if (state.status === 'error') throw new Error(state.error);
  }
}

// Сетка: [число вершин, число граней] uint32, вершины float32, грани uint32.
export async function mesh(path) {
  const buf = await get(path);
  const head = new Uint32Array(buf, 0, 2);
  const vertices = new Float32Array(buf, 8, head[0] * 3);
  const faces = new Uint32Array(buf, 8 + head[0] * 12, head[1] * 3);
  return { vertices, faces };
}
