"""Камера фото по самому снимку: фокус и вертикаль (GeoCalib), расстояние до лица в мм (MoGe-2).

Совмещение фото со сканом (aesthetic.fit_camera) знает размеры зубов, но не перспективу: по зубам расстояние до
камеры и фокус порознь не видны (кейс 7: фокус 21–150 мм с одной ошибкой), поэтому расстояние было «1 м, как
exocad». Здесь — две нейросети, обе ONNX (tools/prepare_camera_models.py):

* **GeoCalib** (Veicht A. и др., ECCV 2024; код Apache-2.0, веса CC BY 4.0): сеть даёт поля «вверх» и «широта»
  с уверенностью; по ним — крен, наклон камеры и фокус, как у авторов (Левенберг — Марквардт с потерей Хьюбера,
  веса — уверенность; здесь — scipy.least_squares). Снимок — короткой стороной 320 px, края — кратно 32 (обрезка
  по центру). В блоке NMF сети начальный базис случайный — как у авторов; итог от запуска к запуску — в доли
  градуса.
* **MoGe-2 ViT-S** (Microsoft, MIT): карта точек с точностью до масштаба и сдвига, масштаб в метрах; фокус и
  сдвиг — по карте (как `recover_focal_shift_numpy` авторов: наименьшие квадраты на сетке 64×64), расстояние —
  глубина в заданной точке снимка (резцы).
"""

import os
from dataclasses import dataclass, field

import numpy as np

GEOCALIB_DIR, MOGE_DIR = "geocalib", "moge2-vits"
GEO_SHORT, GEO_MULTIPLE = 320, 32
GEO_HUBER = 1e-2  # масштаб потери Хьюбера (up_loss_fn_scale, lat_loss_fn_scale авторов)
MOGE_LONG = 960  # длинная сторона снимка для MoGe (сеть сама переводит на свою сетку)
MOGE_TOKENS = 3600  # как у авторов при resolution_level=9
MOGE_GRID = 64
FULL_FRAME_DIAGONAL_MM = 43.27


def geocalib_available(models_dir) -> bool:
    return bool(models_dir) and os.path.isfile(os.path.join(models_dir, GEOCALIB_DIR, "model.onnx"))


def moge_available(models_dir) -> bool:
    return bool(models_dir) and os.path.isfile(os.path.join(models_dir, MOGE_DIR, "model.onnx"))


def _session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _resize(image: np.ndarray, size) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(image[..., :3])).resize(size, Image.BILINEAR), np.float32) / 255.0


@dataclass
class Lens:
    """Камера снимка по нейросетям (пиксели полного снимка)."""

    size: tuple
    focal_px: float | None = None  # GeoCalib
    focal_sigma_px: float | None = None
    roll_deg: float | None = None  # крен: + — камера повёрнута по часовой (горизонт в кадре — под этим углом)
    pitch_deg: float | None = None  # наклон: + — камера смотрит вверх
    moge_focal_px: float | None = None
    distance_mm: float | None = None  # MoGe-2: глубина в точке point_px
    point_px: tuple | None = None
    notes: list = field(default_factory=list)

    @property
    def focal_35mm(self):
        return None if self.focal_px is None else self.focal_px * FULL_FRAME_DIAGONAL_MM / float(np.hypot(*self.size))


# --- GeoCalib -------------------------------------------------------------------------------------------------
def gravity_from_rp(roll, pitch):
    """Вектор «вверх» в осях камеры (x вправо, y вниз, z вперёд) по крену и наклону — как Gravity.from_rp авторов."""
    sr, cr, sp, cp = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)
    return np.array([-sr * cp, -cr * cp, sp])


def perspective_fields(roll, pitch, focal, w, h, step: int = 1):
    """Поля «вверх» (h, w, 2) и широта (h, w) пинхол-камеры с центром в середине снимка (каждый step-й пиксель)."""
    g = gravity_from_rp(roll, pitch)
    x, y = np.meshgrid(np.arange(0, w, step, dtype=float), np.arange(0, h, step, dtype=float))
    u, v = (x - w / 2) / focal, (y - h / 2) / focal
    up = np.stack([g[0] - g[2] * u, g[1] - g[2] * v], -1)
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-12)
    ray = np.stack([u, v, np.ones_like(u)], -1)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    lat = np.arcsin(np.clip(ray @ g, -1 + 1e-6, 1 - 1e-6))
    return up, lat


def geocalib_input(image: np.ndarray):
    """Снимок для сети: короткая сторона 320, края кратно 32 (обрезка по центру); (тензор, масштаб)."""
    H, W = image.shape[:2]
    k = GEO_SHORT / min(H, W)
    w, h = max(1, round(W * k)), max(1, round(H * k))
    img = _resize(image, (w, h))
    w2, h2 = (w // GEO_MULTIPLE) * GEO_MULTIPLE, (h // GEO_MULTIPLE) * GEO_MULTIPLE
    x0, y0 = (w - w2) // 2, (h - h2) // 2
    img = img[y0:y0 + h2, x0:x0 + w2]
    return img.transpose(2, 0, 1)[None].astype(np.float32), (w / W, h / H)


def calibrate_fields(up, up_conf, lat, lat_conf, step: int = 2):
    """Крен, наклон (рад) и фокус (px сети) по полям сети; (roll, pitch, focal, σ фокуса)."""
    from scipy.optimize import least_squares

    H, W = up_conf.shape
    up, up_conf = up[:, ::step, ::step].transpose(1, 2, 0), up_conf[::step, ::step]
    lat, lat_conf = lat[0, ::step, ::step], lat_conf[::step, ::step]
    sw_up, sw_lat = np.sqrt(up_conf).ravel(), np.sqrt(lat_conf).ravel()
    s_lat = np.sin(lat).ravel()

    def residual(x):
        pu, pl = perspective_fields(x[0], x[1], np.exp(x[2]), W, H, step)
        r_up = (up - pu).reshape(-1, 2) * sw_up[:, None]
        r_lat = (s_lat - np.sin(pl).ravel()) * sw_lat
        return np.r_[r_up.ravel(), r_lat]

    x0 = np.array([0.0, 0.0, np.log(0.7 * max(H, W))])
    sol = least_squares(residual, x0, loss="huber", f_scale=GEO_HUBER, x_scale=[0.05, 0.05, 0.1])
    sigma = None
    try:
        J = sol.jac
        cov = np.linalg.inv(J.T @ J) * max(2 * sol.cost / max(len(sol.fun) - 3, 1), 1e-12)
        sigma = float(np.exp(sol.x[2]) * np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        pass
    return float(sol.x[0]), float(sol.x[1]), float(np.exp(sol.x[2])), sigma


class GeoCalibOnnx:
    def __init__(self, models_dir):
        self.path = os.path.join(models_dir, GEOCALIB_DIR, "model.onnx")
        self._s = None

    def __call__(self, image: np.ndarray):
        """(roll, pitch, фокус в px полного снимка, σ фокуса)."""
        if self._s is None:
            self._s = _session(self.path)
        x, (sx, _sy) = geocalib_input(image)
        up, up_c, lat, lat_c = self._s.run(None, {"image": x})
        roll, pitch, f, sigma = calibrate_fields(up[0], up_c[0], lat[0], lat_c[0])
        return roll, pitch, f / sx, None if sigma is None else sigma / sx


# --- MoGe-2 ---------------------------------------------------------------------------------------------------
def view_plane_uv(width: int, height: int):
    """Нормированные координаты пикселей MoGe: углы — (±w/диагональ, ±h/диагональ)."""
    a = width / height
    sx, sy = a / (1 + a * a) ** 0.5, 1 / (1 + a * a) ** 0.5
    u = np.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width)
    v = np.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height)
    return np.stack(np.meshgrid(u, v, indexing="xy"), -1)


def recover_focal_shift(points: np.ndarray, mask: np.ndarray):
    """Фокус (в долях половины диагонали) и сдвиг глубины: min |f·xy/(z + s) − uv| (как у авторов)."""
    from scipy.optimize import least_squares

    h, w = points.shape[:2]
    uv = view_plane_uv(w, h)
    iy = np.linspace(0, h - 1, MOGE_GRID).round().astype(int)
    ix = np.linspace(0, w - 1, MOGE_GRID).round().astype(int)
    sel = mask[np.ix_(iy, ix)]
    P, UV = points[np.ix_(iy, ix)][sel], uv[np.ix_(iy, ix)][sel]
    if len(P) < 10:
        raise ValueError("MoGe: на снимке мало точек")
    xy, z = P[:, :2], P[:, 2]

    def fn(s):
        proj = xy / (z + s[0])[:, None]
        f = (proj * UV).sum() / np.square(proj).sum()
        return (f * proj - UV).ravel()

    s = float(least_squares(fn, x0=[0.0], ftol=1e-3, method="lm").x[0])
    proj = xy / (z + s)[:, None]
    return float((proj * UV).sum() / np.square(proj).sum()), s


class MoGeOnnx:
    def __init__(self, models_dir):
        self.path = os.path.join(models_dir, MOGE_DIR, "model.onnx")
        self._s = None

    def __call__(self, image: np.ndarray, point_px=None):
        """(фокус в px полного снимка, глубина в мм в точке point_px — или в середине кадра)."""
        if self._s is None:
            self._s = _session(self.path)
        H, W = image.shape[:2]
        k = min(1.0, MOGE_LONG / max(H, W))
        w, h = max(1, round(W * k)), max(1, round(H * k))
        x = _resize(image, (w, h)).transpose(2, 0, 1)[None].astype(np.float32)
        points, _normal, mask, scale = self._s.run(None, {"image": x, "num_tokens": np.array(MOGE_TOKENS, np.int64)})
        points, mask, scale = points[0], mask[0] > 0.5, float(np.ravel(scale)[0])
        focal, shift = recover_focal_shift(points, mask)
        depth = (points[..., 2] + shift) * scale  # метры
        u, v = (W / 2, H / 2) if point_px is None else point_px
        j, i = int(np.clip(u * k, 0, w - 1)), int(np.clip(v * k, 0, h - 1))
        r = max(2, round(0.01 * max(w, h)))
        win = depth[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
        ok = mask[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1] & np.isfinite(win) & (win > 0)
        d = float(np.median(win[ok])) if ok.any() else float("nan")
        # фокус MoGe — в долях половины диагонали
        return focal * np.hypot(W, H) / 2, d * 1000.0


def estimate(image: np.ndarray, models_dir, point_px=None) -> Lens:
    """Камера снимка: что смогли нейросети (нет модели — поле пустое, причина — в notes)."""
    H, W = image.shape[:2]
    lens = Lens(size=(W, H), point_px=None if point_px is None else tuple(float(c) for c in point_px))
    if geocalib_available(models_dir):
        r, p, f, s = GeoCalibOnnx(models_dir)(image)
        lens.roll_deg, lens.pitch_deg, lens.focal_px, lens.focal_sigma_px = float(np.degrees(r)), float(np.degrees(p)), f, s
    else:
        lens.notes.append("нет GeoCalib (models/geocalib)")
    if moge_available(models_dir):
        f, d = MoGeOnnx(models_dir)(image, point_px)
        lens.moge_focal_px = f
        lens.distance_mm = d if np.isfinite(d) else None
    else:
        lens.notes.append("нет MoGe-2 (models/moge2-vits)")
    return lens
