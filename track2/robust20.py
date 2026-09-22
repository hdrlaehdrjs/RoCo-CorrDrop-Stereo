"""RobustSpring-style 20-corruption synthetic validation suite (held-out Spring validation scenes).

The 20 corruption names are exactly those of the official RobustSpring benchmark. RobustSpring's
generation code is not available locally, so the corruptions follow the ImageNet-C definitions
(Hendrycks & Dietterich 2019) as packaged by `imagecorruptions` 1.1.2, at SEVERITY 3, re-implemented
here for numpy 2 / scikit-image 0.26 (that package's own code no longer runs). RobustSpring-inspired
changes, all documented per function:
  * fog is depth-aware: transmission from each view's OWN GT disparity (disp1_left / disp1_right),
    modulated by the ImageNet-C plasma fractal;
  * rain uses the Track 3 streak overlay (ImageNet-C has no rain);
  * glass_blur's per-pixel sequential swap loop is replaced by a vectorised random local
    displacement (same sigma / max_delta / iterations), zoom_blur uses cv2 bilinear zooms.
Each view is corrupted with its own seed (independent sensor realisation). Nothing here was
calibrated against RobustSpring test images. This is a local proxy, not the official benchmark.
"""
from __future__ import annotations

import math
import random
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from skimage import color as skcolor
from skimage.filters import gaussian

from . import q4_primitives as q4

SEVERITY = 3
CORRUPTIONS20 = ("brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur",
                 "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
                 "rain", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur")
GROUPS20 = {"brightness": "color", "contrast": "color", "saturate": "color",
            "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise", "speckle_noise": "noise",
            "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur", "gaussian_blur": "blur",
            "fog": "weather", "frost": "weather", "snow": "weather", "spatter": "weather", "rain": "weather",
            "elastic_transform": "digital", "pixelate": "digital", "jpeg_compression": "digital"}
FROST_DIR = Path("/opt/conda/lib/python3.11/site-packages/imagecorruptions/frost")
S = SEVERITY - 1


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- helpers (ImageNet-C)
def _disk(radius, alias_blur=0.1):
    L = np.arange(-8, 9) if radius <= 8 else np.arange(-radius, radius + 1)
    ksize = (3, 3) if radius <= 8 else (5, 5)
    X, Y = np.meshgrid(L, L)
    d = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32)
    d /= d.sum()
    return cv2.GaussianBlur(d, ksize=ksize, sigmaX=alias_blur)


def _plasma_fractal(rs, mapsize=256, wibbledecay=3):
    m = np.empty((mapsize, mapsize), dtype=np.float64)
    m[0, 0] = 0
    step, wibble = mapsize, 100.0

    def wmean(a):
        return a / 4 + wibble * rs.uniform(-wibble, wibble, a.shape)

    while step >= 2:
        c = m[0:mapsize:step, 0:mapsize:step]
        sq = c + np.roll(c, -1, axis=0)
        sq += np.roll(sq, -1, axis=1)
        m[step // 2:mapsize:step, step // 2:mapsize:step] = wmean(sq)
        dr = m[step // 2:mapsize:step, step // 2:mapsize:step]
        ul = m[0:mapsize:step, 0:mapsize:step]
        m[0:mapsize:step, step // 2:mapsize:step] = wmean(dr + np.roll(dr, 1, axis=0) + ul + np.roll(ul, -1, axis=1))
        m[step // 2:mapsize:step, 0:mapsize:step] = wmean(dr + np.roll(dr, 1, axis=1) + ul + np.roll(ul, -1, axis=0))
        step //= 2
        wibble /= wibbledecay
    m -= m.min()
    return m / m.max()


def _clipped_zoom(img, z):
    h, w = img.shape[:2]
    ch, cw = int(np.ceil(h / z)), int(np.ceil(w / z))
    t, l = (h - ch) // 2, (w - cw) // 2
    out = cv2.resize(img[t:t + ch, l:l + cw], (int(round(cw * z)), int(round(ch * z))), interpolation=cv2.INTER_LINEAR)
    if out.ndim == 2 and img.ndim == 3:
        out = out[..., None]
    return out[:h, :w]


def _shift(image, dx, dy):
    if dx < 0:
        s = np.roll(image, shift=image.shape[1] + dx, axis=1); s[:, dx:] = s[:, dx - 1:dx]
    elif dx > 0:
        s = np.roll(image, shift=dx, axis=1); s[:, :dx] = s[:, dx:dx + 1]
    else:
        s = image
    if dy < 0:
        s = np.roll(s, shift=image.shape[0] + dy, axis=0); s[dy:, :] = s[dy - 1:dy, :]
    elif dy > 0:
        s = np.roll(s, shift=dy, axis=0); s[:dy, :] = s[dy:dy + 1, :]
    return s


def _motion_blur(x, radius, sigma, angle):
    width = radius * 2 + 1
    k = np.exp(-np.arange(width) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    k /= k.sum()
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hyp = math.hypot(*point)
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hyp) - 0.5)
        dx = -math.ceil(((i * point[1]) / hyp) - 0.5)
        if abs(dy) >= x.shape[0] or abs(dx) >= x.shape[1]:
            break
        out = out + k[i] * _shift(x, dx, dy)
    return out


# ---------------------------------------------------------------- corruptions (uint8 HWC RGB in/out)
def gaussian_noise(x, rs, sev=SEVERITY, **_):
    c = [.08, .12, 0.18, 0.26, 0.38][sev - 1]
    return _u8(np.clip(x / 255. + rs.normal(size=x.shape, scale=c), 0, 1) * 255)


def shot_noise(x, rs, sev=SEVERITY, **_):
    c = [60, 25, 12, 5, 3][sev - 1]
    return _u8(np.clip(rs.poisson(x / 255. * c) / float(c), 0, 1) * 255)


def impulse_noise(x, rs, sev=SEVERITY, **_):
    c = [.03, .06, .09, 0.17, 0.27][sev - 1]
    y = x / 255.
    u = rs.random(x.shape)  # skimage 's&p': flip ~amount of values, half salt half pepper
    y = np.where(u < c / 2, 0.0, np.where(u < c, 1.0, y))
    return _u8(y * 255)


def speckle_noise(x, rs, sev=SEVERITY, **_):
    c = [.15, .2, 0.35, 0.45, 0.6][sev - 1]
    y = x / 255.
    return _u8(np.clip(y + y * rs.normal(size=x.shape, scale=c), 0, 1) * 255)


def gaussian_blur(x, rs, sev=SEVERITY, **_):
    c = [1, 2, 3, 4, 6][sev - 1]
    return _u8(np.clip(gaussian(x / 255., sigma=c, channel_axis=-1), 0, 1) * 255)


def glass_blur(x, rs, sev=SEVERITY, **_):
    sigma, delta, iters = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][sev - 1]
    y = _u8(gaussian(x / 255., sigma=sigma, channel_axis=-1) * 255)
    h, w = y.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    for _ in range(iters):  # vectorised local shuffle (see module doc)
        dy = rs.integers(-delta, delta, size=(h, w))
        dx = rs.integers(-delta, delta, size=(h, w))
        y = y[np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)]
    return _u8(np.clip(gaussian(y / 255., sigma=sigma, channel_axis=-1), 0, 1) * 255)


def defocus_blur(x, rs, sev=SEVERITY, **_):
    r, a = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][sev - 1]
    k = _disk(r, a)
    y = (x / 255.).astype(np.float32)
    return _u8(np.clip(np.stack([cv2.filter2D(y[..., d], -1, k) for d in range(3)], -1), 0, 1) * 255)


def motion_blur(x, rs, sev=SEVERITY, **_):
    r, s = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][sev - 1]
    return _u8(_motion_blur(x.astype(np.float32), r, s, rs.uniform(-45, 45)))


def zoom_blur(x, rs, sev=SEVERITY, **_):
    zs = [np.arange(1, 1.11, 0.01), np.arange(1, 1.16, 0.01), np.arange(1, 1.21, 0.02),
          np.arange(1, 1.26, 0.02), np.arange(1, 1.31, 0.03)][sev - 1]
    y = (x / 255.).astype(np.float32)
    out = np.zeros_like(y)
    for z in zs:
        out += _clipped_zoom(y, z)
    return _u8(np.clip((y + out) / (len(zs) + 1), 0, 1) * 255)


def fog(x, rs, disp=None, sev=SEVERITY, **_):
    """Depth-aware fog: t = exp(-beta * z / z_med), z = 1 / disparity (sky / d<=0 -> t = 0), light A
    modulated by the ImageNet-C plasma fractal (severity-3 strength 2.5, decay 1.7)."""
    strength, decay = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][sev - 1]
    h, w = x.shape[:2]
    frac = _plasma_fractal(rs, mapsize=2 ** math.ceil(math.log2(max(h, w))), wibbledecay=decay)[:h, :w]
    y = x / 255.
    d = np.nan_to_num(disp, nan=0.0) if disp is not None else np.ones((h, w), np.float32)
    z = np.where(d > 1e-3, 1.0 / np.maximum(d, 1e-3), np.inf)
    zmed = np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 1.0
    beta = strength / 2.5 * 1.2          # severity 3 -> t = exp(-1.2) = 0.30 at the median depth
    t = np.exp(-beta * np.where(np.isfinite(z), z / zmed, 1e6))[..., None]
    A = (0.55 + 0.4 * frac)[..., None]   # bright, spatially varying airlight
    return _u8(np.clip(y * t + A * (1 - t), 0, 1) * 255)


def frost(x, rs, sev=SEVERITY, **_):
    a, b = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][sev - 1]
    files = ["frost1.png", "frost2.png", "frost3.png", "frost4.jpg", "frost5.jpg", "frost6.jpg"]
    f = cv2.imread(str(FROST_DIR / files[int(rs.integers(5))]))
    fh, fw = f.shape[:2]
    h, w = x.shape[:2]
    scale = max(h / fh, w / fw, 1.0) * 1.1
    f = cv2.resize(f, (int(np.ceil(fw * scale)), int(np.ceil(fh * scale))), interpolation=cv2.INTER_CUBIC)
    ys, xs = int(rs.integers(0, f.shape[0] - h)), int(rs.integers(0, f.shape[1] - w))
    f = f[ys:ys + h, xs:xs + w][..., ::-1]
    return _u8(a * x.astype(np.float32) + b * f)


def snow(x, rs, sev=SEVERITY, **_):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.7), (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65), (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][sev - 1]
    y = x.astype(np.float32) / 255.
    layer = rs.normal(size=y.shape[:2], loc=c[0], scale=c[1]).astype(np.float32)
    layer = _clipped_zoom(layer, c[2])
    layer[layer < c[3]] = 0
    layer = np.clip(layer, 0, 1)
    layer = _motion_blur(layer, c[4], c[5], rs.uniform(-135, -45))
    layer = (np.round(layer * 255).astype(np.uint8) / 255.)[..., None]
    y = c[6] * y + (1 - c[6]) * np.maximum(y, cv2.cvtColor(y, cv2.COLOR_RGB2GRAY)[..., None] * 1.5 + 0.5)
    return _u8(np.clip(y + layer + np.rot90(layer, k=2), 0, 1) * 255)


def spatter(x, rs, sev=SEVERITY, **_):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0), (0.65, 0.3, 3, 0.68, 0.6, 0), (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1), (0.67, 0.4, 1, 0.65, 1.5, 1)][sev - 1]
    y = x.astype(np.float32) / 255.
    liquid = gaussian(rs.normal(size=y.shape[:2], loc=c[0], scale=c[1]), sigma=c[2])
    liquid[liquid < c[3]] = 0
    # severity 3 uses the water branch (c[5] == 0); the image is RGB here, so colours are given in RGB
    lq = (liquid * 255).astype(np.uint8)
    dist = 255 - cv2.Canny(lq, 50, 150)
    dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
    _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
    dist = cv2.equalizeHist(cv2.blur(dist, (3, 3)).astype(np.uint8))
    dist = cv2.blur(cv2.filter2D(dist, cv2.CV_8U, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])), (3, 3)).astype(np.float32)
    m = (liquid * dist).astype(np.float32)
    m = m / max(float(m.max()), 1e-6) * c[4]
    water = np.array([238, 238, 175], np.float32) / 255.  # pale turquoise (BGR 175,238,238 in the original)
    return _u8(np.clip(y + m[..., None] * water, 0, 1) * 255)


def contrast(x, rs, sev=SEVERITY, **_):
    c = [0.4, .3, .2, .1, .05][sev - 1]
    y = x / 255.
    mu = y.mean(axis=(0, 1), keepdims=True)
    return _u8(np.clip((y - mu) * c + mu, 0, 1) * 255)


def brightness(x, rs, sev=SEVERITY, **_):
    c = [.1, .2, .3, .4, .5][sev - 1]
    hsv = skcolor.rgb2hsv(x / 255.)
    hsv[..., 2] = np.clip(hsv[..., 2] + c, 0, 1)
    return _u8(np.clip(skcolor.hsv2rgb(hsv), 0, 1) * 255)


def saturate(x, rs, sev=SEVERITY, **_):
    a, b = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][sev - 1]
    hsv = skcolor.rgb2hsv(x / 255.)
    hsv[..., 1] = np.clip(hsv[..., 1] * a + b, 0, 1)
    return _u8(np.clip(skcolor.hsv2rgb(hsv), 0, 1) * 255)


def jpeg_compression(x, rs, sev=SEVERITY, **_):
    q = [25, 18, 15, 10, 7][sev - 1]
    buf = BytesIO()
    Image.fromarray(x).save(buf, "JPEG", quality=q)
    return np.asarray(Image.open(buf).convert("RGB")).copy()


def pixelate(x, rs, sev=SEVERITY, **_):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][sev - 1]
    h, w = x.shape[:2]
    im = Image.fromarray(x).resize((int(w * c), int(h * c)), Image.BOX).resize((w, h), Image.NEAREST)
    return np.asarray(im).copy()


def elastic_transform(x, rs, sev=SEVERITY, **_):
    y = x.astype(np.float32) / 255.
    h, w = y.shape[:2]
    sigma = np.array((h, w)) * 0.01
    alpha = [250 * 0.05, 250 * 0.065, 250 * 0.085, 250 * 0.1, 250 * 0.12][sev - 1]
    mx = h * 0.005
    dx = (gaussian(rs.uniform(-mx, mx, size=(h, w)), sigma, mode="reflect", truncate=3) * alpha).astype(np.float32)
    dy = (gaussian(rs.uniform(-mx, mx, size=(h, w)), sigma, mode="reflect", truncate=3) * alpha).astype(np.float32)
    gy, gx = np.mgrid[:h, :w].astype(np.float32)
    out = np.stack([map_coordinates(y[..., ch], (gy + dy, gx + dx), order=1, mode="reflect") for ch in range(3)], -1)
    return _u8(np.clip(out, 0, 1) * 255)


def rain(x, rs, sev=SEVERITY, **_):
    """Track 3 streak overlay (corresguard_core.weather_mask 'rain' + white overlay), severity 0.5."""
    seed = int(rs.integers(2**31))
    return q4._overlay(x, q4.weather_mask(x.shape[:2], "rain", 0.5, random.Random(seed), np.random.default_rng(seed)),
                       np.random.default_rng(seed + 1), white=True)


FUNCS = {n: globals()[n] for n in CORRUPTIONS20}


def seed_for(ci: int, fi: int, view: int) -> int:
    return 20260917 + 100000 * (ci + 1) + 10 * fi + view


def apply(name: str, img: np.ndarray, seed: int, disp: np.ndarray | None = None, severity: int = SEVERITY) -> np.ndarray:
    out = FUNCS[name](img, np.random.default_rng(seed), disp=disp, sev=int(severity))
    assert out.shape == img.shape and out.dtype == np.uint8, (name, out.shape, out.dtype)
    return out
