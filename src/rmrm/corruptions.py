"""Controlled visual corruptions for driving video streams.

Design principles

* Formulas follow the spirit of ImageNet-C (Hendrycks & Dietterich, 2019) but are
  re-implemented with numpy/OpenCV so they work at arbitrary resolution and are
  *temporally coherent*: spatial patterns that would physically persist in a video
  (fog density field, occluder position, lens droplets) are sampled ONCE per stream
  (``stream_seed``) and reused for every post-change frame; per-frame stochastic
  components (sensor noise, rain streak positions, snow flakes) are resampled per
  frame with a seed derived from (stream_seed, frame_index) so results are exactly
  reproducible.
* Three severities (1=mild, 2=moderate, 3=severe). Parameters were fixed a priori
  from ImageNet-C levels {1,3,5} (roughly) before any detector was run and are NOT
  tuned to favour any method.  They are recorded in SEVERITY_TABLE for the paper.

All functions take and return uint8 RGB arrays of shape (H, W, 3).
"""
from __future__ import annotations

import numpy as np
import cv2

CORRUPTIONS = [
    "gaussian_noise", "motion_blur", "defocus_blur", "fog", "low_light",
    "brightness", "contrast", "occlusion", "rain", "snow",
]

# a-priori severity parameters (mild, moderate, severe)
SEVERITY_TABLE = {
    "gaussian_noise": [0.04, 0.08, 0.14],          # sigma on [0,1] scale
    "motion_blur": [(7, 0), (13, 0), (21, 0)],       # kernel length px @ 640 width, angle deg (+/- jitter)
    "defocus_blur": [2, 4, 7],                       # disk radius px @ 640 width
    "fog": [(0.35, 2.0), (0.6, 1.7), (0.85, 1.4)],   # (max opacity, fractal decay)
    "low_light": [(0.55, 0.02), (0.35, 0.035), (0.2, 0.05)],  # (gain after gamma 2.2 darkening, noise sigma)
    "brightness": [0.15, 0.3, 0.45],                 # add to V channel in HSV [0,1]
    "contrast": [0.6, 0.4, 0.25],                    # contrast factor around mean
    "occlusion": [0.04, 0.10, 0.20],                 # fraction of image area occluded (lens smudge blobs)
    "rain": [(250, 0.25), (600, 0.45), (1200, 0.65)],  # (streaks per frame @640x360, blend + haze)
    "snow": [(0.3, 0.25), (0.55, 0.4), (0.8, 0.6)],  # (flake density, whitening)
}


def _rng(stream_seed: int, frame_idx: int | None = None) -> np.random.Generator:
    if frame_idx is None:
        return np.random.default_rng([int(stream_seed), 7])
    return np.random.default_rng([int(stream_seed), int(frame_idx), 11])


def _scale(img: np.ndarray) -> float:
    """Resolution factor relative to 640 px width (so severities are resolution-invariant)."""
    return img.shape[1] / 640.0


def _plasma(h: int, w: int, decay: float, rng: np.random.Generator) -> np.ndarray:
    """Spectral-synthesis fractal noise in [0,1] (fog density field)."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1.0
    amp = 1.0 / f ** decay
    amp[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=amp.shape)
    spec = amp * np.exp(1j * phase)
    field = np.fft.irfft2(spec, s=(h, w))
    field = (field - field.min()) / (field.max() - field.min() + 1e-8)
    return field.astype(np.float32)


def apply_corruption(img: np.ndarray, name: str, severity: int, stream_seed: int,
                     frame_idx: int) -> np.ndarray:
    assert 1 <= severity <= 3, severity
    p = SEVERITY_TABLE[name][severity - 1]
    x = img.astype(np.float32) / 255.0
    h, w = x.shape[:2]
    s = _scale(img)
    fr = _rng(stream_seed, frame_idx)
    st = _rng(stream_seed)

    if name == "gaussian_noise":
        x = x + fr.normal(0, p, size=x.shape).astype(np.float32)

    elif name == "motion_blur":
        length = max(3, int(round(p[0] * s)))
        angle = st.uniform(-15, 15)  # fixed per stream (camera shake direction)
        k = np.zeros((length, length), np.float32)
        k[length // 2, :] = 1.0
        M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        k = cv2.warpAffine(k, M, (length, length))
        k /= k.sum() + 1e-8
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    elif name == "defocus_blur":
        r = max(1, int(round(p * s)))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        k = ((xx ** 2 + yy ** 2) <= r ** 2).astype(np.float32)
        k /= k.sum()
        x = cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

    elif name == "fog":
        opacity, decay = p
        field = _plasma(h, w, decay, st)
        # denser fog towards the horizon/top half, as in real driving scenes
        vert = np.linspace(1.0, 0.6, h, dtype=np.float32)[:, None]
        a = np.clip(opacity * (0.5 + 0.5 * field) * vert, 0, 1)[..., None]
        airlight = 0.8
        x = x * (1 - a) + airlight * a

    elif name == "low_light":
        gain, sigma = p
        x = (x ** 2.2) * gain  # darken in linear space
        x = x ** (1 / 2.2)
        x = x * 0.9 + 0.1 * x.mean(axis=2, keepdims=True)  # mild desaturation
        x = x + fr.normal(0, sigma, size=x.shape).astype(np.float32)

    elif name == "brightness":
        hsv = cv2.cvtColor(np.clip(x, 0, 1), cv2.COLOR_RGB2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] + p, 0, 1)
        x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    elif name == "contrast":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = (x - m) * p + m

    elif name == "occlusion":
        # soft opaque blobs (mud / smudge on the lens), fixed across frames
        area = p * h * w
        mask = np.zeros((h, w), np.float32)
        n_blobs = int(st.integers(1, 4))
        for b in range(n_blobs):
            a_b = area / n_blobs
            ry = np.sqrt(a_b / np.pi) * st.uniform(0.7, 1.3)
            rx = a_b / (np.pi * ry)
            cy, cx = st.uniform(0.15, 0.85) * h, st.uniform(0.1, 0.9) * w
            cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), st.uniform(0, 180), 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 3 * s)[..., None]
        color = np.array([0.18, 0.15, 0.12], np.float32)
        x = x * (1 - mask) + color * mask

    elif name == "rain":
        n, blend = p
        n = int(n * (h * w) / (640 * 360))
        layer = np.zeros((h, w), np.float32)
        ang = st.uniform(-20, 20)
        L = int(18 * s)
        dx, dy = L * np.sin(np.deg2rad(ang)), L * np.cos(np.deg2rad(ang))
        ys = fr.uniform(0, h, n)
        xs = fr.uniform(0, w, n)
        for yy, xx in zip(ys, xs):
            cv2.line(layer, (int(xx), int(yy)), (int(xx + dx), int(yy + dy)), float(fr.uniform(0.5, 1.0)), 1)
        layer = cv2.GaussianBlur(layer, (3, 3), 0)[..., None]
        x = x * (1 - 0.35 * blend) + 0.35 * blend * 0.7  # haze + reduced contrast of wet scene
        x = np.maximum(x, layer * 0.85 * blend + x * (1 - layer * blend))
        x = cv2.GaussianBlur(x, (0, 0), 0.6 * s)  # slight droplet blur

    elif name == "snow":
        dens, white = p
        n = int(dens * 1500 * (h * w) / (640 * 360))
        layer = np.zeros((h, w), np.float32)
        ys = fr.uniform(0, h, n)
        xs = fr.uniform(0, w, n)
        rs = fr.uniform(0.6, 2.2, n) * s
        for yy, xx, rr in zip(ys, xs, rs):
            cv2.circle(layer, (int(xx), int(yy)), max(1, int(rr)), 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.8 * s)[..., None]
        gray = x.mean(axis=2, keepdims=True)
        x = (1 - white) * x + white * np.maximum(x, gray * 1.5 + 0.1)  # snow cover brightening
        x = np.maximum(x, layer)
    else:
        raise ValueError(name)

    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
