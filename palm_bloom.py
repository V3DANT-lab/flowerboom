"""
Palm Bloom 3D — Fast & Stable  (real-time, on-camera flower)

Install:  pip install -r requirements.txt
Run:      python palm_bloom.py

Massive performance + stability overhaul:
  • Petals pre-rendered ONCE as sprites (no per-frame HSV math)
  • Sprite cache keyed by (angle_bucket, size_bucket)
  • Small-ROI blur instead of full-frame blur
  • One Euro Filter on hand position (kills jitter without lag)
  • Finger-count hysteresis (no flicker at boundaries)
  • Camera at 640x480 for speed; window scales it up
"""

import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import random
import sys

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
SMOOTH = 0.25                  # bloom responsiveness
CAM_W, CAM_H = 640, 480        # smaller = faster. Flower is drawn upscaled.
CAM_FPS = 30
DISPLAY_W, DISPLAY_H = 1280, 720   # window size (frame gets upscaled)

IRIUN_INDEX = 0
LAPTOP_INDEX = 1

# Flower
NUM_PETALS = 9
PETAL_LEN_RATIO = 0.95
MIN_PETAL_SCALE = 0.25
BASE_PETAL_W_RATIO = 0.32

# Filters
POS_MIN_CUTOFF = 1.2           # lower = smoother but laggier
POS_BETA = 0.04                # higher = less lag, more jitter
FINGER_HYSTERESIS = 0.35       # threshold gap to change finger count

# Sprite cache
ANGLE_BUCKETS = 48             # cache petal at 48 angular orientations
SIZE_BUCKETS = 24              # cache at 24 size levels

# MediaPipe landmark indices
TIP_IDX = [4, 8, 12, 16, 20]
PIP_IDX = [3, 6, 10, 14, 18]
WRIST = 0
MIDDLE_MCP = 9

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ──────────────────────────────────────────────────────────────
# ONE EURO FILTER (kills jitter without adding lag)
# ──────────────────────────────────────────────────────────────

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


# ──────────────────────────────────────────────────────────────
# CAMERA
# ──────────────────────────────────────────────────────────────

def _probe_camera(index, tries=10):
    if not os.path.exists(f"/dev/video{index}"):
        return None, False
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None, False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimize latency
    for _ in range(tries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return cap, True
        time.sleep(0.05)
    cap.release()
    return None, False


def open_best_camera():
    print(f"[camera] Trying Iriun at /dev/video{IRIUN_INDEX} ...")
    cap, ok = _probe_camera(IRIUN_INDEX)
    if ok:
        print(f"[camera] ✓ Using Iriun (phone) at /dev/video{IRIUN_INDEX}")
        return cap, "Iriun (phone)"
    print(f"[camera] Trying laptop at /dev/video{LAPTOP_INDEX} ...")
    cap, ok = _probe_camera(LAPTOP_INDEX)
    if ok:
        print(f"[camera] ✓ Using EasyCamera (laptop) at /dev/video{LAPTOP_INDEX}")
        return cap, "EasyCamera (laptop)"
    print("[camera] ✗ No usable camera found.")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────
# HAND MATH
# ──────────────────────────────────────────────────────────────

def finger_states(lm):
    """Return list of 5 booleans: [thumb, index, middle, ring, pinky]."""
    states = [False] * 5
    if abs(lm[TIP_IDX[0]].x - lm[PIP_IDX[0]].x) > 0.04:
        states[0] = True
    for i in range(1, 5):
        if lm[TIP_IDX[i]].y < lm[PIP_IDX[i]].y - 0.02:
            states[i] = True
    return states


def count_fingers(lm):
    return sum(finger_states(lm))


def palm_center_and_scale(lm, w, h):
    xs = [lm[i].x * w for i in (WRIST, 5, 9, 13, 17)]
    ys = [lm[i].y * h for i in (WRIST, 5, 9, 13, 17)]
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    dx = (lm[MIDDLE_MCP].x - lm[WRIST].x) * w
    dy = (lm[MIDDLE_MCP].y - lm[WRIST].y) * h
    palm = math.hypot(dx, dy)
    return cx, cy, palm


# ──────────────────────────────────────────────────────────────
# PETAL SPRITE CACHE (the big performance win)
# ──────────────────────────────────────────────────────────────

def _hsv_to_bgr_arr(h, s, v):
    c = v * s
    x = c * (1 - abs(((h / 60.0) % 2) - 1))
    m = v - c
    if   h < 60:  r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return np.array([(b + m) * 255, (g + m) * 255, (r + m) * 255], dtype=np.float32)


class PetalSprite:
    """A pre-rendered petal: BGR image + alpha, ready to blit with cv2."""
    __slots__ = ("bgr", "alpha", "w", "h")

    def __init__(self, bgr, alpha):
        self.bgr = bgr
        self.alpha = alpha
        self.h, self.w = bgr.shape[:2]


def build_petal_sprite(base_hue, tip_hue, length, width, curl=0.35):
    """
    Render a single petal once into a small BGR+alpha sprite.
    Petal points 'up' (angle = -pi/2). Rotation happens at blit time.
    """
    n = 40
    t = np.linspace(0, 1, n)

    # Centerline: base at bottom, tip at top with a sideways curl
    cxs = curl * length * (t ** 2)
    cys = -length * t

    # Width profile: fat middle, tapered ends
    w_profile = np.sin(np.pi * t) ** 0.75 * (width * 0.5)

    # Perp is horizontal (angle of 0) since petal points up
    left  = np.stack([cxs + w_profile, cys], axis=1)
    right = np.stack([cxs - w_profile, cys], axis=1)
    poly = np.vstack([left, right[::-1]])

    # Bounding box with padding
    minx, miny = np.min(poly, axis=0) - 4
    maxx, maxy = np.max(poly, axis=0) + 4
    minx = int(math.floor(minx)); miny = int(math.floor(miny))
    maxx = int(math.ceil(maxx));  maxy = int(math.ceil(maxy))
    sw, sh = maxx - minx, maxy - miny

    # Local mask
    mask = np.zeros((sh, sw), dtype=np.uint8)
    poly_local = (poly - [minx, miny]).astype(np.int32)
    cv2.fillPoly(mask, [poly_local], 255)

    # Distance from petal tip (0 at base, 1 at tip)
    yy, xx = np.mgrid[0:sh, 0:sw].astype(np.float32)
    px = xx + minx
    py = yy + miny
    # Approx distance from base (0,0) in local coords
    dist = np.sqrt(px * px + py * py)
    dist_norm = np.clip(dist / max(length, 1), 0, 1)

    # Gradient hue base → tip
    hue = base_hue + (tip_hue - base_hue) * dist_norm
    # Brightness: bright middle
    val = 0.70 + 0.30 * np.sin(np.pi * np.clip(dist_norm, 0, 1))
    # Saturation falls off at tip
    sat = 0.85 - 0.25 * dist_norm

    hsv = np.stack([hue / 2.0, sat * 255, val * 255], axis=-1)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

    # Specular highlight streak
    highlight = np.exp(-((dist_norm - 0.35) ** 2) / 0.04) * 60
    bgr = np.clip(bgr + highlight[..., None], 0, 255)

    # Edge darkening
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge = cv2.GaussianBlur(edge, (5, 5), 0).astype(np.float32) / 255.0
    bgr = np.clip(bgr * (1 - 0.35 * edge[..., None]), 0, 255).astype(np.uint8)

    # Alpha: fade slightly at tip
    alpha = (mask.astype(np.float32) * (1 - 0.15 * dist_norm)).astype(np.uint8)

    return PetalSprite(bgr, alpha)


def _rotate_sprite(sprite, angle_rad, max_dim):
    """Rotate a sprite and paste into a square canvas of size max_dim."""
    canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    alpha = np.zeros((max_dim, max_dim), dtype=np.uint8)
    cx, cy = max_dim // 2, max_dim // 2

    # Place sprite so its base (bottom-center) is at canvas center
    x0 = cx - sprite.w // 2
    y0 = cy - sprite.h + 1
    canvas[y0:y0 + sprite.h, x0:x0 + sprite.w] = sprite.bgr
    alpha[y0:y0 + sprite.h, x0:x0 + sprite.w] = sprite.alpha

    # Rotate around canvas center
    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad) + 90, 1.0)
    # angle+90 so "up" petal rotates to the given angle
    rot_c = cv2.warpAffine(canvas, M, (max_dim, max_dim),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rot_a = cv2.warpAffine(alpha, M, (max_dim, max_dim),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_c, rot_a


class PetalCache:
    """
    Cache pre-rotated petal sprites keyed by:
        (hue_bucket, size_bucket, angle_bucket)
    Only a few hundred combinations are ever used → render once, reuse forever.
    """
    def __init__(self, max_dim=260):
        self.max_dim = max_dim
        self._cache = {}

    def get(self, base_hue, tip_hue, length, width, angle_rad):
        # Quantize
        hue_b = int(base_hue / 10) * 10
        size_b = int(length / 6)             # ~6px size steps
        ang_b = int((angle_rad % (2 * math.pi)) /
                    (2 * math.pi / ANGLE_BUCKETS))
        key = (hue_b, size_b, ang_b)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        length_q = size_b * 6
        width_q = max(6, int(length_q * (width / max(length, 1))))
        sprite = build_petal_sprite(hue_b, hue_b + 25, length_q, width_q)
        rot_c, rot_a = _rotate_sprite(sprite, angle_rad, self.max_dim)
        self._cache[key] = (rot_c, rot_a)
        return rot_c, rot_a


# ──────────────────────────────────────────────────────────────
# FAST ALPHA BLIT (only touches petal bbox, not full frame)
# ──────────────────────────────────────────────────────────────

def blit_sprite(frame, sprite_bgr, sprite_alpha, cx, cy, scale=1.0, depth=1.0):
    """
    Paste a petal sprite centered at (cx, cy) into frame using the sprite's
    own bbox as ROI. Modifies frame in place.
    """
    sh, sw = sprite_alpha.shape
    half_w = sw // 2
    half_h = sh // 2

    # Source ROI clipped to frame
    fx0 = int(cx - half_w)
    fy0 = int(cy - half_h)
    fx1 = fx0 + sw
    fy1 = fy0 + sh

    sx0 = max(0, -fx0)
    sy0 = max(0, -fy0)
    sx1 = sw - max(0, fx1 - frame.shape[1])
    sy1 = sh - max(0, fy1 - frame.shape[0])
    if sx1 <= sx0 or sy1 <= sy0:
        return

    fx0c = max(0, fx0); fy0c = max(0, fy0)
    fx1c = fx0c + (sx1 - sx0); fy1c = fy0c + (sy1 - sy0)

    a = sprite_alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0
    if depth < 1.0:
        a = a * depth
    s = sprite_bgr[sy0:sy1, sx0:sx1].astype(np.float32)

    roi = frame[fy0c:fy1c, fx0c:fx1c].astype(np.float32)
    a3 = a[..., None]
    frame[fy0c:fy1c, fx0c:fx1c] = (roi * (1 - a3) + s * a3).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# RENDER
# ──────────────────────────────────────────────────────────────

def draw_flower_fast(frame, cache, cx, cy, palm_px, openness, t, sparkles):
    h, w = frame.shape[:2]

    max_len = palm_px * PETAL_LEN_RATIO * 1.6
    petal_len = max_len * (MIN_PETAL_SCALE + (1 - MIN_PETAL_SCALE) * openness)
    petal_w = petal_len * BASE_PETAL_W_RATIO

    if petal_len < 6:
        petal_len = 6

    sway = math.sin(t * 1.3) * 0.05 * openness

    # Draw back-to-front
    petals = []
    for i in range(NUM_PETALS):
        angle = (2 * math.pi * i / NUM_PETALS) + sway
        depth = math.sin(angle)  # -1 back, +1 front
        petals.append((angle, depth, i))
    petals.sort(key=lambda p: p[1])

    # Pre-compute center radius
    center_r = max(6, int(petal_len * 0.28))

    # Ground shadow — small blurred ROI only
    if openness > 0.05:
        sr = int(petal_len * 1.0)
        sx0 = max(0, int(cx) - sr); sy0 = max(0, int(cy) + 4)
        sx1 = min(w, int(cx) + sr); sy1 = min(h, int(cy) + sr // 2 + 8)
        if sx1 > sx0 and sy1 > sy0:
            patch = frame[sy0:sy1, sx0:sx1].astype(np.float32)
            local = np.zeros((sy1 - sy0, sx1 - sx0), dtype=np.float32)
            cv2.ellipse(local, (int(cx) - sx0, int(cy) + 4 - sy0),
                        (sr, sr // 3), 0, 0, 360, 1.0, -1)
            local = cv2.GaussianBlur(local, (0, 0), sigmaX=12)
            patch *= (1 - 0.35 * local[..., None] * openness)
            frame[sy0:sy1, sx0:sx1] = patch.astype(np.uint8)

    # Glow behind flower — small blurred ROI
    if openness > 0.15:
        gr = int(petal_len * 1.2)
        gx0 = max(0, int(cx) - gr); gy0 = max(0, int(cy) - gr)
        gx1 = min(w, int(cx) + gr); gy1 = min(h, int(cy) + gr)
        if gx1 > gx0 and gy1 > gy0:
            patch = frame[gy0:gy1, gx0:gx1].astype(np.float32)
            local = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.float32)
            cv2.circle(local, (int(cx) - gx0, int(cy) - gy0), gr,
                       (200, 140, 90), -1)
            local = cv2.GaussianBlur(local, (0, 0), sigmaX=gr * 0.35)
            patch += local * (0.5 * openness)
            frame[gy0:gy1, gx0:gx1] = np.clip(patch, 0, 255).astype(np.uint8)

    # Petals
    for angle, depth, i in petals:
        depth_shade = 0.65 + 0.35 * ((depth + 1) / 2)
        scale = 0.82 + 0.18 * ((depth + 1) / 2)

        base_hue = (330 + (i / NUM_PETALS) * 60) % 360

        sprite_bgr, sprite_a = cache.get(
            base_hue, base_hue + 25,
            petal_len * scale, petal_w * scale,
            angle
        )
        # Blit with base at (cx,cy) — the sprite center is at its bottom-center
        # because we placed it that way in _rotate_sprite → offset by half height
        sh = sprite_a.shape[0]
        blit_sprite(frame, sprite_bgr, sprite_a,
                    cx, cy + (sh // 2) - sh + 1,  # base at cx,cy
                    scale=1.0, depth=depth_shade)

    # Center dome — simple & fast
    ccx, ccy = int(cx), int(cy)
    cv2.circle(frame, (ccx, ccy), center_r, (60, 90, 180), -1, cv2.LINE_AA)
    cv2.circle(frame, (ccx, ccy), int(center_r * 0.78), (40, 150, 240), -1, cv2.LINE_AA)
    cv2.circle(frame, (ccx, ccy), int(center_r * 0.52), (70, 210, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (int(ccx - center_r * 0.2), int(ccy - center_r * 0.2)),
               max(2, int(center_r * 0.22)), (230, 245, 255), -1, cv2.LINE_AA)

    # Stamen dots
    for k in range(10):
        ang = 2 * math.pi * k / 10
        sxx = int(cx + math.cos(ang) * center_r * 1.15)
        syy = int(cy + math.sin(ang) * center_r * 1.15)
        cv2.circle(frame, (sxx, syy), max(1, center_r // 9),
                   (50, 200, 255), -1, cv2.LINE_AA)

    # Sparkles
    for s in sparkles:
        x, y, vx, alpha = s
        if alpha > 0:
            r = max(1, int(alpha // 40))
            cv2.circle(frame, (int(x), int(y)), r, (180, 240, 255), -1, cv2.LINE_AA)


def spawn_sparkle(cx, cy, openness, sparkles, palm_px):
    if openness > 0.35 and len(sparkles) < 40 and random.random() < 0.35:
        angle = random.random() * 2 * math.pi
        r = random.uniform(palm_px * 0.4, palm_px * 1.1)
        x = cx + math.cos(angle) * r
        y = cy + math.sin(angle) * r
        vx = random.uniform(-0.5, 0.5)
        sparkles.append([x, y, vx, 255])


def update_sparkles(sparkles):
    for s in sparkles:
        s[0] += s[2]
        s[1] -= 0.8
        s[3] -= 6
    return [s for s in sparkles if s[3] > 0]


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    BaseOptions = mp.tasks.BaseOptions

    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found. Download from:")
        print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task")
        sys.exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap, cam_label = open_best_camera()
    print(f"[camera] Active source: {cam_label}")

    win_name = "Palm Bloom 3D  (ESC to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_W, DISPLAY_H)

    # Filters — one per axis
    fx = OneEuroFilter(min_cutoff=POS_MIN_CUTOFF, beta=POS_BETA)
    fy = OneEuroFilter(min_cutoff=POS_MIN_CUTOFF, beta=POS_BETA)
    fp = OneEuroFilter(min_cutoff=0.8, beta=0.02)

    # Petal sprite cache
    cache = PetalCache(max_dim=int(CAM_H * 0.9))

    open_amount = 0.0
    sparkles = []
    t0 = time.time()
    timestamp_ms = 0

    # Finger hysteresis state
    fingers_smooth = 0.0        # continuous 0-5
    last_finger_int = 0

    # Fallback values when no hand detected
    last_cx, last_cy, last_palm = CAM_W / 2, CAM_H / 2, 200

    # FPS counter
    fps_t = time.time(); fps_n = 0; fps = 0.0

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # MediaPipe detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_present = bool(results.hand_landmarks)
        now = time.time()

        if hand_present:
            lm = results.hand_landmarks[0]

            # Raw finger count
            raw_fingers = count_fingers(lm)

            # Hysteresis: only change integer count if we're clearly past
            # the boundary (kills 4↔5 flicker)
            fingers_smooth += (raw_fingers - fingers_smooth) * 0.35
            target_open = float(np.clip(fingers_smooth / 5.0, 0.0, 1.0))

            # Position with One Euro Filter
            raw_cx, raw_cy, raw_palm = palm_center_and_scale(lm, w, h)
            cx = fx(raw_cx, now)
            cy = fy(raw_cy, now)
            palm = fp(raw_palm, now)
            last_cx, last_cy, last_palm = cx, cy, palm
        else:
            target_open = 0.0
            cx, cy, palm = last_cx, last_cy, last_palm

        # Smooth openness
        open_amount += (target_open - open_amount) * SMOOTH
        open_amount = max(0.0, min(1.0, open_amount))

        # Draw on camera
        if hand_present or open_amount > 0.02:
            t = now - t0
            spawn_sparkle(cx, cy, open_amount, sparkles, palm)
            sparkles = update_sparkles(sparkles)
            draw_flower_fast(frame, cache, cx, cy, palm, open_amount, t, sparkles)

            # Skeleton (thin, under flower visual weight)
            if hand_present:
                lm = results.hand_landmarks[0]
                for a, b in HAND_CONNECTIONS:
                    ax, ay = int(lm[a].x * w), int(lm[a].y * h)
                    bx, by = int(lm[b].x * w), int(lm[b].y * h)
                    cv2.line(frame, (ax, ay), (bx, by), (0, 180, 0), 1, cv2.LINE_AA)

        # FPS
        fps_n += 1
        if now - fps_t >= 0.5:
            fps = fps_n / (now - fps_t)
            fps_t = now; fps_n = 0

        # HUD
        hud = f"{cam_label}  |  Bloom: {int(open_amount*100)}%  |  {fps:.0f} FPS"
        cv2.putText(frame, hud, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Open palm to bloom  -  ESC to quit",
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (210, 210, 210), 1, cv2.LINE_AA)

        # Upscale for display
        if DISPLAY_W and (w != DISPLAY_W or h != DISPLAY_H):
            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = frame

        cv2.imshow(win_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    sys.exit(0)


if __name__ == "__main__":
    main()