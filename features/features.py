# code by xyanua - github.com/imxyanua
"""Tiền xử lý, phân đoạn đối tượng, trích đặc trưng màu và hình dạng."""
import cv2
import numpy as np

PROCESS_SIZE = 192
HUE_BINS = 8
SAT_BINS = 4
VAL_BINS = 4

COLOR_DIMS = 6 + HUE_BINS + SAT_BINS + VAL_BINS
GEOMETRY_DIMS = 5
HU_DIMS = 7
FEATURE_GROUPS = {
    "color": slice(0, COLOR_DIMS),
    "geometry": slice(COLOR_DIMS, COLOR_DIMS + GEOMETRY_DIMS),
    "hu": slice(COLOR_DIMS + GEOMETRY_DIMS, COLOR_DIMS + GEOMETRY_DIMS + HU_DIMS),
}


def load_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def preprocess(bgr, size=PROCESS_SIZE):
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized, (5, 5), 0)


def segment_pet(bgr):
    """GrabCut + morph + giữ blob lớn nhất."""
    img = preprocess(bgr)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mx, my = int(w * 0.08), int(h * 0.08)
    rw, rh = max(1, w - 2 * mx), max(1, h - 2 * my)
    rect = (mx, my, rw, rh)
    try:
        cv2.grabCut(img, mask, rect, bgd, fgd, 1, cv2.GC_INIT_WITH_RECT)
        binary = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
    except cv2.error:
        binary = np.full((h, w), 255, dtype=np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean = np.zeros_like(binary)
        cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
        binary = clean
    return img, binary


def overlay_mask(img, mask, alpha=0.45):
    color = np.zeros_like(img)
    color[:, :] = (80, 180, 80)
    overlay = cv2.addWeighted(img, 1.0, color, alpha, 0)
    mask3 = (mask > 0)[:, :, None]
    return np.where(mask3, overlay, img)


def _safe_hist(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(12, dtype=np.float64)

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    _x, _y, bw, bh = cv2.boundingRect(cnt)
    rect_area = float(max(bw * bh, 1))
    hull = cv2.convexHull(cnt)
    hull_area = float(max(cv2.contourArea(hull), 1))

    aspect = bw / max(bh, 1)
    extent = area / rect_area
    solidity = area / hull_area
    circularity = 0.0 if peri < 1e-6 else (4.0 * np.pi * area) / (peri * peri)

    (_cx, _cy), (mw, mh), _angle = cv2.minAreaRect(cnt)
    eccentricity = min(mw, mh) / max(max(mw, mh), 1e-6)

    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    return np.concatenate(
        [
            np.array([aspect, extent, solidity, circularity, eccentricity], dtype=np.float64),
            hu,
        ]
    )


def extract_features(bgr, mask=None):
    if mask is None:
        img, mask = segment_pet(bgr)
    else:
        img = preprocess(bgr) if bgr.shape[:2] != mask.shape[:2] else bgr

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if len(pixels) == 0:
        pixels = hsv.reshape(-1, 3)

    mean_hsv = pixels.mean(axis=0)
    std_hsv = pixels.std(axis=0)
    hist_h = _safe_hist(pixels[:, 0], HUE_BINS, (0, 180))
    hist_s = _safe_hist(pixels[:, 1], SAT_BINS, (0, 256))
    hist_v = _safe_hist(pixels[:, 2], VAL_BINS, (0, 256))
    shape = _shape_features(mask)

    vector = np.concatenate([mean_hsv, std_hsv, hist_h, hist_s, hist_v, shape])
    return vector.astype(np.float64), img, mask


def features_and_preview(path):
    bgr = load_bgr(path)
    vector, img, mask = extract_features(bgr)
    preview = overlay_mask(img, mask)
    return vector, img, preview
