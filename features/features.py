# code by xyanua - github.com/imxyanua
"""Tiền xử lý, phân đoạn đối tượng, trích đặc trưng màu và hình dạng."""
import cv2
import numpy as np

PROCESS_SIZE = 192
GRABCUT_ITERS = 5
# Thú cưng thường chạm mép ảnh, nên chỉ 1% ngoài cùng bị coi là nền chắc chắn.
BORDER_SURE_BG = 0.01
BORDER_PROB_BG = 0.05
CENTER_SURE_FG = 0.20
HUE_BINS = 16
SAT_BINS = 16
VAL_BINS = 16
# Pixel dưới ngưỡng này có H gần như nhiễu (đen / xám / trắng).
CHROMA_S_MIN = 40
CHROMA_V_MIN = 40

COLOR_STAT_DIMS = 7  # μ_H,S,V + σ_H,S,V + tỷ lệ pixel không màu; còn lại là histogram
COLOR_DIMS = COLOR_STAT_DIMS + HUE_BINS + SAT_BINS + VAL_BINS
GEOMETRY_DIMS = 5
HU_DIMS = 7
FEATURE_GROUPS = {
    "color": slice(0, COLOR_DIMS),
    "geometry": slice(COLOR_DIMS, COLOR_DIMS + GEOMETRY_DIMS),
    "hu": slice(COLOR_DIMS + GEOMETRY_DIMS, COLOR_DIMS + GEOMETRY_DIMS + HU_DIMS),
}


def select_feature_groups(X, groups):
    """Lấy các cột thuộc những nhóm đặc trưng đã chọn; X có thể là 1 vector hoặc ma trận."""
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"Nhóm đặc trưng không hợp lệ: {', '.join(unknown)}")
    return np.concatenate([X[..., FEATURE_GROUPS[g]] for g in groups], axis=-1)


def histogram_mask(groups):
    """Mask boolean: cột nào trong ma trận đã chọn là bin histogram HSV (chưa z-score khi dùng χ²)."""
    width = sum(FEATURE_GROUPS[g].stop - FEATURE_GROUPS[g].start for g in groups)
    mask = np.zeros(width, dtype=bool)
    offset = 0
    for g in groups:
        sl = FEATURE_GROUPS[g]
        group_width = sl.stop - sl.start
        if g == "color":
            mask[offset + COLOR_STAT_DIMS : offset + group_width] = True
        offset += group_width
    return mask


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


def grabcut_init_mask(h, w):
    """Viền ngoài cùng là nền chắc chắn, dải viền là có thể là nền,
    elip nhỏ ở tâm là tiền cảnh chắc chắn, phần còn lại là có thể là tiền cảnh."""
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    for frac, value in ((BORDER_PROB_BG, cv2.GC_PR_BGD), (BORDER_SURE_BG, cv2.GC_BGD)):
        bx, by = max(1, int(w * frac)), max(1, int(h * frac))
        mask[:by, :] = value
        mask[-by:, :] = value
        mask[:, :bx] = value
        mask[:, -bx:] = value
    axes = (max(1, int(w * CENTER_SURE_FG / 2)), max(1, int(h * CENTER_SURE_FG / 2)))
    cv2.ellipse(mask, (w // 2, h // 2), axes, 0, 0, 360, cv2.GC_FGD, thickness=-1)
    return mask


def segment_pet(bgr):
    """GrabCut + morph + giữ blob lớn nhất."""
    img = preprocess(bgr)
    h, w = img.shape[:2]
    mask = grabcut_init_mask(h, w)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(img, mask, None, bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)
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

    # H chỉ tin cậy khi S và V đủ cao; S/V vẫn tính trên mọi pixel trong mask.
    chromatic = (pixels[:, 1] >= CHROMA_S_MIN) & (pixels[:, 2] >= CHROMA_V_MIN)
    achromatic_ratio = 1.0 - float(np.count_nonzero(chromatic)) / float(len(pixels))

    mean_s = float(pixels[:, 1].mean())
    mean_v = float(pixels[:, 2].mean())
    std_s = float(pixels[:, 1].std())
    std_v = float(pixels[:, 2].std())
    if np.any(chromatic):
        hue = pixels[chromatic, 0].astype(np.float64)
        mean_h = float(hue.mean())
        std_h = float(hue.std())
        hist_h = _safe_hist(hue, HUE_BINS, (0, 180))
    else:
        mean_h = 0.0
        std_h = 0.0
        hist_h = np.zeros(HUE_BINS, dtype=np.float64)

    hist_s = _safe_hist(pixels[:, 1], SAT_BINS, (0, 256))
    hist_v = _safe_hist(pixels[:, 2], VAL_BINS, (0, 256))
    shape = _shape_features(mask)

    color_stats = np.array(
        [mean_h, mean_s, mean_v, std_h, std_s, std_v, achromatic_ratio],
        dtype=np.float64,
    )
    vector = np.concatenate([color_stats, hist_h, hist_s, hist_v, shape])
    return vector.astype(np.float64), img, mask


def features_and_preview(path):
    bgr = load_bgr(path)
    vector, img, mask = extract_features(bgr)
    preview = overlay_mask(img, mask)
    return vector, img, preview
