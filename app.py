import base64
import io
import os
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance, ImageOps


app = Flask(__name__)

CCCD_ASPECT = 85.6 / 53.98
OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = round(OUTPUT_WIDTH / CCCD_ASPECT)
JPEG_QUALITY = 85
PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = round(PREVIEW_WIDTH / CCCD_ASPECT)


def decode_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise ValueError("Cannot decode image")


def encode_jpg(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Cannot encode output JPG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")

    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]

    s = pts.sum(axis=1)
    start = int(np.argmin(s))
    ordered = np.roll(pts, -start, axis=0)

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    left_height = np.linalg.norm(ordered[3] - ordered[0])

    if (top_width + bottom_width) < (right_height + left_height):
        ordered = np.array(
            [ordered[3], ordered[0], ordered[1], ordered[2]],
            dtype="float32",
        )

    return ordered


def contour_score(points: np.ndarray, image_area: float) -> float:
    area = cv2.contourArea(points.astype("float32"))
    if area <= 0:
        return -1

    rect = cv2.minAreaRect(points.astype("float32"))
    width, height = rect[1]
    if width <= 0 or height <= 0:
        return -1

    aspect = max(width, height) / min(width, height)
    aspect_error = abs(aspect - CCCD_ASPECT) / CCCD_ASPECT
    area_ratio = area / image_area
    rect_area = width * height
    extent = area / rect_area if rect_area > 0 else 0

    if area_ratio < 0.06 or area_ratio > 0.96:
        return -1
    if aspect_error > 0.45:
        return -1
    if extent < 0.55:
        return -1

    return area_ratio * 5.0 + extent * 1.2 - aspect_error * 2.5


def expand_quad(points: np.ndarray, factor: float, width: int, height: int) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    center = np.mean(pts, axis=0)
    expanded = center + (pts - center) * factor
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return expanded.astype("float32")


def background_color_mask(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    patch = max(12, min(h, w) // 25)

    samples = np.vstack(
        [
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, w - patch :].reshape(-1, 3),
            image[h - patch :, :patch].reshape(-1, 3),
            image[h - patch :, w - patch :].reshape(-1, 3),
        ]
    )
    background = np.median(samples, axis=0).astype("float32")
    diff_bgr = np.linalg.norm(image.astype("float32") - background, axis=2)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32")
    bg_lab = cv2.cvtColor(background.reshape(1, 1, 3).astype("uint8"), cv2.COLOR_BGR2LAB)
    diff_lab = np.linalg.norm(lab - bg_lab.reshape(1, 1, 3).astype("float32"), axis=2)
    diff = np.maximum(diff_bgr, diff_lab)

    mask = np.zeros((h, w), dtype=np.uint8)
    diff_u8 = np.clip(diff, 0, 255).astype("uint8")
    threshold, _ = cv2.threshold(diff_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[diff > max(24, threshold * 0.85)] = 255
    return mask


def resize_for_detection(image: np.ndarray, target_long_side: int = 1000) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= target_long_side:
        return image.copy(), 1.0

    ratio = long_side / float(target_long_side)
    resized = cv2.resize(image, (int(w / ratio), int(h / ratio)), interpolation=cv2.INTER_AREA)
    return resized, ratio


def keep_largest_components(mask: np.ndarray, max_components: int = 3) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    indexes = np.argsort(areas)[::-1][:max_components] + 1
    kept = np.zeros_like(mask)
    for idx in indexes:
        kept[labels == idx] = 255
    return kept


def add_contour_candidates(
    candidates: list[tuple[float, np.ndarray]],
    contours,
    image_area: float,
    weight: float = 1.0,
) -> None:
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:45]:
        if cv2.contourArea(contour) < image_area * 0.015:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 0.025 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2).astype("float32")
            score = contour_score(points, image_area)
            if score >= 0:
                candidates.append((score * 1.08 * weight, points))

        rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
        score = contour_score(rect, image_area)
        if score >= 0:
            candidates.append((score * weight, rect))


def rotate_points_to_original(
    points: np.ndarray,
    angle: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    mapped = np.zeros_like(pts)

    if angle == 0:
        return pts
    if angle == 90:
        mapped[:, 0] = pts[:, 1]
        mapped[:, 1] = original_height - 1 - pts[:, 0]
        return mapped
    if angle == 180:
        mapped[:, 0] = original_width - 1 - pts[:, 0]
        mapped[:, 1] = original_height - 1 - pts[:, 1]
        return mapped
    if angle == 270:
        mapped[:, 0] = original_width - 1 - pts[:, 1]
        mapped[:, 1] = pts[:, 0]
        return mapped

    return pts


def warp_quad_preview(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    src = order_points(quad)
    dst = np.array(
        [
            [0, 0],
            [PREVIEW_WIDTH - 1, 0],
            [PREVIEW_WIDTH - 1, PREVIEW_HEIGHT - 1],
            [0, PREVIEW_HEIGHT - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (PREVIEW_WIDTH, PREVIEW_HEIGHT))


def cccd_content_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    inner = np.zeros_like(gray)
    inner[int(h * 0.05) : int(h * 0.95), int(w * 0.05) : int(w * 0.95)] = 255

    cyan = cv2.inRange(hsv, np.array([70, 18, 70]), np.array([105, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([14, 25, 80]), np.array([45, 255, 255]))
    red1 = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([163, 45, 45]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    dark = cv2.inRange(gray, 0, 145)
    edges = cv2.Canny(gray, 45, 140)

    cyan_density = cv2.countNonZero(cv2.bitwise_and(cyan, inner)) / cv2.countNonZero(inner)
    yellow_density = cv2.countNonZero(cv2.bitwise_and(yellow, inner)) / cv2.countNonZero(inner)
    red_density = cv2.countNonZero(cv2.bitwise_and(red, inner)) / cv2.countNonZero(inner)
    dark_density = cv2.countNonZero(cv2.bitwise_and(dark, inner)) / cv2.countNonZero(inner)
    edge_density = cv2.countNonZero(cv2.bitwise_and(edges, inner)) / cv2.countNonZero(inner)

    color_density = cyan_density + yellow_density
    if color_density < 0.035 and red_density < 0.0015:
        return -1
    if dark_density < 0.012 or dark_density > 0.40:
        return -1
    if edge_density < 0.012:
        return -1

    card_color = cv2.bitwise_or(cyan, yellow)
    card_color = cv2.bitwise_or(card_color, red)
    card_color = cv2.morphologyEx(
        card_color,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
        iterations=2,
    )
    color_points = cv2.findNonZero(cv2.bitwise_and(card_color, inner))
    if color_points is None:
        return -1

    x, y, bw, bh = cv2.boundingRect(color_points)
    color_width_ratio = bw / w
    color_height_ratio = bh / h
    if x > w * 0.22 or (x + bw) < w * 0.78:
        return -1
    if y > h * 0.18 or (y + bh) < h * 0.82:
        return -1
    if color_width_ratio < 0.62 or color_height_ratio < 0.58:
        return -1

    low_saturation = hsv[:, :, 1] < 24
    bright = hsv[:, :, 2] > 165
    background_like = np.where(low_saturation & bright, 255, 0).astype("uint8")
    side_bands = [
        background_like[int(h * 0.20) : int(h * 0.80), : int(w * 0.10)],
        background_like[int(h * 0.20) : int(h * 0.80), int(w * 0.90) :],
        background_like[: int(h * 0.10), int(w * 0.20) : int(w * 0.80)],
        background_like[int(h * 0.90) :, int(w * 0.20) : int(w * 0.80)],
    ]
    for band in side_bands:
        if cv2.countNonZero(band) / max(1, band.size) > 0.78:
            return -1

    score = 0.0
    score += min(cyan_density, 0.45) * 10.0
    score += min(yellow_density, 0.30) * 3.0
    score += min(red_density, 0.08) * 20.0
    score += min(dark_density, 0.16) * 10.0
    score += min(edge_density, 0.18) * 6.0
    score += color_width_ratio * 2.0 + color_height_ratio * 2.0
    return score


def find_card_quad_single_orientation(image: np.ndarray) -> Optional[tuple[float, np.ndarray]]:
    resized, ratio = resize_for_detection(image)
    image_area = resized.shape[0] * resized.shape[1]

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 120)

    candidates = []
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (47, 31))

    # Method 0: background subtraction from corner colors. This is useful when
    # the card sits on a white sheet/table and rounded corners weaken the border.
    bg_mask = background_color_mask(resized)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    bg_mask = cv2.dilate(bg_mask, kernel_medium, iterations=1)
    bg_mask = keep_largest_components(bg_mask)
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(candidates, contours, image_area, weight=1.55)

    # Method 1: normal border/edge detection. Works well when the card border
    # contrasts with the background.
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    edges_closed = cv2.dilate(edges_closed, kernel_small, iterations=1)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(candidates, contours, image_area, weight=1.0)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(candidates, contours, image_area, weight=0.75)

    # Method 2: saturation mask. This helps when the card sits on a white
    # background and the outer edge is weak, but the CCCD artwork/text is colored.
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = cv2.inRange(saturation, 28, 255)
    bright_mask = cv2.inRange(value, 70, 255)
    card_mask = cv2.bitwise_and(color_mask, bright_mask)

    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    card_mask = cv2.dilate(card_mask, kernel_medium, iterations=1)
    card_mask = keep_largest_components(card_mask)

    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(candidates, contours, image_area, weight=1.25)

    # Method 3: adaptive threshold for low-contrast photos.
    adaptive = cv2.adaptiveThreshold(
        gray_eq,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5,
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    adaptive = keep_largest_components(adaptive)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_contour_candidates(candidates, contours, image_area, weight=0.9)

    verified_candidates = []
    for geometry_score, points in candidates:
        expanded = expand_quad(points, 1.015, resized.shape[1], resized.shape[0])
        preview = warp_quad_preview(resized, expanded)
        content_score = cccd_content_score(preview)
        if content_score < 0:
            continue
        verified_candidates.append((geometry_score + content_score, expanded))

    if verified_candidates:
        verified_candidates.sort(key=lambda item: item[0], reverse=True)
        return verified_candidates[0][0], verified_candidates[0][1] * ratio

    return None


def find_card_quad(image: np.ndarray) -> Optional[np.ndarray]:
    h, w = image.shape[:2]
    candidates = []

    for angle in (0, 90, 180, 270):
        rotated = rotate_image(image, angle)
        result = find_card_quad_single_orientation(rotated)
        if result is None:
            continue

        score, quad = result
        mapped = rotate_points_to_original(quad, angle, w, h)
        candidates.append((score, mapped))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def warp_card(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    src = order_points(quad)
    dst = np.array(
        [
            [0, 0],
            [OUTPUT_WIDTH - 1, 0],
            [OUTPUT_WIDTH - 1, OUTPUT_HEIGHT - 1],
            [0, OUTPUT_HEIGHT - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (OUTPUT_WIDTH, OUTPUT_HEIGHT))


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def ensure_landscape(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h > w:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def qr_position_score(image: np.ndarray) -> float:
    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(image)
    if not ok or points is None:
        return 0.0

    pts = points.reshape(-1, 2)
    center_x = float(np.mean(pts[:, 0])) / image.shape[1]
    center_y = float(np.mean(pts[:, 1])) / image.shape[0]

    score = 4.0
    if center_x > 0.45:
        score += 2.0
    else:
        score -= 1.0

    if center_y < 0.55:
        score += 2.0
    else:
        score -= 2.0

    return score


def bottom_text_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 120)

    h, w = dark.shape
    top = dark[: int(h * 0.40), :]
    bottom = dark[int(h * 0.52) :, :]

    top_density = cv2.countNonZero(top) / max(1, top.size)
    bottom_density = cv2.countNonZero(bottom) / max(1, bottom.size)

    return (bottom_density - top_density) * 30.0


def mrz_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dark = cv2.inRange(gray, 0, 135)

    h, w = dark.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 28), 3))
    rows = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)

    top = rows[: int(h * 0.38), int(w * 0.05) : int(w * 0.95)]
    bottom = rows[int(h * 0.58) :, int(w * 0.05) : int(w * 0.95)]

    top_density = cv2.countNonZero(top) / max(1, top.size)
    bottom_density = cv2.countNonZero(bottom) / max(1, bottom.size)

    return (bottom_density - top_density) * 55.0


def red_header_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 55, 45]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([163, 55, 45]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    h, w = red.shape
    upper = red[: int(h * 0.48), int(w * 0.08) : int(w * 0.92)]
    lower = red[int(h * 0.52) :, int(w * 0.08) : int(w * 0.92)]

    upper_density = cv2.countNonZero(upper) / max(1, upper.size)
    lower_density = cv2.countNonZero(lower) / max(1, lower.size)

    return (upper_density - lower_density) * 35.0


def red_emblem_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, np.array([0, 70, 60]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 70, 60]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    h, w = red.shape
    top_left = red[: int(h * 0.35), : int(w * 0.35)]
    other = red[int(h * 0.35) :, :] | 0

    top_left_density = cv2.countNonZero(top_left) / max(1, top_left.size)
    other_density = cv2.countNonZero(other) / max(1, other.size)

    return (top_left_density - other_density) * 25.0


def normalize_card_orientation(image: np.ndarray) -> tuple[np.ndarray, int, float]:
    image = ensure_landscape(image)

    candidates = []
    for angle in (0, 180):
        rotated = rotate_image(image, angle)
        score = (
            qr_position_score(rotated)
            + bottom_text_score(rotated)
            + mrz_score(rotated)
            + red_emblem_score(rotated)
            + red_header_score(rotated)
        )
        candidates.append((score, angle, rotated))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_angle, best_image = candidates[0]
    return best_image, best_angle, best_score


def enhance_image(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = ImageEnhance.Contrast(pil).enhance(1.05)
    pil = ImageEnhance.Sharpness(pil).enhance(1.10)
    pil = ImageEnhance.Color(pil).enhance(1.02)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.post("/crop-cccd")
def crop_cccd():
    try:
        token = os.environ.get("CROP_API_TOKEN", "")
        if token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                return jsonify({"ok": False, "error": "Unauthorized"}), 401

        body = request.get_json(silent=True) or {}
        image_b64 = body.get("imageBase64")
        if not image_b64:
            return jsonify({"ok": False, "error": "Missing imageBase64"}), 400

        image = decode_image(image_b64)
        quad = find_card_quad(image)
        if quad is None:
            return jsonify(
                {
                    "ok": False,
                    "error": "Cannot find CCCD rectangle. Retake photo with all 4 card corners visible.",
                }
            ), 422

        cropped = warp_card(image, quad)
        cropped, rotate_angle, orientation_score = normalize_card_orientation(cropped)
        enhanced = enhance_image(cropped)

        return jsonify(
            {
                "ok": True,
                "method": "opencv_quad",
                "format": "jpg",
                "rotateAngle": rotate_angle,
                "orientationScore": round(float(orientation_score), 4),
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "imageBase64": encode_jpg(enhanced),
            }
        )

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
