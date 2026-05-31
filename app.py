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
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]

    top_width = np.linalg.norm(ordered[1] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])

    if top_width < right_height:
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

    if area_ratio < 0.08 or area_ratio > 0.95:
        return -1
    if aspect_error > 0.45:
        return -1

    return area_ratio * 5.0 - aspect_error * 2.0


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
    diff = np.linalg.norm(image.astype("float32") - background, axis=2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[diff > 28] = 255
    return mask


def find_card_quad(image: np.ndarray) -> Optional[np.ndarray]:
    ratio = image.shape[0] / 700.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 700))
    image_area = resized.shape[0] * resized.shape[1]

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    candidates = []

    def add_candidates_from_contours(contours, weight=1.0):
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:30]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype("float32")
                score = contour_score(points, image_area)
                if score >= 0:
                    candidates.append((score * weight, points))

            rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
            score = contour_score(rect, image_area)
            if score >= 0:
                candidates.append((score * 0.85 * weight, rect))

    # Method 0: background subtraction from corner colors. This is useful when
    # the card sits on a white sheet/table and the outer border is weak.
    bg_mask = background_color_mask(resized)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 25))
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, bg_kernel, iterations=4)
    bg_mask = cv2.dilate(bg_mask, kernel_small, iterations=2)
    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, weight=1.45)

    # Method 1: normal border/edge detection. Works well when the card border
    # contrasts with the background.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, weight=1.0)

    # Method 2: saturation mask. This helps when the card sits on a white
    # background and the outer edge is weak, but the CCCD artwork/text is colored.
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = cv2.inRange(saturation, 28, 255)
    bright_mask = cv2.inRange(value, 70, 255)
    card_mask = cv2.bitwise_and(color_mask, bright_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 21))
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    card_mask = cv2.dilate(card_mask, kernel_small, iterations=2)

    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, weight=1.2)

    # Method 3: adaptive threshold for low-contrast photos.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_big, iterations=2)
    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, weight=0.9)

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best = expand_quad(candidates[0][1], 1.03, resized.shape[1], resized.shape[0])
        return best * ratio

    return None


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
    dark = cv2.inRange(gray, 0, 105)

    h, w = dark.shape
    top = dark[: int(h * 0.35), :]
    bottom = dark[int(h * 0.55) :, :]

    top_density = cv2.countNonZero(top) / max(1, top.size)
    bottom_density = cv2.countNonZero(bottom) / max(1, bottom.size)

    return (bottom_density - top_density) * 30.0


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
            + red_emblem_score(rotated)
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
