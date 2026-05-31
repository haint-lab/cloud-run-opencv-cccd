import base64
import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance


app = Flask(__name__)

CCCD_ASPECT = 85.6 / 53.98
OUTPUT_WIDTH = 1200
OUTPUT_HEIGHT = round(OUTPUT_WIDTH / CCCD_ASPECT)


def decode_image(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image


def encode_png(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Cannot encode output PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def resize_for_detection(image: np.ndarray, max_side: int = 1400) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
      return image.copy(), 1.0
    resized = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def contour_score(poly: np.ndarray, area: float, image_area: float) -> float:
    rect = cv2.minAreaRect(poly)
    (rw, rh) = rect[1]
    if rw <= 0 or rh <= 0:
        return -1

    aspect = max(rw, rh) / min(rw, rh)
    aspect_error = abs(aspect - CCCD_ASPECT) / CCCD_ASPECT
    area_ratio = area / image_area

    if area_ratio < 0.08 or area_ratio > 0.95:
        return -1
    if aspect_error > 0.45:
        return -1

    # Prefer large rectangular candidates near CCCD ratio.
    return area_ratio * 5.0 - aspect_error * 2.0


def find_card_quad(image: np.ndarray) -> Optional[np.ndarray]:
    work, scale = resize_for_detection(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates = []
    image_area = work.shape[0] * work.shape[1]

    edge_sets = []
    edge_sets.append(cv2.Canny(blur, 40, 120))
    edge_sets.append(cv2.Canny(blur, 70, 180))

    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 4
    )
    edge_sets.append(cv2.bitwise_not(adaptive))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    for edges in edge_sets:
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * peri, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                score = contour_score(approx, area, image_area)
                if score >= 0:
                    candidates.append((score, approx.reshape(4, 2)))

            rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
            score = contour_score(rect, area, image_area)
            if score >= 0:
                candidates.append((score * 0.85, rect))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1].astype("float32")
    return best / scale


def fallback_center_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    current = w / h

    if current > CCCD_ASPECT:
        new_w = round(h * CCCD_ASPECT)
        x = max(0, (w - new_w) // 2)
        return image[:, x:x + new_w]

    new_h = round(w / CCCD_ASPECT)
    y = max(0, (h - new_h) // 2)
    return image[y:y + new_h, :]


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


def enhance_image(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil = ImageEnhance.Contrast(pil).enhance(1.08)
    pil = ImageEnhance.Sharpness(pil).enhance(1.18)
    pil = ImageEnhance.Color(pil).enhance(1.03)

    out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.4, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


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
            cropped = fallback_center_crop(image)
            method = "fallback_center_crop"
            cropped = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
        else:
            cropped = warp_card(image, quad)
            method = "opencv_quad"

        enhanced = enhance_image(cropped)

        return jsonify(
            {
                "ok": True,
                "method": method,
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "imageBase64": encode_png(enhanced),
            }
        )

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
