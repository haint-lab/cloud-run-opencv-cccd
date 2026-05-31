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


def find_card_quad(image: np.ndarray) -> Optional[np.ndarray]:
    ratio = image.shape[0] / 700.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 700))
    image_area = resized.shape[0] * resized.shape[1]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    candidates = []
    for contour in contours[:20]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2).astype("float32")
            score = contour_score(points, image_area)
            if score >= 0:
                candidates.append((score, points))

        rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
        score = contour_score(rect, image_area)
        if score >= 0:
            candidates.append((score * 0.85, rect))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1] * ratio

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
        enhanced = enhance_image(cropped)

        return jsonify(
            {
                "ok": True,
                "method": "opencv_quad",
                "format": "jpg",
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "imageBase64": encode_jpg(enhanced),
            }
        )

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
