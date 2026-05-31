import base64
import io
import os
from typing import Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageEnhance, ImageOps


app = Flask(__name__)

CCCD_ASPECT = 85.6 / 53.98

OUTPUT_WIDTH = int(os.environ.get("OUTPUT_WIDTH", "1800"))
OUTPUT_HEIGHT = round(OUTPUT_WIDTH / CCCD_ASPECT)
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "88"))

EARLY_EXIT_SCORE = float(os.environ.get("EARLY_EXIT_SCORE", "4.8"))
MIN_ACCEPT_SCORE = float(os.environ.get("MIN_ACCEPT_SCORE", "0.8"))

# Crop sát hơn bản trước
EXPAND_FACTOR = float(os.environ.get("EXPAND_FACTOR", "1.03"))

# Nên để false để không lưu ảnh gốc resize vào cột ảnh chuẩn
ENABLE_FALLBACK_RESIZE = os.environ.get("ENABLE_FALLBACK_RESIZE", "false").lower() == "true"


def decode_image(image_b64: str) -> np.ndarray:
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]

    raw = base64.b64decode(image_b64)
    try:
        pil = Image.open(io.BytesIO(raw))
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        raise ValueError("Cannot decode image")


def encode_jpg(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError("Cannot encode output JPG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


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

    if area_ratio < 0.06 or area_ratio > 0.96:
        return -1

    if aspect_error > 0.42:
        return -1

    return area_ratio * 5.0 - aspect_error * 2.2


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
            image[:patch, w - patch:].reshape(-1, 3),
            image[h - patch:, :patch].reshape(-1, 3),
            image[h - patch:, w - patch:].reshape(-1, 3),
        ]
    )

    background = np.median(samples, axis=0).astype("float32")
    diff = np.linalg.norm(image.astype("float32") - background, axis=2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[diff > 28] = 255
    return mask


def best_candidate(candidates):
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0]


def add_candidates_from_contours(
    contours,
    candidates,
    image_area: float,
    weight: float = 1.0,
    limit: int = 15,
):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:limit]:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)

            if len(approx) == 4 and cv2.isContourConvex(approx):
                points = approx.reshape(4, 2).astype("float32")
                score = contour_score(points, image_area)
                if score >= 0:
                    candidates.append((score * weight, points))
                    break

        rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype("float32")
        score = contour_score(rect, image_area)
        if score >= 0:
            candidates.append((score * 0.85 * weight, rect))


def find_card_quad(image: np.ndarray) -> Tuple[Optional[np.ndarray], float, str]:
    h0, w0 = image.shape[:2]
    ratio = h0 / 700.0
    resized = cv2.resize(image, (int(w0 / ratio), 700))
    rh, rw = resized.shape[:2]
    image_area = rh * rw

    candidates = []

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 21))

    def maybe_exit(method_name: str):
        best = best_candidate(candidates)
        if best and best[0] >= EARLY_EXIT_SCORE:
            quad = expand_quad(best[1], EXPAND_FACTOR, rw, rh)
            return quad * ratio, float(best[0]), method_name
        return None

    edges = cv2.Canny(gray, 35, 120)
    edges = cv2.dilate(edges, kernel_small, iterations=1)
    edges = cv2.erode(edges, kernel_small, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, candidates, image_area, weight=1.35, limit=15)

    early = maybe_exit("edge_bilateral_early")
    if early:
        return early

    bg_mask = background_color_mask(resized)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 25))

    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, bg_kernel, iterations=4)
    bg_mask = cv2.dilate(bg_mask, kernel_small, iterations=2)

    contours, _ = cv2.findContours(bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, candidates, image_area, weight=1.25, limit=15)

    early = maybe_exit("background_mask_early")
    if early:
        return early

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    color_mask = cv2.inRange(saturation, 28, 255)
    bright_mask = cv2.inRange(value, 70, 255)
    card_mask = cv2.bitwise_and(color_mask, bright_mask)

    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel_big, iterations=3)
    card_mask = cv2.dilate(card_mask, kernel_small, iterations=2)

    contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    add_candidates_from_contours(contours, candidates, image_area, weight=1.15, limit=15)

    early = maybe_exit("saturation_mask_early")
    if early:
        return early

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
    add_candidates_from_contours(contours, candidates, image_area, weight=0.9, limit=15)

    best = best_candidate(candidates)
    if best and best[0] >= MIN_ACCEPT_SCORE:
        quad = expand_quad(best[1], EXPAND_FACTOR, rw, rh)
        return quad * ratio, float(best[0]), "opencv_best_candidate"

    return None, -1.0, "not_found"


def find_card_quad_any_rotation(
    image: np.ndarray,
) -> Tuple[Optional[np.ndarray], float, str, np.ndarray, int]:
    results = []

    # Trước crop: thử 4 hướng để bắt cạnh
    for angle in (0, 90, 180, 270):
        rotated = rotate_image(image, angle)
        quad, score, method = find_card_quad(rotated)

        if quad is not None:
            results.append((score, angle, rotated, quad, method))

    if not results:
        return None, -1.0, "not_found", image, 0

    results.sort(key=lambda item: item[0], reverse=True)
    score, angle, rotated_image, quad, method = results[0]

    return quad, float(score), f"{method}_source_rot{angle}", rotated_image, angle


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


def qr_position_score(image: np.ndarray) -> float:
    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(image)

    if not ok or points is None:
        return 0.0

    pts = points.reshape(-1, 2)
    center_x = float(np.mean(pts[:, 0])) / image.shape[1]
    center_y = float(np.mean(pts[:, 1])) / image.shape[0]

    # Mặt trước đúng chiều: QR ở góc trên phải
    if center_x > 0.50 and center_y < 0.50:
        return 10.0

    # QR ở góc dưới trái thường là ngược 180
    if center_x < 0.50 and center_y > 0.50:
        return -6.0

    return -1.0


def bottom_text_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = cv2.inRange(gray, 0, 105)

    h, _ = dark.shape
    top = dark[: int(h * 0.35), :]
    bottom = dark[int(h * 0.55):, :]

    top_density = cv2.countNonZero(top) / max(1, top.size)
    bottom_density = cv2.countNonZero(bottom) / max(1, bottom.size)

    return (bottom_density - top_density) * 30.0


def red_emblem_score(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(
        hsv,
        np.array([0, 70, 60]),
        np.array([12, 255, 255]),
    )
    red2 = cv2.inRange(
        hsv,
        np.array([165, 70, 60]),
        np.array([179, 255, 255]),
    )

    red = cv2.bitwise_or(red1, red2)

    h, w = red.shape

    top_left = red[: int(h * 0.40), : int(w * 0.40)]
    bottom_right = red[int(h * 0.60):, int(w * 0.60):]
    other = red[int(h * 0.40):, :]

    top_left_density = cv2.countNonZero(top_left) / max(1, top_left.size)
    bottom_right_density = cv2.countNonZero(bottom_right) / max(1, bottom_right.size)
    other_density = cv2.countNonZero(other) / max(1, other.size)

    return (top_left_density * 30.0) - (bottom_right_density * 20.0) - (other_density * 8.0)


def mrz_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, _ = gray.shape
    top = gray[: int(h * 0.45), :]
    bottom = gray[int(h * 0.52):, :]

    top_dark = cv2.inRange(top, 0, 130)
    bottom_dark = cv2.inRange(bottom, 0, 130)

    top_density = cv2.countNonZero(top_dark) / max(1, top_dark.size)
    bottom_density = cv2.countNonZero(bottom_dark) / max(1, bottom_dark.size)

    # Mặt sau đúng chiều: MRZ nằm phía dưới
    return (bottom_density - top_density) * 100.0


def detect_side_from_filename(filename: str) -> str:
    name = (filename or "").upper()

    if "ANH_TRUOC" in name or "ANH_TRƯỚC" in name:
        return "front"

    if "ANH_SAU" in name or "ANH_SAY" in name:
        return "back"

    return "unknown"


def normalize_card_orientation(
    image: np.ndarray,
    filename: str = "",
) -> Tuple[np.ndarray, int, float, str]:
    side = detect_side_from_filename(filename)

    candidates = []

    # Sau khi crop/warp, ảnh đã là khung ngang.
    # Chỉ xét 0 và 180. Không xoay 90/270 nữa.
    for angle in (0, 180):
        rotated = rotate_image(image, angle)

        if side == "front":
            score = (
                qr_position_score(rotated)
                + red_emblem_score(rotated)
                + bottom_text_score(rotated)
            )

        elif side == "back":
            score = mrz_score(rotated)

        else:
            front_score = (
                qr_position_score(rotated)
                + red_emblem_score(rotated)
                + bottom_text_score(rotated)
            )
            back_score = mrz_score(rotated)

            score = max(front_score, back_score)

        candidates.append((score, angle, rotated))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_angle, best_image = candidates[0]

    return best_image, best_angle, float(best_score), side


def enhance_image(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    pil = ImageEnhance.Contrast(pil).enhance(1.06)
    pil = ImageEnhance.Sharpness(pil).enhance(1.12)
    pil = ImageEnhance.Color(pil).enhance(1.02)

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    max_w = 900
    if gray.shape[1] > max_w:
        new_h = int(gray.shape[0] * max_w / gray.shape[1])
        gray = cv2.resize(gray, (max_w, new_h))

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_to_output_landscape(image: np.ndarray) -> np.ndarray:
    image = ensure_landscape(image)
    return cv2.resize(
        image,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


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
        filename = (
            body.get("fileName")
            or body.get("filename")
            or body.get("name")
            or body.get("path")
            or body.get("imagePath")
            or ""
        )

        if not image_b64:
            return jsonify({"ok": False, "error": "Missing imageBase64"}), 400

        image = decode_image(image_b64)
        source_blur_score = blur_score(image)

        quad, detect_score, method, image_for_crop, source_rotate_angle = find_card_quad_any_rotation(image)

        if quad is None:
            if ENABLE_FALLBACK_RESIZE:
                fallback = resize_to_output_landscape(image)
                fallback = enhance_image(fallback)

                return jsonify(
                    {
                        "ok": True,
                        "method": "fallback_original_resized",
                        "cropStatus": "NEED_REVIEW",
                        "cropNote": "Cannot find CCCD rectangle. Returned resized original image.",
                        "format": "jpg",
                        "fileName": filename,
                        "sourceRotateAngle": 0,
                        "rotateAngle": 0,
                        "orientationScore": 0,
                        "detectScore": round(float(detect_score), 4),
                        "blurScore": round(float(source_blur_score), 4),
                        "width": OUTPUT_WIDTH,
                        "height": OUTPUT_HEIGHT,
                        "imageBase64": encode_jpg(fallback),
                    }
                )

            return jsonify(
                {
                    "ok": False,
                    "error": "Cannot find CCCD rectangle.",
                    "cropStatus": "FAILED",
                    "method": method,
                    "fileName": filename,
                    "detectScore": round(float(detect_score), 4),
                    "blurScore": round(float(source_blur_score), 4),
                }
            ), 422

        cropped = warp_card(image_for_crop, quad)

        cropped, rotate_angle, orientation_score, side = normalize_card_orientation(
            cropped,
            filename,
        )

        enhanced = enhance_image(cropped)

        return jsonify(
            {
                "ok": True,
                "method": method,
                "cropStatus": "CROPPED",
                "cropNote": "Crop completed.",
                "format": "jpg",
                "fileName": filename,
                "side": side,
                "sourceRotateAngle": source_rotate_angle,
                "rotateAngle": rotate_angle,
                "orientationScore": round(float(orientation_score), 4),
                "detectScore": round(float(detect_score), 4),
                "blurScore": round(float(source_blur_score), 4),
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
                "imageBase64": encode_jpg(enhanced),
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "cropStatus": "ERROR",
            }
        ), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
    )
