
from flask import Flask, render_template, request, jsonify, send_file, url_for
import cv2
import os
import numpy as np
import uuid

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
RESULT_FOLDER = os.path.join(STATIC_DIR, "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_imread(path):
    file_bytes = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def safe_imwrite(path, image, quality=95):
    try:
        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)

        ext = os.path.splitext(path)[1].lower()

        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        elif ext == ".png":
            compression = max(0, min(9, int((100 - quality) / 10)))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(quality)]
        else:
            params = []

        success, buffer = cv2.imencode(ext, image, params)
        if not success:
            return False

        with open(path, "wb") as f:
            f.write(buffer.tobytes())

        return True
    except Exception as e:
        print("SAVE ERROR:", e)
        return False


def get_bgr_color(color_name):
    color_map = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "gray": (160, 160, 160)
    }
    return color_map.get(color_name, (255, 255, 255))


def apply_preset(result, preset):
    if preset == "none":
        return result

    if preset == "vivid":
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.35
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        result = cv2.convertScaleAbs(result, alpha=1.08, beta=8)

    elif preset == "cinematic":
        result = cv2.convertScaleAbs(result, alpha=1.05, beta=-5)
        b, g, r = cv2.split(result)
        b = np.clip(b * 1.08, 0, 255).astype(np.uint8)
        r = np.clip(r * 0.95, 0, 255).astype(np.uint8)
        result = cv2.merge([b, g, r])

    elif preset == "vintage":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        result = cv2.transform(result, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)

    elif preset == "cool":
        b, g, r = cv2.split(result)
        b = np.clip(b * 1.12, 0, 255).astype(np.uint8)
        result = cv2.merge([b, g, r])

    return result


def get_text_position(image_shape, text, scale, thickness, position):
    h, w = image_shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    margin = 20

    if position == "top-left":
        return margin, margin + text_h
    elif position == "top-right":
        return w - text_w - margin, margin + text_h
    elif position == "center":
        return (w - text_w) // 2, (h + text_h) // 2
    elif position == "bottom-left":
        return margin, h - margin
    else:
        return w - text_w - margin, h - margin


def add_text_overlay(image, text, size, color_name, position):
    if not text.strip():
        return image

    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, float(size))
    thickness = max(1, int(size * 2))
    color = get_bgr_color(color_name)
    x, y = get_text_position(img.shape, text, scale, thickness, position)

    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    return img


def add_watermark(image, text, size, color_name, position, opacity):
    if not text.strip():
        return image

    base = image.copy()
    overlay = image.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, float(size))
    thickness = max(1, int(size * 2))
    color = get_bgr_color(color_name)
    x, y = get_text_position(base.shape, text, scale, thickness, position)

    cv2.putText(overlay, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    alpha = max(0.0, min(1.0, float(opacity)))
    result = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    return result


def resize_image(img, resize_preset, custom_width, custom_height):
    preset_sizes = {
        "original": None,
        "instagram_post": (1080, 1080),
        "instagram_story": (1080, 1920),
        "youtube_thumbnail": (1280, 720),
        "facebook_post": (1200, 630),
        "square_small": (800, 800)
    }

    if resize_preset == "custom":
        if custom_width > 0 and custom_height > 0:
            return cv2.resize(img, (custom_width, custom_height), interpolation=cv2.INTER_AREA)
        return img

    size = preset_sizes.get(resize_preset)
    if size is None:
        return img

    width, height = size
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def crop_image(img, crop_x, crop_y, crop_w, crop_h):
    if crop_w <= 0 or crop_h <= 0:
        return img

    h, w = img.shape[:2]

    x = max(0, min(crop_x, w - 1))
    y = max(0, min(crop_y, h - 1))
    cw = max(1, min(crop_w, w - x))
    ch = max(1, min(crop_h, h - y))

    cropped = img[y:y + ch, x:x + cw]
    if cropped.size == 0:
        return img
    return cropped


def process_image(
    img,
    filter_type,
    brightness,
    contrast,
    saturation,
    rotate,
    flip,
    blur_strength,
    sharpen_strength,
    preset,
    overlay_text,
    text_size,
    text_color,
    text_position,
    watermark_text,
    watermark_size,
    watermark_color,
    watermark_position,
    watermark_opacity,
    resize_preset,
    custom_width,
    custom_height,
    crop_x,
    crop_y,
    crop_w,
    crop_h
):
    result = img.copy()

    result = resize_image(result, resize_preset, custom_width, custom_height)
    result = crop_image(result, crop_x, crop_y, crop_w, crop_h)

    if filter_type == "grayscale":
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    elif filter_type == "blur":
        k = max(1, blur_strength)
        if k % 2 == 0:
            k += 1
        result = cv2.GaussianBlur(result, (k, k), 0)

    elif filter_type == "edges":
        result = cv2.Canny(result, 100, 200)

    elif filter_type == "sharpen":
        s = float(sharpen_strength)
        kernel = np.array([
            [0, -1, 0],
            [-1, 5 + s, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)

    elif filter_type == "sepia":
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        result = cv2.transform(result, kernel)
        result = np.clip(result, 0, 255).astype(np.uint8)

    elif filter_type == "invert":
        result = cv2.bitwise_not(result)

    elif filter_type == "emboss":
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        result = cv2.filter2D(result, -1, kernel)

    elif filter_type == "cartoon":
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9, 9
        )
        color = cv2.bilateralFilter(result, 9, 250, 250)
        result = cv2.bitwise_and(color, color, mask=edges)

    elif filter_type == "warm":
        result = result.astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.15, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)
        result = result.astype(np.uint8)

    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    result = apply_preset(result, preset)
    result = cv2.convertScaleAbs(result, alpha=contrast, beta=brightness)

    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rotate == "left":
        result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == "right":
        result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == "180":
        result = cv2.rotate(result, cv2.ROTATE_180)

    if flip == "horizontal":
        result = cv2.flip(result, 1)
    elif flip == "vertical":
        result = cv2.flip(result, 0)

    result = add_text_overlay(result, overlay_text, text_size, text_color, text_position)
    result = add_watermark(
        result,
        watermark_text,
        watermark_size,
        watermark_color,
        watermark_position,
        watermark_opacity
    )

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("image")
    filter_type = request.form.get("filter", "none")
    brightness = int(float(request.form.get("brightness", 0)))
    contrast = float(request.form.get("contrast", 1))
    saturation = float(request.form.get("saturation", 1))
    rotate = request.form.get("rotate", "none")
    flip = request.form.get("flip", "none")
    blur_strength = int(float(request.form.get("blur_strength", 15)))
    sharpen_strength = float(request.form.get("sharpen_strength", 0))
    preset = request.form.get("preset", "none")

    overlay_text = request.form.get("overlay_text", "")
    text_size = float(request.form.get("text_size", 1))
    text_color = request.form.get("text_color", "white")
    text_position = request.form.get("text_position", "bottom-right")

    watermark_text = request.form.get("watermark_text", "")
    watermark_size = float(request.form.get("watermark_size", 0.8))
    watermark_color = request.form.get("watermark_color", "white")
    watermark_position = request.form.get("watermark_position", "bottom-right")
    watermark_opacity = float(request.form.get("watermark_opacity", 0.35))

    resize_preset = request.form.get("resize_preset", "original")
    custom_width = int(float(request.form.get("custom_width", 0) or 0))
    custom_height = int(float(request.form.get("custom_height", 0) or 0))

    crop_x = int(float(request.form.get("crop_x", 0) or 0))
    crop_y = int(float(request.form.get("crop_y", 0) or 0))
    crop_w = int(float(request.form.get("crop_w", 0) or 0))
    crop_h = int(float(request.form.get("crop_h", 0) or 0))

    export_format = request.form.get("export_format", "jpg").lower()
    export_quality = int(float(request.form.get("export_quality", 95)))

    if export_format not in {"jpg", "png", "webp"}:
        export_format = "jpg"

    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed formats: PNG, JPG, JPEG, WEBP."}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_id = uuid.uuid4().hex

    upload_filename = f"upload_{unique_id}.{ext}"
    result_filename = f"result_{unique_id}.{export_format}"

    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
    result_path = os.path.join(app.config["RESULT_FOLDER"], result_filename)

    for old_file in os.listdir(RESULT_FOLDER):
        try:
            os.remove(os.path.join(RESULT_FOLDER, old_file))
        except Exception:
            pass

    file.save(upload_path)

    img = safe_imread(upload_path)
    if img is None:
        return jsonify({"error": "The image could not be loaded."}), 400

    original_h, original_w = img.shape[:2]

    result = process_image(
        img,
        filter_type,
        brightness,
        contrast,
        saturation,
        rotate,
        flip,
        blur_strength,
        sharpen_strength,
        preset,
        overlay_text,
        text_size,
        text_color,
        text_position,
        watermark_text,
        watermark_size,
        watermark_color,
        watermark_position,
        watermark_opacity,
        resize_preset,
        custom_width,
        custom_height,
        crop_x,
        crop_y,
        crop_w,
        crop_h
    )

    processed_h, processed_w = result.shape[:2]

    saved = safe_imwrite(result_path, result, export_quality)
    if not saved:
        return jsonify({"error": f"Failed to save processed image to: {result_path}"}), 500

    return jsonify({
        "original_image": url_for("static", filename=f"uploads/{upload_filename}"),
        "result_image": url_for("static", filename=f"results/{result_filename}"),
        "info": {
            "original_width": original_w,
            "original_height": original_h,
            "processed_width": processed_w,
            "processed_height": processed_h,
            "input_format": ext.upper(),
            "export_format": export_format.upper()
        }
    })


@app.route("/download")
def download():
    files = os.listdir(RESULT_FOLDER)
    if not files:
        return "No processed image available."

    latest_file = max(
        [os.path.join(RESULT_FOLDER, f) for f in files],
        key=os.path.getctime
    )
    return send_file(latest_file, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
