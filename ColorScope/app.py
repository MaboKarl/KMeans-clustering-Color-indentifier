from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import base64
import re
import cv2
import numpy as np
import sqlite3
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # change in production!

DB_NAME = "users.db"

# -------------------------------
# Database Setup
# -------------------------------
def init_db():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Database created successfully!")

# -------------------------------
# Login / Registration Routes
# -------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
            conn.close()
            flash("✅ Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("⚠️ Username already taken.", "danger")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["username"] = username
            # ✅ Instead of flash + redirect, render success page
            return render_template("login_success.html")
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template("login.html")



@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("👋 Logged out.", "info")
    return redirect(url_for("login"))

# -------------------------------
# Require login decorator
# -------------------------------
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            flash("⚠️ Please log in first!", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function

# -------------------------------
# Color classification (HSV-based)
# -------------------------------
def get_color_name(h, s, v):
    def shade(v):
        if v < 50:
            return "Dark"
        elif v > 200:
            return "Light"
        else:
            return "Medium"

    if s <= 25:
        if v <= 50:
            return "Black"
        elif v >= 200:
            return "White"
        else:
            return "Gray"

    if 10 <= h <= 25 and v < 170 and s > 50:
        return f"{shade(v)} Brown"
    if 160 <= h <= 175 and v > 150 and s > 40:
        return f"{shade(v)} Pink"

    hue = h * 2
    color_name = "Undefined"

    if hue < 5 or hue >= 345:
        color_name = "Red"
    elif hue < 15:
        color_name = "Red-Orange"
    elif hue < 25:
        color_name = "Orange"
    elif hue < 35:
        color_name = "Yellow-Orange"
    elif hue < 45:
        color_name = "Yellow"
    elif hue < 55:
        color_name = "Yellow-Green"
    elif hue < 78:
        color_name = "Green"
    elif hue < 90:
        color_name = "Teal"
    elif hue < 110:
        color_name = "Cyan"
    elif hue < 140:
        color_name = "Sky Blue"
    elif hue < 160:
        color_name = "Blue"
    elif hue < 170:
        color_name = "Blue-Purple"
    elif hue < 190:
        color_name = "Purple"
    elif hue < 330:
        color_name = "Magenta"
    else:
        color_name = "Red"

    return f"{shade(v)} {color_name}"

# -------------------------------
# Image analysis
# -------------------------------
def analyze_bgr_image(bgr_img, roi_half=6):
    if bgr_img is None or bgr_img.size == 0:
        raise ValueError("Empty image.")

    h_img, w_img = bgr_img.shape[:2]
    cx, cy = w_img // 2, h_img // 2
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    x1 = max(0, cx - roi_half)
    y1 = max(0, cy - roi_half)
    x2 = min(w_img, cx + roi_half + 1)
    y2 = min(h_img, cy + roi_half + 1)

    roi = hsv[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("ROI out of bounds.")

    avg_h = int(np.mean(roi[:, :, 0]))
    avg_s = int(np.mean(roi[:, :, 1]))
    avg_v = int(np.mean(roi[:, :, 2]))

    color_name = get_color_name(avg_h, avg_s, avg_v)
    patch_bgr = cv2.cvtColor(np.uint8([[[avg_h, avg_s, avg_v]]]), cv2.COLOR_HSV2BGR)[0, 0, :]
    r, g, b = int(patch_bgr[2]), int(patch_bgr[1]), int(patch_bgr[0])
    hex_color = f"#{r:02X}{g:02X}{b:02X}"

    return {"name": color_name, "hsv": {"h": avg_h, "s": avg_s, "v": avg_v}, "rgb": {"r": r, "g": g, "b": b}, "hex": hex_color}

def decode_base64_image(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("Missing image data.")
    match = re.match(r"^data:image/\w+;base64,(.+)", data_url)
    b64_data = match.group(1) if match else data_url
    img_bytes = base64.b64decode(b64_data)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image.")
    return bgr

# -------------------------------
# Protected Routes
# -------------------------------
@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/webcam")
@login_required
def webcam():
    return render_template("webcam.html")

@app.route("/analyze_frame", methods=["POST"])
@login_required
def analyze_frame():
    try:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image")
        bgr = decode_base64_image(data_url)
        result = analyze_bgr_image(bgr)
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "GET":
        return render_template("upload.html", result=None, error=None)

    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("upload.html", result=None, error="No file selected.")

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        return render_template("upload.html", result=None, error="Unsupported file type.")

    try:
        data = file.read()
        if not data:
            return render_template("upload.html", result=None, error="Empty file.")
        file_bytes = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Could not decode image bytes.")
        result = analyze_bgr_image(bgr)
        return render_template("upload.html", result=result, error=None)
    except Exception as e:
        return render_template("upload.html", result=None, error=str(e))

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=True)
