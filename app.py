"""
app.py
Flask-based Face Recognition Attendance System (Mobile & Desktop Support)
"""
import os
import time
import platform
import base64
import cv2
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, jsonify, flash
)

import database as db
import face_utils as fu

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

ADMIN_USERNAME = "muzammil"
ADMIN_PASSWORD = "muzammil182921"

db.init_db()

capture_progress = {}     
last_marked = {}          
camera = None             
CAMERA_INDEX = 0          


def _open_camera(index):
    if platform.system() == "Windows":
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(index, cv2.CAP_MSMF)
    else:
        cam = cv2.VideoCapture(index)
    return cam


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = _open_camera(CAMERA_INDEX)
        if camera.isOpened():
            for _ in range(5):
                camera.read()
                time.sleep(0.02)
    return camera


def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    release_camera()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    stats = db.get_stats()
    today_logs = db.get_today_attendance()
    pending = db.get_pending_approvals()
    return render_template("dashboard.html", stats=stats, today_logs=today_logs, pending=pending)


@app.route("/employees/approve/<int:employee_id>", methods=["POST"])
@login_required
def approve_employee(employee_id):
    db.approve_employee(employee_id)
    flash("Employee approved — their face will now be recognized in Live Attendance.", "success")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/employees/revoke/<int:employee_id>", methods=["POST"])
@login_required
def revoke_employee(employee_id):
    db.revoke_approval(employee_id)
    flash("Approval revoked.", "success")
    return redirect(request.referrer or url_for("employees"))


@app.route("/employees")
@login_required
def employees():
    all_employees = db.get_all_employees()
    return render_template("employees.html", employees=all_employees)


@app.route("/employees/delete/<int:employee_id>", methods=["POST"])
@login_required
def delete_employee(employee_id):
    db.delete_employee(employee_id)
    flash("Employee removed.", "success")
    return redirect(url_for("employees"))


@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    departments = db.get_all_departments()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        department_id = request.form.get("department_id")
        if not name or not department_id:
            flash("Name and department are required.", "error")
            return render_template("register.html", departments=departments)

        new_id = db.add_employee(name, int(department_id))
        capture_progress[new_id] = {"count": 0, "done": False}
        return redirect(url_for("capture_page", employee_id=new_id))

    return render_template("register.html", departments=departments)


@app.route("/departments/add", methods=["POST"])
@login_required
def add_department():
    name = request.form.get("dept_name", "").strip()
    if name:
        ok = db.add_department(name)
        flash("Department added." if ok else "Department already exists.", "success" if ok else "error")
    return redirect(url_for("register"))


@app.route("/capture/<int:employee_id>")
@login_required
def capture_page(employee_id):
    employee = db.get_employee(employee_id)
    if employee is None:
        return redirect(url_for("register"))
    return render_template("capture.html", employee=employee, needed=fu.SAMPLES_NEEDED)


# -------- Mobile Camera Frame Upload API --------
@app.route("/api/upload_capture_frame/<int:employee_id>", methods=["POST"])
@login_required
def upload_capture_frame(employee_id):
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image provided"})

    try:
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid image decode"})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fu.detect_faces(gray)

        progress = capture_progress.setdefault(employee_id, {"count": 0, "done": False})
        if progress["done"]:
            return jsonify({"success": True, "count": progress["count"], "done": True})

        count = progress["count"]
        for (x, y, w, h) in faces:
            count += 1
            fu.save_face_sample(employee_id, gray[y:y + h, x:x + w], count)
            progress["count"] = count
            if count >= fu.SAMPLES_NEEDED:
                progress["done"] = True
                db.update_samples_captured(employee_id, count)
            break  

        return jsonify({
            "success": True,
            "count": progress["count"],
            "done": progress["done"],
            "faces_detected": len(faces)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/capture_status/<int:employee_id>")
@login_required
def capture_status(employee_id):
    progress = capture_progress.get(employee_id, {"count": 0, "done": False})
    return jsonify(progress)


@app.route("/train/<int:employee_id>", methods=["POST"])
@login_required
def train(employee_id):
    ok, message = fu.train_model()
    if ok:
        db.mark_trained(employee_id)
        flash(f"Face data trained! {message} Now waiting for admin approval.", "success")
    else:
        flash(f"Training failed: {message}", "error")
    return redirect(url_for("employees"))


@app.route("/attendance")
@login_required
def attendance_page():
    return render_template("attendance_live.html")


@app.route("/api/recognize_frame", methods=["POST"])
@login_required
def recognize_frame():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "No image provided"})

    try:
        header, encoded = data["image"].split(",", 1)
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"})

        recognizer = fu.load_recognizer()
        trained_map = {e["id"]: e for e in db.get_trained_employees()}

        if recognizer is None:
            return jsonify({"success": True, "results": [], "message": "No trained faces yet."})

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fu.detect_faces(gray)
        results = []

        for (x, y, w, h) in faces:
            employee_id, confidence = fu.recognize_face(recognizer, gray[y:y + h, x:x + w])
            status = "Unknown"

            if employee_id is not None and employee_id in trained_map:
                emp = trained_map[employee_id]
                status = f"{emp['name']} | {emp['department_name']} ({confidence:.0f})"

                now = time.time()
                if now - last_marked.get(employee_id, 0) > 5:
                    last_marked[employee_id] = now
                    newly_marked = db.mark_attendance(employee_id)
                    if newly_marked:
                        status += " [MARKED]"

            results.append({
                "box": [int(x), int(y), int(w), int(h)],
                "label": status,
                "recognized": employee_id in trained_map if employee_id else False
            })

        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/reports")
@login_required
def reports():
    date_filter = request.args.get("date") or None
    dept_filter = request.args.get("department_id") or None
    dept_filter = int(dept_filter) if dept_filter else None

    logs = db.get_attendance_filtered(date=date_filter, department_id=dept_filter)
    departments = db.get_all_departments()
    return render_template("reports.html", logs=logs, departments=departments,
                           selected_date=date_filter, selected_dept=dept_filter)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)