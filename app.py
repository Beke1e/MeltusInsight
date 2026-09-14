import csv
import io
import os
import pickle
import secrets
import sqlite3
from datetime import date, datetime
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


def load_secret_key():
    configured = os.environ.get("SECRET_KEY")
    if configured:
        if len(configured) < 32:
            raise RuntimeError("SECRET_KEY must contain at least 32 characters.")
        return configured
    key_path = os.path.join(BASE_DIR, ".secret_key")
    if os.path.exists(key_path):
        with open(key_path, "r", encoding="ascii") as file:
            return file.read().strip()
    key = secrets.token_hex(32)
    with open(key_path, "w", encoding="ascii") as file:
        file.write(key)
    return key


app.config.update(
    SECRET_KEY=load_secret_key(),
    DATABASE=os.path.join(BASE_DIR, "diabetes.db"),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("HTTPS", "").lower() == "true",
)
FEATURES = ["Treatment_Type", "Enrollment_Timing", "Edu_Category", "Marital_Status", "Loc_Type", "Age_Group", "Sex", "Blood_Sugar", "Diabetes_Type"]
EDUCATION = ["No Formal Education", "Primary Education", "Secondary Education", "Diploma", "Bachelor's Degree", "Master's Degree", "Doctorate", "Other", "Unknown"]


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_compatible_model(path):
    try:
        return load_model(path, compile=False)
    except (TypeError, ValueError):
        inputs = Input(shape=(len(FEATURES),))
        layer = Dense(64, activation="relu")(inputs)
        layer = Dropout(0.3)(layer)
        layer = Dense(64, activation="relu")(layer)
        layer = Dropout(0.3)(layer)
        layer = Dense(512, activation="relu")(layer)
        layer = Dropout(0.3)(layer)
        model = Model(inputs=inputs, outputs=Dense(1, activation="sigmoid")(layer))
        model.load_weights(path)
        return model


model = load_compatible_model(os.path.join(BASE_DIR, "model.h5"))
with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as file:
    scaler = CompatibleUnpickler(file).load()


def db():
    if "db" not in g:
        g.db = sqlite3.connect(app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    connection = g.pop("db", None)
    if connection:
        connection.close()


def init_db():
    connection = db()
    existing = {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "patients" in existing:
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(patients)")}
        if "full_name" not in columns:
            connection.execute("ALTER TABLE patients RENAME TO patients_legacy")
            if "predictions" in existing:
                connection.execute("ALTER TABLE predictions RENAME TO predictions_legacy")
            if "prediction_inputs" in existing:
                connection.execute("ALTER TABLE prediction_inputs RENAME TO prediction_inputs_legacy")
            connection.commit()
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user', created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, age INTEGER NOT NULL, sex TEXT NOT NULL, phone TEXT, residence TEXT NOT NULL, marital_status TEXT NOT NULL, education_level TEXT NOT NULL, family_history_dm TEXT NOT NULL, socioeconomic_status TEXT NOT NULL, date_of_registration TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL, user_id INTEGER NOT NULL, treatment_type INTEGER NOT NULL, enrollment_timing INTEGER NOT NULL, edu_category INTEGER NOT NULL, marital_status_code INTEGER NOT NULL, loc_type INTEGER NOT NULL, age_group INTEGER NOT NULL, sex_code INTEGER NOT NULL, blood_sugar INTEGER NOT NULL, diabetes_type INTEGER NOT NULL, outcome INTEGER NOT NULL, probability REAL NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE, FOREIGN KEY(user_id) REFERENCES users(id));
    """)
    if "patients_legacy" in {row["name"] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}:
        connection.execute("""INSERT OR IGNORE INTO patients(id, full_name, age, sex, phone, residence, marital_status, education_level, family_history_dm, socioeconomic_status, date_of_registration, created_at, updated_at)
            SELECT id, patient_name, 0, 'Unknown', phone, 'Unknown', 'Unknown', 'Unknown', 'Unknown', 'Unknown', date(created_at), created_at, updated_at FROM patients_legacy""")
    connection.commit()


@app.before_request
def load_user():
    user_id = session.get("user_id")
    g.user = db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone() if user_id else None


@app.before_request
def validate_csrf():
    if request.method == "POST":
        token = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
        if not token or token != session.get("csrf_token"):
            abort(400, description="Invalid or missing CSRF token.")


@app.context_processor
def csrf_context():
    token = session.setdefault("csrf_token", secrets.token_urlsafe(32))
    return {"csrf_token": token}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        if g.user["role"] != "admin":
            flash("Administrator access is required.", "error")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)
    return wrapped


def prediction(values):
    frame = pd.DataFrame([values], columns=FEATURES)
    scores = model.predict(scaler.transform(frame), verbose=0)
    probability = float(scores[0][0]) if scores.shape[1] == 1 else float(np.max(scores[0]))
    return int(probability >= 0.5), probability


def numeric_values(source):
    try:
        values = [int(source.get(field)) for field in FEATURES]
    except (TypeError, ValueError):
        return None
    return values if all(0 <= value <= 10 for value in values) else None


def patient_age(source):
    try:
        age = int(source.get("age", ""))
    except (TypeError, ValueError):
        return None
    return age if 1 <= age <= 120 else None


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def patient_form_values(source):
    required = ["full_name", "sex", "residence", "marital_status", "education_level", "family_history_dm", "socioeconomic_status"]
    age = patient_age(source)
    if age is None or any(not source.get(field, "").strip() for field in required):
        return None
    return {
        "full_name": source["full_name"].strip(),
        "age": age,
        "sex": source["sex"],
        "phone": source.get("phone", "").strip(),
        "residence": source["residence"],
        "marital_status": source["marital_status"],
        "education_level": source["education_level"],
        "family_history_dm": source["family_history_dm"],
        "socioeconomic_status": source["socioeconomic_status"],
    }


def save_prediction(patient_id, user_id, values):
    outcome, probability = prediction(values)
    cursor = db().execute("INSERT INTO predictions(patient_id, user_id, treatment_type, enrollment_timing, edu_category, marital_status_code, loc_type, age_group, sex_code, blood_sugar, diabetes_type, outcome, probability, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (patient_id, user_id, *values, outcome, probability, now_iso()))
    return cursor.lastrowid


def prediction_filter_args():
    outcome = request.args.get("outcome", "")
    patient_id = request.args.get("patient_id", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    if outcome not in {"", "0", "1"}:
        outcome = ""
    if patient_id and not patient_id.isdigit():
        patient_id = ""
    return outcome, patient_id, date_from, date_to


@app.route("/")
@login_required
def dashboard():
    connection = db()
    total = connection.execute("SELECT COUNT(*) AS value FROM predictions").fetchone()["value"]
    positive = connection.execute("SELECT COUNT(*) AS value FROM predictions WHERE outcome = 1").fetchone()["value"]
    patients = connection.execute("SELECT COUNT(*) AS value FROM patients").fetchone()["value"]
    recent = connection.execute("SELECT p.*, pt.full_name FROM predictions p JOIN patients pt ON pt.id = p.patient_id ORDER BY p.id DESC LIMIT 8").fetchall()
    return render_template("dashboard.html", total=total, positive=positive, patients=patients, recent=recent)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if len(username) < 3 or len(password) < 8:
            flash("Username must have 3 characters and password must have 8 characters.", "error")
        else:
            try:
                role = "admin" if db().execute("SELECT COUNT(*) AS value FROM users").fetchone()["value"] == 0 else "user"
                db().execute("INSERT INTO users(username, password_hash, role, created_at) VALUES (?, ?, ?, ?)", (username, generate_password_hash(password), role, datetime.now().isoformat(timespec="seconds")))
                db().commit()
                flash("Account created. Sign in to continue.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("That username is already registered.", "error")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = db().execute("SELECT * FROM users WHERE username = ?", (request.form.get("username", "").strip(),)).fetchone()
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            session.clear()
            session["user_id"] = user["id"]
            return redirect(request.args.get("next") or url_for("dashboard"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/patients", methods=["GET", "POST"])
@login_required
def patients():
    if request.method == "POST":
        values = patient_form_values(request.form)
        if values is None:
            flash("Complete all required patient fields.", "error")
        else:
            now = now_iso()
            db().execute("INSERT INTO patients(full_name, age, sex, phone, residence, marital_status, education_level, family_history_dm, socioeconomic_status, date_of_registration, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (request.form["full_name"].strip(), age, request.form["sex"], request.form.get("phone", "").strip(), request.form["residence"], request.form["marital_status"], request.form["education_level"], request.form["family_history_dm"], request.form["socioeconomic_status"], date.today().isoformat(), now, now))
            db().execute("INSERT INTO patients(full_name, age, sex, phone, residence, marital_status, education_level, family_history_dm, socioeconomic_status, date_of_registration, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*values.values(), date.today().isoformat(), now, now))
            db().commit()
            flash("Patient record created.", "success")
        return redirect(url_for("patients"))
    query = request.args.get("q", "").strip()
    records = db().execute("SELECT * FROM patients WHERE ? = '' OR full_name LIKE ? OR phone LIKE ? ORDER BY full_name", (query, f"%{query}%", f"%{query}%")).fetchall()
    return render_template("patients.html", patients=records, query=query, education=EDUCATION)


@app.route("/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_patient(patient_id):
    patient = db().execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for("patients"))
    if request.method == "POST":
        values = patient_form_values(request.form)
        if values is None:
            flash("Complete all patient fields. Age must be between 1 and 120.", "error")
            return render_template("patient_form.html", patient=patient, education=EDUCATION)
        db().execute("UPDATE patients SET full_name=?, age=?, sex=?, phone=?, residence=?, marital_status=?, education_level=?, family_history_dm=?, socioeconomic_status=?, updated_at=? WHERE id=?", (*values.values(), now_iso(), patient_id))
        db().commit()
        flash("Patient record updated.", "success")
        return redirect(url_for("patients"))
    return render_template("patient_form.html", patient=patient, education=EDUCATION)


@app.route("/patients/<int:patient_id>")
@login_required
def patient_detail(patient_id):
    patient = db().execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if not patient:
        flash("Patient not found.", "error")
        return redirect(url_for("patients"))
    records = db().execute("SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,)).fetchall()
    return render_template("patient_detail.html", patient=patient, predictions=records)


@app.route("/patients/<int:patient_id>/delete", methods=["POST"])
@admin_required
def delete_patient(patient_id):
    db().execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    db().commit()
    flash("Patient record deleted.", "success")
    return redirect(url_for("patients"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    patients = db().execute("SELECT id, full_name FROM patients ORDER BY full_name").fetchall()
    if request.method == "GET":
        return render_template("predict.html", patients=patients)
    patient_id = request.form.get("patient_id", type=int)
    values = numeric_values(request.form)
    if not patient_id or not values or not db().execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone():
        flash("Select a patient and complete every model input.", "error")
        return redirect(url_for("predict"))
    prediction_id = save_prediction(patient_id, g.user["id"], values)
    db().commit()
    flash("Prediction saved and linked to the patient.", "success")
    return redirect(url_for("prediction_result", prediction_id=prediction_id))


@app.route("/predictions/<int:prediction_id>/edit", methods=["GET", "POST"])
@admin_required
def edit_prediction(prediction_id):
    record = db().execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    patients = db().execute("SELECT id FROM patients ORDER BY id").fetchall()
    if not record:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))
    if request.method == "POST":
        patient_id = request.form.get("patient_id", type=int)
        values = numeric_values(request.form)
        if not patient_id or not values or not db().execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone():
            flash("Select a valid Patient ID and complete every model input.", "error")
        else:
            outcome, probability = prediction(values)
            db().execute("UPDATE predictions SET patient_id=?, treatment_type=?, enrollment_timing=?, edu_category=?, marital_status_code=?, loc_type=?, age_group=?, sex_code=?, blood_sugar=?, diabetes_type=?, outcome=?, probability=? WHERE id=?", (patient_id, *values, outcome, probability, prediction_id))
            db().commit()
            flash("Prediction record updated.", "success")
            return redirect(url_for("prediction_result", prediction_id=prediction_id))
    return render_template("prediction_edit.html", record=record, patients=patients)


@app.route("/predictions/<int:prediction_id>/delete", methods=["POST"])
@admin_required
def delete_prediction(prediction_id):
    db().execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    db().commit()
    flash("Prediction record deleted.", "success")
    return redirect(url_for("history"))


@app.route("/predictions/<int:prediction_id>")
@login_required
def prediction_result(prediction_id):
    record = db().execute("SELECT p.*, pt.* FROM predictions p JOIN patients pt ON pt.id = p.patient_id WHERE p.id = ?", (prediction_id,)).fetchone()
    if not record:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))
    return render_template("prediction_result.html", record=record)


@app.route("/predictions/<int:prediction_id>/download")
@login_required
def download_prediction(prediction_id):
    record = db().execute("SELECT id AS prediction_id, patient_id, treatment_type, enrollment_timing, edu_category, marital_status_code, loc_type, age_group, sex_code, blood_sugar, diabetes_type, outcome, created_at FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if not record:
        flash("Prediction not found.", "error")
        return redirect(url_for("history"))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(record.keys())
    writer.writerow(tuple(record))
    response = app.response_class(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=prediction_{prediction_id}.csv"
    return response


@app.route("/history")
@login_required
def history():
    query = request.args.get("q", "").strip()
    outcome = request.args.get("outcome", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    filters = ["(? = '' OR pt.full_name LIKE ? OR CAST(pt.id AS TEXT) = ?)", "(? = '' OR p.outcome = ?)", "(? = '' OR date(p.created_at) >= date(?))", "(? = '' OR date(p.created_at) <= date(?))"]
    params = [query, f"%{query}%", query, outcome, outcome or -1, date_from, date_from, date_to, date_to]
    records = db().execute(f"SELECT p.*, pt.full_name FROM predictions p JOIN patients pt ON pt.id = p.patient_id WHERE {' AND '.join(filters)} ORDER BY p.created_at DESC", params).fetchall()
    return render_template("history.html", records=records, query=query, outcome=outcome, date_from=date_from, date_to=date_to)


@app.route("/import", methods=["POST"])
@admin_required
def import_data():
    upload = request.files.get("file")
    if not upload or not upload.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        flash("Upload a CSV or Excel file.", "error")
        return redirect(url_for("history"))
    try:
        frame = pd.read_csv(upload) if upload.filename.lower().endswith(".csv") else pd.read_excel(upload)
        missing = [field for field in FEATURES + ["patient_id"] if field not in frame.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        saved = 0
        for _, row in frame.iterrows():
            values = numeric_values({field: row[field] for field in FEATURES})
            patient_id = int(row["patient_id"])
            if values and db().execute("SELECT id FROM patients WHERE id = ?", (patient_id,)).fetchone():
                save_prediction(patient_id, g.user["id"], values)
                saved += 1
        db().commit()
        flash(f"Imported {saved} validated rows.", "success")
    except Exception as error:
        flash(f"Import failed: {error}", "error")
    return redirect(url_for("history"))


@app.route("/download")
@login_required
def download():
    outcome, patient_id, date_from, date_to = prediction_filter_args()
    filters = ["(? = '' OR p.outcome = ?)", "(? = '' OR p.patient_id = ?)", "(? = '' OR date(p.created_at) >= date(?))", "(? = '' OR date(p.created_at) <= date(?))"]
    params = [outcome, outcome or -1, patient_id, patient_id or -1, date_from, date_from, date_to, date_to]
    rows = db().execute(f"SELECT p.id AS prediction_id, p.patient_id, p.treatment_type, p.enrollment_timing, p.edu_category, p.marital_status_code, p.loc_type, p.age_group, p.sex_code, p.blood_sugar, p.diabetes_type, p.outcome, p.created_at FROM predictions p WHERE {' AND '.join(filters)} ORDER BY p.created_at DESC", params).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    if rows:
        writer.writerow(rows[0].keys())
        writer.writerows([tuple(row) for row in rows])
    response = app.response_class(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=diabetes_prediction_report.csv"
    return response


@app.route("/about")
def about():
    return render_template("about.html")


with app.app_context():
    init_db()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
