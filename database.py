"""
database.py
Handles all SQLite database operations for the Face Recognition Attendance System.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")

DEFAULT_DEPARTMENTS = [
    "Information Technology",
    "Human Resources",
    "Finance",
    "Marketing",
    "Sales",
    "Operations",
    "Administration",
    "Production",
    "Quality Assurance",
    "Research & Development",
    "Customer Support",
    "Logistics & Supply Chain",
]


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create tables and seed default departments if the DB is new."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            samples_captured INTEGER DEFAULT 0,
            trained INTEGER DEFAULT 0,
            approved INTEGER DEFAULT 0,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    """)

    # Migration: add 'approved' column if this DB was created before this feature existed
    existing_cols = [row["name"] for row in cur.execute("PRAGMA table_info(employees)").fetchall()]
    if "approved" not in existing_cols:
        cur.execute("ALTER TABLE employees ADD COLUMN approved INTEGER DEFAULT 0")
        conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    """)

    conn.commit()

    # Seed departments only if table is empty
    cur.execute("SELECT COUNT(*) as c FROM departments")
    if cur.fetchone()["c"] == 0:
        for dept in DEFAULT_DEPARTMENTS:
            cur.execute("INSERT INTO departments (name) VALUES (?)", (dept,))
        conn.commit()

    conn.close()


# ---------------- Departments ----------------

def get_all_departments():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM departments ORDER BY name").fetchall()
    conn.close()
    return rows


def add_department(name):
    conn = get_connection()
    try:
        conn.execute("INSERT INTO departments (name) VALUES (?)", (name.strip(),))
        conn.commit()
        ok = True
    except sqlite3.IntegrityError:
        ok = False
    conn.close()
    return ok


# ---------------- Employees ----------------

def add_employee(name, department_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO employees (name, department_id, created_at) VALUES (?, ?, ?)",
        (name.strip(), department_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def update_samples_captured(employee_id, count):
    conn = get_connection()
    conn.execute("UPDATE employees SET samples_captured = ? WHERE id = ?", (count, employee_id))
    conn.commit()
    conn.close()


def mark_trained(employee_id):
    conn = get_connection()
    conn.execute("UPDATE employees SET trained = 1 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()


def approve_employee(employee_id):
    conn = get_connection()
    conn.execute("UPDATE employees SET approved = 1 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()


def revoke_approval(employee_id):
    conn = get_connection()
    conn.execute("UPDATE employees SET approved = 0 WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()


def get_pending_approvals():
    """Employees whose face model is trained but not yet approved by admin."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT e.id, e.name, e.created_at, d.name as department_name
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE e.trained = 1 AND e.approved = 0
        ORDER BY e.created_at DESC
    """).fetchall()
    conn.close()
    return rows


def get_employee(employee_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    conn.close()
    return row


def get_all_employees():
    conn = get_connection()
    rows = conn.execute("""
        SELECT e.id, e.name, e.created_at, e.trained, e.approved, e.samples_captured,
               d.name as department_name, d.id as department_id
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        ORDER BY e.name
    """).fetchall()
    conn.close()
    return rows


def get_trained_employees():
    """Only employees who are trained AND approved by admin are recognizable
    during live attendance."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT e.id, e.name, d.name as department_name
        FROM employees e
        JOIN departments d ON e.department_id = d.id
        WHERE e.trained = 1 AND e.approved = 1
    """).fetchall()
    conn.close()
    return rows


def delete_employee(employee_id):
    conn = get_connection()
    conn.execute("DELETE FROM attendance WHERE employee_id = ?", (employee_id,))
    conn.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
    conn.commit()
    conn.close()


# ---------------- Attendance ----------------

def mark_attendance(employee_id):
    """Mark attendance once per day per employee. Returns True if newly marked."""
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%H:%M:%S")

    conn = get_connection()
    cur = conn.cursor()
    existing = cur.execute(
        "SELECT id FROM attendance WHERE employee_id = ? AND date = ?",
        (employee_id, today),
    ).fetchone()

    if existing:
        conn.close()
        return False

    cur.execute(
        "INSERT INTO attendance (employee_id, date, time) VALUES (?, ?, ?)",
        (employee_id, today, now_time),
    )
    conn.commit()
    conn.close()
    return True


def get_today_attendance():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.id, e.name, d.name as department_name, a.date, a.time
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        JOIN departments d ON e.department_id = d.id
        WHERE a.date = ?
        ORDER BY a.time DESC
    """, (today,)).fetchall()
    conn.close()
    return rows


def get_attendance_filtered(date=None, department_id=None):
    query = """
        SELECT a.id, e.name, d.name as department_name, a.date, a.time
        FROM attendance a
        JOIN employees e ON a.employee_id = e.id
        JOIN departments d ON e.department_id = d.id
        WHERE 1=1
    """
    params = []
    if date:
        query += " AND a.date = ?"
        params.append(date)
    if department_id:
        query += " AND d.id = ?"
        params.append(department_id)
    query += " ORDER BY a.date DESC, a.time DESC"

    conn = get_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_stats():
    conn = get_connection()
    total_employees = conn.execute("SELECT COUNT(*) c FROM employees").fetchone()["c"]
    total_departments = conn.execute("SELECT COUNT(*) c FROM departments").fetchone()["c"]
    today = datetime.now().strftime("%Y-%m-%d")
    today_present = conn.execute(
        "SELECT COUNT(*) c FROM attendance WHERE date = ?", (today,)
    ).fetchone()["c"]

    dept_counts = conn.execute("""
        SELECT d.name as department_name, COUNT(a.id) as present_count
        FROM departments d
        LEFT JOIN employees e ON e.department_id = d.id
        LEFT JOIN attendance a ON a.employee_id = e.id AND a.date = ?
        GROUP BY d.id
        ORDER BY d.name
    """, (today,)).fetchall()

    conn.close()
    return {
        "total_employees": total_employees,
        "total_departments": total_departments,
        "today_present": today_present,
        "dept_counts": [dict(r) for r in dept_counts],
    }
