import sqlite3
import json
import os

DB_NAME = "data.db"
TABLE_NAME = "images"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            dimensions TEXT,
            analysis TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_metadata(filename: str, dimensions: tuple, analysis: dict = None):
    dims_str = json.dumps(dimensions)
    analysis_str = json.dumps(analysis) if analysis is not None else None
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(f"""
        INSERT OR REPLACE INTO {TABLE_NAME} (filename, dimensions, analysis)
        VALUES (?, ?, ?)
    """, (filename, dims_str, analysis_str))
    conn.commit()
    conn.close()


def get_metadata(filename: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT dimensions, analysis "
        f"FROM {TABLE_NAME} WHERE filename=?", (filename,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        dimensions, analysis = row
        return {
            "dimensions": json.loads(dimensions),
            "analysis": json.loads(analysis) if analysis else None
        }
    return None


# Initialize DB upon module load.
if not os.path.exists(DB_NAME):
    init_db()
