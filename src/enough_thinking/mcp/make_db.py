
import sqlite3
from pathlib import Path

DB = Path("enough-thinking/data/local/workflows.db")
DB.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS calendar")
cur.execute("DROP TABLE IF EXISTS expenses")

cur.execute("""
CREATE TABLE calendar (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT,
  start TEXT,
  end TEXT
)
""")

cur.execute("""
CREATE TABLE expenses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  category TEXT,
  amount REAL,
  date TEXT
)
""")

# sample events
events = [
  ("Research meeting", "2026-01-16 10:00", "2026-01-16 11:00"),
  ("Gym", "2026-01-16 18:00", "2026-01-16 19:00"),
  ("Project deadline", "2026-01-20 09:00", "2026-01-20 09:30"),
]
cur.executemany("INSERT INTO calendar(title,start,end) VALUES(?,?,?)", events)

# sample expenses
expenses = [
  ("Food", 24.50, "2026-01-12"),
  ("Uber", 18.25, "2026-01-12"),
  ("Food", 12.90, "2026-01-13"),
  ("Books", 45.00, "2026-01-13"),
]
cur.executemany("INSERT INTO expenses(category,amount,date) VALUES(?,?,?)", expenses)

conn.commit()
conn.close()
print("✅ DB created at:", DB)
