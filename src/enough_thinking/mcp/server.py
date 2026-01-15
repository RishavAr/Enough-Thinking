
import sqlite3
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

DB = Path("enough-thinking/data/local/workflows.db")

app = FastAPI(title="EnoughThinking MCP-Style Server")

class SQLQuery(BaseModel):
    sql: str

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/query")
def query(body: SQLQuery):
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    try:
        cur.execute(body.sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return {"columns": cols, "rows": rows}
    finally:
        conn.close()
