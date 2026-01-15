
import argparse, json, requests
from datetime import datetime
from collections import defaultdict

def call_tool(server_url: str, sql: str):
    r = requests.post(f"{server_url}/query", json={"sql": sql}, timeout=30)
    r.raise_for_status()
    return r.json()

def parse_dt(s: str):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")

def compute_overlaps(events):
    overlaps = []
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            a = events[i]; b = events[j]
            if a[1] < b[2] and b[1] < a[2]:
                overlaps.append((a[0], b[0]))
    return overlaps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--question", required=True)
    args = ap.parse_args()

    q = args.question.lower()

    # ---------- CALENDAR ----------
    if "calendar" in q or "meeting" in q or "event" in q:
        tool = call_tool(args.server, "SELECT title,start,end FROM calendar ORDER BY start LIMIT 3;")
        cols, rows = tool["columns"], tool["rows"]

        events = []
        for r in rows:
            d = dict(zip(cols, r))
            events.append((d["title"], parse_dt(d["start"]), parse_dt(d["end"])))

        overlap_pairs = compute_overlaps(events)

        out = {
            "next_3_events": [
                {
                    "title": t,
                    "start": s.strftime("%Y-%m-%d %H:%M"),
                    "end": e.strftime("%Y-%m-%d %H:%M")
                } for t,s,e in events
            ],
            "overlap": "No" if not overlap_pairs else "Yes",
            "overlap_details": "" if not overlap_pairs else ", ".join(
                [f"{a} overlaps {b}" for a,b in overlap_pairs]
            )
        }

        print(json.dumps(out, indent=2))
        return

    # ---------- EXPENSES ----------
    if "expense" in q or "spend" in q or "category" in q:
        tool = call_tool(args.server, "SELECT category, amount FROM expenses;")
        cols, rows = tool["columns"], tool["rows"]

        totals = defaultdict(float)
        for r in rows:
            d = dict(zip(cols, r))
            totals[d["category"]] += float(d["amount"])

        out = {
            "total_spend_per_category": {
                k: round(v, 2) for k, v in totals.items()
            }
        }

        print(json.dumps(out, indent=2))
        return

    # ---------- FALLBACK ----------
    print(json.dumps({
        "error": "No supported tool route for this question"
    }, indent=2))

if __name__ == "__main__":
    main()
