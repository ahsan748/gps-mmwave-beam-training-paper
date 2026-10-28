# Checks that DeepSense seq_index order = real time order.
# Usage: python check_time_order.py path\to\scenario9.csv
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8-sig")))
cols = rows[0].keys()
tcol = next(c for c in cols if "time" in c.lower())
scol = next(c for c in cols if "seq" in c.lower())
def secs(v):  # handles "11-37-50-140", "['11:37:50.140']", "2021-05-03 11:37:50.140"
    d = re.findall(r"\d+", v)
    if len(d) >= 6 and int(d[0]) > 1900: d = d[3:]     # drop a date part
    h, m, s = int(d[0]), int(d[1]), int(d[2])
    frac = int(d[3]) / 10 ** len(d[3]) if len(d) > 3 else 0.0   # "050" -> 0.050
    return h * 3600 + m * 60 + s + frac
t = [secs(r[tcol]) for r in rows]; q = [int(float(r[scol])) for r in rows]
back = sum(t[i + 1] < t[i] for i in range(len(t) - 1))
within_back = sum(t[i + 1] < t[i] and q[i + 1] == q[i] for i in range(len(t) - 1))
starts = {}
for ti, qi in zip(t, q): starts.setdefault(qi, ti)
ids = sorted(starts); start_t = [starts[i] for i in ids]
seq_back = sum(start_t[i + 1] < start_t[i] for i in range(len(ids) - 1))
print(f"time column: {tcol} | seq column: {scol} | rows: {len(rows)} | sequences: {len(ids)}")
print(f"first time {rows[0][tcol]}  last time {rows[-1][tcol]}")
print(f"rows where time goes backwards: {back} (inside a sequence: {within_back})")
print(f"sequences that start earlier than the previous sequence: {seq_back}")
print("RESULT:", "PASS - seq_index order is time order" if seq_back == 0 and within_back == 0
      else "CHECK - some order breaks (if the capture spans midnight or several days, look at the dates)")
