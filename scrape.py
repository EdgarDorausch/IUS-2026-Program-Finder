#!/usr/bin/env python3
"""Scrape IUS 2026 session schedules from epapers2.org into data.json.

Usage: python3 scrape.py [sched_id ...]   (default: 1 2 = lectures and posters)
       python3 scrape.py --refresh 2      (ignore cached HTML)
       python3 scrape.py --details        (also fetch per-paper abstracts and keywords)
       python3 scrape.py --details --limit 20   (stop after 20 papers; for spot checks)
"""
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://epapers2.org/ius2026/ESR"
ROOT = Path(__file__).parent
CACHE = ROOT / "cache"


def fetch(url, cache_name, refresh=False):
    path = CACHE / cache_name
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ius-2026-overview)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        text = r.read().decode("utf-8", errors="replace")
    CACHE.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    time.sleep(0.5)
    return text


def clean(s):
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


ROW_HEAD = re.compile(r'class="SessionRowHeading"[^>]*>(.*?)</td>', re.S)
CELL = re.compile(
    r'class="SessionCell"[^>]*>\s*([^<]+?)<br>\s*'
    r"<a class=\"session\" href=\"javascript:js_popup\('/ius2026/ESR/session_view\.php\?session_id=(\d+)'[^>]*>(.*?)</a>"
    r"\s*\((\d+)&nbsp;papers\)(.*?)</td>",
    re.S,
)


def parse_schedule(text):
    """Yield session dicts; each SessionCell belongs to the nearest preceding row heading."""
    heads = [(m.start(), m.group(1)) for m in ROW_HEAD.finditer(text)]
    sessions = []
    for m in CELL.finditer(text):
        head = next((h for pos, h in reversed(heads) if pos < m.start()), "")
        parts = [clean(p) for p in re.split(r"<br\s*/?>", head)]
        rest = m.group(5)
        chair = re.search(r"Chr:\s*(.*?)<br>", rest)
        track = re.search(r"Track:\s*(.*?)<br>", rest)
        sessions.append({
            "id": int(m.group(2)),
            "code": clean(m.group(1)),
            "title": clean(m.group(3)),
            "expected": int(m.group(4)),
            "day": parts[0] if parts else "",
            "date": parts[1] if len(parts) > 1 else "",
            "time": parts[2] if len(parts) > 2 else "",
            "chair": clean(chair.group(1)) if chair else "",
            "track": clean(track.group(1)) if track else "",
        })
    return sessions


PAPER = re.compile(
    r"paper_details\.php\?paper_id=(\d+)'.*?"
    r"Tip\('([^']*)'\)[^>]*>([^<]*)</a></td>\s*"
    r'<td[^>]*>(.*?)</td>',
    re.S,
)
FIELD = re.compile(r"<b>([^<]+?):&nbsp;</b></td><td>(.*?)</td>", re.S)


def parse_session(text):
    fields = {clean(k): clean(v) for k, v in FIELD.findall(text)}
    papers = []
    for pid, topic_name, topic, cell in PAPER.findall(text):
        title, _, authors = cell.partition("<i>")
        papers.append({
            "id": int(pid),
            "topic": clean(topic),
            "topicName": clean(topic_name),
            "title": clean(title),
            "authors": [a for a in (x.strip() for x in clean(authors).split(",")) if a],
        })
    return fields, papers


# Paper detail pages use a different table markup than session pages: the label sits in a
# <b> inside a FormLabelL cell and the value in the FormLabelL cell that follows.
PAPER_FIELD = re.compile(
    r'<td class="FormLabelL"[^>]*><b>([^<]+?):</b></td>\s*'
    r'<td class="FormLabelL"[^>]*>(.*?)</td>',
    re.S,
)


def parse_paper_details(text):
    return {clean(k): clean(v) for k, v in PAPER_FIELD.findall(text)}


def add_details(sessions, refresh=False, limit=None):
    """Fetch paper_details.php for every paper; attach abstract and keywords in place."""
    papers = [p for s in sessions for p in s["papers"]]
    if limit:
        papers = papers[:limit]
    missing = []
    for i, p in enumerate(papers, 1):
        text = fetch(f"{BASE}/paper_details.php?paper_id={p['id']}", f"paper_{p['id']}.html", refresh)
        f = parse_paper_details(text)
        p["abstract"] = f.get("Abstract", "")
        p["keywords"] = [k for k in (x.strip() for x in f.get("Keywords", "").split(",")) if k]
        # The session-page regex truncates a few titles (e.g. paper 7204 -> "A"); the detail page is authoritative.
        detail_title = f.get("Paper Title", "")
        if len(detail_title) > len(p["title"]):
            print(f"  title fixed for {p['id']}: {p['title']!r} -> {detail_title[:60]!r}")
            p["title"] = detail_title
        if not p["abstract"]:
            missing.append(p["id"])
        if i % 50 == 0 or i == len(papers):
            print(f"  details {i}/{len(papers)}")
    print(f"\n{len(papers)} papers detailed, {len(missing)} without an abstract"
          + (f": {missing[:10]}" if missing else ""))


def main():
    args = sys.argv[1:]
    refresh = "--refresh" in args
    details = "--details" in args
    limit = None
    if "--limit" in args:
        at = args.index("--limit")
        limit = int(args[at + 1])
        del args[at:at + 2]
    sched_ids = [a for a in args if a not in ("--refresh", "--details")] or ["1", "2"]

    sessions, seen, problems = [], set(), 0
    for sid in sched_ids:
        sched = fetch(f"{BASE}/session_sched_view.php?sched_id={sid}", f"sched_{sid}.html", refresh)
        for s in parse_schedule(sched):
            if s["id"] in seen:
                continue
            seen.add(s["id"])
            text = fetch(f"{BASE}/session_view.php?session_id={s['id']}", f"session_{s['id']}.html", refresh)
            fields, papers = parse_session(text)
            s.update(type=fields.get("Session Type", ""), location=fields.get("Location", ""),
                     datetime=fields.get("Date & Time", ""), papers=papers)
            if s["chair"] == "":
                s["chair"] = fields.get("Chair", "")
            ok = len(papers) == s["expected"]
            problems += not ok
            print(f"{s['code']:8} id={s['id']:<4} {len(papers):>3}/{s['expected']:<3} {'ok' if ok else 'MISMATCH'}  {s['title'][:60]}")
            sessions.append(s)

    if details:
        print(f"\nFetching paper details for {sum(len(s['papers']) for s in sessions)} papers...")
        add_details(sessions, refresh, limit)

    total = sum(len(s["papers"]) for s in sessions)
    authors = {a for s in sessions for p in s["papers"] for a in p["authors"]}
    print(f"\n{len(sessions)} sessions, {total} papers, {len(authors)} unique authors, {problems} count mismatches")
    data = {"source": f"{BASE}/session_sched_view.php?sched_id={','.join(sched_ids)}",
            "fetched": time.strftime("%Y-%m-%d"), "sessions": sessions}
    (ROOT / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
