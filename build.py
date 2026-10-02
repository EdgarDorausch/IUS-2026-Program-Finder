#!/usr/bin/env python3
"""Embed data.json and embeddings.json into template.html -> index.html."""
import json
from pathlib import Path

ROOT = Path(__file__).parent


def inline(s):
    """Serialize compactly and keep the payload from closing the host <script> element."""
    return json.dumps(s, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


data = json.loads((ROOT / "data.json").read_text(encoding="utf-8"))
page = (ROOT / "template.html").read_text(encoding="utf-8")
page = page.replace("/*DATA*/null", inline(data))

emb_path = ROOT / "embeddings.json"
if emb_path.exists():
    emb = json.loads(emb_path.read_text(encoding="utf-8"))
    page = page.replace("/*EMB*/null", inline(emb))
    print(f"embeddings: {emb['n']}x{emb['d']} int8, fp={emb['fp']}, {len(emb['b64']) // 1024} KB base64")
else:
    print("embeddings.json missing - building without semantic search (run: node embed.mjs)")

(ROOT / "index.html").write_text(page, encoding="utf-8")
print(f"index.html written ({len(page) // 1024} KB)")
