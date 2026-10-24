"""Offline check of the memory layer with REAL embeddings (Chroma's default MiniLM).
No Nebius key and no Gmail needed. Uses a throwaway temp dir, never touches data/chroma.

Usage: python3 scripts/test_memory.py

It tells you: does recall find the right item, does an unrelated question get nothing,
does re-ingesting dedupe, does it survive a restart, and what max_distance to use.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.memory import Memory  # noqa: E402

EMAILS = [
    {"id": "t1", "from": "billing@acme.com", "subject": "Invoice 42 due Friday",
     "date": "Fri, 23 Oct 2026", "snippet": "Your invoice of $120 is due Friday."},
    {"id": "t2", "from": "hr@example.com", "subject": "Team offsite on Nov 14",
     "date": "Thu, 22 Oct 2026", "snippet": "The offsite is in Goa on Nov 14. Please RSVP by Nov 1."},
    {"id": "t3", "from": "noreply@github.com", "subject": "[nebius-personal-ai] new pull request",
     "date": "Fri, 23 Oct 2026", "snippet": "A collaborator opened PR #1: add retry logic to the LLM client."},
    {"id": "t4", "from": "deals@shop.example", "subject": "50% off sneakers",
     "date": "Sat, 24 Oct 2026", "snippet": "Flash sale ends tonight. Free shipping on all shoes."},
]

# (question, text that must appear in the top hit, or None if nothing should be recalled)
CASES = [
    ("when is the invoice due?", "Invoice 42"),
    ("what is happening in Goa?", "offsite"),
    ("did anyone open a pull request on my repo?", "PR #1"),
    ("any discounts on shoes?", "sneakers"),
    ("what is the weather in Paris?", None),
    ("how do I bake sourdough bread?", None),
]


def main() -> int:
    s = get_settings()
    thr = s.memory_max_distance
    fails = 0
    relevant, irrelevant = [], []

    with tempfile.TemporaryDirectory() as d:
        m = Memory(d)  # default embedding fn, same as the app
        m.ingest_tool_result("search_emails", json.dumps(EMAILS))
        m.add_turn("any invoices due?", "Invoice 42 ($120) from Acme is due Friday.")
        print(f"seeded {m.count()} items (4 emails + 1 turn); threshold = {thr}\n")

        for q, must in CASES:
            top = m.search(q, k=1, max_distance=9.0)[0]  # unfiltered best match
            dist = top["distance"]
            kept = dist <= thr
            if must is None:
                ok = not kept
                irrelevant.append(dist)
            else:
                ok = kept and must.lower() in top["text"].lower()
                relevant.append(dist)
            fails += not ok
            print(f"[{'PASS' if ok else 'FAIL'}] d={dist:<6} {q!r}\n         top: {top['text'][:70]!r}")

        # dedupe: same ids again must not grow the store
        before = m.count()
        m.ingest_tool_result("search_emails", json.dumps(EMAILS))
        ok = m.count() == before
        fails += not ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] re-ingest dedupes ({before} -> {m.count()})")

        # snippet must not clobber a stored full body
        m.ingest_tool_result("get_email", json.dumps({**EMAILS[0], "body": "FULL BODY: pay to IBAN ending 99"}))
        m.ingest_tool_result("search_emails", json.dumps([EMAILS[0]]))
        ok = "FULL BODY" in m.col.get(ids=["email:t1"])["documents"][0]
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] full body survives a later snippet-only hit")

        # persistence: reopen the same directory
        n = m.count()
        ok = Memory(d).count() == n
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] data persists across a fresh client ({n} items)")

        if relevant and irrelevant:
            hi, lo = max(relevant), min(irrelevant)
            print(f"\nrelevant hits: up to d={hi} | irrelevant best: d={lo}")
            if hi < lo:
                print(f"suggested MEMORY_MAX_DISTANCE ~ {round((hi + lo) / 2, 2)} (put it in .env)")
            else:
                print("relevant and irrelevant distances overlap; a threshold alone won't separate them")

    print(f"\n{'ALL GOOD' if not fails else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
