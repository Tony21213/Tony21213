"""Survey a lab archive (exocad case folders and webview exports) before learning from it.

Walks a folder tree and reports, without patient-identifying names, what can
be used for training: case folders, exocad project/construction files,
scans, finished designs and webview HTML exports - and, for webviews, which
teeth have both a preparation and a single-tooth wax-up.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

_KEEP = re.compile(r"(prep|die|stumpf|jaw|kiefer|upper|lower|ober|unter|antag|margin|scan|cad|cam|design|crown|"
                   r"krone|wax|model|gingiva|restor|result|препар|челюст|скан|ваксап|антагон|десн|коронк)", re.I)
CASE_MARKERS = (".dentalproject", ".constructioninfo", ".html")


def anonymize(name: str) -> str:
    """Keep technical words, tooth numbers and the extension; mask everything else."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    parts = re.split(r"([\-_. ()])", stem)
    out = [p if (not p or re.fullmatch(r"[\-_. ()]", p) or _KEEP.search(p) or re.fullmatch(r"\d{1,2}", p))
           else "*" for p in parts]
    return "".join(out) + (f".{ext}" if ext else "")


def survey(root: str | Path, *, inspect_webviews: bool = False, limit: int | None = None) -> dict:
    root = Path(root)
    ext_count: Counter = Counter()
    cases: dict[Path, Counter] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        ext_count[ext] += 1
        if ext in CASE_MARKERS:
            cases.setdefault(p.parent, Counter())
    for folder in cases:
        for p in folder.iterdir():
            if p.is_file():
                cases[folder][p.suffix.lower()] += 1
    report = {"root": str(root), "files_by_extension": dict(ext_count.most_common()),
              "case_folders": len(cases), "examples": []}
    usable = Counter()
    for k, folder in enumerate(sorted(cases)):
        if limit is not None and k >= limit:
            break
        entry = {"folder": anonymize(folder.name), "files": dict(cases[folder]),
                 "names": sorted({anonymize(p.name) for p in folder.iterdir() if p.is_file()})[:20]}
        if inspect_webviews:
            entry["webviews"] = []
            for html in folder.glob("*.html"):
                entry["webviews"].append(_webview_summary(html))
                for t in entry["webviews"][-1].get("trainable_teeth", []):
                    usable["anterior" if t % 10 <= 3 else "posterior"] += 1
        report["examples"].append(entry)
    if inspect_webviews:
        report["trainable_teeth"] = dict(usable)
    return report


def _webview_summary(path: Path) -> dict:
    from .webview import classify, load_webview

    try:
        objects = load_webview(path)
    except Exception as exc:  # not a webview, password protected, ...
        return {"file": anonymize(path.name), "error": str(exc)[:120]}
    kinds = Counter()
    preps, waxups = set(), set()
    for o in objects:
        c = classify(o)
        kinds[c["kind"]] += 1
        if c["kind"] == "prep":
            preps.update(c["teeth"])
        if c["kind"] == "waxup" and len(c["teeth"]) == 1:
            waxups.update(c["teeth"])
    return {"file": anonymize(path.name), "objects": dict(kinds),
            "trainable_teeth": sorted(preps & waxups)}
