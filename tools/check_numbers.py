"""List every \\num{key} used in the paper sources that the experiment code did not write, and every '??' in a PDF.

    python tools/check_numbers.py [paper_dir] [pdf ...]

The first part reads paper/sec_*.tex and the generated numbers.tex; the second part scans compiled PDFs for the
placeholder that \\num prints for a missing key, and for unresolved references.
"""
import glob
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(argv):
    paper = argv[0] if argv else os.path.join(ROOT, "paper")
    pdfs = argv[1:]
    keys = set()
    nf = os.path.join(paper, "generated", "numbers.tex")
    if os.path.exists(nf):
        keys = set(re.findall(r"csname num@(.*?)\\endcsname", open(nf, encoding="utf-8").read()))
    missing = {}
    for f in sorted(glob.glob(os.path.join(paper, "*.tex"))):
        txt = open(f, encoding="utf-8").read()
        txt = re.sub(r"(?m)(?<!\\)%.*$", "", txt)
        txt = "\n".join(line for line in txt.splitlines() if "\\newcommand" not in line)       # macro definitions use #1
        used = re.findall(r"\\num\{([^}]+)\}", txt) + re.findall(r"\\numif\{([^}]+)\}", txt)
        for m in re.finditer(r"\\ci\{([^}]+)\}", txt):
            used += [f"d:{m.group(1)}:lo", f"d:{m.group(1)}:hi"]
        for k in used:
            if k not in keys:
                missing.setdefault(k, set()).add(os.path.basename(f))
    optional = {k for k in missing if k.startswith("repl:")}          # second-machine numbers appear only if a replication exists
    bad = {k: v for k, v in missing.items() if k not in optional}
    print(f"numbers.tex has {len(keys)} keys; {len(bad)} keys used in the text are missing" + (f" ({len(optional)} optional)" if optional else ""))
    for k, v in sorted(bad.items()):
        print(f"  MISSING {k}   in {', '.join(sorted(v))}")
    problems = len(bad)
    if pdfs:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        for p in pdfs:
            doc = fitz.open(p)
            text = "\n".join(pg.get_text() for pg in doc)
            q = len(re.findall(r"\?\?", text))
            print(f"{os.path.basename(p)}: {doc.page_count} pages, {q} '??' marks")
            problems += q
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
