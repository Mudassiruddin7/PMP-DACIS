"""Build notebooks/pmp_dacis_cv4eo.py: a single-file marimo notebook that embeds the library and paper sources.

    python tools/build_notebook.py
"""
import base64
import io
import os
import tarfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def tar_b64(base_dir, rel_dir, keep=lambda p: True, arc_prefix=""):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
        top = os.path.join(base_dir, rel_dir)
        for dp, dn, fn in os.walk(top):
            dn[:] = [d for d in dn if d not in ("__pycache__", "generated")]
            for f in sorted(fn):
                full = os.path.join(dp, f)
                rel = os.path.relpath(full, top)
                if not keep(rel):
                    continue
                info = tf.gettarinfo(full, arcname=os.path.join(arc_prefix, rel).replace("\\", "/"))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 1760000000  # fixed, after 1980, so extracted files can be zipped
                with open(full, "rb") as fh:
                    tf.addfile(info, fh)
    b = base64.b64encode(buf.getvalue()).decode()
    return "\n".join(b[i:i + 110] for i in range(0, len(b), 110))


def main():
    pkg = tar_b64(ROOT, "pmp_dacis", keep=lambda r: r.endswith(".py"), arc_prefix="pmp_dacis")
    paper = tar_b64(ROOT, "paper", keep=lambda r: not r.endswith((".pdf", ".aux", ".log", ".out", ".bbl", ".blg")))
    tpl = open(os.path.join(ROOT, "tools", "notebook_template.py"), encoding="utf-8").read()
    assert "@@PKG_B64@@" in tpl and "@@PAPER_B64@@" in tpl
    nb = tpl.replace("@@PKG_B64@@", pkg).replace("@@PAPER_B64@@", paper)
    out = os.path.join(ROOT, "notebooks", "pmp_dacis_cv4eo.py")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", newline="\n").write(nb)
    print(f"wrote {out}  ({len(nb) / 1e3:.0f} kB; package {len(pkg) / 1e3:.0f} kB, paper {len(paper) / 1e3:.0f} kB base64)")


if __name__ == "__main__":
    main()
