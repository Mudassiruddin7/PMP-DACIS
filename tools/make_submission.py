"""Collect everything that goes to OpenReview into one folder.

    python tools/make_submission.py OUT_DIR

* paper_review.pdf            anonymous PDF (upload this)
* paper_authors.pdf           same paper with the author block (for the camera-ready and for preprint servers)
* supplement.pdf              extra tables (anonymous)
* supplementary_anonymous.zip notebook + library + tests, no names
* openreview_fields.md        title, abstract, keywords and the author list to type into the submission form
"""
import os
import re
import shutil
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AUTHORS = [
    ("Mohammed Mudassir Uddin", "mohd.mudassiruddin7@gmail.com", "Osmania University, Hyderabad, India"),
    ("Fahmina Taranum", "ftaranum@mjcollege.ac.in", "Osmania University, Hyderabad, India"),
    ("Tasneem Bano Rehman", "tasneem.bano@mjcollege.ac.in", "Muffakham Jah College of Engineering and Technology, Hyderabad, India"),
    ("Sumrana Siddiqui", "sumrana.siddiqui@mjcollege.ac.in", "Muffakham Jah College of Engineering and Technology, Hyderabad, India"),
    ("Sara Rizwan", "sara.rizwan014@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Samaanah Abdus Salam", "samaanah345@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Quratulain Nayeem", "quratulainnayeem@gmail.com", "Osmania University, Hyderabad, India"),
    ("Summaiya Unnisa Begum", "summaiyaunisa@gmail.com", "Jawaharlal Nehru Technological University Hyderabad, India"),
    ("Sadaf Kauser", "sadafkauser8143@gmail.com", "Birla Institute of Technology and Science Pilani, Dubai Campus"),
    ("Muzammil Shareef", "mshareef1@learn.athabascau.ca", "Athabasca University, Canada"),
    ("Abdul Raheem Ahmed", "Aahmed011@my.wilmu.edu", "Wilmington University, United States"),
    ("Mohammed Aqib Abdullah", "aqib.abdullah13@gmail.com", "(affiliation not provided)"),
]


def abstract_text(tex_path):
    s = open(tex_path, encoding="utf-8").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def main(out):
    os.makedirs(out, exist_ok=True)
    paper = os.path.join(ROOT, "paper")
    for src, dst in (("paper_review.pdf", "paper_review.pdf"), ("paper_final.pdf", "paper_authors.pdf"), ("paper_supp.pdf", "supplement.pdf")):
        if os.path.exists(os.path.join(paper, src)):
            shutil.copy(os.path.join(paper, src), os.path.join(out, dst))
    with zipfile.ZipFile(os.path.join(out, "supplementary_anonymous.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for rel in ("notebooks/pmp_dacis_cv4eo.py", "requirements.txt", "tests/test_core.py", "tools/build_notebook.py",
                    "tools/notebook_template.py"):
            if os.path.exists(os.path.join(ROOT, rel)):
                z.write(os.path.join(ROOT, rel), "pmp_dacis_code/" + rel)
        for base, dirs, files in os.walk(os.path.join(ROOT, "pmp_dacis")):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(base, f)
                    z.write(full, "pmp_dacis_code/" + os.path.relpath(full, ROOT).replace("\\", "/"))
        for base, dirs, files in os.walk(os.path.join(ROOT, "paper", "generated")):
            for f in files:
                full = os.path.join(base, f)
                z.write(full, "pmp_dacis_code/results_tables/" + f)
    title = re.search(r"\\title\{(.*?)\}", open(os.path.join(paper, "paper_title.tex"), encoding="utf-8").read(), re.S).group(1)
    with open(os.path.join(out, "openreview_fields.md"), "w", encoding="utf-8") as f:
        f.write("# OpenReview submission form (4th CV4EO workshop at WACV 2027)\n\n")
        f.write(f"**Title:** {title}\n\n**Track / type:** full paper (8 pages plus references), archival, double-blind\n\n")
        f.write(f"**Abstract:** {abstract_text(os.path.join(paper, 'sec_abstract.tex'))}\n\n")
        f.write("**Keywords:** channel pruning; few-shot learning; Sentinel-2 multispectral imagery; plant disease; edge inference\n\n")
        f.write("**Authors (enter in this order; every author needs an up-to-date OpenReview profile):**\n\n")
        for i, (n, e, a) in enumerate(AUTHORS, 1):
            f.write(f"{i}. {n} <{e}>, {a}\n")
        f.write("\nUpload `paper_review.pdf` as the paper (it is anonymous) and `supplementary_anonymous.zip` as supplementary material.\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "submission"))
