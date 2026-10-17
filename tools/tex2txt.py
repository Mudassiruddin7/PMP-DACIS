"""Strip LaTeX to plain prose so that the human-writing linter (tell_check.py) can read it.

    python tools/tex2txt.py paper/paper_body.tex paper/sec_*.tex > prose.txt
"""
import re
import sys


def clean(tex):
    tex = re.sub(r"(?m)(?<!\\)%.*$", "", tex)                                     # comments
    tex = re.sub(r"\\begin\{(figure\*?|table\*?|tabular|equation\*?|align\*?|tikzpicture|abstract)\}.*?\\end\{\1\}",
                 lambda m: (re.sub(r"\\begin\{abstract\}|\\end\{abstract\}", "", m.group(0)) if m.group(1) == "abstract" else " "),
                 tex, flags=re.S)
    tex = re.sub(r"\$[^$]*\$", " X ", tex)                                          # inline math
    tex = re.sub(r"\\(?:cite|citep|citet)\{[^}]*\}", "[1]", tex)
    tex = re.sub(r"\\(?:cref|Cref|ref|eqref)\{[^}]*\}", "Fig. 1", tex)
    tex = re.sub(r"\\num\{[^}]*\}", "12.3", tex)
    tex = re.sub(r"\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}", r"\n\n\1.\n\n", tex)
    tex = re.sub(r"\\paragraph\{([^}]*)\}", r"\n\n\1 ", tex)
    tex = re.sub(r"\\label\{[^}]*\}|\\input\{[^}]*\}|\\centering|\\small|\\footnotesize|\\noindent", "", tex)
    tex = re.sub(r"\\(?:emph|textbf|textit|texttt|textsc|mathrm|text)\{([^}]*)\}", r"\1", tex)
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", tex)
    tex = tex.replace("~", " ").replace("``", '"').replace("''", '"').replace("--", "-").replace("\\%", "%")
    tex = re.sub(r"[{}]", "", tex)
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r"\n\s*\n+", "\n\n", tex)
    return tex.strip()


if __name__ == "__main__":
    out = []
    for p in sys.argv[1:]:
        out.append(clean(open(p, encoding="utf-8").read()))
    sys.stdout.write("\n\n".join(out) + "\n")
