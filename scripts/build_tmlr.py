"""Build the TMLR versions of the paper from main_v2.tex.

  tmlr/anonymous   double-blind review PDF (no author names, no repository URL)
  tmlr/preprint    same with authors and the public repository link

Both are flat folders that compile with pdflatex + bibtex alone.
"""
from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "main_v2.tex").read_text(encoding="utf-8").replace("\r\n", "\n")
FILES = ["numbers_generated.tex", "tables_generated.tex", "tables_appendix_generated.tex", "theory_section.tex", "theory_appendix.tex",
         "sec_intro_related_method.tex", "sec_protocol.tex", "sec_results_synthetic.tex", "sec_results_real.tex", "rtp_replication_text.tex",
         "sec_discussion.tex", "references_v2.bib", "new_refs.bib"]
STY = ["tmlr.sty", "tmlr.bst", "fancyhdr.sty"]

PRE_OLD = SRC[:SRC.index(r"\input{numbers_generated}")]
PREAMBLE = r"""\documentclass{article}
\usepackage[%s]{tmlr}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{url}

\newtheorem{theorem}{Theorem}
\newtheorem{lemma}{Lemma}
\newtheorem{assumption}{Assumption}
\newtheorem{definition}{Definition}
\newtheorem{proposition}{Proposition}
\newtheorem{corollary}{Corollary}
\newtheorem{remark}{Remark}

"""
AUTHORS = r"""\title{Conformal Risk Control for Foundation Models Under Prompt Shift:\\ What Weighting Adds to the Online Update}
\author{\name Jeffery Opoku \\
\addr The University of Texas Rio Grande Valley, Edinburg, Texas, USA \\
\texttt{jeffery.opoku01@utrgv.edu}
\AND
\name David Banahene \\
\addr Florida International University, Miami, Florida, USA \\
\texttt{abanahene54@gmail.com}}
\def\month{10}
\def\year{2026}
\def\openreview{\url{https://openreview.net/forum?id=XXXXXXXXXX}}
\makeatletter
\def\AND{\par\vskip \interauthorskip}
\makeatother
"""


def build(kind: str) -> Path:
    out = ROOT / "tmlr" / kind
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(ROOT / f, out / f)
    for f in STY:
        shutil.copy2(ROOT / "tmlr" / f, out / f)
    shutil.copytree(ROOT / "figures_v2", out / "figures_v2")
    body = SRC[SRC.index(r"\input{numbers_generated}"):]
    # drop the arXiv-style title/author/maketitle block, keep from the abstract on
    abstract = body[body.index(r"\begin{abstract}"):]
    authors_block = AUTHORS if kind == "preprint" else r"\title{Conformal Risk Control for Foundation Models Under Prompt Shift:\\ What Weighting Adds to the Online Update}" + "\n" + r"\author{Anonymous}" + "\n"
    head = r"\input{numbers_generated}" + "\n" + authors_block + "\n\\begin{document}\n\\maketitle\n"
    tex = PREAMBLE % ("preprint" if kind == "preprint" else "") + head + abstract
    tex = tex.replace(r"\bibliographystyle{plainnat}", r"\bibliographystyle{tmlr}")
    tex = tex.replace(r"\bibliography{references,new_refs}", r"\bibliography{references_v2,new_refs}")
    tex = re.sub(r"\\noindent\\textbf\{Keywords:\}.*?\n", "", tex)
    if kind == "anonymous":
        _anon = ("The code, the recorded model outputs (risk scores, features and the identifiers needed to rebuild the streams), and the result tables are provided as supplementary material, and a public repository will be linked on acceptance. Question and prompt text from MMLU and TruthfulQA can be re-downloaded from the public datasets; the RealToxicityPrompts prompt and continuation text is withheld because it contains offensive language, and the generation script regenerates it. Sampled continuations are not bit-reproducible across hardware, so the released risk scores are canonical. Appendix~\\ref{app:repro} gives the commands.")
        # a function, not a string: in a re.sub template the backslash of \ref is read as an escape and turns \r into a carriage return
        tex = re.sub(r"The code, the recorded model outputs.*?Appendix~\\ref\{app:repro\} gives the commands\.", lambda m: _anon, tex, flags=re.S)
        assert "jopoku16" not in tex, "repository URL leaked into the anonymous version"
    (out / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"], ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        subprocess.run(cmd, cwd=out, capture_output=True)
    log = (out / "main.log").read_text(encoding="utf-8", errors="ignore")
    for junk in ("main.log", "main.aux", "main.blg", "main.out"):
        pass
    errs = re.findall(r"^!.*$", log, flags=re.M)
    und = re.findall(r"Reference `[^']*' on page \d+ undefined", log) + re.findall(r"Citation `[^']*' on page \d+ undefined", log)
    pages = re.search(r"Output written on main.pdf \((\d+) pages", log)
    if kind == "anonymous":
        src_check = " ".join(p.read_text(encoding="utf-8", errors="ignore").lower() for p in out.glob("*.tex")) + (out / "new_refs.bib").read_text(encoding="utf-8").lower()
        assert not any(w in src_check for w in ("opoku", "banahene", "utrgv", "jopoku", "gmail", "fable")), "identifying text in the anonymous source"
        for junk in ("main.log", "main.aux", "main.blg", "main.out"):
            (out / junk).unlink(missing_ok=True)
    print(kind, "pages:", pages.group(1) if pages else None, "errors:", errs[:3], "undefined:", und[:3])
    return out


if __name__ == "__main__":
    build("anonymous")
    build("preprint")
