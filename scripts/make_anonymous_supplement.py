"""Anonymous supplementary zip for the TMLR double-blind review: the release folder without the paper, authors, licence or citation files."""
from __future__ import annotations
import shutil, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "release" / "promptshift-crc"
OUT = ROOT / "release" / "anonymous_supplement"
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(REL, OUT, ignore=shutil.ignore_patterns("paper", "CITATION.cff", "LICENSE", "build_tmlr.py", "make_release.py", "make_anonymous_supplement.py", "__pycache__", ".git"))
readme = (OUT / "README.md").read_text(encoding="utf-8")
readme = readme.replace("Code, model outputs, result tables and the paper source for\n\n> Opoku, J. and Banahene, D. *PromptShift-CRC: What Drift-Aware Weighting Adds to Online Conformal Risk Control for Foundation Models.*",
                        "Code, model outputs and result tables for the submission *PromptShift-CRC: What Drift-Aware Weighting Adds to Online Conformal Risk Control for Foundation Models* (anonymous supplementary material).")
readme = readme.split("## Licence and citation")[0].rstrip() + "\n\n## Licence\n\nMIT licence; the full text and the authors' names are added on acceptance.\n"
readme = readme.replace("`figures_v2/` the figures. `paper/` the paper source and PDF.", "`figures_v2/` the figures.")
(OUT / "README.md").write_text(readme, encoding="utf-8")
bad = [p for p in OUT.rglob("*") if p.is_file() and any(w in p.read_bytes().lower() for w in (b"opoku", b"banahene", b"utrgv", b"jopoku", b"gmail"))]
assert not bad, f"identifying strings in {bad}"
zp = ROOT / "release" / "PromptShift_anonymous_supplement.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print("anonymous supplement:", zp, round(zp.stat().st_size / 1e6, 1), "MB")
