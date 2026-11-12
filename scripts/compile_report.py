#!/usr/bin/env python3
"""
Report Compilation CLI for AI4MIA Assignment 4.

Compiles report/StudentID_Name_Report.tex to PDF using pdflatex,
verifies page count, and updates root submission deliverables:
- 20267105171_shimsoonyong_Report.pdf
- StudentID_Name_Report.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compile_pdf(clean: bool = True) -> Path:
    report_dir = REPO_ROOT / "report"
    tex_file = report_dir / "StudentID_Name_Report.tex"
    pdf_file = report_dir / "StudentID_Name_Report.pdf"

    if not tex_file.exists():
        raise FileNotFoundError(f"LaTeX file not found at: {tex_file}")

    print(f"\n{'='*70}")
    print(" Compiling LaTeX Report with pdflatex...")
    print(f" Source: {tex_file}")
    print(f"{'='*70}\n")

    # Pass 1
    print("Pass 1/2...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "StudentID_Name_Report.tex"],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
    )
    if res1.returncode != 0:
        print("pdflatex Pass 1 Error Output:")
        print(res1.stdout[-1500:])
        raise RuntimeError("Pass 1 failed!")

    # Pass 2 for cross-references & balance
    print("Pass 2/2...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "StudentID_Name_Report.tex"],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
    )
    if res2.returncode != 0:
        print("pdflatex Pass 2 Error Output:")
        print(res2.stdout[-1500:])
        raise RuntimeError("Pass 2 failed!")

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not generated: {pdf_file}")

    # Check page count
    page_count = None
    try:
        info = subprocess.run(["pdfinfo", str(pdf_file)], capture_output=True, text=True, check=True)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                page_count = int(line.split(":")[1].strip())
                break
    except Exception:
        pass

    print(f"\nPDF successfully compiled: {pdf_file}")
    if page_count:
        print(f"Total Page Count: {page_count} pages")

    # Copy to required submission deliverables
    sub_targets = [
        REPO_ROOT / "20267105171_shimsoonyong_Report.pdf",
        REPO_ROOT / "StudentID_Name_Report.pdf",
    ]
    for target in sub_targets:
        shutil.copy2(pdf_file, target)
        print(f"Copied deliverable -> {target.name} ({target.stat().st_size / (1024*1024):.2f} MB)")

    # Clean intermediate files if requested
    if clean:
        for ext in [".aux", ".log", ".out", ".toc", ".synctex.gz"]:
            for f in report_dir.glob(f"*{ext}"):
                f.unlink(missing_ok=True)
        print("Cleaned temporary LaTeX build artifacts (.aux, .log, .out).")

    return pdf_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile report LaTeX to PDF")
    parser.add_argument("--no-clean", action="store_true", help="Keep intermediate .aux/.log files")
    args = parser.parse_args()

    compile_pdf(clean=not args.no_clean)


if __name__ == "__main__":
    main()
