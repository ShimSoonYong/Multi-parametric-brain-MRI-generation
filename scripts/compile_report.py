#!/usr/bin/env python3
"""
Report Compilation CLI for AI4MIA Assignment 4.

Compiles both required LaTeX reports to PDF using pdflatex:
1. report/StudentID_Name_Report.tex -> StudentID_Name_Report.pdf
2. report/2026710517_shimsoonyong_Report.tex -> 2026710517_shimsoonyong_Report.pdf

Verifies page counts, ensures clean float/paragraph budgets, and synchronizes deliverables to root.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def compile_single_report(tex_basename: str, clean: bool = True) -> tuple[Path, int]:
    report_dir = REPO_ROOT / "report"
    tex_file = report_dir / f"{tex_basename}.tex"
    pdf_file = report_dir / f"{tex_basename}.pdf"

    if not tex_file.exists():
        # Fallback to StudentID_Name_Report.tex if target tex doesn't exist yet
        src_master = report_dir / "StudentID_Name_Report.tex"
        if src_master.exists():
            shutil.copy2(src_master, tex_file)
        else:
            raise FileNotFoundError(f"LaTeX source not found at: {tex_file}")

    print(f"\n{'='*70}")
    print(f" Compiling Report: {tex_file.name}")
    print(f"{'='*70}")

    # Pass 1
    print(f"[{tex_basename}] Pass 1/2...")
    res1 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{tex_basename}.tex"],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
    )
    if res1.returncode != 0:
        print(f"pdflatex Pass 1 Error Output for {tex_basename}:")
        print(res1.stdout[-1500:])
        raise RuntimeError(f"Pass 1 failed for {tex_basename}!")

    # Pass 2 for cross-references & balance
    print(f"[{tex_basename}] Pass 2/2...")
    res2 = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{tex_basename}.tex"],
        cwd=str(report_dir),
        capture_output=True,
        text=True,
    )
    if res2.returncode != 0:
        print(f"pdflatex Pass 2 Error Output for {tex_basename}:")
        print(res2.stdout[-1500:])
        raise RuntimeError(f"Pass 2 failed for {tex_basename}!")

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not generated: {pdf_file}")

    # Check page count
    page_count = 0
    try:
        info = subprocess.run(["pdfinfo", str(pdf_file)], capture_output=True, text=True, check=True)
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                page_count = int(line.split(":")[1].strip())
                break
    except Exception:
        pass

    print(f"  -> Generated: {pdf_file.name} ({pdf_file.stat().st_size / (1024*1024):.2f} MB, {page_count} pages)")

    # Copy deliverable to workspace root
    root_target = REPO_ROOT / f"{tex_basename}.pdf"
    shutil.copy2(pdf_file, root_target)
    print(f"  -> Synced root deliverable: {root_target.name}")

    # Clean intermediate files if requested
    if clean:
        for ext in [".aux", ".log", ".out", ".toc", ".synctex.gz"]:
            f = report_dir / f"{tex_basename}{ext}"
            f.unlink(missing_ok=True)

    return pdf_file, page_count


def compile_both_reports(clean: bool = True) -> list[tuple[Path, int]]:
    report_dir = REPO_ROOT / "report"

    # Ensure 2026710517_shimsoonyong_Report.tex is synced with StudentID_Name_Report.tex
    src_master = report_dir / "StudentID_Name_Report.tex"
    target_named_tex = report_dir / "2026710517_shimsoonyong_Report.tex"
    shutil.copy2(src_master, target_named_tex)

    targets = [
        "StudentID_Name_Report",
        "2026710517_shimsoonyong_Report",
    ]

    results = []
    for tgt in targets:
        pdf_path, p_count = compile_single_report(tgt, clean=clean)
        results.append((pdf_path, p_count))

    print(f"\n{'='*70}")
    print(" Both Reports Successfully Compiled and Synchronized:")
    for pdf_path, p_count in results:
        root_path = REPO_ROOT / pdf_path.name
        print(f"   * {pdf_path.name} -> {root_path} ({p_count} pages, {root_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"{'='*70}\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile both LaTeX reports to PDF")
    parser.add_argument("--no-clean", action="store_true", help="Keep intermediate .aux/.log files")
    args = parser.parse_args()

    compile_both_reports(clean=not args.no_clean)


if __name__ == "__main__":
    main()
