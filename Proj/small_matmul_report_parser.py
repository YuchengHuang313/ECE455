#!/usr/bin/env python3
"""
small_matmul_report_parser.py

Parse console output from small_matmul.cpp and optionally generate plots.

Features
- Extracts *all* numeric variables it can find:
  - Timings in ms (incl. GPU copy/compute breakdown)
  - Speedups (×)
  - GFLOPS
  - Number of matrices, verification samples, etc.
- Writes a tidy CSV with columns: run_id, section, metric, value, unit
- Optionally generates three plots for the performance summary:
  1) Timings (ms)
  2) Speedups (×)
  3) GFLOPS
- Supports multiple input files (merged CSV, separate plots per file)

Usage
------
# Single file, create plots into ./plots
python small_matmul_report_parser.py run1.txt --plots --outdir plots

# Multiple files, merged CSV + one set of plots per input
python small_matmul_report_parser.py out/runA.txt out/runB.txt --plots --outdir plots

# Parse only, no plots
python small_matmul_report_parser.py out/runA.txt --outcsv all_runs.csv

Notes
- Plots require matplotlib. CSV always written if --outcsv given (default: parsed_metrics.csv).
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def _lazy_import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def parse_small_matmul_output(text: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    variables: List[dict] = []
    timings_summary: Dict[str, float] = {}
    speedups_summary: Dict[str, float] = {}
    gflops_summary: Dict[str, float] = {}

    in_perf_summary = False
    in_speedup = False
    in_gflops = False

    ms_line = re.compile(r"^\s*(?P<label>[^:]+):\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*ms\s*$")
    gpu_breakdown_prefix = re.compile(r"^\s*-\s*(?P<label>[^:]+):\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*ms\s*$")
    speedup_line = re.compile(r"^\s*(?P<label>OpenMP|GPU\s*\(kernel only\)|GPU\s*\(total\)):\s*(?P<val>[-+]?\d+(?:\.\d+)?)x\s*$")
    gflops_line = re.compile(r"^\s*(?P<label>CPU\s*\(single\)|CPU\s*\(OpenMP\)|GPU\s*\(kernel\)):\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*GFLOPS\s*$", re.IGNORECASE)
    num_matrices_line = re.compile(r"^\s*Number of matrices:\s*(?P<val>\d+)\s*$", re.IGNORECASE)
    saved_verify_line = re.compile(r"^\s*Saved\s+(?P<val>\d+)\s+elements\s+for\s+verification", re.IGNORECASE)

    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if "========== PERFORMANCE SUMMARY ==========" in line:
            in_perf_summary = True
            in_speedup = False
            in_gflops = False
            continue
        if in_perf_summary and line.strip().startswith("Speedup vs single-threaded CPU"):
            in_speedup = True
            in_gflops = False
            continue
        if in_perf_summary and line.strip().startswith("Performance (GFLOPS)"):
            in_gflops = True
            in_speedup = False
            continue

        m = num_matrices_line.match(line)
        if m:
            variables.append({"section": "Header", "metric": "Number of matrices", "value": float(m.group("val")), "unit": ""})
            continue
        m = saved_verify_line.search(line)
        if m:
            variables.append({"section": "Verification", "metric": "Verification samples", "value": float(m.group("val")), "unit": "elements"})
            continue
        m = gpu_breakdown_prefix.match(line)
        if m:
            label = m.group("label").strip()
            val = float(m.group("val"))
            variables.append({"section": "GPU", "metric": f"GPU breakdown - {label}", "value": val, "unit": "ms"})
            continue
        m = ms_line.match(line)
        if m:
            label = re.sub(r"\s+", " ", m.group("label")).strip()
            val = float(m.group("val"))
            variables.append({"section": "Timings", "metric": label, "value": val, "unit": "ms"})
            if in_perf_summary and not in_speedup and not in_gflops:
                timings_summary[label] = val
            continue
        m = speedup_line.match(line)
        if m:
            label = re.sub(r"\s+", " ", m.group("label")).strip()
            val = float(m.group("val"))
            variables.append({"section": "Speedups", "metric": f"Speedup - {label}", "value": val, "unit": "x"})
            if in_perf_summary and in_speedup:
                speedups_summary[label] = val
            continue
        m = gflops_line.match(line)
        if m:
            label = re.sub(r"\s+", " ", m.group("label")).strip()
            val = float(m.group("val"))
            variables.append({"section": "GFLOPS", "metric": f"GFLOPS - {label}", "value": val, "unit": "GFLOPS"})
            if in_perf_summary and in_gflops:
                gflops_summary[label] = val
            continue

    df_all = pd.DataFrame(variables, columns=["section", "metric", "value", "unit"])
    return df_all, timings_summary, speedups_summary, gflops_summary

def plot_performance_summaries(run_id: str, timings, speedups, gflops, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt = _lazy_import_matplotlib()
    def _plot_bar(d, title, ylabel, fname):
        if not d:
            return
        labels = list(d.keys())
        values = [d[k] for k in labels]
        fig = plt.figure()
        ax = fig.gca()
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {run_id}")
        fig.tight_layout()
        fpath = outdir / fname
        fig.savefig(fpath, dpi=150)
        plt.close(fig)
        print(f"[PLOT] Wrote {fpath}")
    _plot_bar(timings, "Performance Summary: Timings", "ms", f"{run_id}_timings.png")
    _plot_bar(speedups, "Performance Summary: Speedups", "×", f"{run_id}_speedups.png")
    _plot_bar(gflops, "Performance Summary: GFLOPS", "GFLOPS", f"{run_id}_gflops.png")

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse small_matmul.cpp runtime report(s).")
    ap.add_argument("inputs", nargs="+", help="Path(s) to console output text files.")
    ap.add_argument("--outcsv", default="parsed_metrics.csv", help="Output CSV path.")
    ap.add_argument("--plots", action="store_true", help="Generate plots for performance summary.")
    ap.add_argument("--outdir", default="plots", help="Directory to write plots.")
    args = ap.parse_args()

    all_rows = []
    for inp in args.inputs:
        path = Path(inp)
        run_id = path.stem
        text = path.read_text(encoding="utf-8", errors="ignore")
        df_all, timings, speedups, gflops = parse_small_matmul_output(text)
        if df_all.empty:
            print(f"[WARN] No variables parsed from {path}")
        df_all.insert(0, "run_id", run_id)
        all_rows.append(df_all)
        if args.plots:
            plot_performance_summaries(run_id, timings, speedups, gflops, Path(args.outdir))

    if all_rows:
        merged = pd.concat(all_rows, ignore_index=True)
        outcsv = Path(args.outcsv)
        outcsv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(outcsv, index=False)
        print(f"[CSV] Wrote {outcsv} with {len(merged)} rows")
    else:
        print("[WARN] No rows parsed from any input.")

if __name__ == "__main__":
    main()
