from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize coverage intervals separated by major gaps.")
    parser.add_argument("--intervals-csv", required=True)
    parser.add_argument("--gaps-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--major-gap-hours", type=float, default=24.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    intervals = pd.read_csv(args.intervals_csv, parse_dates=["interval_start", "interval_end"])
    gaps = pd.read_csv(args.gaps_csv, parse_dates=["gap_start", "gap_end"])

    if intervals.empty:
        pd.DataFrame().to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        return

    threshold_minutes = float(args.major_gap_hours) * 60.0
    major_gaps = gaps[gaps["gap_minutes"] >= threshold_minutes].sort_values("gap_start")
    break_starts = list(major_gaps["gap_start"])
    break_ends = list(major_gaps["gap_end"])

    rows = []
    current = []
    break_idx = 0
    for _, interval in intervals.sort_values("interval_start").iterrows():
        while break_idx < len(break_starts) and interval["interval_start"] >= break_ends[break_idx]:
            if current:
                rows.append(current)
                current = []
            break_idx += 1
        current.append(interval)
    if current:
        rows.append(current)

    out_rows = []
    for idx, group in enumerate(rows, start=1):
        frame = pd.DataFrame(group)
        start = frame["interval_start"].min()
        end = frame["interval_end"].max()
        duration_days = (end - start).total_seconds() / 86400.0
        out_rows.append(
            {
                "period_no": idx,
                "data_start": start,
                "data_end": end,
                "rows": int(frame["rows"].sum()),
                "duration_days": duration_days,
                "duration_months_approx": duration_days / 30.4375,
                "merged_exact_intervals": int(len(frame.index)),
            }
        )

    out = pd.DataFrame(out_rows)
    out.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
