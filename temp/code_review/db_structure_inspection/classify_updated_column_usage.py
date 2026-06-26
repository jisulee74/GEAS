from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATED_ROOT = REPO_ROOT / "GEAS3.0"
COLUMN_INFO = Path(__file__).resolve().parent / "column_info.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "updated_column_usage.csv"


PRIMARY_RUNTIME_FILES = {
    "GEAS3.0/source/inner_layer/core/config.py",
    "GEAS3.0/source/inner_layer/core/preprocessing.py",
    "GEAS3.0/source/inner_layer/core/sensor_QC.py",
    "GEAS3.0/source/inner_layer/core/features.py",
    "GEAS3.0/source/inner_layer/core/solar_eta.py",
    "GEAS3.0/source/inner_layer/policy/controller.py",
    "GEAS3.0/source/inner_layer/policy/controller_legacy.py",
    "GEAS3.0/source/inner_layer/utilities/sensor_qc_runtime.py",
    "GEAS3.0/source/inner_layer/utilities/utils.py",
    "GEAS3.0/source/controller_layer/runtime.py",
    "GEAS3.0/source/controller_layer/policy_common.py",
    "GEAS3.0/source/outer_layer/data_case_classifier.py",
    "GEAS3.0/source/outer_layer/daily_policy_update.py",
    "GEAS3.0/source/outer_layer/daily_modeling_batch.py",
    "GEAS3.0/source/outer_layer/theta_limited.py",
    "GEAS3.0/source/outer_layer/theta_sufficient.py",
}


def py_files() -> list[Path]:
    return sorted(UPDATED_ROOT.rglob("*.py"))


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def classify_file(path: Path) -> str:
    relative = rel(path)
    if relative in PRIMARY_RUNTIME_FILES:
        return "primary_runtime"
    if relative.startswith("GEAS3.0/source/"):
        return "geas30_aux_or_legacy"
    if relative.startswith("GEAS3.0/evaluation/"):
        return "evaluation"
    return "other"


def find_occurrences(column: str, files: list[Path]) -> list[tuple[str, str]]:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])")
    out: list[tuple[str, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="cp949", errors="ignore")
        if pattern.search(text):
            out.append((rel(path), classify_file(path)))
    return out


def main() -> None:
    info = pd.read_csv(COLUMN_INFO)
    columns = [str(col) for col in info["컬럼명"].tolist()]
    files = py_files()

    rows = []
    for column in columns:
        occurrences = find_occurrences(column, files)
        files_by_kind: dict[str, list[str]] = {
            "primary_runtime": [],
            "geas30_aux_or_legacy": [],
            "evaluation": [],
            "other": [],
        }
        for file_path, kind in occurrences:
            files_by_kind[kind].append(file_path)

        if files_by_kind["primary_runtime"]:
            classification = "used_by_primary_updated_runtime"
        elif files_by_kind["geas30_aux_or_legacy"]:
            classification = "used_by_updated_geas30_aux_or_legacy_only"
        elif files_by_kind["evaluation"]:
            classification = "used_by_updated_evaluation_only"
        else:
            classification = "not_referenced_in_updated_python"

        rows.append(
            {
                "column": column,
                "classification": classification,
                "used_by_primary_runtime": bool(files_by_kind["primary_runtime"]),
                "used_by_geas30_aux_or_legacy": bool(files_by_kind["geas30_aux_or_legacy"]),
                "used_by_evaluation": bool(files_by_kind["evaluation"]),
                "primary_runtime_files": "; ".join(files_by_kind["primary_runtime"]),
                "geas30_aux_or_legacy_files": "; ".join(files_by_kind["geas30_aux_or_legacy"]),
                "evaluation_files": "; ".join(files_by_kind["evaluation"]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"wrote {OUTPUT_CSV}")
    print(out[["column", "classification"]].to_string(index=False))


if __name__ == "__main__":
    main()
