from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pymysql
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
SENSOR_QC_PATH = REPO_ROOT / "GEAS3.0" / "source" / "inner_layer" / "core" / "sensor_QC.py"
DEFAULT_COLUMNS = [
    "reg_date",
    "in_temp",
    "in_hum",
    "in_co2",
    "out_temp",
    "out_hum",
    "out_light",
    "out_windsp",
    "out_winddirec",
    "cont_fan_run",
    "cont_skyl_vol",
    "cont_heater_run",
]


def load_sensor_qc_module():
    if importlib.util.find_spec("sqlalchemy") is None:
        sqlalchemy_stub = types.ModuleType("sqlalchemy")

        def unavailable_create_engine(*args, **kwargs):
            raise RuntimeError(
                "SQLAlchemy is unavailable. Use this diagnostics script's pymysql DB reader."
            )

        sqlalchemy_stub.create_engine = unavailable_create_engine
        sys.modules["sqlalchemy"] = sqlalchemy_stub

    spec = importlib.util.spec_from_file_location("geas_sensor_qc", SENSOR_QC_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load sensor QC module: {SENSOR_QC_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quote_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return "`" + value.replace("`", "``") + "`"


def fetch_period(args: argparse.Namespace) -> pd.DataFrame:
    columns_sql = ", ".join(quote_identifier(col) for col in DEFAULT_COLUMNS)
    sql = f"""
        SELECT {columns_sql}
        FROM {quote_identifier(args.table)}
        WHERE {quote_identifier(args.id_column)} = %s
          AND {quote_identifier("reg_date")} >= %s
          AND {quote_identifier("reg_date")} < %s
        ORDER BY {quote_identifier("reg_date")} ASC
        LIMIT %s
    """
    connection = pymysql.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        database=args.db_name,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET time_zone = %s", ("+09:00",))
            cursor.execute(sql, (args.farm_sn, args.start, args.end, args.limit))
            rows = cursor.fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(
            f"No rows found for farm_sn={args.farm_sn}, "
            f"period=[{args.start}, {args.end})."
        )
    return pd.DataFrame(rows)


def make_qc_frame(raw: pd.DataFrame, qc: Any) -> pd.DataFrame:
    def get_numeric(column: str, default: float = 0.0) -> pd.Series:
        if column in raw.columns:
            return qc.numify(raw[column])
        return pd.Series(default, index=raw.index, dtype=float)

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(raw["reg_date"], errors="coerce"),
            "Tin": get_numeric("in_temp", np.nan),
            "RHin": get_numeric("in_hum", np.nan),
            "CO2": get_numeric("in_co2", np.nan),
            "Tout": get_numeric("out_temp", np.nan),
            "RHout": get_numeric("out_hum", np.nan),
            "Rin": get_numeric("out_light", np.nan),
            "vWind": get_numeric("out_windsp", np.nan),
            "wDir": get_numeric("out_winddirec", np.nan),
            "fan": get_numeric("cont_fan_run", 0.0),
            "skyl": get_numeric("cont_skyl_vol", 0.0),
            "heat": get_numeric("cont_heater_run", 0.0),
        }
    )
    return frame.sort_values("time").dropna(subset=["time"]).reset_index(drop=True)


def update_last_diagnostics(
    target: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    fitted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    residual: np.ndarray,
    flag: np.ndarray,
) -> None:
    target["fitted"][indices] = fitted
    target["lower"][indices] = lower
    target["upper"][indices] = upper
    target["residual"][indices] = residual
    target["flag_count"][indices] += flag.astype(int)
    target["evaluated_count"][indices] += np.isfinite(fitted).astype(int)


def empty_diagnostics(n: int) -> dict[str, np.ndarray]:
    return {
        "fitted": np.full(n, np.nan),
        "lower": np.full(n, np.nan),
        "upper": np.full(n, np.nan),
        "residual": np.full(n, np.nan),
        "flag_count": np.zeros(n, dtype=int),
        "evaluated_count": np.zeros(n, dtype=int),
    }


def run_temperature_humidity_qc(
    frame: pd.DataFrame,
    qc: Any,
    *,
    win: int,
    step: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    starts = list(range(0, n - win, step))
    if not starts:
        raise ValueError(
            f"Not enough rows for QC: rows={n}, win={win}. "
            "The original GEAS logic requires rows > win."
        )

    U = frame[["Tout", "RHout", "Rin", "vWind"]].to_numpy(dtype=float)
    temp_corrected = frame["Tin"].to_numpy(dtype=float).copy()
    hum_corrected = frame["RHin"].to_numpy(dtype=float).copy()

    temp_flag = np.zeros(n, dtype=bool)
    hum_flag = np.zeros(n, dtype=bool)
    multivariate_flag = np.zeros(n, dtype=bool)
    multivariate_flag_count = np.zeros(n, dtype=int)
    mahalanobis_d2 = np.full(n, np.nan)
    mahalanobis_threshold = float(chi2.ppf(1 - alpha, df=2))

    temp_diag = empty_diagnostics(n)
    hum_diag = empty_diagnostics(n)
    parameter_rows: list[dict[str, Any]] = []

    for window_id, start in enumerate(starts, start=1):
        stop = start + win
        tr = slice(start, stop)
        indices = np.arange(start, stop)

        fit_temp = qc.fit_arx(temp_corrected[tr], U[tr])
        det_temp = qc.detect_and_correct_clip(
            temp_corrected[tr],
            fit_temp["fitted"],
            np.full(win, fit_temp["sigma"]),
            alpha,
        )
        fit_hum = qc.fit_arx(hum_corrected[tr], U[tr])
        det_hum = qc.detect_and_correct_clip(
            hum_corrected[tr],
            fit_hum["fitted"],
            np.full(win, fit_hum["sigma"]),
            alpha,
        )

        residual_matrix = np.column_stack(
            [
                temp_corrected[tr] - fit_temp["fitted"],
                hum_corrected[tr] - fit_hum["fitted"],
            ]
        )
        valid = np.all(np.isfinite(residual_matrix), axis=1)
        flag_mv = np.zeros(win, dtype=bool)
        d2 = np.full(win, np.nan)
        if valid.sum() > 2:
            covariance = np.cov(residual_matrix[valid].T)
            try:
                inverse_covariance = np.linalg.inv(covariance)
                d2 = np.einsum(
                    "ij,jk,ik->i",
                    residual_matrix,
                    inverse_covariance,
                    residual_matrix,
                )
                flag_mv = np.isfinite(d2) & (d2 > mahalanobis_threshold)
            except np.linalg.LinAlgError:
                pass

        temp_flag[indices] |= det_temp["flag"]
        hum_flag[indices] |= det_hum["flag"]
        multivariate_flag[indices] |= flag_mv
        multivariate_flag_count[indices] += flag_mv.astype(int)
        mahalanobis_d2[indices] = d2

        update_last_diagnostics(
            temp_diag,
            indices,
            fitted=fit_temp["fitted"],
            lower=det_temp["L"],
            upper=det_temp["U"],
            residual=temp_corrected[tr] - fit_temp["fitted"],
            flag=det_temp["flag"],
        )
        update_last_diagnostics(
            hum_diag,
            indices,
            fitted=fit_hum["fitted"],
            lower=det_hum["L"],
            upper=det_hum["U"],
            residual=hum_corrected[tr] - fit_hum["fitted"],
            flag=det_hum["flag"],
        )

        for target_name, fit in (("temperature", fit_temp), ("humidity", fit_hum)):
            row = {
                "window_id": window_id,
                "target": target_name,
                "window_start": frame["time"].iloc[start],
                "window_end": frame["time"].iloc[stop - 1],
                "row_start": start,
                "row_stop_exclusive": stop,
                "alpha": alpha,
                "a_alpha": float(fit["a"]),
                "phi": float(fit["phi"]),
                "raw_phi_logit": float(
                    np.log(float(fit["phi"]) / (1.0 - float(fit["phi"])))
                ),
                "sigma_rmse": float(fit["sigma"]),
            }
            row.update(
                {
                    f"beta_{name}": float(value)
                    for name, value in zip(
                        ["out_temp", "out_hum", "out_light", "out_windsp"],
                        fit["b"],
                    )
                }
            )
            parameter_rows.append(row)

        temp_corrected[tr] = det_temp["ycorr"]
        hum_corrected[tr] = det_hum["ycorr"]

    result = frame.copy()
    result["Tin_corr"] = temp_corrected
    result["RHin_corr"] = hum_corrected
    result["temp_univariate_flag"] = temp_flag
    result["hum_univariate_flag"] = hum_flag
    result["multivariate_flag"] = multivariate_flag
    result["multivariate_flag_count"] = multivariate_flag_count
    result["mahalanobis_d2_last"] = mahalanobis_d2
    result["mahalanobis_threshold"] = mahalanobis_threshold

    for prefix, diagnostics in (("temp", temp_diag), ("hum", hum_diag)):
        for name, values in diagnostics.items():
            result[f"{prefix}_{name}_last" if name not in {"flag_count", "evaluated_count"} else f"{prefix}_{name}"] = values

    return result, pd.DataFrame(parameter_rows)


def run_co2_qc(
    frame: pd.DataFrame,
    qc: Any,
    *,
    win: int,
    step: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    n = len(result)
    starts = list(range(0, n - win, step))

    co2 = result["CO2"].to_numpy(dtype=float).copy()
    co2[(~np.isfinite(co2)) | (co2 <= 0) | (co2 > 5000)] = np.nan
    temp = result["Tin_corr"].to_numpy(dtype=float)
    hum = result["RHin_corr"].to_numpy(dtype=float)
    d_temp = np.concatenate([[np.nan], np.diff(temp)])
    d_hum = np.concatenate([[np.nan], np.diff(hum)])
    day = (result["Rin"].to_numpy(dtype=float) > 5).astype(float)
    X = np.column_stack(
        [
            day,
            result["Rin"].to_numpy(dtype=float),
            result["vWind"].to_numpy(dtype=float),
            result["fan"].fillna(0).to_numpy(dtype=float),
            result["skyl"].fillna(0).to_numpy(dtype=float),
            result["heat"].fillna(0).to_numpy(dtype=float),
            d_temp,
            d_hum,
        ]
    )

    co2_flag = np.zeros(n, dtype=bool)
    diagnostics = empty_diagnostics(n)
    parameter_rows: list[dict[str, Any]] = []

    for window_id, start in enumerate(starts, start=1):
        stop = start + win
        tr = slice(start, stop)
        indices = np.arange(start, stop)
        valid = np.all(np.isfinite(np.column_stack([co2[tr], X[tr]])), axis=1)
        if valid.sum() < 200:
            continue

        fit = qc.fit_co2_delta(co2[tr][valid], X[tr][valid])
        if fit is None:
            continue

        fitted = np.full(win, np.nan)
        fitted_valid = qc.predict_co2_1step(co2[tr][valid], X[tr][valid], fit)
        fitted[np.where(valid)[0]] = fitted_valid
        detection = qc.detect_and_correct_clip(
            co2[tr],
            fitted,
            np.full(win, fit["sigma"]),
            alpha,
        )

        co2_flag[indices] |= detection["flag"]
        update_last_diagnostics(
            diagnostics,
            indices,
            fitted=fitted,
            lower=detection["L"],
            upper=detection["U"],
            residual=co2[tr] - fitted,
            flag=detection["flag"],
        )
        co2[tr] = detection["ycorr"]

        row = {
            "window_id": window_id,
            "target": "co2",
            "window_start": result["time"].iloc[start],
            "window_end": result["time"].iloc[stop - 1],
            "row_start": start,
            "row_stop_exclusive": stop,
            "alpha": alpha,
            "b0": float(fit["b0"]),
            "b1": float(fit["b1"]),
            "cstar": float(fit["Cstar"]),
            "sigma_rmse": float(fit["sigma"]),
            "valid_rows": int(valid.sum()),
        }
        row.update(
            {
                f"beta_{name}": float(value)
                for name, value in zip(
                    [
                        "day",
                        "out_light",
                        "out_windsp",
                        "fan",
                        "skyl",
                        "heat",
                        "d_temp",
                        "d_hum",
                    ],
                    fit["b"],
                )
            }
        )
        parameter_rows.append(row)

    result["CO2_corr"] = co2
    result["co2_univariate_flag"] = co2_flag
    for name, values in diagnostics.items():
        result[f"co2_{name}_last" if name not in {"flag_count", "evaluated_count"} else f"co2_{name}"] = values

    return result, pd.DataFrame(parameter_rows)


def build_output_table(result: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "timestamp": result["time"],
            "temp_before": result["Tin"],
            "temp_pred_last": result["temp_fitted_last"],
            "temp_lower_last": result["temp_lower_last"],
            "temp_upper_last": result["temp_upper_last"],
            "temp_univariate_outlier": result["temp_univariate_flag"],
            "temp_univariate_hit_count": result["temp_flag_count"],
            "temp_evaluated_count": result["temp_evaluated_count"],
            "temp_after": result["Tin_corr"],
            "hum_before": result["RHin"],
            "hum_pred_last": result["hum_fitted_last"],
            "hum_lower_last": result["hum_lower_last"],
            "hum_upper_last": result["hum_upper_last"],
            "hum_univariate_outlier": result["hum_univariate_flag"],
            "hum_univariate_hit_count": result["hum_flag_count"],
            "hum_evaluated_count": result["hum_evaluated_count"],
            "hum_after": result["RHin_corr"],
            "mahalanobis_d2_last": result["mahalanobis_d2_last"],
            "mahalanobis_threshold": result["mahalanobis_threshold"],
            "multivariate_outlier": result["multivariate_flag"],
            "multivariate_hit_count": result["multivariate_flag_count"],
            "co2_before": result["CO2"],
            "co2_pred_last": result["co2_fitted_last"],
            "co2_lower_last": result["co2_lower_last"],
            "co2_upper_last": result["co2_upper_last"],
            "co2_univariate_outlier": result["co2_univariate_flag"],
            "co2_univariate_hit_count": result["co2_flag_count"],
            "co2_evaluated_count": result["co2_evaluated_count"],
            "co2_after": result["CO2_corr"],
        }
    )
    output["temp_correction"] = output["temp_after"] - output["temp_before"]
    output["hum_correction"] = output["hum_after"] - output["hum_before"]
    output["co2_correction"] = output["co2_after"] - output["co2_before"]
    output["any_outlier"] = output[
        [
            "temp_univariate_outlier",
            "hum_univariate_outlier",
            "multivariate_outlier",
            "co2_univariate_outlier",
        ]
    ].any(axis=1)
    return output


def print_table(frame: pd.DataFrame, *, title: str, max_rows: int) -> None:
    print(f"\n=== {title} (rows={len(frame)}) ===")
    if frame.empty:
        print("(empty)")
        return
    shown = frame.head(max_rows) if max_rows > 0 else frame
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        320,
        "display.float_format",
        lambda value: f"{value:.5f}",
    ):
        print(shown.to_string(index=False))
    if max_rows > 0 and len(frame) > max_rows:
        print(f"... {len(frame) - max_rows} additional rows are available in the CSV.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GEAS3.0 sensor QC against a DB period."
    )
    parser.add_argument("--start", required=True, help="Inclusive start timestamp.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp.")
    parser.add_argument("--farm-sn", type=int, default=int(os.getenv("FARM_SN", "97")))
    parser.add_argument("--win", type=int, default=288)
    parser.add_argument("--step", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--table", default=os.getenv("MAIN_TABLE_NAME", "data_silla_enc"))
    parser.add_argument("--id-column", default=os.getenv("ID_IDX", "iot_data_idx"))
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "3306")))
    parser.add_argument("--db-user", default=os.getenv("DB_USER", "root"))
    parser.add_argument("--db-password", default=os.getenv("DB_PASSWORD", ""))
    parser.add_argument("--db-name", default=os.getenv("DB_NAME", "farmstom"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs",
    )
    parser.add_argument(
        "--print-mode",
        choices=["all", "outliers", "none"],
        default="outliers",
    )
    parser.add_argument(
        "--max-print-rows",
        type=int,
        default=200,
        help="Use 0 to print every selected row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.win <= 1:
        raise ValueError("--win must be greater than 1.")
    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must be between 0 and 1.")

    qc = load_sensor_qc_module()
    raw = fetch_period(args)
    frame = make_qc_frame(raw, qc)
    print(
        f"[DB] rows={len(frame)}, period=[{frame['time'].min()}, {frame['time'].max()}], "
        f"farm_sn={args.farm_sn}"
    )
    print(f"[QC] win={args.win}, step={args.step}, alpha={args.alpha}")

    th_result, arx_parameters = run_temperature_humidity_qc(
        frame,
        qc,
        win=args.win,
        step=args.step,
        alpha=args.alpha,
    )
    full_result, co2_parameters = run_co2_qc(
        th_result,
        qc,
        win=args.win,
        step=args.step,
        alpha=args.alpha,
    )
    output = build_output_table(full_result)
    parameters = pd.concat([arx_parameters, co2_parameters], ignore_index=True, sort=False)
    outliers = output[output["any_outlier"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "sensor_qc_all_rows.csv"
    outlier_path = args.output_dir / "sensor_qc_outliers_only.csv"
    parameter_path = args.output_dir / "sensor_qc_window_parameters.csv"
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    outliers.to_csv(outlier_path, index=False, encoding="utf-8-sig")
    parameters.to_csv(parameter_path, index=False, encoding="utf-8-sig")

    print(
        "[SUMMARY] "
        f"temp_univariate={int(output['temp_univariate_outlier'].sum())}, "
        f"hum_univariate={int(output['hum_univariate_outlier'].sum())}, "
        f"multivariate={int(output['multivariate_outlier'].sum())}, "
        f"co2_univariate={int(output['co2_univariate_outlier'].sum())}, "
        f"any={int(output['any_outlier'].sum())}"
    )

    if args.print_mode == "all":
        print_table(output, title="QC diagnostics: all rows", max_rows=args.max_print_rows)
    elif args.print_mode == "outliers":
        print_table(outliers, title="QC diagnostics: outliers", max_rows=args.max_print_rows)
    print_table(parameters, title="Optimized parameters by window", max_rows=args.max_print_rows)

    print("\n[OUTPUT]")
    print(output_path.resolve())
    print(outlier_path.resolve())
    print(parameter_path.resolve())


if __name__ == "__main__":
    main()
