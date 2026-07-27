from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


SEASON_BY_MONTH = {
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
    12: "winter",
}

# Domain-review sensor ranges. Each tuple is
# (lower, upper, lower_is_valid, upper_is_valid); None means no bound.
DOMAIN_NORMAL_RANGES = {
    "in_temp": (2.0, 50.0, True, False),
    "in_temp2": (2.0, 50.0, True, False),
    "in_hum": (0.0, 100.0, True, True),
    "in_hum2": (0.0, 100.0, True, True),
    "in_co2": (200.0, 5000.0, True, False),
    "in_co2_2": (200.0, 5000.0, True, False),
    "out_temp": (None, 50.0, True, False),
    "out_winddirec": (0.0, 359.0, True, True),
    "out_windsp": (0.0, 30.0, True, True),
    "out_rain": (0.0, 1.0, True, True),
    "out_light": (0.0, 1400.0, True, True),
    "out_light_sum": (None, 10000.0, True, False),
    "in_medium_hum1": (0.0, 100.0, True, True),
    "in_medium_hum2": (0.0, 100.0, True, True),
    "in_medium_temp1": (0.0, 41.0, False, False),
    "in_medium_temp2": (0.0, 41.0, False, False),
}

SEASONAL_CONSISTENCY_Q999 = {
    ("in_temp", "in_temp2"): {
        "spring": 25.950000000000003,
        "summer": 2.049999999999997,
        "autumn": 29.300000000000004,
        "winter": 1.549999999999999,
    },
    ("in_hum", "in_hum2"): {
        "spring": 21.87,
        "summer": 9.579999999999998,
        "autumn": 34.96,
        "winter": 28.84,
    },
    ("in_co2", "in_co2_2"): {
        "spring": 1430.0,
        "summer": 158.0,
        "autumn": 2408.0,
        "winter": 4021.0,
    },
    ("in_medium_temp1", "in_medium_temp2"): {
        "spring": 42.1,
        "summer": 57.1,
        "autumn": 57.1,
        "winter": 26.9,
    },
    ("in_medium_hum1", "in_medium_hum2"): {
        "spring": 53.2,
        "summer": 48.5,
        "autumn": 79.7,
        "winter": 9.11,
    },
    ("in_medium_ec1", "in_medium_ec2"): {
        "spring": 7.14,
        "summer": 2.8,
        "autumn": 8.0,
        "winter": 0.2800000000000002,
    },
}

PAIRED_SENSOR_MAP = {
    "in_temp": "in_temp2",
    "in_temp2": "in_temp",
    "in_hum": "in_hum2",
    "in_hum2": "in_hum",
    "in_co2": "in_co2_2",
    "in_co2_2": "in_co2",
    "in_medium_temp1": "in_medium_temp2",
    "in_medium_temp2": "in_medium_temp1",
    "in_medium_hum1": "in_medium_hum2",
    "in_medium_hum2": "in_medium_hum1",
    "in_medium_ec1": "in_medium_ec2",
    "in_medium_ec2": "in_medium_ec1",
}

BINARY_COLUMNS = {"out_rain"}
DOMAIN_ALLOWED_VALUES = {
    "out_rain": {0.0, 1.0},
}


@dataclass(frozen=True)
class MultivariateModelSpec:
    name: str
    columns: tuple[str, ...]


MULTIVARIATE_MODELS = (
    MultivariateModelSpec(
        "indoor_climate",
        ("tin_rep", "rhin_rep", "co2_rep", "vpd", "tdew", "dtcond"),
    ),
    MultivariateModelSpec(
        "substrate",
        ("substrate_temp_rep", "substrate_water_rep", "substrate_ec_rep"),
    ),
    MultivariateModelSpec(
        "outdoor_weather",
        ("out_temp", "out_hum", "out_windsp", "out_light", "out_light_sum", "out_airpress"),
    ),
)

FEATURE_TO_SENSOR_COLUMNS = {
    "tin_rep": ("in_temp", "in_temp2"),
    "rhin_rep": ("in_hum", "in_hum2"),
    "co2_rep": ("in_co2", "in_co2_2"),
    "vpd": ("in_temp", "in_temp2", "in_hum", "in_hum2"),
    "tdew": ("in_temp", "in_temp2", "in_hum", "in_hum2"),
    "dtcond": ("in_temp", "in_temp2", "in_hum", "in_hum2"),
    "substrate_temp_rep": ("in_medium_temp1", "in_medium_temp2"),
    "substrate_water_rep": ("in_medium_hum1", "in_medium_hum2"),
    "substrate_ec_rep": ("in_medium_ec1", "in_medium_ec2"),
    "out_temp": ("out_temp",),
    "out_hum": ("out_hum",),
    "out_windsp": ("out_windsp",),
    "out_light": ("out_light",),
    "out_light_sum": ("out_light_sum",),
    "out_airpress": ("out_airpress",),
}


def _available(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _flag_col(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def _domain_rule_flag(
    values: pd.Series,
    rule: tuple[float | None, float | None, bool, bool],
    allowed_values: set[float] | None = None,
) -> pd.Series:
    lower, upper, lower_is_valid, upper_is_valid = rule
    present = values.notna()
    flag = pd.Series(False, index=values.index)
    if lower is not None:
        flag |= present & ((values < lower) if lower_is_valid else (values <= lower))
    if upper is not None:
        flag |= present & ((values > upper) if upper_is_valid else (values >= upper))
    if allowed_values is not None:
        flag |= present & ~values.isin(allowed_values)
    return flag


def _season_series(df: pd.DataFrame) -> pd.Series:
    if "reg_date" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="object")
    return pd.to_datetime(df["reg_date"], errors="coerce").dt.month.map(SEASON_BY_MONTH)


def _dewpoint_magnus(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(rh_pct, errors="coerce").clip(1e-6, 100.0)
    a, b = 17.62, 243.12
    gamma = np.log(rh / 100.0) + (a * temp) / (b + temp)
    return (b * gamma) / (a - gamma)


def _vpd_kpa(temp_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    temp = pd.to_numeric(temp_c, errors="coerce")
    rh = pd.to_numeric(rh_pct, errors="coerce").clip(0.0, 100.0)
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    ea = es * (rh / 100.0)
    return (es - ea).clip(lower=0.0)


def _domain_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    out = pd.DataFrame(index=df.index)
    any_flag = pd.Series(False, index=df.index)
    counts: Dict[str, int] = {}

    for col, rule in DOMAIN_NORMAL_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        flag = _domain_rule_flag(values, rule, DOMAIN_ALLOWED_VALUES.get(col))
        out[_flag_col("batch_point", col)] = flag.astype(int)
        any_flag |= flag
        counts[col] = int(flag.sum())

    return out, any_flag, counts


def _pair_flags(df: pd.DataFrame, seasons: pd.Series) -> tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    out = pd.DataFrame(index=df.index)
    any_flag = pd.Series(False, index=df.index)
    counts: Dict[str, int] = {}

    for (var1, var2), thresholds in SEASONAL_CONSISTENCY_Q999.items():
        if var1 not in df.columns or var2 not in df.columns:
            continue
        s1 = pd.to_numeric(df[var1], errors="coerce")
        s2 = pd.to_numeric(df[var2], errors="coerce")
        diff = (s1 - s2).abs()
        flag = pd.Series(False, index=df.index)
        for season, threshold in thresholds.items():
            valid = s1.notna() & s2.notna() & seasons.eq(season)
            flag |= valid & (diff > float(threshold))

        name = f"{var1}_vs_{var2}"
        out[_flag_col("batch_pair", name)] = flag.astype(int)
        any_flag |= flag
        counts[name] = int(flag.sum())

    return out, any_flag, counts


def _pair_key(col: str) -> Optional[tuple[str, str]]:
    other = PAIRED_SENSOR_MAP.get(col)
    if other is None:
        return None
    return tuple(sorted((col, other)))


def _pair_flag_for_col(flags: pd.DataFrame, col: str) -> pd.Series:
    pair = _pair_key(col)
    if pair is None:
        return pd.Series(False, index=flags.index)
    pair_col = _flag_col("batch_pair", f"{pair[0]}_vs_{pair[1]}")
    if pair_col in flags.columns:
        return flags[pair_col].fillna(0).astype(bool)
    reverse_col = _flag_col("batch_pair", f"{pair[1]}_vs_{pair[0]}")
    if reverse_col in flags.columns:
        return flags[reverse_col].fillna(0).astype(bool)
    return pd.Series(False, index=flags.index)


def _clean_series(df: pd.DataFrame, flags: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = pd.to_numeric(df[col], errors="coerce").astype(float)
    point_col = _flag_col("batch_point", col)
    point = flags[point_col].fillna(0).astype(bool) if point_col in flags.columns else False
    return values.mask(point)


def _representative(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    var1: str,
    var2: str,
    pair_flag_name: str,
) -> pd.Series:
    s1 = _clean_series(df, flags, var1)
    s2 = _clean_series(df, flags, var2)
    pair_flag_col = _flag_col("batch_pair", pair_flag_name)
    pair_flag = flags[pair_flag_col].fillna(0).astype(bool) if pair_flag_col in flags.columns else False
    both = s1.notna() & s2.notna() & ~pair_flag
    one = s1.notna() & s2.isna()
    two = s2.notna() & s1.isna()
    rep = pd.Series(np.nan, index=df.index, dtype=float)
    rep.loc[both] = (s1.loc[both] + s2.loc[both]) / 2.0
    rep.loc[one] = s1.loc[one]
    rep.loc[two] = s2.loc[two]
    return rep


def _build_multivariate_features(df: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["tin_rep"] = _representative(df, flags, "in_temp", "in_temp2", "in_temp_vs_in_temp2")
    feat["rhin_rep"] = _representative(df, flags, "in_hum", "in_hum2", "in_hum_vs_in_hum2")
    feat["co2_rep"] = _representative(df, flags, "in_co2", "in_co2_2", "in_co2_vs_in_co2_2")
    feat["tdew"] = _dewpoint_magnus(feat["tin_rep"], feat["rhin_rep"])
    feat["vpd"] = _vpd_kpa(feat["tin_rep"], feat["rhin_rep"])
    feat["dtcond"] = feat["tin_rep"] - feat["tdew"]

    feat["substrate_temp_rep"] = _representative(
        df, flags, "in_medium_temp1", "in_medium_temp2", "in_medium_temp1_vs_in_medium_temp2"
    )
    feat["substrate_water_rep"] = _representative(
        df, flags, "in_medium_hum1", "in_medium_hum2", "in_medium_hum1_vs_in_medium_hum2"
    )
    feat["substrate_ec_rep"] = _representative(
        df, flags, "in_medium_ec1", "in_medium_ec2", "in_medium_ec1_vs_in_medium_ec2"
    )

    for col in ("out_temp", "out_hum", "out_windsp", "out_light", "out_light_sum", "out_airpress"):
        feat[col] = _clean_series(df, flags, col)
    return feat


def _mahalanobis_flags(features: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    flags = pd.DataFrame(index=features.index)
    report: Dict[str, Dict[str, Any]] = {}

    for spec in MULTIVARIATE_MODELS:
        cols = _available(spec.columns, features)
        flag_col = _flag_col("batch_multivariate", spec.name)
        score_col = f"{flag_col}_score"
        flags[flag_col] = 0
        flags[score_col] = np.nan
        if len(cols) < 2:
            report[spec.name] = {"status": "skipped", "reason": "too_few_columns", "columns": cols}
            continue

        x = features[cols].apply(pd.to_numeric, errors="coerce")
        valid = x.notna().all(axis=1)
        min_samples = max(30, len(cols) * 10)
        if int(valid.sum()) < min_samples:
            report[spec.name] = {
                "status": "skipped",
                "reason": "too_few_complete_rows",
                "columns": cols,
                "complete_rows": int(valid.sum()),
            }
            continue

        x_fit = x.loc[valid].astype(float)
        mu = x_fit.mean(axis=0).to_numpy()
        cov = np.cov(x_fit.to_numpy(), rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        centered = x_fit.to_numpy() - mu
        dist2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
        threshold = float(np.nanquantile(dist2, 0.999))
        model_flag = dist2 > threshold
        culprit_counts: Dict[str, int] = {}

        if np.any(model_flag):
            scale = x_fit.std(axis=0, ddof=0).replace(0.0, np.nan).to_numpy()
            z = np.abs((x_fit.to_numpy() - mu) / scale)
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            top_positions = np.argmax(z[model_flag], axis=1)
            flagged_index = x_fit.index[model_flag]
            top_features = pd.Series([cols[pos] for pos in top_positions], index=flagged_index)

            for feature_name, feature_index in top_features.groupby(top_features).groups.items():
                culprit_counts[str(feature_name)] = int(len(feature_index))
                for source_col in FEATURE_TO_SENSOR_COLUMNS.get(str(feature_name), (str(feature_name),)):
                    impute_col = _flag_col("batch_multivariate_impute", source_col)
                    if impute_col not in flags.columns:
                        flags[impute_col] = 0
                    flags.loc[list(feature_index), impute_col] = 1

        flags.loc[x_fit.index, score_col] = dist2
        flags.loc[x_fit.index, flag_col] = model_flag.astype(int)
        report[spec.name] = {
            "status": "ok",
            "columns": cols,
            "complete_rows": int(valid.sum()),
            "threshold_q999": threshold,
            "flagged_rows": int(model_flag.sum()),
            "culprit_counts_by_feature": culprit_counts,
        }

    multi_cols = [f"batch_multivariate_{spec.name}" for spec in MULTIVARIATE_MODELS if f"batch_multivariate_{spec.name}" in flags.columns]
    flags["batch_multivariate_flag"] = flags[multi_cols].sum(axis=1).gt(0).astype(int) if multi_cols else 0
    return flags, report


def _mode_or_nan(series: pd.Series) -> float:
    mode = series.dropna().mode()
    if mode.empty:
        return np.nan
    return float(mode.iloc[0])


def _fallback_values(values: pd.Series, seasons: pd.Series, hours: pd.Series, binary: bool) -> pd.Series:
    fallback = pd.Series(np.nan, index=values.index, dtype=float)
    tmp = pd.DataFrame({"value": values, "season": seasons, "hour": hours})
    if binary:
        by_season_hour = tmp.groupby(["season", "hour"], dropna=False)["value"].transform(_mode_or_nan)
        by_season = tmp.groupby("season", dropna=False)["value"].transform(_mode_or_nan)
        global_value = _mode_or_nan(values)
    else:
        by_season_hour = tmp.groupby(["season", "hour"], dropna=False)["value"].transform("median")
        by_season = tmp.groupby("season", dropna=False)["value"].transform("median")
        global_value = float(values.median()) if values.notna().any() else np.nan
    fallback = fallback.fillna(by_season_hour).fillna(by_season).fillna(global_value)
    return fallback


def _impute_column(
    df: pd.DataFrame,
    flags: pd.DataFrame,
    col: str,
    seasons: pd.Series,
    hours: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    values = pd.to_numeric(df[col], errors="coerce").astype(float)
    point_col = _flag_col("batch_point", col)
    point_flag = flags[point_col].fillna(0).astype(bool) if point_col in flags.columns else pd.Series(False, index=df.index)
    multivariate_col = _flag_col("batch_multivariate_impute", col)
    multivariate_flag = (
        flags[multivariate_col].fillna(0).astype(bool)
        if multivariate_col in flags.columns
        else pd.Series(False, index=df.index)
    )
    impute_mask = point_flag | multivariate_flag
    qc = values.copy()
    method = pd.Series("unchanged", index=df.index, dtype="object")

    other = PAIRED_SENSOR_MAP.get(col)
    if other in df.columns:
        other_values = pd.to_numeric(df[other], errors="coerce")
        other_point_col = _flag_col("batch_point", other)
        other_point = (
            flags[other_point_col].fillna(0).astype(bool)
            if other_point_col in flags.columns
            else pd.Series(False, index=df.index)
        )
        other_multivariate_col = _flag_col("batch_multivariate_impute", other)
        other_multivariate = (
            flags[other_multivariate_col].fillna(0).astype(bool)
            if other_multivariate_col in flags.columns
            else pd.Series(False, index=df.index)
        )
        pair_flag = _pair_flag_for_col(flags, col)
        use_other = impute_mask & other_values.notna() & ~other_point & ~other_multivariate & ~pair_flag
        qc.loc[use_other] = other_values.loc[use_other]
        method.loc[use_other & point_flag] = "paired_sensor"
        method.loc[use_other & ~point_flag & multivariate_flag] = "multivariate_paired_sensor"

    remaining = impute_mask & method.eq("unchanged")
    masked = qc.mask(remaining)
    if "reg_date" in df.columns:
        timed = pd.DataFrame({"reg_date": pd.to_datetime(df["reg_date"], errors="coerce"), "value": masked})
        timed = timed.sort_values("reg_date")
        interpolated_sorted = timed["value"].interpolate(method="linear", limit=12, limit_direction="both")
        interpolated = interpolated_sorted.reindex(timed.index).sort_index()
    else:
        interpolated = masked.interpolate(method="linear", limit=12, limit_direction="both")

    fill_linear = remaining & interpolated.notna()
    qc.loc[fill_linear] = interpolated.loc[fill_linear]
    method.loc[fill_linear & point_flag] = "time_linear"
    method.loc[fill_linear & ~point_flag & multivariate_flag] = "multivariate_time_linear"

    remaining = impute_mask & method.eq("unchanged")
    fallback = _fallback_values(masked, seasons, hours, col in BINARY_COLUMNS)
    fill_fallback = remaining & fallback.notna()
    qc.loc[fill_fallback] = fallback.loc[fill_fallback]
    method.loc[fill_fallback & point_flag] = "season_hour_fallback"
    method.loc[fill_fallback & ~point_flag & multivariate_flag] = "multivariate_season_hour_fallback"

    still_missing = impute_mask & qc.isna()
    method.loc[still_missing] = "unresolved"
    if col in BINARY_COLUMNS:
        qc = qc.round().clip(0, 1)
    return qc, impute_mask.astype(int), method


def _apply_imputation(df: pd.DataFrame, flags: pd.DataFrame, seasons: pd.Series) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = df.copy()
    if "reg_date" in out.columns:
        hours = pd.to_datetime(out["reg_date"], errors="coerce").dt.hour
    else:
        hours = pd.Series(pd.NA, index=out.index)

    counts: Dict[str, int] = {}
    for col in _available(DOMAIN_NORMAL_RANGES.keys(), out):
        if f"{col}_raw" not in out.columns:
            out[f"{col}_raw"] = out[col]
        qc, imputed_flag, method = _impute_column(out, flags, col, seasons, hours)
        out[f"{col}_qc"] = qc
        out[f"batch_imputed_{col}"] = imputed_flag
        out[f"batch_imputation_method_{col}"] = method
        out[col] = qc
        counts[col] = int(imputed_flag.sum())
    return out, counts


def run_batch_sensor_qc(df_history: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Run batch sensor QC and return a corrected copy plus an audit report.

    Raw columns are preserved as ``*_raw``. Corrected values are written both to
    ``*_qc`` columns and back to the original sensor columns so downstream theta
    estimation naturally consumes QC-adjusted values.
    """

    if df_history is None or df_history.empty:
        return df_history, {"status": "skipped", "reason": "empty_dataframe"}

    df = df_history.copy()
    if "reg_date" in df.columns:
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce")
        df = df.sort_values("reg_date").reset_index(drop=True)

    sensor_cols = _available(DOMAIN_NORMAL_RANGES.keys(), df)
    for col in sensor_cols:
        df[f"{col}_raw"] = df[col]
        df[col] = pd.to_numeric(df[col], errors="coerce")

    seasons = _season_series(df)
    point_flags, point_any, point_counts = _domain_flags(df)
    pair_flags, pair_any, pair_counts = _pair_flags(df, seasons)
    flags = pd.concat([point_flags, pair_flags], axis=1)
    flags["batch_point_flag"] = point_any.astype(int)
    flags["batch_pair_flag"] = pair_any.astype(int)

    features = _build_multivariate_features(df, flags)
    multi_flags, multi_report = _mahalanobis_flags(features)
    flags = pd.concat([flags, multi_flags], axis=1)
    flags["batch_outlier_flag"] = (
        flags["batch_point_flag"].astype(bool)
        | flags["batch_pair_flag"].astype(bool)
        | flags["batch_multivariate_flag"].astype(bool)
    ).astype(int)

    df_qc = pd.concat([df, features.add_prefix("batch_feature_"), flags], axis=1)
    df_qc, imputation_counts = _apply_imputation(df_qc, flags, seasons)

    report = {
        "status": "ok",
        "row_count": int(len(df_qc.index)),
        "point_flagged_rows": int(flags["batch_point_flag"].sum()),
        "pair_flagged_rows": int(flags["batch_pair_flag"].sum()),
        "multivariate_flagged_rows": int(flags["batch_multivariate_flag"].sum()),
        "any_flagged_rows": int(flags["batch_outlier_flag"].sum()),
        "point_counts_by_column": point_counts,
        "pair_counts_by_pair": pair_counts,
        "multivariate_models": multi_report,
        "imputation_counts_by_column": imputation_counts,
        "imputation_policy": (
            "raw_preserved; point outliers and multivariate culprit variables masked; "
            "paired sensor used first when uncontaminated; then linear time interpolation; "
            "then season-hour fallback"
        ),
    }
    return df_qc, report
