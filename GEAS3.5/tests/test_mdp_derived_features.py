import numpy as np
import pandas as pd
import pytest

from geas35.features.derived import (
    _condensation_margin_forecast_10m,
    add_mdp_derived_features,
)


def _frame(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    count = len(timestamps)
    return pd.DataFrame(
        {
            "reg_date": timestamps,
            "obs_indoor_temp_c": np.full(count, 30.0),
            "obs_indoor_humidity_pct": np.full(count, 70.0),
            "obs_outdoor_temp_c": np.full(count, 20.0),
            "obs_outdoor_light": np.full(count, 100.0),
            "obs_current_target_temp_c": np.full(count, 25.0),
            "_mdp_v1_logged_vent_pct": np.full(count, 50.0),
        }
    )


def test_documented_light_features_and_condensation_name() -> None:
    df = _frame(pd.date_range("2026-06-21 09:00", periods=3, freq="5min"))
    df["out_light_sum"] = [10.0, 20.0, 30.0]
    df["cs_sum_jcm2"] = [20.0, 40.0, 60.0]
    df["target_jcm2"] = 50.0

    result = add_mdp_derived_features(df)

    assert "dTcond_pred_10m" in result
    assert "obs_derived_condensation_margin_10m_c" not in result
    assert result.loc[1, "obs_derived_dli_mol_m2"] == pytest.approx(0.0606)
    assert result["obs_derived_light_sum_ratio"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert result["obs_derived_light_eta_minutes"].tolist() == pytest.approx([10.0, 5.0, 0.0])
    assert result.loc[0, "obs_derived_vent_loss_proxy"] == pytest.approx(4.5)


def test_haurwitz_curve_is_generated_from_site_coordinates() -> None:
    df = _frame(pd.date_range("2026-06-21 05:00", periods=181, freq="5min"))
    result = add_mdp_derived_features(
        df,
        latitude=36.5,
        longitude=127.5,
        timezone="Asia/Seoul",
        target_light_sum_j_cm2=100.0,
    )

    clear_sum = result["obs_derived_clear_sky_sum"]
    assert clear_sum.max() > 0.0
    assert clear_sum.is_monotonic_increasing
    assert np.isfinite(result["obs_derived_light_sum_ratio"]).all()
    assert result.loc[0, "obs_derived_light_eta_minutes"] > 0.0


def _legacy_condensation_forecast(
    timestamps: pd.Series, margin: pd.Series
) -> pd.Series:
    t = pd.to_datetime(timestamps, errors="coerce")
    y = pd.to_numeric(margin, errors="coerce")
    out = []
    for idx, now in enumerate(t):
        current = y.iloc[idx]
        if pd.isna(now) or not np.isfinite(current):
            out.append(np.nan)
            continue
        mask = (t >= now - pd.Timedelta(minutes=30)) & (t <= now)
        hist_t = t.loc[mask]
        hist_y = y.loc[mask]
        valid = hist_t.notna() & np.isfinite(hist_y.to_numpy(dtype=float))
        hist_t = hist_t.loc[valid]
        hist_y = hist_y.loc[valid]
        if len(hist_y) < 2:
            out.append(float(current))
            continue
        seconds = (hist_t - hist_t.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(seconds), seconds])
        beta, *_ = np.linalg.lstsq(design, hist_y.to_numpy(dtype=float), rcond=None)
        pred_t = (now - hist_t.iloc[0]).total_seconds() + 600.0
        out.append(float(beta[0] + beta[1] * pred_t))
    return pd.Series(out, index=margin.index, dtype="float64")


def test_condensation_forecast_matches_previous_recent_30_minute_definition() -> None:
    timestamps = pd.Series(pd.date_range("2026-06-01", periods=20, freq="5min"))
    margin = pd.Series(np.sin(np.arange(20) / 3.0) + 2.0)
    margin.iloc[5] = np.nan
    expected = _legacy_condensation_forecast(timestamps, margin)
    actual = _condensation_margin_forecast_10m(timestamps, margin)
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


def test_condensation_forecast_resets_at_episode_and_time_gap() -> None:
    timestamps = pd.Series(pd.to_datetime([
        "2026-06-01 00:00", "2026-06-01 00:05",
        "2026-06-01 00:10", "2026-06-01 00:15",
    ]))
    margin = pd.Series([5.0, 4.0, 1.0, 2.0])
    boundary = pd.DataFrame({"episode_id": ["a", "a", "b", "b"]})
    result = _condensation_margin_forecast_10m(
        timestamps, margin, boundary_frame=boundary
    )
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(4.0)

    gap_timestamps = pd.Series(pd.to_datetime([
        "2026-06-01 00:00", "2026-06-01 00:05",
        "2026-06-01 00:20", "2026-06-01 00:25",
    ]))
    gap_result = _condensation_margin_forecast_10m(gap_timestamps, margin)
    assert gap_result.iloc[2] == pytest.approx(1.0)
    assert gap_result.iloc[3] == pytest.approx(4.0)


def test_condensation_forecast_handles_large_sequence_without_quadratic_scan() -> None:
    rows = 20_000
    timestamps = pd.Series(pd.date_range("2026-06-01", periods=rows, freq="5min"))
    margin = pd.Series(np.linspace(2.0, 1.0, rows))
    result = _condensation_margin_forecast_10m(timestamps, margin)
    assert len(result) == rows
    assert np.isfinite(result).all()
