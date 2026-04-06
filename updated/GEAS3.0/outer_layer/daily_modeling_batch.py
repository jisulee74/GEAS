from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import runpy
from typing import Any, Dict, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_UPDATED_ROOT = _THIS_DIR.parent
_EVAL_DIR = _UPDATED_ROOT.parent / "evaluation"


def _history_period(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    if df is None or df.empty or 'reg_date' not in df.columns:
        return None, None
    ts = pd.to_datetime(df['reg_date'], errors='coerce').dropna()
    if ts.empty:
        return None, None
    return ts.iloc[0].isoformat(), ts.iloc[-1].isoformat()


def _load(path_name: str, *, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    target_dir = base_dir or _THIS_DIR
    for _path in (target_dir, _THIS_DIR, _UPDATED_ROOT):
        if str(_path) not in sys.path:
            sys.path.insert(0, str(_path))
    return runpy.run_path(str(target_dir / path_name))


def run_daily_modeling_batch(
    df_history: pd.DataFrame,
    *,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    store_path: Optional[str] = None,
    sufficient_method: str = 'generalized',
    area_m2: float = 200.0,
    cover_type: str = 'single_film',
    height_m: float = 4.0,
    policy_compare: bool = False,
    compare_window_start: Optional[str] = None,
    compare_window_end: Optional[str] = None,
    n_mc: int = 20,
    alpha: float = 0.9,
) -> Dict[str, Any]:
    classifier = _load('data_case_classifier.py')
    decision = classifier['classify_data_availability'](df_history, time_col='reg_date')
    valid_from, valid_to = _history_period(df_history)
    out: Dict[str, Any] = {
        'case_info': {
            'data_case': decision.data_case,
            'metrics': dict(decision.metrics),
            'reasons': list(decision.reasons),
            'valid_from': valid_from,
            'valid_to': valid_to,
        }
    }

    if decision.data_case == 'no_data':
        out['theta_identification'] = None
        out['message'] = 'no_data 환경에서는 물리파라미터 식별을 수행하지 않음'
        return out

    if decision.data_case == 'limited':
        mod = _load('theta_limited.py')
        theta_limited = mod['identify_physical_params_auto'](
            df_history,
            area_m2=area_m2,
            cover_type=cover_type,
            height_m=height_m,
            auto_save=True,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name='default',
            store_path=store_path,
        )
        out['theta_identification'] = theta_limited
        out['theta_limited'] = theta_limited
        return out

    mod_365 = _load('theta_sufficient.py')
    theta_sufficient = mod_365['identify_physical_params'](
        df_history,
        auto_save=True,
        farm_sn=farm_sn,
        stage_name=stage_name,
        model_name='default',
        store_path=store_path,
    )
    out['theta_identification'] = theta_sufficient
    out['theta_sufficient'] = theta_sufficient

    if policy_compare and compare_window_start and compare_window_end:
        mod_366 = _load('3_6_6__외기_조건_고전_기반_정책_비교_시뮬레이션.py', base_dir=_EVAL_DIR)
        map_columns_raw = mod_366['map_columns_raw']
        regularize_time_std = mod_366['regularize_time_std']
        derive_features = mod_366['derive_features']
        run_fit_then_window = mod_366['run_fit_then_window']

        dt0 = df_history.copy()
        dt0['reg_date'] = pd.to_datetime(dt0['reg_date'])
        dt0 = dt0.sort_values('reg_date').reset_index(drop=True)
        dt_std = map_columns_raw(dt0)
        dt = regularize_time_std(dt_std, step_mins=10)
        dt = derive_features(dt)

        compare_result = run_fit_then_window(
            dt_full=dt,
            window_start=compare_window_start,
            window_end=compare_window_end,
            n_mc=n_mc,
            alpha=alpha,
            auto_save_theta=True,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name='fit_then_window',
            store_path=store_path,
        )
        out['policy_compare'] = compare_result

    return out
