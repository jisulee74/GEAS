from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import runpy
from typing import Any, Dict, Optional

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent


def _load(path_name: str) -> Dict[str, Any]:
    if str(_THIS_DIR) not in sys.path:
        sys.path.insert(0, str(_THIS_DIR))
    return runpy.run_path(str(_THIS_DIR / path_name))


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
    out: Dict[str, Any] = {
        'case_info': {
            'data_case': decision.data_case,
            'metrics': dict(decision.metrics),
            'reasons': list(decision.reasons),
        }
    }

    if decision.data_case == 'no_data':
        mod = _load('3_6_3__무데이터_환경에서의_견고성_검증_방법론.py')
        theta = mod['sample_theta'](
            auto_save=True,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name='default',
            store_path=store_path,
        )
        out['theta_result'] = theta
        return out

    if decision.data_case == 'limited':
        mod = _load('3_6_4__제한적_데이터_환경에서의_부분_검증_방법론.py')
        theta = mod['sample_theta'](
            auto_save=True,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name='default',
            store_path=store_path,
        )
        out['theta_result'] = theta
        return out

    mod_365 = _load('3_6_5__충분한_데이터_환경에서의_검증_방법론.py')
    theta_result = mod_365['estimate_theta_sufficient'](
        df_history,
        method=sufficient_method,
        auto_save=True,
        farm_sn=farm_sn,
        stage_name=stage_name,
        store_path=store_path,
        area_m2=area_m2,
        cover_type=cover_type,
        height_m=height_m,
    )
    out['theta_result'] = theta_result

    if policy_compare and compare_window_start and compare_window_end:
        mod_366 = _load('3_6_6__외기_조건_고전_기반_정책_비교_시뮬레이션.py')
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
