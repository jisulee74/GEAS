"""
3.6.5. 충분한 데이터 환경에서의 검증 방법론
기존 1_ / 2_ 식별 코드를 감싸서 충분한 데이터 환경의 물리 파라미터를
공통 module 3 저장 포맷으로 정규화하고, 필요 시 저장소에 자동 적재한다.
"""

from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any, Dict, Optional

import pandas as pd

from physics_store_sync import approve_and_save_theta_case, evaluate_theta_candidate

_THIS_DIR = Path(__file__).resolve().parent
_BASIC_PATH = _THIS_DIR / '1__단일_구획_모델_회귀_기반_식별_방법.py'
_GENERAL_PATH = _THIS_DIR / '2__단일_구획_모델_회귀_기반_식별_방법_일반화.py'


def _load_basic_api() -> Dict[str, Any]:
    return runpy.run_path(str(_BASIC_PATH))


def _load_general_api() -> Dict[str, Any]:
    return runpy.run_path(str(_GENERAL_PATH))


def _theta_from_basic_result(coefs: Dict[str, float], c_est: float) -> Dict[str, float]:
    return {
        'C': float(c_est),
        'k_heat': float(coefs['a1'] * c_est),
        'eta': float(coefs['a2'] * c_est),
        'UA': float(max(-coefs['a3'] * c_est, 0.0)),
        'a0': 0.05,
        'a1': float(max(-coefs['a4'] * c_est, 0.0)),
        'a2': 0.10,
        'a3': 0.0,
    }


def _theta_from_general_result(result: Dict[str, Any]) -> Dict[str, float]:
    vals = list(result['par_est'])
    return {
        'C': float(vals[0]),
        'k_heat': float(vals[1]),
        'eta': float(vals[2]),
        'UA': float(vals[3]),
        'a0': 0.05,
        'a1': float(vals[4]),
        'a2': 0.10,
        'a3': 0.0,
    }


def estimate_theta_sufficient_basic(
    df_raw: pd.DataFrame,
    *,
    auto_save: bool = False,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    model_name: str = 'basic',
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    api = _load_basic_api()
    preprocess = api['preprocess']
    fit_huber = api['fit_huber']
    estimate_C = api['estimate_C']

    df_proc = preprocess(df_raw)
    coefs, df_reg = fit_huber(df_proc)
    df_sub = df_proc.iloc[::5].reset_index(drop=True)
    c_est = estimate_C(df_sub, coefs)
    theta = _theta_from_basic_result(coefs, c_est)

    metadata = {
        'method': 'basic',
        'n_obs': int(len(df_proc.index)),
        'n_subsample': int(len(df_sub.index)),
    }
    out: Dict[str, Any] = {
        'theta': theta,
        'method': 'basic',
        'coefs': dict(coefs),
        'n_obs': int(len(df_proc.index)),
        'n_subsample': int(len(df_sub.index)),
        'approval': evaluate_theta_candidate(theta, data_case='sufficient', metadata=metadata),
    }
    if auto_save:
        saved = approve_and_save_theta_case(
            theta,
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case='sufficient',
            model_name=model_name,
            store_path=store_path,
            source='modeling_sufficient_basic',
            metadata=metadata,
        )
        out['approval'] = saved['approval']
        if 'record' in saved:
            out['store_key'] = saved['record'].key
    return out


def estimate_theta_sufficient_general(
    df_raw: pd.DataFrame,
    *,
    area_m2: float,
    cover_type: str = 'single_film',
    height_m: float = 4.0,
    lambda_C: float = 0.0,
    lambda_UA: float = 0.0,
    n_start: int = 3,
    auto_save: bool = False,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    model_name: str = 'generalized',
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    api = _load_general_api()
    preprocess_for_ident = api['preprocess_for_ident']
    identify = api['identify_physical_params_auto']

    df_proc = preprocess_for_ident(df_raw)
    result = identify(
        df=df_proc,
        area_m2=area_m2,
        cover_type=cover_type,
        height_m=height_m,
        lambda_C=lambda_C,
        lambda_UA=lambda_UA,
        n_start=n_start,
        auto_save=False,
    )
    theta = _theta_from_general_result(result)

    metadata = {
        'method': 'generalized',
        'n_obs': int(len(df_proc.index)),
        'area_m2': area_m2,
        'cover_type': cover_type,
        'height_m': height_m,
        'lambda_C': lambda_C,
        'lambda_UA': lambda_UA,
        'n_start': n_start,
        'raw_par_est': list(result['par_est']),
        'value': result.get('value'),
        'convergence': result.get('convergence'),
    }
    out: Dict[str, Any] = {
        'theta': theta,
        'method': 'generalized',
        'raw_result': result,
        'n_obs': int(len(df_proc.index)),
        'approval': evaluate_theta_candidate(theta, data_case='sufficient', metadata=metadata),
    }
    if auto_save:
        saved = approve_and_save_theta_case(
            theta,
            farm_sn=farm_sn,
            stage_name=stage_name,
            data_case='sufficient',
            model_name=model_name,
            store_path=store_path,
            source='modeling_sufficient_generalized',
            metadata=metadata,
        )
        out['approval'] = saved['approval']
        if 'record' in saved:
            out['store_key'] = saved['record'].key
    return out


def estimate_theta_sufficient(
    df_raw: pd.DataFrame,
    *,
    method: str = 'generalized',
    auto_save: bool = False,
    farm_sn: Optional[int] = None,
    stage_name: Optional[str] = None,
    model_name: Optional[str] = None,
    store_path: Optional[str] = None,
    area_m2: float = 200.0,
    cover_type: str = 'single_film',
    height_m: float = 4.0,
    lambda_C: float = 0.0,
    lambda_UA: float = 0.0,
    n_start: int = 3,
) -> Dict[str, Any]:
    method = str(method).lower()
    if method == 'basic':
        return estimate_theta_sufficient_basic(
            df_raw,
            auto_save=auto_save,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name=model_name or 'basic',
            store_path=store_path,
        )
    if method == 'generalized':
        return estimate_theta_sufficient_general(
            df_raw,
            area_m2=area_m2,
            cover_type=cover_type,
            height_m=height_m,
            lambda_C=lambda_C,
            lambda_UA=lambda_UA,
            n_start=n_start,
            auto_save=auto_save,
            farm_sn=farm_sn,
            stage_name=stage_name,
            model_name=model_name or 'generalized',
            store_path=store_path,
        )
    raise ValueError(f'Unknown sufficient-data method: {method}')


if __name__ == '__main__':
    print('모듈 로드 완료.')
