"""
3.6.3. AI 자율제어기 견고성 검증 방법론 - 무데이터 환경
합성 외기 시나리오 + Monte Carlo 파라미터 샘플링으로
AI 제어 정책의 견고성을 검증한다.

핵심 no-data 제어 정책(후보 생성, 제약 필터링, 비용 최소화)은
GEAS3.0/source/inner_layer/policy/controller.py 로 이관되었고,
본 파일은 해당 정책을 호출해 시뮬레이션/평가를 수행한다.
"""

from pathlib import Path
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

_UPDATED_ROOT = Path(__file__).resolve().parents[1] / 'source'
_INNER_ROOT = _UPDATED_ROOT / 'inner_layer'
for _path in (_UPDATED_ROOT, _INNER_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from policy.controller import (
    sat_vp_kpa_no_data as sat_vp_kpa,
    dewpoint_c_no_data as dewpoint_c,
    vpd_kpa_no_data as vpd_kpa,
    sample_theta_no_data as sample_theta,
    step_dynamics_no_data as step_dynamics,
    project_action_no_data as project_action,
    check_feasible_no_data as check_feasible,
    make_action_grid_no_data as make_action_grid,
    policy_optimal_no_data as policy_optimal,
)


_RESULTS_ROOT = Path(__file__).resolve().parent / "eval_results" / 'no_data'
_DOC_METRIC_COLUMNS = [
    'temp_viol_rate',
    'temp_viol_maxrun',
    'cond_viol_rate',
    'cond_viol_maxrun',
    'rh_viol_rate',
    'vpd_viol_rate',
    'hv_ineff_rate',
    'tv_vent',
    'sw_heat',
]


def _ensure_results_dir() -> Path:
    _RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = _RESULTS_ROOT / datetime.utcnow().strftime('run_%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_builtin(value):
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _round_floats(value, digits: int = 3):
    if isinstance(value, dict):
        return {str(k): _round_floats(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_floats(v, digits) for v in value]
    if isinstance(value, tuple):
        return tuple(_round_floats(v, digits) for v in value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, digits)
    return value


def save_eval_outputs(
    output_dir: Path,
    robust_result: dict,
    episode_df: pd.DataFrame,
    tuned_result: dict,
    config: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / 'metrics.csv'
    q90_path = output_dir / 'q90.json'
    cvar90_path = output_dir / 'cvar90.json'
    episode_path = output_dir / 'episode.csv'
    tuned_path = output_dir / 'tuned_weights.json'
    config_path = output_dir / 'run_config.json'

    metrics_to_save = robust_result['metrics'][_DOC_METRIC_COLUMNS].copy()
    metrics_to_save = metrics_to_save.round(3)
    episode_to_save = episode_df.copy()
    float_cols = episode_to_save.select_dtypes(include=['float', 'float16', 'float32', 'float64']).columns
    episode_to_save[float_cols] = episode_to_save[float_cols].round(3)
    q90_to_save = _round_floats({k: robust_result['q90'][k] for k in _DOC_METRIC_COLUMNS})
    cvar90_to_save = _round_floats({k: robust_result['cvar90'][k] for k in _DOC_METRIC_COLUMNS})
    tuned_to_save = _round_floats(_to_builtin(tuned_result))
    config_to_save = _round_floats(_to_builtin(config))

    metrics_to_save.to_csv(metrics_path, index=False)
    episode_to_save.to_csv(episode_path, index=False)
    q90_path.write_text(json.dumps(q90_to_save, ensure_ascii=False, indent=2), encoding='utf-8')
    cvar90_path.write_text(json.dumps(cvar90_to_save, ensure_ascii=False, indent=2), encoding='utf-8')
    tuned_path.write_text(json.dumps(tuned_to_save, ensure_ascii=False, indent=2), encoding='utf-8')
    config_path.write_text(json.dumps(config_to_save, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'output_dir': str(output_dir),
        'metrics_csv': str(metrics_path),
        'q90_json': str(q90_path),
        'cvar90_json': str(cvar90_path),
        'episode_csv': str(episode_path),
        'tuned_weights_json': str(tuned_path),
        'run_config_json': str(config_path),
    }


def save_metric_plots(output_dir: Path, metrics_df: pd.DataFrame) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import koreanize_matplotlib  # noqa: F401  # 한글 폰트 자동 설정
    import matplotlib.pyplot as plt

    plt.rcParams['axes.unicode_minus'] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = [
        ('temp_viol_rate', 'temp_viol_rate.png', '온도 제약 위반률', '온도 제약 위반률', '#d62728'),
        ('temp_viol_maxrun', 'temp_viol_maxrun.png', '온도 연속 위반 최대 길이', '연속 위반 길이', '#ff7f0e'),
        ('cond_viol_rate', 'cond_viol_rate.png', '결로 위험 위반률', '결로 위험 위반률', '#1f77b4'),
        ('cond_viol_maxrun', 'cond_viol_maxrun.png', '결로 위험 연속 위반 최대 길이', '연속 위반 길이', '#17becf'),
        ('hv_ineff_rate', 'hv_ineff_rate.png', '난방-환기 동시 사용 비효율', '비효율 비율', '#8c564b'),
        ('tv_vent', 'm_deltax_tv_vent.png', '환기 개도율의 총 변화량', '총 변화량', '#2ca02c'),
        ('sw_heat', 'm_sw_heat_switch_count.png', '난방 스위칭 횟수', '스위칭 횟수', '#9467bd'),
    ]

    saved = {}
    for col, filename, title, xlabel, color in plot_specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = pd.to_numeric(metrics_df[col], errors='coerce').dropna().values
        ax.hist(values, bins=min(20, max(5, len(values))), color=color, edgecolor='black', alpha=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel('빈도', fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(alpha=0.25)
        plot_path = output_dir / filename
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        saved[col] = str(plot_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    rh_values = pd.to_numeric(metrics_df['rh_viol_rate'], errors='coerce').dropna().values
    vpd_values = pd.to_numeric(metrics_df['vpd_viol_rate'], errors='coerce').dropna().values
    bins = min(20, max(5, len(metrics_df)))
    ax.hist(rh_values, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.55, label='과습 위반률')
    ax.hist(vpd_values, bins=bins, color='#d62728', edgecolor='black', alpha=0.55, label='저VPD 위반률')
    ax.set_title('과습 위반률 / 저VPD 위반률', fontsize=12)
    ax.set_xlabel('위반률', fontsize=10)
    ax.set_ylabel('빈도', fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plot_path = output_dir / 'rh_vpd_violation_rates.png'
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    saved['rh_vpd_combined'] = str(plot_path)

    return saved


# ── 1. 합성 외기 조건 생성 ──────────────────────────────────────────────────

def gen_boundary_simple(n: int = 22 * 288, dt_min: int = 5, seed: int = 1) -> pd.DataFrame:
    """합성 외기 시계열을 생성한다 (실측 데이터 없는 경우 사용)."""
    rng = np.random.default_rng(seed)
    t_sec = np.arange(n) * dt_min * 60
    daysec = 24 * 3600
    phase = (t_sec % daysec) / daysec
    hour = (t_sec / 3600) % 24

    ar_noise = np.zeros(n)
    for i in range(1, n):
        ar_noise[i] = 0.9 * ar_noise[i - 1] + rng.normal(0, 0.6)
    Tout = 5 + 6 * np.sin(2 * np.pi * (phase - 0.25)) + ar_noise

    sun = np.maximum(0, np.sin(np.pi * (phase - 0.25) / 0.5))
    sun[(phase < 0.25) | (phase > 0.75)] = 0
    It = 650 * sun * (0.3 + 0.7 * rng.random(n))

    RHout = np.clip(0.75 - 0.02 * (Tout - 5) + rng.normal(0, 0.05, n), 0.3, 1.0)
    wind = np.maximum(0.2, np.abs(2 + rng.normal(0, 0.6, n)))
    CO2out = 420 + 5 * np.cos(2 * np.pi * (phase - 0.1)) + rng.normal(0, 5, n)

    return pd.DataFrame({
        'k': np.arange(1, n + 1),
        't_sec': t_sec,
        'hour': hour,
        'Tout': Tout,
        'RHout': RHout,
        'It': It,
        'wind': wind,
        'CO2out': CO2out,
    })


# ── 2. 에피소드 시뮬레이션 ──────────────────────────────────────────────────

def simulate_episode(boundary: pd.DataFrame, theta: dict, weights: dict,
                     dt_min=5, A=200, V=300,
                     Tin0=None, RHin0=None, CO20=None,
                     Tref=None, T_min=12, T_max=28, RH_max=None, VPD_min=0.30,
                     dTcond_min=0.8, alpha=0.25,
                     show_progress=False) -> pd.DataFrame:
    """외기 시나리오와 물리 파라미터 θ에 대해 에피소드를 시뮬레이션한다."""
    n = len(boundary)
    first = boundary.iloc[0]
    if Tin0 is None:
        Tin0 = float(first['Tout'])
    if RHin0 is None:
        RHin0 = float(np.clip(first['RHout'], 0, 1))
    if CO20 is None:
        CO20 = float(first['CO2out'])
    if Tref is None:
        Tref = 0.5 * (T_min + T_max)

    Tin = np.zeros(n); Tin[0] = Tin0
    e_in = np.zeros(n); e_in[0] = RHin0 * float(sat_vp_kpa(Tin0))
    CO2 = np.zeros(n); CO2[0] = CO20

    u_heat = np.zeros(n); x_vent = np.zeros(n)
    curtain = np.zeros(n); u_co2 = np.zeros(n)
    ACH = np.zeros(n); Q_heat = np.zeros(n); Q_ventloss = np.zeros(n)

    prev_a = {'u_heat': 0, 'x_vent': 0, 'curtain': 0, 'u_co2': 0}

    it = range(1, n)
    if show_progress:
        it = tqdm(it, desc='simulate_episode')

    for t in it:
        state = {'Tin': Tin[t - 1], 'e_in': e_in[t - 1], 'CO2': CO2[t - 1]}
        u = boundary.iloc[t - 1].to_dict()
        a = policy_optimal(
            state, u, prev_a, theta,
            weights=weights, Tref=Tref, VPD_min=VPD_min,
            T_min=T_min, T_max=T_max, dTcond_min=dTcond_min, alpha=alpha,
            dt_min=dt_min, A=A, V=V,
        )
        pred = step_dynamics(state, u, a, theta, dt_min, A, V)

        Tin[t] = pred['Tin']; e_in[t] = pred['e_in']; CO2[t] = pred['CO2']
        u_heat[t] = a['u_heat']; x_vent[t] = a['x_vent']
        curtain[t] = a['curtain']; u_co2[t] = a['u_co2']
        ACH[t] = pred['ACH']; Q_heat[t] = pred['Q_heat']; Q_ventloss[t] = pred['Q_ventloss']
        prev_a = a

    sat_arr = np.array([sat_vp_kpa(float(t)) for t in Tin], dtype=float)
    RHin = np.clip(e_in / sat_arr, 0, 1)
    VPD = np.array([vpd_kpa(float(t), float(rh)) for t, rh in zip(Tin, RHin)], dtype=float)
    Td = np.array([dewpoint_c(float(t), float(rh)) for t, rh in zip(Tin, RHin)], dtype=float)
    dTcond = Tin - Td

    return pd.DataFrame({
        'k': boundary['k'].values, 'hour': boundary['hour'].values,
        'Tout': boundary['Tout'].values, 'RHout': boundary['RHout'].values,
        'It': boundary['It'].values, 'wind': boundary['wind'].values,
        'CO2out': boundary['CO2out'].values,
        'Tin': Tin, 'RHin': RHin, 'VPD': VPD, 'dTcond': dTcond, 'CO2': CO2,
        'u_heat': u_heat, 'x_vent': x_vent, 'curtain': curtain, 'u_co2': u_co2,
        'ACH': ACH, 'Q_heat': Q_heat, 'Q_ventloss': Q_ventloss,
    })


# ── 3. KPI 계산 ─────────────────────────────────────────────────────────────

def max_run(b: np.ndarray) -> int:
    if not b.any():
        return 0
    lengths = []
    count = 0
    for v in b:
        if v:
            count += 1
            lengths.append(count)
        else:
            count = 0
    return max(lengths)


def calc_metrics(df: pd.DataFrame,
                 T_min=12, T_max=28, RH_max=0.90,
                 VPD_min=0.30, dTcond_min=0.8, alpha=0.25) -> pd.DataFrame:
    temp_viol = (df['Tin'] < T_min) | (df['Tin'] > T_max)
    cond_viol = df['dTcond'] < dTcond_min
    rh_viol = df['RHin'] > RH_max
    vpd_viol = df['VPD'] < VPD_min

    idx_heat = df['u_heat'] > 0.5
    if idx_heat.any():
        hv_ineff = float((df.loc[idx_heat, 'Q_ventloss'] >
                          alpha * df.loc[idx_heat, 'Q_heat'].clip(lower=1e-6)).mean())
    else:
        hv_ineff = 0.0

    return pd.DataFrame([{
        'temp_viol_rate': float(temp_viol.mean()),
        'temp_viol_maxrun': max_run(temp_viol.values),
        'cond_viol_rate': float(cond_viol.mean()),
        'cond_viol_maxrun': max_run(cond_viol.values),
        'rh_viol_rate': float(rh_viol.mean()),
        'vpd_viol_rate': float(vpd_viol.mean()),
        'hv_ineff_rate': hv_ineff,
        'tv_vent': float(df['x_vent'].diff().abs().sum()),
        'sw_heat': int((df['u_heat'].diff().abs() > 0).sum()),
    }])


def cvar(x: np.ndarray, q: float = 0.9) -> float:
    x = x[np.isfinite(x)]
    thr = float(np.quantile(x, q))
    return float(x[x >= thr].mean())


def _single_mc_run(task: dict) -> dict:
    seed = int(task['seed'])
    n = int(task['n'])
    dt_min = int(task['dt_min'])
    weights = dict(task['weights'])

    rng = np.random.default_rng(seed)
    theta = sample_theta(rng)
    boundary_seed = int(rng.integers(0, 2**31 - 1))
    boundary = gen_boundary_simple(n=n, dt_min=dt_min, seed=boundary_seed)
    ep = simulate_episode(
        boundary, theta, weights, dt_min=dt_min,
        Tref=task.get('Tref'),
        T_min=float(task['T_min']), T_max=float(task['T_max']),
        VPD_min=float(task['VPD_min']), dTcond_min=float(task['dTcond_min']),
        alpha=float(task['alpha']),
        show_progress=False,
    )
    return calc_metrics(
        ep,
        T_min=float(task['T_min']), T_max=float(task['T_max']),
        RH_max=float(task['RH_max']), VPD_min=float(task['VPD_min']),
        dTcond_min=float(task['dTcond_min']), alpha=float(task['alpha']),
    ).iloc[0].to_dict()


# ── 4. 견고성 평가 (Monte Carlo) ────────────────────────────────────────────

def robust_eval(N=50, days=22, dt_min=5, seed=1, weights=None,
                Tref=None, T_min=12, T_max=28, RH_max=0.90, VPD_min=0.30,
                dTcond_min=0.8, alpha=0.25,
                show_progress=True, n_jobs: int = 1) -> dict:
    if weights is None:
        weights = {'wT': 1, 'wVPD': 1, 'wE': 1e-8, 'wDx': 0.2, 'wSlack': 50}

    n = days * (24 * 60 // dt_min)
    n_jobs = int(max(1, n_jobs))
    tasks = [
        {
            'seed': int(seed) + i,
            'n': n,
            'dt_min': int(dt_min),
            'weights': dict(weights),
            'Tref': Tref,
            'T_min': float(T_min),
            'T_max': float(T_max),
            'RH_max': float(RH_max),
            'VPD_min': float(VPD_min),
            'dTcond_min': float(dTcond_min),
            'alpha': float(alpha),
        }
        for i in range(int(N))
    ]

    if n_jobs == 1:
        iterator = tasks if not show_progress else tqdm(tasks, desc='robust_eval')
        rows = [_single_mc_run(task) for task in iterator]
    else:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            mapped = ex.map(_single_mc_run, tasks)
            if show_progress:
                mapped = tqdm(mapped, total=len(tasks), desc=f'robust_eval[{n_jobs}t]')
            rows = list(mapped)

    M = pd.DataFrame(rows)
    cols = M.columns.tolist()
    q90 = {c: float(np.quantile(M[c], 0.9)) for c in cols}
    c90 = {c: cvar(M[c].values, 0.9) for c in cols}
    return {'metrics': M, 'q90': q90, 'cvar90': c90}


# ── 5. 가중치 튜닝 (그리드 탐색) ───────────────────────────────────────────

def tune_weights_grid(grid: pd.DataFrame, N=40, seed=1,
                      Tref=None, T_min=12, T_max=28, RH_max=0.90, VPD_min=0.30,
                      dTcond_min=0.8, alpha=0.25,
                      show_progress=True) -> dict:
    best = None
    best_obj = np.inf

    it = grid.iterrows() if not show_progress else tqdm(grid.iterrows(), total=len(grid))
    for _, row in it:
        w = {k: float(row[k]) for k in ['wT', 'wVPD', 'wE', 'wDx', 'wSlack']}
        res = robust_eval(
            N=N, seed=seed, weights=w, Tref=Tref,
            T_min=T_min, T_max=T_max, RH_max=RH_max,
            VPD_min=VPD_min, dTcond_min=dTcond_min, alpha=alpha,
            show_progress=False, n_jobs=1,
        )
        obj = (res['cvar90']['temp_viol_rate']
               + res['cvar90']['cond_viol_rate']
               + 0.1 * res['cvar90']['hv_ineff_rate'])
        if obj < best_obj:
            best_obj = obj
            best = w

    return {'best_weights': best, 'best_obj': best_obj}


if __name__ == '__main__':
    run_dir = _ensure_results_dir()

    robust_cfg = {'N': 60, 'days': 22, 'dt_min': 5, 'seed': 42, 'n_jobs': max(1, min((os.cpu_count() or 1), 8))}
    constraint_cfg = {'T_min': 12.0, 'T_max': 28.0, 'RH_max': 0.90, 'VPD_min': 0.30, 'dTcond_min': 0.8, 'alpha': 0.25}
    weight_cfg = {'wT': 1, 'wVPD': 1, 'wE': 1e-8, 'wDx': 0.2, 'wSlack': 50}
    tune_cfg = {'enabled': False, 'N': 30, 'seed': 42}

    res = robust_eval(**robust_cfg, weights=weight_cfg, show_progress=True, **constraint_cfg)
    print('q90\n', pd.Series(res['q90']))
    print('cvar90\n', pd.Series(res['cvar90']))
    print(res['metrics'].head())

    metric_plot_paths = save_metric_plots(run_dir, res['metrics'])

    boundary = gen_boundary_simple(n=22 * 288, dt_min=5, seed=42)
    theta = sample_theta(np.random.default_rng(0))
    ep = simulate_episode(boundary, theta, weight_cfg, show_progress=True, **constraint_cfg)

    import matplotlib.pyplot as plt
    plt.plot(ep['Tin']); plt.ylabel('Tin (°C)'); plt.title('Simulated Indoor Temperature')
    plt.tight_layout()
    plot_path = run_dir / 'episode_tin.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()

    tuned = {'best_weights': None, 'best_obj': None}
    if tune_cfg['enabled']:
        grid_vals = list(product([0.5, 1, 2], [0.5, 1], [1e-9, 1e-8], [0.1, 0.2, 0.4], [30, 50, 80]))
        grid = pd.DataFrame(grid_vals, columns=['wT', 'wVPD', 'wE', 'wDx', 'wSlack'])
        tuned = tune_weights_grid(
            grid, N=tune_cfg['N'], seed=tune_cfg['seed'],
            show_progress=True, **constraint_cfg,
        )
        print('best_weights:', tuned['best_weights'])

    saved = save_eval_outputs(
        output_dir=run_dir,
        robust_result=res,
        episode_df=ep,
        tuned_result=tuned,
        config={
            'robust_eval': robust_cfg,
            'weights': weight_cfg,
            'constraints': constraint_cfg,
            'episode_seed': 42,
            'theta_seed': 0,
            'tuning': tune_cfg,
            'plot_path': str(plot_path),
            'metric_plot_paths': metric_plot_paths,
        },
    )
    print('metric_plots:', metric_plot_paths)
    print('saved_outputs:', saved['output_dir'])
