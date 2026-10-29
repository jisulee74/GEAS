"""Paired Validation-only quality-flag ablation reports."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd
from geas35.io_utils import write_json

@dataclass(frozen=True)
class QualityFlagAblationReport:
    crop: str
    model_name: str
    mean_difference: float
    ci_lower: float
    ci_upper: float
    effect: str
    paired_window_count: int
    bootstrap_samples: int
    metric: str = "paired_rollout_weighted_nrmse"
    test_used: bool = False
    def to_artifact(self) -> dict[str, object]:
        return dict(self.__dict__)

def paired_bootstrap_difference(with_flags: np.ndarray, without_flags: np.ndarray, *, samples: int = 2000, confidence: float = 0.95, seed: int = 42) -> tuple[float,float,float]:
    a=np.asarray(with_flags,dtype=float); b=np.asarray(without_flags,dtype=float)
    valid=np.isfinite(a)&np.isfinite(b); delta=a[valid]-b[valid]
    if not len(delta): raise ValueError("No finite paired Validation windows.")
    rng=np.random.default_rng(seed)
    means=np.asarray([np.mean(delta[rng.integers(0,len(delta),len(delta))]) for _ in range(samples)])
    alpha=(1.0-confidence)/2.0
    return float(np.mean(delta)),float(np.quantile(means,alpha)),float(np.quantile(means,1-alpha))

def _window_scores(payload: Mapping[str,object]) -> dict[str,float]:
    parts={"15min":0.10,"30min":0.25,"60min":0.45}; drift_weight=0.10
    accum: dict[str,list[tuple[float,float]]] = {}
    for horizon,weight in parts.items():
        metrics=payload[horizon]; ids=metrics["window_ids"]
        vals=metrics["window_trajectory_nrmse_samples"]
        drift=metrics["window_drift_nmae_samples"]
        for key,value,dvalue in zip(ids,vals,drift):
            accum.setdefault(str(key),[]).extend([(weight,float(value)),(drift_weight/3.0,float(dvalue))])
    return {key: float(sum(w*v for w,v in values)/sum(w for w,v in values)) for key,values in accum.items() if all(np.isfinite(v) for _,v in values)}

def compare_variant_roots(with_root: Path|str, without_root: Path|str, output_root: Path|str, *, dataset_root: Path|str|None = None, crops=("strawberry","melon","cucumber"), models=("persistence","linear_regression","linear_svr","knn","extra_trees","lightgbm","xgboost","mlp"), samples:int=2000) -> list[QualityFlagAblationReport]:
    with_root=Path(with_root); without_root=Path(without_root); output_root=Path(output_root); output_root.mkdir(parents=True,exist_ok=True)
    reports=[]
    for crop in crops:
        for model in models:
            a=json.loads((with_root/crop/crop/model/"rollout_metrics.json").read_text())
            b=json.loads((without_root/crop/crop/model/"rollout_metrics.json").read_text())
            sa,sb=_window_scores(a),_window_scores(b); keys=sorted(set(sa)&set(sb))
            mean,lo,hi=paired_bootstrap_difference(np.array([sa[k] for k in keys]),np.array([sb[k] for k in keys]),samples=samples)
            effect="improved" if hi<0 else "degraded" if lo>0 else "no_significant_difference"
            reports.append(QualityFlagAblationReport(crop,model,mean,lo,hi,effect,len(keys),samples))
    rows=[r.to_artifact() for r in reports]
    pd.DataFrame(rows).to_csv(output_root/"quality_flag_ablation.csv",index=False)
    write_json(output_root/"quality_flag_ablation.json",{"stage":"quality_flag_ablation","rows":rows,"test_used":False},convert=True)
    ranking=[]
    for variant,root in (("deduplicated_with_quality_flags",with_root),("deduplicated_without_quality_flags",without_root)):
        for crop in crops:
            table=pd.read_csv(root/crop/"candidate_metrics.csv")
            for record in table.to_dict(orient="records"):
                ranking.append({"crop":crop,"feature_variant":variant,**record})
    ranking_frame=pd.DataFrame(ranking)
    if "validation_rollout_weighted_score" in ranking_frame:
        ranking_frame["joint_validation_rank"]=ranking_frame.groupby("crop")["validation_rollout_weighted_score"].rank(method="min")
    ranking_frame.to_csv(output_root/"joint_validation_ranking.csv",index=False)
    prevalence=[]
    if dataset_root is not None:
        dataset_root=Path(dataset_root)
        for crop in crops:
            frame=pd.read_parquet(dataset_root/crop/"train.parquet",columns=["obs_quality_missing_imputed_flag","obs_quality_outlier_flag"])
            prevalence.append({"crop":crop,"rows":len(frame),**{f"{col}_count":int((pd.to_numeric(frame[col],errors="coerce").fillna(0)>0).sum()) for col in frame},**{f"{col}_rate":float((pd.to_numeric(frame[col],errors="coerce").fillna(0)>0).mean()) for col in frame}})
        pd.DataFrame(prevalence).to_csv(output_root/"quality_flag_prevalence.csv",index=False)
    lines=["# Quality-Flag Ablation", "", "Test data were not used.", "", "## Paired Bootstrap", "", pd.DataFrame(rows).to_markdown(index=False)]
    if prevalence:
        lines.extend(["", "## Train Quality-Flag Prevalence", "", pd.DataFrame(prevalence).to_markdown(index=False)])
    (output_root/"ablation_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return reports

__all__=["QualityFlagAblationReport","compare_variant_roots","paired_bootstrap_difference"]
