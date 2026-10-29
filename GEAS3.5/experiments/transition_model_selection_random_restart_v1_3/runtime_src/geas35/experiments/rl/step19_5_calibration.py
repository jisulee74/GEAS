"""Learning-curve calibration and immutable Step 19.5 budget artifacts."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, math, random
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

STEP19_5_VERSION="geas35.rl.step19_5.v2"
OFFICIAL_CROPS=("strawberry","melon","cucumber")
DIAGNOSTICS=("policy_loss","value_loss","approx_kl","entropy","explained_variance",
             "gradient_norm","in_support_rate","warning_ood_rate","severe_ood_rate",
             "projection_rate","fallback_rate","termination_rate")

def _read(path: Path)->dict[str,Any]:
    value=json.loads(path.read_text());
    if not isinstance(value,dict): raise ValueError(f"Expected object: {path}")
    return value
def _write(path: Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,default=str)+"\n"); tmp.replace(path)
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def _rank(values: Mapping[str,float])->dict[str,int]:
    ordered=sorted(values,key=lambda key:(-values[key],key)); return {key:i for i,key in enumerate(ordered)}
def kendall_tau(left:Mapping[str,float],right:Mapping[str,float])->float:
    keys=sorted(set(left)&set(right)); concordant=discordant=0
    for i,a in enumerate(keys):
      for b in keys[i+1:]:
        product=(left[a]-left[b])*(right[a]-right[b])
        if product>0:concordant+=1
        elif product<0:discordant+=1
    denominator=concordant+discordant
    return 1.0 if denominator==0 else (concordant-discordant)/denominator

def _top_set(values:Mapping[str,float],fraction:float)->set[str]:
    count=max(1,int(math.ceil(len(values)*fraction)))
    return set(sorted(values,key=lambda key:(-values[key],key))[:count])

def _bootstrap_slope_ci(rows:Sequence[Mapping[str,Any]],steps:Sequence[int],samples:int,
                        confidence:float,seed:int)->tuple[float,float]:
    rng=np.random.default_rng(seed); points=[]
    for step in steps:
      values=[float(r["mean_reward_per_scheduled_valid_step"]) for r in rows if int(r["steps"])==step]
      points.append(values)
    slopes=[]; x=np.arange(len(steps),dtype=float)
    for _ in range(samples):
      y=np.asarray([np.median(rng.choice(values,size=len(values),replace=True)) for values in points])
      slopes.append(float(np.polyfit(x,y,1)[0]))
    alpha=(1-confidence)/2
    return float(np.quantile(slopes,alpha)),float(np.quantile(slopes,1-alpha))

def analyze_crop_learning_curves(rows:Sequence[Mapping[str,Any]],config:Mapping[str,Any])->dict[str,Any]:
    if not rows: raise ValueError("Learning curve rows are empty")
    if any(r.get("split")!="validation" or r.get("test_accessed") is not False for r in rows):
        raise ValueError("Step 19.5 accepts Validation-only metrics with Test locked")
    steps=sorted({int(r["steps"]) for r in rows}); configs=sorted({str(r["config_id"]) for r in rows})
    seeds=sorted({int(r["seed"]) for r in rows})
    expected={(step,cfg,seed) for step in steps for cfg in configs for seed in seeds}
    observed={(int(r["steps"]),str(r["config_id"]),int(r["seed"])) for r in rows}
    if expected!=observed: raise ValueError("Learning curves must form a complete step/config/seed grid")
    medians={step:{cfg:float(np.median([float(r["mean_reward_per_scheduled_valid_step"])
             for r in rows if int(r["steps"])==step and str(r["config_id"])==cfg])) for cfg in configs} for step in steps}
    rc=config["ranking_stability"]; lc=config["learning_stability"]
    comparisons=[]
    for previous,current in zip(steps,steps[1:]):
      tau=kendall_tau(medians[previous],medians[current]); a=_top_set(medians[previous],rc["top_fraction"]); b=_top_set(medians[current],rc["top_fraction"])
      comparisons.append({"previous_steps":previous,"steps":current,"kendall_tau":tau,
                          "top_overlap":len(a&b)/len(a|b),"top_set_stable":a==b,
                          "stable":tau>=rc["kendall_tau_min"] and a==b})
    consecutive=int(rc["consecutive_intervals"]); first=None
    for index in range(consecutive-1,len(comparisons)):
      if all(item["stable"] for item in comparisons[index-consecutive+1:index+1]): first=comparisons[index]["steps"]; break
    plateau=[]; window=int(lc["window_intervals"])
    for index in range(window,len(steps)):
      current=steps[index]; recent=steps[index-window:index+1]
      current_values=np.asarray(list(medians[current].values())); iqr=float(np.quantile(current_values,.75)-np.quantile(current_values,.25))
      scale=max(iqr,np.finfo(float).eps); deltas=[]
      for a,b in zip(recent,recent[1:]):
        deltas.extend(abs(medians[b][cfg]-medians[a][cfg]) for cfg in configs)
      ci=_bootstrap_slope_ci(rows,recent,int(lc["bootstrap_samples"]),float(lc["bootstrap_confidence"]),42+index)
      current_rows=[r for r in rows if int(r["steps"])==current]
      finite=all(np.isfinite(float(r[key])) for r in current_rows for key in DIAGNOSTICS)
      safe=all(bool(r.get("safety_gate_passed",False)) for r in current_rows)
      stable=max(deltas)<=float(lc["reward_delta_fraction_of_current_iqr_max"])*scale and ci[0]<=0<=ci[1] and finite and safe
      plateau.append({"steps":current,"reward_iqr":iqr,"maximum_recent_delta":max(deltas),
                      "bootstrap_slope_ci":list(ci),"finite_diagnostics":finite,"safety_gate_passed":safe,"stable":stable})
    stable_steps={item["steps"] for item in comparisons if item["stable"]}
    final=None
    if first is not None:
      for item in plateau:
        if item["steps"]>=first and item["stable"] and item["steps"] in stable_steps: final=item["steps"]; break
    if first is None or final is None:
      return {"status":"calibration_inconclusive","rung_budgets":[],"steps":steps,"medians":medians,
              "ranking_comparisons":comparisons,"learning_stability":plateau,"test_accessed":False}
    rungs=[first,final] if first!=final else [final]
    ratio=float(config["rung_selection"]["insert_middle_if_final_to_first_ratio_exceeds"])
    if len(rungs)==2 and final/first>ratio:
      target=math.sqrt(first*final); candidates=[step for step in steps if first<step<final and step in stable_steps]
      if candidates:rungs.insert(1,min(candidates,key=lambda step:abs(math.log(step/target))))
    if len(rungs)<2:
      return {"status":"calibration_inconclusive","reason":"distinct_first_and_final_rungs_required",
              "rung_budgets":[],"steps":steps,"medians":medians,"ranking_comparisons":comparisons,
              "learning_stability":plateau,"test_accessed":False}
    return {"status":"passed","rung_budgets":rungs,"steps":steps,"medians":medians,
            "ranking_comparisons":comparisons,"learning_stability":plateau,"test_accessed":False}

def representative_configurations(search_space:Mapping[str,Any],default:Mapping[str,Any],count:int=5,seed:int=42)->list[dict[str,Any]]:
    if count<1:raise ValueError("count must be positive")
    keys=sorted(search_space); rng=random.Random(seed); candidates=[]
    for _ in range(1000): candidates.append({key:rng.choice(search_space[key]) for key in keys})
    def vector(item):
      values=[]
      for key in keys:
        choices=search_space[key]; value=item[key]
        if value in choices: position=choices.index(value)/max(len(choices)-1,1)
        elif all(isinstance(v,(int,float)) for v in choices):
          low,high=float(min(choices)),float(max(choices));position=(float(value)-low)/max(high-low,np.finfo(float).eps)
        else: raise ValueError(f"Default {key} is outside a nonnumeric search space")
        values.append(float(np.clip(position,0,1)))
      return np.asarray(values)
    chosen=[{key:default[key] for key in keys}]
    while len(chosen)<count:
      pick=max(candidates,key=lambda item:min(float(np.linalg.norm(vector(item)-vector(old))) for old in chosen))
      chosen.append(pick); candidates.remove(pick)
    return [{"config_id":f"representative_{i}","parameters":item} for i,item in enumerate(chosen)]

def create_budget_artifact(project_root:str|Path,curves_by_crop:Mapping[str,Sequence[Mapping[str,Any]]])->Path:
    root=Path(project_root).resolve(); out=root/"experiments/rl_policy_training/artifacts/step19_5_v2"
    config_path=root/"experiments/rl_policy_training/configs/step19_5_calibration_v2.json"; config=_read(config_path)
    results={crop:analyze_crop_learning_curves(curves_by_crop[crop],config) for crop in OFFICIAL_CROPS}
    status="passed" if all(v["status"]=="passed" for v in results.values()) else "calibration_inconclusive"
    artifact={"schema_version":"geas35.rl.step19_5.budget.v2","protocol_version":STEP19_5_VERSION,
              "status":status,"created_at_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
              "budget_decision_source":"observed_learning_curve_and_ranking_stability","numeric_budget_precommitted":False,
              "crops":{crop:{"rung_budgets":result["rung_budgets"],"status":result["status"]} for crop,result in results.items()},
              "test_accessed":False}
    out.mkdir(parents=True,exist_ok=True); budget=out/"calibrated_budget.json"; _write(budget,artifact)
    _write(out/"calibration_report.json",{"schema_version":STEP19_5_VERSION,"status":status,"crops":results,"test_accessed":False})
    _write(out/"step19_5_integrity_manifest.json",{"schema_version":STEP19_5_VERSION,"status":status,
           "config_sha256":_sha(config_path),"budget":{"path":budget.relative_to(root).as_posix(),"sha256":_sha(budget)},
           "test_accessed":False,"step20_v2_started":False})
    return budget

__all__=["STEP19_5_VERSION","analyze_crop_learning_curves","create_budget_artifact","kendall_tau","representative_configurations"]
