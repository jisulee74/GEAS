"""Process-vector environment and deterministic seed mapping for RL v2."""
from __future__ import annotations
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class VectorEnvSpec:
    kind:str
    payload:dict[str,Any]

def deterministic_episode_seed(base_seed:int,crop:str,config_id:str,env_index:int,episode_counter:int)->int:
    import hashlib
    value=f"{base_seed}|{crop}|{config_id}|{env_index}|{episode_counter}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4],"big")%(2**31-1)

class _SyntheticCounterEnv:
    def __init__(self,horizon:int=4):self.horizon=horizon;self.position=0
    def reset(self,*,seed=None,options=None):self.position=0;return np.asarray([0.],np.float32),{"seed":seed}
    def step(self,action):
      self.position+=1; done=self.position>=self.horizon
      return np.asarray([float(self.position)],np.float32),float(np.asarray(action).sum()),done,False,{"position":self.position}

def _build(spec:VectorEnvSpec):
    if spec.kind=="synthetic_counter":return _SyntheticCounterEnv(**spec.payload)
    if spec.kind=="geas_model_driven":
      from geas35.rl.model_driven_env import load_model_driven_env_from_step15
      env=load_model_driven_env_from_step15(project_root=Path(spec.payload["project_root"]),crop=spec.payload["crop"],split=spec.payload["split"])
      n_jobs=int(spec.payload.get("transition_n_jobs",1))
      for estimator in getattr(env.model,"estimators_",{}).values():
        if hasattr(estimator,"n_jobs"):estimator.n_jobs=n_jobs
      return env
    raise ValueError(f"Unknown vector environment kind: {spec.kind}")

def _worker(connection,spec):
    env=_build(spec)
    try:
      while True:
        command,payload=connection.recv()
        if command=="reset":connection.send(env.reset(**payload))
        elif command=="step":connection.send(env.step(payload))
        elif command=="feasible":
          if spec.kind=="geas_model_driven":
            from geas35.rl.hybrid_ppo import SupportAwareActionAdapter
            value=SupportAwareActionAdapter(env).feasible_spec()
          else:
            from geas35.rl.hybrid_ppo import FeasibleActionSpec
            value=FeasibleActionSpec.unconstrained()
          connection.send(value)
        elif command=="close":connection.send(True);break
        else:raise ValueError(f"Unknown command: {command}")
    except BaseException as exc:
      connection.send(("worker_error",type(exc).__name__,str(exc)))
    finally:connection.close()

class ProcessVectorEnv:
    def __init__(self,specs:Sequence[VectorEnvSpec],start_method:str="spawn"):
      if not specs:raise ValueError("At least one environment is required")
      context=mp.get_context(start_method);self.connections=[];self.processes=[];self.closed=False
      for spec in specs:
        parent,child=context.Pipe();process=context.Process(target=_worker,args=(child,spec),daemon=True);process.start();child.close()
        self.connections.append(parent);self.processes.append(process)
    def _receive(self,connection):
      value=connection.recv()
      if isinstance(value,tuple) and value and value[0]=="worker_error":raise RuntimeError(":".join(value[1:]))
      return value
    def reset(self,seeds:Sequence[int]):
      if len(seeds)!=len(self.connections):raise ValueError("seed count differs from environment count")
      for connection,seed in zip(self.connections,seeds):connection.send(("reset",{"seed":int(seed)}))
      values=[self._receive(c) for c in self.connections]
      return np.stack([v[0] for v in values]),[v[1] for v in values]
    def feasible_specs(self):
      for connection in self.connections:connection.send(("feasible",None))
      return [self._receive(c) for c in self.connections]
    def reset_indices(self,indices:Sequence[int],seeds:Sequence[int]):
      if len(indices)!=len(seeds):raise ValueError("indices/seeds differ")
      for index,seed in zip(indices,seeds):self.connections[index].send(("reset",{"seed":int(seed)}))
      return {index:self._receive(self.connections[index]) for index in indices}
    def step(self,actions:Sequence[Any]):
      if len(actions)!=len(self.connections):raise ValueError("action count differs from environment count")
      for connection,action in zip(self.connections,actions):connection.send(("step",action))
      values=[self._receive(c) for c in self.connections]
      return (np.stack([v[0] for v in values]),np.asarray([v[1] for v in values],float),
              np.asarray([v[2] for v in values],bool),np.asarray([v[3] for v in values],bool),[v[4] for v in values])
    def close(self):
      if self.closed:return
      for c in self.connections:c.send(("close",None))
      for c in self.connections:self._receive(c);c.close()
      for p in self.processes:p.join(timeout=5)
      self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def collect_vectorized_rollout(vector_env:ProcessVectorEnv,trainer:Any,*,steps:int,base_seed:int,crop:str,config_id:str):
    """Collect an exact, support-aware rollout; steps must divide across environments."""
    import torch
    from geas35.rl.hybrid_ppo import RolloutBuffer
    from geas35.rl.mdp_v1 import MDP_V1_ACTION_COLUMNS
    count=len(vector_env.connections)
    if steps<=0 or steps%count:raise ValueError("Vector rollout steps must be positive and divisible by env count")
    counters=[0]*count;seeds=[deterministic_episode_seed(base_seed,crop,config_id,i,0) for i in range(count)]
    observations,_=vector_env.reset(seeds);buffer=RolloutBuffer()
    for _ in range(steps//count):
      specs=vector_env.feasible_specs();tensor=torch.as_tensor(observations,dtype=torch.float32,device=trainer.device)
      feasible={"continuous_low":torch.as_tensor(np.stack([x.continuous_low for x in specs]),device=trainer.device),"continuous_high":torch.as_tensor(np.stack([x.continuous_high for x in specs]),device=trainer.device),"binary_allow_zero":torch.as_tensor(np.stack([x.binary_allow_zero for x in specs]),device=trainer.device),"binary_allow_one":torch.as_tensor(np.stack([x.binary_allow_one for x in specs]),device=trainer.device)}
      with torch.no_grad():output=trainer.policy.act(tensor,feasible,deterministic=False)
      actions=output["action"].cpu().numpy().astype(np.float32);next_obs,rewards,terminated,truncated,infos=vector_env.step(actions)
      with torch.no_grad():next_values=trainer.policy(torch.as_tensor(next_obs,dtype=torch.float32,device=trainer.device))[-1].cpu().numpy()
      for index in range(count):
       info=infos[index];executed=info.get("executed_action")
       executed_array=np.asarray([executed[column] for column in MDP_V1_ACTION_COLUMNS],np.float32) if executed is not None else actions[index].copy()
       buffer.add(observations=observations[index].copy(),policy_actions=actions[index],executed_actions=executed_array,log_probs=float(output["log_prob"][index].cpu()),values=float(output["value"][index].cpu()),next_values=float(next_values[index]),rewards=float(rewards[index]),terminated=bool(terminated[index]),truncated=bool(truncated[index]),feasible_specs=specs[index],warning_ood_penalties=float(info.get("warning_ood_penalty",0)),projected=bool(info.get("support_projection",{}).get("changes",{})),fallback=bool(info.get("safety_fallback_applied",False)),state_ood_scores=float(info.get("state_ood_score",float("nan"))),joint_ood_scores=float(info.get("joint_ood_score",float("nan"))),joint_ood_levels=str(info.get("joint_ood_level","unknown")),termination_reasons=info.get("termination_reason"))
      observations=next_obs
      done_indices=[i for i in range(count) if terminated[i] or truncated[i]]
      if done_indices:
       reset_seeds=[]
       for i in done_indices:counters[i]+=1;reset_seeds.append(deterministic_episode_seed(base_seed,crop,config_id,i,counters[i]))
       reset=vector_env.reset_indices(done_indices,reset_seeds)
       for i,(observation,_) in reset.items():observations[i]=observation
    return buffer

__all__=["ProcessVectorEnv","VectorEnvSpec","collect_vectorized_rollout","deterministic_episode_seed"]
