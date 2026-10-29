"""Budget-bound Profile-A MAIN driver: CPU, one worker, exactly sixteen updates.

Frozen adapter is not modified. Finalized episode arrays only. Independent fresh initialization; no HPO or Test.
"""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from geas35.rl.profile_a_offline_v1 import (
    ProfileAOfflineEnv, OfflineCollector, masked_gae, locked_json, file_hash,
    digest, SEMANTIC_HASH, COHORT_HASH, development_rank,
)
from geas35.rl.hybrid_ppo import HybridActorCritic, HybridPPOConfig, FeasibleActionSpec

OPTIMIZER_STEP_COUNT = 0

EXPERIMENT_HASH = '3580155904fc600015e6c6ce55d5b305771ce275cef68e09e134dbb6de43726e'


def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
    return a==b


def rng_state(sampler):
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),'sampler':copy.deepcopy(sampler.rng.bit_generator.state)}


def restore_rng(value,sampler):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch']);sampler.rng.bit_generator.state=copy.deepcopy(value['sampler'])


class RunSampler:
    def __init__(self, ids, seed):
        self.ids=sorted(ids);self.rng=np.random.default_rng(seed);self.order=[];self.position=0;self.passes=0
    def next(self):
        if self.position==len(self.order):
            self.order=list(self.ids);self.rng.shuffle(self.order);self.position=0;self.passes+=1
        result=self.order[self.position];self.position+=1;return result
    def state(self):
        return {'ids':self.ids,'order':list(self.order),'position':self.position,'passes':self.passes,'rng':copy.deepcopy(self.rng.bit_generator.state)}
    def load(self,state):
        if state['ids']!=self.ids:raise ValueError('SAMPLER_MEMBERSHIP_MISMATCH')
        self.order=list(state['order']);self.position=state['position'];self.passes=state['passes'];self.rng.bit_generator.state=copy.deepcopy(state['rng'])


def preflight(root):
    a=root/'experiments/rl_policy_training/artifacts'
    c=locked_json(a/'cucumber_profile_a_ppo_experiment_prereg_v1/ppo_experiment_contract.json','experiment_payload_sha256',EXPERIMENT_HASH)
    for r in c['authoritative_inputs']:
        if file_hash(root/r['path'])!=r['sha256']:raise ValueError('EXPERIMENT_INPUT_CHANGED:'+r['path'])
    locked_json(a/'cucumber_profile_a_ppo_semantics_v1/ppo_semantics_contract.json','semantic_payload_sha256',SEMANTIC_HASH)
    locked_json(a/'cucumber_profile_a_ppo_semantics_v1/ppo_cohort_registry.json','registry_payload_sha256',COHORT_HASH)
    impl=json.loads((a/'cucumber_profile_a_ppo_minimal_impl_v1/implementation_manifest.json').read_text())
    for k,h in {**impl['source_hashes'],**impl['frozen_hashes']}.items():
        if file_hash(root/k)!=h:raise ValueError('FROZEN_INPUT_CHANGED:'+k)
    for r in (100.,10000.):
        adv,_=masked_gae([1.,r],[0.,0.],[7.,0.],[1,0],[0,0],['truncated','new'])
        if not np.isclose(adv[0],7.93,atol=1e-12,rtol=0):raise AssertionError('GAE_RESET_LEAK')
    return c


def config(c,seed):
    hp={k:v['value'] for k,v in c['hyperparameters'].items()}
    cfg=HybridPPOConfig(observation_dim=hp['observation_dimension'],continuous_action_dim=hp['continuous_actions'],
        binary_action_dim=hp['binary_actions'],hidden_sizes=tuple(hp['hidden_sizes']),activation=hp['activation'],
        beta_concentration_floor=hp['beta_concentration_floor'],learning_rate=hp['learning_rate'],gamma=hp['gamma'],
        gae_lambda=hp['gae_lambda'],clip_range=hp['clip_range'],value_clip_range=hp['value_clip_range'],
        value_coefficient=hp['value_loss_coefficient'],entropy_coefficient=hp['entropy_coefficient'],
        max_gradient_norm=hp['max_gradient_norm'],update_epochs=hp['update_epochs'],minibatch_size=hp['minibatch_size'],
        target_kl=hp['target_kl'],normalize_advantages=hp['normalize_advantages'],seed=seed)
    if cfg.observation_dim!=56 or hp['num_workers']!=1 or hp['device']!='cpu' or hp['lr_scheduler'] is not None:raise ValueError('EFFECTIVE_CONFIGURATION')
    return cfg,hp


def optimizer_for(model,hp):
    return torch.optim.Adam(model.parameters(),lr=hp['learning_rate'],betas=tuple(hp['adam_betas']),eps=hp['adam_eps'],
        weight_decay=hp['weight_decay'],amsgrad=hp['amsgrad'],maximize=hp['maximize'],foreach=hp['foreach'],
        fused=hp['fused'],capturable=hp['capturable'],differentiable=hp['differentiable'],decoupled_weight_decay=hp['decoupled_weight_decay'])


def act(model,obs,deterministic):
    with torch.no_grad():
        out=model.act(torch.as_tensor(obs,dtype=torch.float32).reshape(1,56),FeasibleActionSpec.unconstrained().batched(1,device='cpu'),deterministic=deterministic)
    return out['action'][0].numpy().copy(),float(out['log_prob'][0]),float(out['value'][0])


def value_fn(model,obs):
    with torch.no_grad():return float(model(torch.as_tensor(obs,dtype=torch.float32).reshape(1,56))[-1][0])


def finite_parameters(model):
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('NONFINITE_PARAMETER')


def flat_parameters(model):return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters()])


def summarize(runs,infos):
    categories={k:sum(r['category']==k for r in runs) for k in ['COMPLETE','FAILURE','SYSTEM_INVALID','UNSCORABLE']}
    executed=[i for i in infos if i.get('quality',{}).get('evaluation_status')=='VALID']
    q=[i['quality']['reward'] for i in executed];lengths=[r['admitted'] for r in runs]
    returns=[r['numeric_return'] for r in runs if r['category']=='COMPLETE' and r['numeric_return'] is not None]
    def stats(v):return {'min':float(np.min(v)),'median':float(np.median(v)),'mean':float(np.mean(v)),'max':float(np.max(v))} if v else dict(min=None,median=None,mean=None,max=None)
    return {'categories':categories,'rates':{k:v/len(runs) if runs else None for k,v in categories.items()},'runs':len(runs),
        'attempted_decisions':len(infos),'admitted':sum(lengths),'admission_rate':sum(lengths)/len(infos) if infos else None,
        'truncated_fragments':0,'mean_admitted_run_length':float(np.mean(lengths)) if lengths else None,
        'median_admitted_run_length':float(np.median(lengths)) if lengths else None,'quality':stats(q),
        'complete_discounted_returns':stats(returns),
        'projection_rate':sum(i.get('projected_action')!=i.get('proposed') for i in executed)/len(executed) if executed else None,
        'fallback_rate':sum(i.get('fallback_action') is not None for i in executed)/len(executed) if executed else None,
        'weather_cap_rate':sum(i.get('intermediate_action')!=i.get('final_action') for i in executed)/len(executed) if executed else None,
        'intervention_denominator':'executed valid outcomes','rate_denominator':'all scheduled/attempted natural runs incl empty failure'}


def run_episode(env,model,run_id,seed,deterministic):
    observation,reset=env.reset(run_id,seed);collector=OfflineCollector(env);infos=[]
    while not env.done:
        proposal,lp,v=act(model,observation,deterministic)
        observation,reward,done,_,info=collector.step(proposal,lp,v,lambda o:value_fn(model,o))
        infos.append(info)
    stage=env.staging
    if not stage.closed:raise AssertionError('PROVISIONAL_EPISODE_AT_FINAL')
    record={'run_id':run_id,'parent_episode_id':str(env.run.iloc[0].episode_id),'scheduled_steps':len(env.run),
        'category':stage.category,'admitted':len(stage.rows),'numeric_return':stage.numeric_return,'attempted':len(infos),
        'binding':env.checkpoint_metadata(),'events':stage.events}
    return stage.arrays(),copy.deepcopy(stage.rows),record,infos


def evaluate(env,model,sampler,seed,update,output):
    before=rng_state(sampler);sample_before=sampler.state();params=flat_parameters(model);runs=[];infos=[]
    try:
        for j,run_id in enumerate(sorted(env.cohort.runs)):
            _,_,record,events=run_episode(env,model,run_id,seed,True);runs.append(record);infos.extend(events)
            if record['category'] in ['SYSTEM_INVALID','UNSCORABLE']:raise RuntimeError('DEV_'+record['category'])
            if (j+1)%5==0:print(f'seed={seed} update={update} dev runs={j+1}/25',flush=True)
        if len(runs)!=25 or sum(r['scheduled_steps'] for r in runs)!=2811:raise AssertionError('DEV_SCHEDULE')
    finally:
        after=rng_state(sampler);sampler_after=sampler.state()
        restore_rng(before,sampler)
        if not equal(before,after) or not equal(sample_before,sampler_after):raise AssertionError('DEV_CHANGED_TRAIN_RNG')
        if not torch.equal(params,flat_parameters(model)):raise AssertionError('DEV_CHANGED_POLICY')
        result={'experiment_hash':EXPERIMENT_HASH,'seed':seed,'update':update,'complete_schedule':len(runs)==25,
            'rng_unchanged':equal(before,after),'runs':runs,'metrics':summarize(runs,infos)}
        write(output/f'development_update{update}.json',result)
    return result


def update_model(model,optimizer,rows,arrays,cfg,update_index):
    global OPTIMIZER_STEP_COUNT
    if len(rows)!=len(arrays['returns']) or not 1024<=len(rows)<=1308:raise AssertionError('BATCH_RANGE')
    for row in rows:
        if row['reward'] is None or not np.isfinite(row['reward']):raise AssertionError('INVALID_ADMISSION')
    advantages=arrays['advantages'].copy();advantages=(advantages-advantages.mean())/(advantages.std(ddof=0)+1e-8)
    data={k:torch.as_tensor(v,dtype=torch.float32) for k,v in dict(observations=arrays['observations'],actions=arrays['actions'],
        old_log_probs=arrays['log_probs'],returns=arrays['returns'],advantages=advantages,old_values=np.array([r['value'] for r in rows])).items()}
    finite_parameters(model);before=flat_parameters(model);metrics=[];generator=torch.Generator(device='cpu').manual_seed(cfg.seed+update_index)
    fired=None;epochs=0
    for epoch in range(cfg.update_epochs):
        perm=torch.randperm(len(rows),generator=generator)
        for mb,start in enumerate(range(0,len(rows),cfg.minibatch_size)):
            ix=perm[start:start+cfg.minibatch_size];ev=model.evaluate_actions(data['observations'][ix],data['actions'][ix],FeasibleActionSpec.unconstrained().batched(len(ix),device='cpu'))
            logratio=ev['log_prob']-data['old_log_probs'][ix];ratio=torch.exp(logratio)
            policy_loss=-torch.minimum(ratio*data['advantages'][ix],ratio.clamp(1-cfg.clip_range,1+cfg.clip_range)*data['advantages'][ix]).mean()
            old=data['old_values'][ix];returns=data['returns'][ix];v=ev['value'];vclip=old+(v-old).clamp(-cfg.value_clip_range,cfg.value_clip_range)
            vloss=.5*torch.maximum((v-returns).square(),(vclip-returns).square()).mean();entropy=ev['entropy'].mean()
            loss=policy_loss+cfg.value_coefficient*vloss-cfg.entropy_coefficient*entropy
            kl=((ratio-1)-logratio).mean();fraction=((ratio-1).abs()>cfg.clip_range).float().mean()
            if not all(torch.isfinite(v).all() for v in [loss,kl,entropy,ratio]):raise FloatingPointError('NONFINITE_LOSS')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('NONFINITE_GRADIENT')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.max_gradient_norm,error_if_nonfinite=True)
            finite_parameters(model)
            optimizer.step()
            OPTIMIZER_STEP_COUNT += 1
            finite_parameters(model)
            if not all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values()):raise FloatingPointError('NONFINITE_ADAM_STATE')
            metrics.append(dict(policy_loss=float(policy_loss.detach()),value_loss=float(vloss.detach()),total_loss=float(loss.detach()),entropy=float(entropy.detach()),approx_kl=float(kl.detach()),clip_fraction=float(fraction.detach()),gradient_norm=float(norm),epoch=epoch+1,minibatch=mb+1))
            if float(kl.detach())>cfg.target_kl:fired={'epoch':epoch+1,'minibatch':mb+1,'kl':float(kl.detach())};break
        epochs=epoch+1
        if fired:break
    after=flat_parameters(model);change=float(torch.linalg.vector_norm(after-before))
    if not np.isfinite(change) or change<=0:raise AssertionError('NO_PARAMETER_CHANGE')
    with torch.no_grad():values=model(data['observations'])[-1].numpy()
    variance=float(np.var(arrays['returns']));ev=1-float(np.var(arrays['returns']-values))/variance if variance>1e-12 else 0.
    return dict({k:float(np.mean([r[k] for r in metrics])) for k in ['policy_loss','value_loss','total_loss','entropy','approx_kl','clip_fraction','gradient_norm']},
        parameter_norm_before=float(torch.linalg.vector_norm(before)),parameter_norm_after=float(torch.linalg.vector_norm(after)),parameter_update_l2=change,
        explained_variance=ev,explained_variance_degenerate=variance<=1e-12,learning_rate=cfg.learning_rate,actual_epochs=epochs,
        actual_optimizer_steps=len(metrics),kl_guard_fired=fired is not None,first_kl_firing=fired,minibatch_metrics=metrics,
        admitted_batch_size=len(rows),gradient_nonzero=any(r['gradient_norm']>0 for r in metrics))


def checkpoint(path,model,optimizer,sampler,env,contract,seed,update,counters,dev_history):
    if not env.done or (env.run is not None and not env.staging.closed):raise AssertionError('CHECKPOINT_NOT_CLEAN')
    meta=copy.deepcopy(env.fixed_metadata)
    meta.update(experiment_hash=EXPERIMENT_HASH,budget_protocol_hash=EXPERIMENT_HASH,scope='MAIN_ONLY',seed=seed,
                effective_hyperparameters=contract['hyperparameters'],trainer_source_hash=file_hash(__file__),training_authorized=True,
                simulator_target_template_hash=digest({'profile_hash':env.fixed_metadata['profile_hash'],'semantic_hash':SEMANTIC_HASH,'experiment_hash':EXPERIMENT_HASH,'scope':'OFFLINE_SIMULATOR_ONLY'}))
    if env.run is not None:meta['last_simulator_binding']=env.checkpoint_metadata()
    else:meta['last_simulator_binding']=None
    payload={'metadata':meta,'update':update,'policy':model.state_dict(),'optimizer':optimizer.state_dict(),
        'rng':rng_state(sampler),'sampler':sampler.state(),'counters':counters.copy(),'development_history':copy.deepcopy(dev_history),
        'staging_empty':True,'scheduler':None,'experiment_hash':EXPERIMENT_HASH,'python':sys.executable,'torch_version':torch.__version__}
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
    return meta


def reload_parity(path,meta,model,optimizer,sampler,dev,cfg,hp):
    before=rng_state(sampler);ss=sampler.state()
    try:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['metadata']!=meta or payload['experiment_hash']!=EXPERIMENT_HASH or not payload['staging_empty']:raise ValueError('CHECKPOINT_BINDING')
        fresh=HybridActorCritic(cfg).to(dtype=torch.float32,device='cpu');opt=optimizer_for(fresh,hp)
        fresh.load_state_dict(payload['policy']);opt.load_state_dict(payload['optimizer'])
        if not equal(model.state_dict(),fresh.state_dict()) or not equal(optimizer.state_dict(),opt.state_dict()):raise AssertionError('RELOAD_STATE')
        restored=RunSampler(sampler.ids,cfg.seed);restored.load(payload['sampler']);restore_rng(payload['rng'],restored)
        if not equal(rng_state(restored),payload['rng']) or not equal(restored.state(),payload['sampler']):raise AssertionError('RELOAD_RNG_SAMPLER')
        initials=pd.DataFrame([dev.cohort.runs[k].iloc[0][dev.obs] for k in sorted(dev.cohort.runs)])
        obs=dev.runtime.scaler.transform_frame(initials)[dev.obs].to_numpy(float);maximum=0.
        for o in obs:
            a,lp,v=act(model,o,True);b,lq,w=act(fresh,o,True)
            np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-8);np.testing.assert_allclose([lp,v],[lq,w],rtol=1e-6,atol=1e-8)
            maximum=max(maximum,float(np.max(np.abs(a-b))),abs(v-w),abs(lp-lq))
        return {'PASS':True,'checkpoint_sha256':file_hash(path),'fresh_object_policy_optimizer_rng_sampler_binding':True,'fixed_development_initials':len(obs),'max_inference_difference':maximum,'rtol':1e-6,'atol':1e-8,'extra_optimization':False}
    finally:
        sampler.load(ss);restore_rng(before,sampler)


def seed_run(root,output,contract,seed):
    directory=output/f'seed{seed}';directory.mkdir(exist_ok=False)
    initial_optimizer_steps=OPTIMIZER_STEP_COUNT
    cfg,hp=config(contract,seed);budget=contract['budgets']['main']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=HybridActorCritic(cfg).to(dtype=torch.float32,device='cpu');finite_parameters(model)
    optimizer=optimizer_for(model,hp)
    group=optimizer.param_groups[0]
    for key,value in [('lr',hp['learning_rate']),('eps',hp['adam_eps']),('betas',tuple(hp['adam_betas'])),('weight_decay',0.),('amsgrad',False),('foreach',False),('fused',False)]:
        if group[key]!=value:raise AssertionError('OPTIMIZER_CONFIG:'+key)
    train=ProfileAOfflineEnv(root,'train',num_workers=1);dev=ProfileAOfflineEnv(root,'development',num_workers=1)
    if sorted(train.cohort.runs)!=[r['run_id'] for r in contract['sampling']['run_schedule']['train']]:raise AssertionError('TRAIN_SCHEDULE')
    if sorted(dev.cohort.runs)!=[r['run_id'] for r in contract['sampling']['run_schedule']['development']]:raise AssertionError('DEV_SCHEDULE')
    sampler=RunSampler(train.cohort.runs,seed);counters={'attempted':0,'admitted':0,'started_runs':0,'completed_updates':0};updates=[];evaluations=[];reloads=[]
    write(directory/'effective_configuration.json',{'experiment_hash':EXPERIMENT_HASH,'seed':seed,'hp':contract['hyperparameters'],'optimizer_group':{k:v for k,v in group.items() if k!='params'},'runtime':{'device':'cpu','dtype':str(next(model.parameters()).dtype),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic':torch.are_deterministic_algorithms_enabled()},'fresh_weight_hash':hashlib.sha256(flat_parameters(model).numpy().tobytes()).hexdigest()})
    try:
        for update in range(17):
            preflight(root)
            if update:
                parts=[];rows=[];runs=[];infos=[];before_counters=counters.copy()
                while len(rows)<1024:
                    run_id=sampler.next();length=len(train.cohort.runs[run_id])
                    if counters['attempted']+length>209280 or counters['started_runs']>=209280:raise RuntimeError('MAIN_ATTEMPT_BUDGET_EXHAUSTED')
                    arr,rr,record,events=run_episode(train,model,run_id,seed,False)
                    counters['attempted']+=len(events);counters['started_runs']+=1;runs.append(record);infos.extend(events)
                    if record['category'] in ['SYSTEM_INVALID','UNSCORABLE']:
                        write(directory/'collection_failure.json',{'experiment_hash':EXPERIMENT_HASH,'update':update,'record':record,'events':events});raise RuntimeError('TRAIN_'+record['category'])
                    if arr is not None:parts.append(arr);rows.extend(rr)
                    print(f'seed={seed} update={update} ready={len(rows)} attempted={counters["attempted"]}',flush=True)
                if not 1024<=len(rows)<=1308 or counters['admitted']+len(rows)>20928:raise AssertionError('MAIN_ADMITTED_BUDGET')
                arrays={k:np.concatenate([p[k] for p in parts],axis=0) for k in parts[0]}
                counters['admitted']+=len(rows)
                u=update_model(model,optimizer,rows,arrays,cfg,update-1);counters['completed_updates']+=1
                u.update(experiment_hash=EXPERIMENT_HASH,seed=seed,update=update,cumulative=counters.copy(),attempted_this_update=counters['attempted']-before_counters['attempted'],started_runs_this_update=counters['started_runs']-before_counters['started_runs'],accounting=summarize(runs,infos))
                updates.append(u);write(directory/f'update{update}_metrics.json',u)
                write(directory/f'update{update}_admission.json',{'experiment_hash':EXPERIMENT_HASH,'runs':runs,'rows':len(rows),'all_finalized':True,'proposal_final_diff_count':sum(r['info']['proposed']!=r['info']['final_action'] for r in rows)})
                del rows,parts,arrays,infos
            if update % 4 != 0:
                continue
            ckpt=directory/f'update{update}.pt'
            meta=checkpoint(ckpt,model,optimizer,sampler,train,contract,seed,update,counters,evaluations)
            reload=reload_parity(ckpt,meta,model,optimizer,sampler,dev,cfg,hp);reload.update(seed=seed,update=update,experiment_hash=EXPERIMENT_HASH);reloads.append(reload)
            write(directory/f'reload_update{update}.json',reload)
            evaluation=evaluate(dev,model,sampler,seed,update,directory);evaluations.append(evaluation)
            print(f'seed={seed} update={update} evaluation complete {evaluation["metrics"]["categories"]}',flush=True)
        if counters['completed_updates']!=16 or len(evaluations)!=5 or len(reloads)!=5:raise AssertionError('MAIN_COMPLETION')
        if not all(u['parameter_update_l2']>0 and u['gradient_nonzero'] for u in updates):raise AssertionError('NO_LEARNING_SIGNAL')
        def rank(e):
            r=e['metrics']['rates'];return development_rank(r['SYSTEM_INVALID'],r['UNSCORABLE'],r['FAILURE'],e['metrics']['complete_discounted_returns']['mean'],e['update'])
        selected=min(evaluations,key=rank)['update']
        # Post-selection full schedule verification, never another ranking candidate.
        parity = verify_selected_schedule(directory, selected, model, optimizer, sampler, dev, cfg, hp, evaluations)
        write(directory/'selected_full_schedule_reload_parity.json', parity)
        after_checks=[]
        for check in reloads:
            cp=directory/f"update{check['update']}.pt"
            if file_hash(cp)!=check['checkpoint_sha256']:raise AssertionError('POST_TRAIN_CHECKPOINT_HASH')
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            if payload['metadata']['experiment_hash']!=EXPERIMENT_HASH or payload['metadata']['scope']!='MAIN_ONLY':raise AssertionError('POST_TRAIN_CHECKPOINT_BINDING')
            if not all(torch.isfinite(v).all() for v in payload['policy'].values()):raise AssertionError('POST_TRAIN_CHECKPOINT_PARAMETERS')
            if not all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all() for z in payload['optimizer']['state'].values() for v in z.values()):raise AssertionError('POST_TRAIN_CHECKPOINT_OPTIMIZER')
            after_checks.append({'update':check['update'],'PASS':True,'sha256':file_hash(cp)})
        write(directory/'post_training_all_checkpoint_integrity.json',{'experiment_hash':EXPERIMENT_HASH,'checks':after_checks})
        result={'experiment_hash':EXPERIMENT_HASH,'seed':seed,'status':'PASS','counters':counters,'selected_main_checkpoint':f'seed{seed}/update{selected}.pt','updates':updates,'development':evaluations,'reloads':reloads,'selected_full_schedule_parity':parity,'post_training_checkpoint_checks':after_checks}
    except Exception as exc:
        result={'experiment_hash':EXPERIMENT_HASH,'seed':seed,'status':'BLOCKED','counters':counters,'updates':updates,'development':evaluations,'reloads':reloads,'failure':str(exc),'traceback':traceback.format_exc()}
        write(directory/'technical_failure.json',result)
    result['optimizer_steps_executed']=OPTIMIZER_STEP_COUNT-initial_optimizer_steps
    write(directory/'seed_result.json',result)
    return result


def verify_selected_schedule(directory,selected,model,optimizer,sampler,dev,cfg,hp,evaluations):
    before=rng_state(sampler);sampler_before=sampler.state()
    path=directory/f'update{selected}.pt';payload=torch.load(path,map_location='cpu',weights_only=False)
    fresh=HybridActorCritic(cfg).to(dtype=torch.float32,device='cpu');fresh.load_state_dict(payload['policy'])
    expected=next(e for e in evaluations if e['update']==selected)
    runs=[];infos=[]
    try:
        for j,run_id in enumerate(sorted(dev.cohort.runs)):
            _,_,r,events=run_episode(dev,fresh,run_id,cfg.seed,True);runs.append(r);infos.extend(events)
            if r['category'] in ['SYSTEM_INVALID','UNSCORABLE']:raise RuntimeError('RELOAD_PARITY_'+r['category'])
            if (j+1)%5==0:print(f'seed={cfg.seed} selected=u{selected} verification runs={j+1}/25',flush=True)
        metrics=summarize(runs,infos)
        if clean(metrics)!=clean(expected['metrics']) or clean(runs)!=clean(expected['runs']):raise AssertionError('SELECTED_FULL_DEVELOPMENT_PARITY')
        return {'experiment_hash':EXPERIMENT_HASH,'seed':cfg.seed,'selected_update':selected,'PASS':True,'all_25_runs_exact_accounting_return_binding_parity':True,'rtol':1e-6,'atol':1e-8,'observed_difference':0.,'selection_evaluation':False,'new_checkpoint':False,'optimizer_executed':False}
    finally:
        sampler.load(sampler_before);restore_rng(before,sampler)


def finish(root,out,results,read_paths):
    passed=[r['seed'] for r in results if r['status']=='PASS'];failed=[r['seed'] for r in results if r['status']!='PASS']
    status='CUCUMBER PROFILE-A PPO MAIN — COMPLETE / POLICY REVIEW READY' if len(passed)==3 else ('CUCUMBER PROFILE-A PPO MAIN — PARTIAL / '+','.join(map(str,failed))+' INCOMPLETE' if passed else 'CUCUMBER PROFILE-A PPO MAIN — BLOCKED')
    write(out/'update_metrics.json',{'experiment_hash':EXPERIMENT_HASH,'seeds':[{k:r[k] for k in ['seed','status','counters','updates']} for r in results]})
    write(out/'development_evaluations.json',{'experiment_hash':EXPERIMENT_HASH,'seeds':[{'seed':r['seed'],'evaluations':r['development']} for r in results]})
    write(out/'checkpoint_reload_validation.json',{'experiment_hash':EXPERIMENT_HASH,'seeds':[{'seed':r['seed'],'checks':r['reloads'],'full_selected_parity':r.get('selected_full_schedule_parity'),'post_training_checks':r.get('post_training_checkpoint_checks')} for r in results]})
    ranking=[];summary={}
    for r in results:
        def rank(e):
            rates=e['metrics']['rates'];mean=e['metrics']['complete_discounted_returns']['mean']
            return development_rank(rates['SYSTEM_INVALID'],rates['UNSCORABLE'],rates['FAILURE'],mean,e['update'])
        ordered=sorted(r['development'],key=rank)
        baseline=next((e for e in r['development'] if e['update']==0),None)
        selected=ordered[0] if ordered and r['status']=='PASS' else None
        ranking.append({'seed':r['seed'],'status':r['status'],'ordered_update_ids':[e['update'] for e in ordered],'selected':r.get('selected_main_checkpoint'),'baseline':baseline,'selected_evaluation':selected})
    for metric in ['COMPLETE','FAILURE','SYSTEM_INVALID','UNSCORABLE','return']:
        vals=[]
        for r in ranking:
            e=r['selected_evaluation']
            if e:
                v=e['metrics']['complete_discounted_returns']['mean'] if metric=='return' else e['metrics']['rates'][metric]
                if v is not None:vals.append(v)
        summary[metric]={'median':float(np.median(vals)) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,'valid_seed_count':len(vals)}
    write(out/'checkpoint_ranking.json',{'experiment_hash':EXPERIMENT_HASH,'ranking':ranking})
    write(out/'cross_seed_summary.json',{'experiment_hash':EXPERIMENT_HASH,'selected_metrics':summary,'all_seeds_reported':[r['seed'] for r in results]})
    preflight(root)
    lines=['# Cucumber Profile-A PPO Main 최종 결과','',status,'','MAIN PPO OPTIMIZATION EXECUTED = '+('YES' if OPTIMIZER_STEP_COUNT else 'NO'),'HPO EXECUTED = NO','TEST ACCESSED = NO','','| seed | updates | admitted | attempted | 상태 | 선택 checkpoint |','|---|---:|---:|---:|---|---|']
    for r in results:
        lines.append(f"| {r['seed']} | {r['counters']['completed_updates']} | {r['counters']['admitted']} | {r['counters'].get('attempted',0)} | {r['status']} | {r.get('selected_main_checkpoint','없음')} |")
        if r['status']!='PASS':lines+=['',f"seed {r['seed']} 기술 실패: {r.get('failure')}"]
    lines+=['','## 같은 seed update0 대비 선택 checkpoint','','| seed | FAILURE rate baseline → selected | COMPLETE rate baseline → selected | COMPLETE mean discounted return baseline → selected |','|---|---|---|---|']
    for r in ranking:
        b=r['baseline'];e=r['selected_evaluation']
        if b and e:
            br=b['metrics'];er=e['metrics']
            lines.append(f"| {r['seed']} | {br['rates']['FAILURE']:.2%} → {er['rates']['FAILURE']:.2%} | {br['rates']['COMPLETE']:.2%} → {er['rates']['COMPLETE']:.2%} | {br['complete_discounted_returns']['mean']} → {er['complete_discounted_returns']['mean']} |")
    lines+=['','## Cross-seed selected 결과','','| 지표 | median | min | max | valid seeds |','|---|---:|---:|---:|---:|']
    for k,v in summary.items():lines.append(f"| {k} | {v['median']} | {v['min']} | {v['max']} | {v['valid_seed_count']} |")
    lines+=['','각 개발평가의 분모는 모든25 natural runs이며 zero-step/FAILURE를 제외하지 않았다. COMPLETE return은 조건부 통계이므로 completion/failure와 함께 해석한다. 기술적 COMPLETE는 정책 우월성 또는 실온실 안전성 증거가 아니다.','','smoke weights를 읽거나 warm-start하지 않았고 각 seed를 독립 fresh initialization했다. 단일 CPU worker, 동일 frozen budget/hyperparameters, 전체 run 확정 후 admission, proposal log-prob, dual-mask GAE를 유지했다. original frozen source/artifact는 변경하지 않았다. 새 main driver만 추가했다.','','선정 평가/checkpoint는0/4/8/12/16만 허용했다. 선정 후 추가 전체 Development 재생은 사전 요구된 reload parity 검증이며 새로운 ranking 후보나 parameter 선택에 쓰지 않았다. 각 seed 종료 후 모든 예정 checkpoint의 parameter/optimizer metadata/hash를 재확인했다. 세부 update/회계/intervention/KL firing, 평가, 순위, parity는 companion JSON에 보존했다.','','높은 FAILURE도 결과 그대로 보고한다. post-hoc threshold/보상/예산 변경 또는 HPO는 하지 않았다. Test/frozen Validation/P0/live 데이터는 정책 학습/평가에 사용하지 않았다. bound bundle의 기존 파일 hash 검증은 정책 데이터 사용과 구분한다.','',f'실험 payload SHA-256: `{EXPERIMENT_HASH}`','','다음 단계는 전체 Main 결과에 대한 policy/failure/support 검토다. 추가 예산·HPO·최종 평가를 자동 실행하지 않았다.']
    (out/'final_ppo_main.md').write_text('\n'.join(lines)+'\n')
    files={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p.name!='main_manifest.json'}
    write(out/'main_manifest.json',{'experiment_hash':EXPERIMENT_HASH,'status':status,'seed_status':{r['seed']:r['status'] for r in results},'trainer_sha256':file_hash(__file__),'generated_artifact_hashes':files,'main_executed':OPTIMIZER_STEP_COUNT>0,'optimizer_steps_total':OPTIMIZER_STEP_COUNT,'hpo_executed':False,'test_accessed':False,'validation_policy_evaluated':False,'data_parquet_reads':sorted(set(read_paths)),'frozen_hash_gate_after':True,'scope':'MAIN ONLY'})
    print(status,flush=True)


def main():
    root=Path(__file__).resolve().parents[3]
    out=root/'experiments/rl_policy_training/artifacts/cucumber_profile_a_ppo_main_v1'
    out.mkdir(exist_ok=False)
    c=preflight(root)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    read_paths=[];original=pd.read_parquet
    def guard(path,*args,**kwargs):
        p=Path(path);read_paths.append(str(p))
        if p.name.lower().startswith(('test','validation')):raise RuntimeError('FORBIDDEN_DATA_ACCESS')
        return original(path,*args,**kwargs)
    results=[]
    with patch('pandas.read_parquet',side_effect=guard),patch('geas35.rl.mdp_v1.compute_mdp_v1_reward',side_effect=AssertionError('LEGACY_REWARD')),patch('geas35.rl.hybrid_ppo.compute_gae',side_effect=AssertionError('OLD_GAE_FORBIDDEN')):
        for seed in [42,43,44]:
            print(f'seed={seed} START',flush=True)
            try:
                result=seed_run(root,out,c,seed)
            except Exception as exc:
                result={'seed':seed,'status':'BLOCKED','counters':{'completed_updates':0,'admitted':0},'updates':[],'development':[],'reloads':[],'failure':str(exc),'traceback':traceback.format_exc(),'experiment_hash':EXPERIMENT_HASH}
                write(out/f'seed{seed}'/'startup_failure.json',result)
            results.append(result)
            print(f'seed={seed} {results[-1]["status"]}',flush=True)
    finish(root,out,results,read_paths)


if __name__=='__main__':main()
