"""Evidence tables and a Korean diagnostic report; no official decisions are mutated."""
from pathlib import Path
from collections import Counter, defaultdict
import gzip
import json

import numpy as np
import pandas as pd

from .step19_5_diagnostics import (ART, OUT, CROPS, RTOL, ATOL, digest, clean, read_json,
    write_json, sha256, checkpoint, make_job, job_id, stage_jobs, verify_inputs)


def training_history_review(root):
    summary=[];window_rows=[];resume=[]
    for crop in CROPS:
        for n in range(5):
            for seed in (42,43,44):
                config=f'representative_{n}'
                source=checkpoint(root,crop,config,seed,524288).with_suffix('.training_history.json')
                prior=read_json(source)['history'];hist=prior
                if crop=='strawberry':
                    extended=checkpoint(root,crop,config,seed,1048576).with_suffix('.training_history.json')
                    hist=read_json(extended)['history']
                    prefix=[x for x in hist if x['training_steps']<=524288]
                    resume.append({'crop':crop,'config_id':config,'seed':seed,'source_prefix_identical':prefix==prior,
                        'strictly_increasing':all(b['training_steps']>a['training_steps'] for a,b in zip(hist,hist[1:])),
                        'last_step':hist[-1]['training_steps'],
                        'continuous_execution_equivalence_proven':False})
                frame=pd.DataFrame(hist)
                for lo,hi in ((0,262144),(262144,524288),(524288,1048576)):
                    sub=frame[(frame.training_steps>lo)&(frame.training_steps<=hi)]
                    if sub.empty:continue
                    for metric in ('policy_loss','value_loss','approx_kl','entropy','clip_fraction','gradient_norm',
                                   'explained_variance','fallback_rate','severe_state_termination_rate','parameter_update_l2'):
                        if metric in sub:
                            values=sub[metric].to_numpy(dtype=float)
                            window_rows.append({'crop':crop,'config_id':config,'seed':seed,'from_step':lo,'to_step':hi,
                                'metric':metric,'median':float(np.median(values)),'q90':float(np.quantile(values,.9)),
                                'minimum':float(np.min(values)),'maximum':float(np.max(values)),
                                'finite':bool(np.isfinite(values).all())})
    # Preserve and check the actual snapshot taken at the sequential-to-parallel boundary.
    backup=root/'experiments/rl_policy_training/backups/parallel_resume_20260923_082748/strawberry/representative_1/seed_42.training_history.json'
    boundary={'snapshot_available':backup.exists(),'continuous_execution_equivalence_proven':False}
    if backup.exists():
        old=read_json(backup)['history'];new=read_json(checkpoint(root,'strawberry','representative_1',42,1048576).with_suffix('.training_history.json'))['history']
        last=old[-1]['training_steps'];boundary.update({'saved_step':last,'snapshot_sha256':sha256(backup),
            'saved_prefix_preserved':[x for x in new if x['training_steps']<=last]==old,
            'next_saved_step':next((x['training_steps'] for x in new if x['training_steps']>last),None)})
    return window_rows,{'histories':resume,'parallel_resume_boundary':boundary,
        'limitation':'No uninterrupted control run; optimizer/RNG presence and prefix continuity do not prove identical continuous training.'}


def trace_findings(root,result):
    directory=root/OUT/'jobs'/job_id(result['job']);top=Counter();horizon=defaultdict(list);modified=0;predicted=0
    first_events=[]
    for rec in result['episode_files']:
        p=(directory/rec['path']).with_suffix('.jsonl.gz')
        with gzip.open(p,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                contribution=row.get('state_distance')
                if contribution:
                    components=contribution['squared_distance_components'];total=sum(components.values())
                    for name,value in components.items():top[name]+=value/max(total,1e-30)
                    if row.get('info',{}).get('state_ood_level')=='severe':
                        first_events.append({'episode_index':row['episode_index'],'step':row['step'],
                            'top_features':sorted(components,key=components.get,reverse=True)[:5],
                            'state_score':contribution['state_score']})
                if 'recorded_path_error_physical' in row:
                    predicted+=1;modified+=int(row['recorded_action_modified'])
                    for target,value in row['recorded_path_error_physical'].items():
                        band=next((x for x in (1,6,12,24,48,96,192,288) if row['step']+1<=x),999)
                        horizon[(target,band)].append(value)
    return {'top_distance_features':top.most_common(10),'first_ood_events':first_events,
        'recorded_actions_modified':modified,'recorded_prediction_steps':predicted,
        'rollout_errors':[{'target':t,'horizon_bin_upper':h,'count':len(v),'mae':float(np.abs(v).mean()),'rmse':float(np.sqrt(np.square(v).mean()))} for (t,h),v in sorted(horizon.items())],
        'error_interpretation':'Recorded-action rollout errors include existing safety projections; these are not errors for counterfactual PPO actions.'}


def generate_report(root,manifest,device='cpu'):
    root=Path(root);out=root/OUT;report=out/'report';report.mkdir(parents=True,exist_ok=True)
    checked=verify_inputs(root,manifest)
    expected={job_id(j):j for stage in ('evaluator-parity','rollout-diagnosis','learning-analysis') for j in stage_jobs(stage,CROPS,device)}
    results={};missing=[]
    for ident,job in expected.items():
        p=out/'jobs'/ident/'result.json'
        if not p.exists():missing.append(ident);continue
        result=read_json(p)
        if result['fingerprint']!=digest({'manifest':digest(manifest),'job':job}):raise ValueError('Stale report input')
        for item in result['episode_files']:
            ep_path=p.parent/item['path'];ep=read_json(ep_path)
            if sha256(ep_path)!=item['sha256'] or sha256(ep_path.with_suffix('.jsonl.gz'))!=ep['trace_sha256']:
                raise ValueError(f'Corrupt trace input: {ep_path}')
        results[ident]=result
    job_rows=[];episode_rows=[]
    for result in results.values():
        job=result['job'];metrics=result['metrics']
        job_rows.append({**job,**{k:v for k,v in metrics.items() if not isinstance(v,dict)},
            **{f'reward_{k}':v for k,v in metrics['reward_components'].items()}})
        for e in result['episodes']:
            episode_rows.append({**job,'episode_index':e['episode_index'],'episode_id':e['episode_id'],
                **{k:e.get(k) for k in ('scheduled_valid_steps','executed_steps','reward_sum','severe_ood_rate','termination_reason','first_severe_step')},
                **{f'reward_{k}':v for k,v in e['reward_components'].items()},
                **{f'term_{k}':v for k,v in e.get('reward_terms',{}).items() if k.endswith('_penalty') and not k.startswith(('raw_','normalized_'))}})
    pd.DataFrame(job_rows).to_csv(report/'job_summary.csv',index=False)
    pd.DataFrame(episode_rows).to_csv(report/'episode_summary.csv',index=False)
    windows,resume=training_history_review(root)
    pd.DataFrame(windows).to_csv(report/'training_diagnostics.csv',index=False)
    write_json(report/'resume_review.json',resume)
    parity={crop:read_json(out/'parity'/f'{crop}.json') for crop in CROPS if (out/'parity'/f'{crop}.json').exists()}
    crop_evidence={};one_step_rows=[];distance_rows=[];horizon_rows=[]
    for crop in ('melon','cucumber'):
        evidence={}
        for split in ('train','validation'):
            for mode in ('recorded-one-step','recorded-rollout'):
                job=make_job(crop,split=split,mode=mode,device=device);result=results.get(job_id(job))
                if result is None:continue
                key=f'{split}/{mode}';evidence[key]=dict(result['metrics'])
                evidence[key]['first_state_ood_steps']=[e['first_severe_step'] for e in result['episodes']]
                if mode=='recorded-one-step':
                    totals=defaultdict(Counter)
                    for e in result['episodes']:
                        for t,v in e['prediction_errors'].items():totals[t].update(v)
                    evidence[key]['state_severe_rate']=sum(e['state_severe_steps'] for e in result['episodes'])/sum(e['executed_steps'] for e in result['episodes'])
                    for t,v in totals.items():one_step_rows.append({'crop':crop,'split':split,'target':t,'count':v['count'],
                        'mae':v['sum_abs']/v['count'],'rmse':np.sqrt(v['sum_square']/v['count']),'bias':v['sum']/v['count']})
                findings=trace_findings(root,result);evidence[key]['trace_findings']=findings
                for feature,value in findings['top_distance_features']:distance_rows.append({'crop':crop,'split':split,'mode':mode,'feature':feature,'summed_distance_fraction':value})
                horizon_rows.extend({'crop':crop,'split':split,**x} for x in findings['rollout_errors'])
        policies=[r for r in results.values() if r['job']['crop']==crop and r['job']['mode']=='audit' and r['job']['repeat']==0]
        evidence['policies']=[{'job':r['job'],'metrics':r['metrics']} for r in policies]
        # Detailed representative policy traces link common termination locations to state dimensions.
        rep=results.get(job_id(make_job(crop,device=device)))
        if rep:evidence['representative_policy_trace']=trace_findings(root,rep)
        crop_evidence[crop]=evidence
    for name,rows in [('one_step_errors',one_step_rows),('distance_contributions',distance_rows),('recorded_rollout_errors',horizon_rows)]:
        pd.DataFrame(rows).to_csv(report/f'{name}.csv',index=False)
    strawberry=[r for r in results.values() if r['job']['crop']=='strawberry' and r['job']['mode']=='audit' and r['job']['repeat']==0]
    pairs=[];seed_rows=[];changes=[]
    for n in range(5):
        config=f'representative_{n}'
        for seed in (42,43,44):
            a=results.get(job_id(make_job('strawberry',config_id=config,seed=seed,steps=524288,device=device)))
            b=results.get(job_id(make_job('strawberry',config_id=config,seed=seed,steps=1048576,device=device)))
            if not a or not b:continue
            delta=b['metrics']['mean_reward_per_scheduled_valid_step']-a['metrics']['mean_reward_per_scheduled_valid_step']
            pairs.append({'config_id':config,'seed':seed,'reward_delta':delta,
                **{f'delta_{k}':b['metrics']['reward_components'][k]-a['metrics']['reward_components'][k] for k in ('base','warning','terminal','clamp_or_override')}})
            for ea,eb in zip(a['episodes'],b['episodes']):
                if ea['episode_index']!=eb['episode_index']:raise ValueError('Pair alignment mismatch')
                changes.append({'config_id':config,'seed':seed,'episode_index':ea['episode_index'],'episode_id':ea['episode_id'],
                    'reward_sum_delta':eb['reward_sum']-ea['reward_sum'],
                    'terminal_delta':eb['reward_components']['terminal']-ea['reward_components']['terminal'],
                    'executed_steps_delta':eb['executed_steps']-ea['executed_steps']})
        for step in (524288,1048576):
            values=[r['metrics']['mean_reward_per_scheduled_valid_step'] for r in strawberry if r['job']['config_id']==config and r['job']['steps']==step]
            if values:seed_rows.append({'config_id':config,'steps':step,'n_seeds':len(values),'median':float(np.median(values)),
                'minimum':min(values),'maximum':max(values),'std':float(np.std(values,ddof=1)) if len(values)>1 else None})
    pd.DataFrame(pairs).to_csv(report/'strawberry_reward_changes.csv',index=False)
    pd.DataFrame(changes).to_csv(report/'strawberry_episode_changes.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(report/'strawberry_seed_dispersion.csv',index=False)
    rank={str(s):[x['config_id'] for x in sorted((x for x in seed_rows if x['steps']==s),key=lambda x:-x['median'])] for s in (524288,1048576)}
    strawberry_evidence={'evaluated_jobs':len(strawberry),'ranking_by_median':rank,
        'reward_deltas':pairs,'seed_dispersion':seed_rows,
        'absolute_delta_median':float(np.median([abs(x['reward_delta']) for x in pairs])) if pairs else None,
        'largest_episode_changes':sorted(changes,key=lambda x:-abs(x['reward_sum_delta']))[:20],
        'source_history_prefixes_preserved':all(x['source_prefix_identical'] for x in resume['histories']),
        'limitation':'Three training seeds; no causal claim from PPO diagnostic correlation alone.'}
    evidence={'status':'complete' if not missing else 'incomplete','missing_jobs':missing,
        'completed_jobs':len(results),'expected_jobs':len(expected),'verified_input_files':checked,
        'source_code_drift':manifest.get('source_code_drift',[]),'historical_environment_equivalence_proven':manifest.get('historical_environment_equivalence_proven',False),'parity':parity,'crop_evidence':crop_evidence,'strawberry':strawberry_evidence,
        'test_accessed':False,'official_step19_5_decision_changed':False,'new_training_started':False}
    write_json(report/'evidence.json',clean(evidence))
    plot_report(report,job_rows,seed_rows,horizon_rows)
    lines=['# GEAS3.5 Step 19.5 원인 진단 보고서','',
        '## 전체 흐름과 현재 위치','',
        'Transition Model 확정(Step 14) → 환경·안전성·PPO 기본 검증(Step 15~19) → 학습량 calibration(Step 19.5) → HPO(Step 20) → 최종 학습·평가·Test·인계.',
        '현재는 Step 19.5 기준 미충족의 원인을 진단하는 단계다. 기존 최종 판정은 유지하며 수정·추가 학습·HPO는 실행하지 않았다.',
        '',f"진단 상태: **{evidence['status']}**, 작업 {len(results)}/{len(expected)}, 입력 파일 {checked}개 해시 검증.",'',
        '## 평가기와 재개 검증','']
    for drift in manifest.get('source_code_drift',[]):lines.append(f"- **기존 동결 코드와 불일치:** {drift['path']}. 현재 코드로 재평가하며 과거 학습 환경과의 동일성은 증명되지 않았다.")
    for crop,p in parity.items():lines.append(f"- {crop}: 반복 재현={p['reproducible']}, 기존/v2.2 평가 일치={p['evaluators_equivalent']}.")
    lines += [f"- 딸기 source 이력 prefix 보존: {strawberry_evidence['source_history_prefixes_preserved']}.",
        '- optimizer·RNG 상태 존재와 이력 연속성은 검사했지만, 중단 없는 대조 실행과의 완전 동일성은 입증하지 않았다.',
        '- 262,144 step 가중치는 보존되지 않아 해당 시점은 기존 평가 결과만 활용한다.','', '## 작물별 근거와 개선 방향','']
    for crop,ev in crop_evidence.items():
        lines += [f'### {crop}','']
        for key,value in ev.items():
            if '/' in key:
                lines.append(f"- {key}: OOD {value['severe_ood_rate']:.4%}, 실행 {value['executed_steps']}/{value['scheduled_valid_steps']}, 종료 {value['termination_reasons']}.")
        recorded=ev.get('validation/recorded-one-step');rollout=ev.get('validation/recorded-rollout')
        if recorded and rollout:
            lines.append(f"- 실제 기록 state-only severe 비율: {recorded['state_severe_rate']:.4%}.")
            if rollout['termination_reasons'].get('severe_state_ood',0)>0:
                lines.append('- **확인됨:** 기록 행동을 사용한 누적 모델 경로에서도 상태 OOD 종료가 발생한다. PPO만의 문제로 설명할 수 없다.')
                lines.append('- **유력하지만 추가 검증 필요:** 공통 전이모델 누적 오차·이력 갱신·지원 범위의 영향을 먼저 분리해야 한다. 1-step 오차와 horizon별 오차를 근거로 대상 변수를 선정한다.')
            if recorded['state_severe_rate']>0:
                lines.append('- **확인됨:** 실제 기록 상태에도 일부 OOD가 존재한다. 발생 에피소드·변수와 정책 종료 위치가 겹치는지 확인 후 분포 이동의 기여도를 판단한다.')
            lines += ['- 거리 기여도는 가중 k번째 이웃 거리의 구성 성분이며 인과적 중요도가 아니다.',
                '- **개선안:** 입력 schema·scaler·이력 갱신 검사 → 기록 행동 1-step/누적 비교 → 필요 시 전이모델·support 재검토 순서. 수정 전후 동일 에피소드의 최초 OOD·실행 길이·물리 단위 오차를 재검증한다.',
                '- **제한:** 기록 행동도 기존 safety projection을 통과한다. 투영 횟수는 evidence.json에 기록했으며 PPO 경로를 실제 기록과 비교한 차이를 반사실적 예측 오차로 부르지 않는다.','']
    lines += ['### strawberry','',f"- 동일 평가기로 재평가한 체크포인트: {len(strawberry)}/30.",f'- 중앙값 순위: {rank}.']
    if pairs:lines.append(f"- 설정·seed별 절대 보상 변화 중앙값: {strawberry_evidence['absolute_delta_median']:.8f}.")
    lines += ['- 에피소드별 보상/종료 비용 변화, seed 편차, 업데이트 진단 지표는 각각 CSV로 분리했다.',
        '- **확인됨/근거 범위:** 평가 일관성이 통과하고 보상 변화가 유지되면 평가기 차이만으로 변동을 설명할 수 없다. 순위·seed 편차 자체는 관측 사실이며 PPO 업데이트가 원인이라는 인과적 결론은 아니다.',
        '- **개선안:** 가장 큰 보상 변화 에피소드에서 종료 비용과 행동 비용 기여를 먼저 확인하고, KL·entropy·gradient 변화와의 정렬 결과에 따라 다음 학습 프로토콜의 가설을 사전 정의한다.',
        '- **재검증 조건:** 독립된 새 프로토콜에서 동일 평가 조건·seed grid를 사용하고 순위 및 보상 안정성 기준을 다시 적용한다.','',
        '## 후속 결정','',
        'HPO 보류 유지. 진단은 통과 기준을 바꾸지 않는다. 평가 오류이면 보존 정책 재평가, 환경 오류이면 영향받은 Step 15~19 재검증, 모델/support 문제이면 의존성 변경 범위 재검증, PPO 문제이면 새 calibration 프로토콜을 제안한다.',
        '원인이 분리되지 않은 항목은 근거 부족으로 남기며 추가 장기 학습을 자동 실행하지 않는다.','',
        '근거 파일: evidence.json, job_summary.csv, episode_summary.csv, one_step_errors.csv, recorded_rollout_errors.csv, distance_contributions.csv, strawberry_reward_changes.csv, strawberry_episode_changes.csv, strawberry_seed_dispersion.csv, training_diagnostics.csv, resume_review.json.']
    if missing:lines += ['', '**미완료:** 남은 작업이 있으므로 이 보고서는 중간 보고서다. 최종 판단 자료로 사용하지 않는다.']
    (report/'diagnosis_report_ko.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report/'diagnosis_report_ko.md'


def plot_report(report,job_rows,seed_rows,horizon_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if seed_rows:
        frame=pd.DataFrame(seed_rows);fig,ax=plt.subplots(figsize=(8,4))
        for name,sub in frame.groupby('config_id'):
            sub=sub.sort_values('steps');ax.errorbar(sub.steps,sub['median'],yerr=[sub['median']-sub.minimum,sub.maximum-sub['median']],marker='o',label=name)
        ax.set(xlabel='Training steps',ylabel='Validation reward (median, seed range)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(report/'strawberry_seed_dispersion.png',dpi=160);plt.close(fig)
    policy=[x for x in job_rows if x['mode']=='audit' and x['repeat']==0 and x['split']=='validation']
    if policy:
        frame=pd.DataFrame(policy);fig,ax=plt.subplots(figsize=(8,4))
        for crop,sub in frame.groupby('crop'):ax.scatter([crop]*len(sub),sub.severe_ood_rate*100,alpha=.5,label=crop)
        ax.set(ylabel='Severe OOD rate (%)');fig.tight_layout();fig.savefig(report/'policy_ood_rates.png',dpi=160);plt.close(fig)
    if horizon_rows:
        frame=pd.DataFrame(horizon_rows);targets=list(frame.target.unique());fig,axes=plt.subplots(1,len(targets),figsize=(5*len(targets),4),squeeze=False)
        for ax,target in zip(axes[0],targets):
            for (crop,split),sub in frame[frame.target==target].groupby(['crop','split']):ax.plot(sub.horizon_bin_upper,sub.rmse,marker='o',label=f'{crop}/{split}')
            ax.set(title=target,xlabel='Horizon bin upper (steps)',ylabel='Recorded rollout RMSE');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(report/'recorded_rollout_errors.png',dpi=160);plt.close(fig)
