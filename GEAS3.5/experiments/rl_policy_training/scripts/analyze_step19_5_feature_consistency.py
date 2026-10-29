#!/usr/bin/env python3
"""Post-hoc support-score sensitivity; does not change or replay an environment."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,gzip
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from geas35.rl.support_v1 import StateActionSupportModel
from geas35.experiments.rl.step19_5_diagnostics import OUT,ART,read_json,write_json,sha256,verify_inputs


def score_replacement(support,state,recorded,action,columns):
    """Replace selected coordinates on a COPY for score sensitivity only."""
    altered=state.copy(deep=True)
    for column in columns:altered[column]=recorded[column]
    return float(support.scores(altered,action)['state'][0])


def main():
    root=ROOT;out=root/OUT;manifest=read_json(out/'input_manifest.json');verify_inputs(root,manifest)
    dependency_path=out/'feature_dependency_manifest.json'
    dependencies=[{'path':str(p.relative_to(root)),'sha256':sha256(p)} for p in sorted((root/'src/geas35/features').glob('*.py'))]
    if dependency_path.exists():
        if read_json(dependency_path)['files']!=dependencies:raise ValueError('Feature implementation changed during diagnosis')
    else:write_json(dependency_path,{'files':dependencies,'test_accessed':False})
    protocol=read_json(root/ART/'step15_v1_4/rl_protocol_manifest.json')
    splits=read_json(root/protocol['split_access_policy']['path']);events=[];statistics=[];inputs=[]
    for crop in ('melon','cucumber'):
        support=StateActionSupportModel(root/protocol['crops'][crop]['state_action_support']['path'])
        columns=manifest['trace_state_columns'][crop]
        scaler=read_json(root/protocol['crops'][crop]['observation_scaler']['path'])
        frames={}
        for split in ('train','validation'):
            frames[split]=pd.read_parquet(splits['datasets'][crop][split]['path'])
            x=frames[split]['obs_derived_light_sum'].to_numpy(dtype=float)*scaler['scale']['obs_derived_light_sum']+scaler['mean']['obs_derived_light_sum']
            statistics.append({'crop':crop,'split':split,'count':len(x),'zero_fraction':float(np.isclose(x,0,atol=1e-10).mean()),
                'minimum':float(x.min()),'maximum':float(x.max()),'mean':float(x.mean()),'std':float(x.std()),
                'frozen_train_mean':scaler['mean']['obs_derived_light_sum'],'frozen_train_scale':scaler['scale']['obs_derived_light_sum']})
        for directory in sorted((out/'jobs').glob(f'{crop}__*')):
            result_path=directory/'result.json'
            if not result_path.exists():continue
            result=read_json(result_path);job=result['job']
            if job['repeat']!=0 or job['mode'] not in ('audit','recorded-rollout'):continue
            frame=frames[job['split']]
            inputs.append({'path':str(result_path.relative_to(root)),'sha256':sha256(result_path)})
            for record in result['episode_files']:
                ep=read_json(directory/record['path'])
                if ep['termination_reason']!='severe_state_ood':continue
                trace=(directory/record['path']).with_suffix('.jsonl.gz')
                if sha256(trace)!=ep['trace_sha256']:raise ValueError('Changed trace')
                with gzip.open(trace,'rt') as f:
                    for line in f:
                        row=json.loads(line)
                        if row.get('info',{}).get('state_ood_level')!='severe':continue
                        state=pd.DataFrame([row['scaled_state']],columns=columns)
                        logged=frame.iloc[row['source_position']]
                        action=pd.DataFrame([row['info']['proposed_action']],columns=support.config['action_columns'])
                        before=float(support.scores(state,action)['state'][0]);threshold=support.config['thresholds']['state_severe']
                        if not np.isclose(before,row['info']['state_ood_score'],rtol=1e-6,atol=1e-8):raise ValueError('Original score not reproduced')
                        light=score_replacement(support,state,logged,action,['obs_derived_light_sum'])
                        external=[x for x in ('obs_derived_light_sum','obs_derived_dli_mol_m2','obs_derived_clear_sky_sum','obs_derived_light_sum_ratio','obs_derived_light_eta_minutes') if x in columns]
                        group=score_replacement(support,state,logged,action,external)
                        i=columns.index('obs_derived_light_sum')
                        events.append({'crop':crop,'split':job['split'],'mode':job['mode'],'config_id':job['config_id'],'seed':job['seed'],
                            'episode_index':row['episode_index'],'step':row['step'],'source_position':row['source_position'],
                            'state_score':before,'threshold':threshold,'light_replaced_score':light,'external_group_replaced_score':group,
                            'light_replacement_below_threshold':light<=threshold,'external_group_replacement_below_threshold':group<=threshold,
                            'recorded_light_physical':float(logged['obs_derived_light_sum']*scaler['scale']['obs_derived_light_sum']+scaler['mean']['obs_derived_light_sum']),
                            'rollout_light_physical':row['physical_state'][i]})
    target=out/'report';target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(statistics).to_csv(target/'recorded_light_feature_distribution.csv',index=False)
    pd.DataFrame(events).to_csv(target/'state_ood_feature_sensitivity.csv',index=False)
    summary={}
    for crop in ('melon','cucumber'):
        selected=[e for e in events if e['crop']==crop and e['split']=='validation']
        summary[crop]={'events':len(selected),'light_replacement_below_threshold':sum(e['light_replacement_below_threshold'] for e in selected),
            'external_group_replacement_below_threshold':sum(e['external_group_replacement_below_threshold'] for e in selected)}
    payload={'feature_dependency_manifest_sha256':sha256(dependency_path),'input_results':inputs,'script_sha256':sha256(Path(__file__)),'summary':summary,'distribution':statistics,
        'interpretation':'Diagnostic coordinate substitution explains support threshold crossings; no environment fix or policy performance improvement has been tested.',
        'test_accessed':False,'environment_modified':False}
    write_json(target/'feature_consistency_evidence.json',payload)
    lines=['# 누적 광량 일관성 추가 진단','',
        'Step 19.5 보류 원인을 구분하기 위한 관측·점수 민감도 분석이다. 환경이나 정책을 수정하지 않았다.','',
        '## 관측 사실','', '종료 건수는 설정·seed·에피소드 조합별 관측 수이며, 독립적인 에피소드 표본 수를 뜻하지 않는다.','']
    for s in statistics:lines.append(f"- {s['crop']}/{s['split']}: 기록 누적 광량 0 비율 {s['zero_fraction']:.4%}, 평균 {s['mean']:.6g}, 표준편차 {s['std']:.6g}.")
    for crop,s in summary.items():lines.append(f"- {crop}: Validation OOD 종료 {s['events']}건 중 누적 광량 한 좌표만 같은 시점 기록값으로 대체한 진단 점수가 기준 이하가 된 경우 {s['light_replacement_below_threshold']}건; 외생 파생변수 묶음은 {s['external_group_replacement_below_threshold']}건.")
    lines += ['', '## 코드 경로와 판단','',
        '`CausalStateUpdater.advance`는 상태 이력에서 `add_mdp_derived_features`를 호출한다. 데이터에 raw 누적 광량 컬럼이 없으면 이 함수는 `obs_outdoor_light`를 적분해 `obs_derived_light_sum`을 다시 만든다. 기록 데이터의 누적 광량 정의·분포와 rollout 중 재계산 분포의 일치 여부가 핵심이다.',
        '- **확인됨:** 실제 저장 feature 분포와 rollout의 누적 광량 값 사이에 불일치가 존재한다. 좌표 대체로 임계값 아래로 내려간 건은 그 종료 시점의 support 판정에 해당 좌표가 기여했음을 보여준다.',
        '- **유력하지만 추가 검증 필요:** 이 외생 파생변수 계약 불일치가 공통 조기 종료를 유발하는 주요 기전이다. 데이터의 원천 누적값이 0이 된 전처리 경로와 일사량 단위의 타당성은 별도 원천 데이터 검증이 필요하다.',
        '- **근거 부족:** 이 좌표를 실제 환경에서 바꾸면 정책 성능·안전성이 개선된다는 결론. 본 분석은 저장 상태에 대한 점수 민감도이며 수정된 rollout을 실행하지 않았다.',
        '', '## 개선안과 재검증 조건','',
        '기록 데이터와 rollout에서 누적 광량·DLI·clear-sky/ratio의 동일한 정의, 단위, 일 경계와 초기 이력을 사용하도록 계약을 명시한다. 원천 기록값의 결측·0 처리 오류인지, rollout 재계산 규칙 문제인지 확인한 뒤 수정 대상을 정한다. 데이터/scaler/support 또는 전이모델 입력이 변경되면 해당 의존성을 새 버전으로 재생성한다.',
        '수정 후 동일 Train/Validation 에피소드에서 기록 상태 및 기록 행동 rollout을 먼저 대조하고, OOD 최초 위치·실행 길이·물리 예측 오차가 개선되는지 검증한다. 이후 PPO 평가와 필요한 Step 15~19 재검증을 거쳐 새 Step 19.5 계획을 정한다. 현재 HPO 보류와 공식 임계값은 유지한다.']
    (target/'feature_consistency_review_ko.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
