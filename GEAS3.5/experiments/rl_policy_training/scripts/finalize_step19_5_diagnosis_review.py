#!/usr/bin/env python3
"""Complete post-hoc diagnosis from validated, finished diagnostic jobs only."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import sys,json,gzip,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from geas35.experiments.rl.step19_5_diagnostics import OUT,read_json,write_json,verify_inputs,sha256,lock


def correlation(x,y):
    a,b=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    valid=np.isfinite(a)&np.isfinite(b);a,b=a[valid],b[valid]
    return None if len(a)<3 or a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])


def main():
    out=ROOT/OUT;report=out/'report';evidence=read_json(report/'evidence.json')
    if evidence['status']!='complete':raise ValueError('Final diagnosis requires all planned jobs')
    manifest=read_json(out/'input_manifest.json');verify_inputs(ROOT,manifest)
    with lock(out/'final_review.lock'):
        subprocess.run([sys.executable,str(Path(__file__).with_name('analyze_step19_5_feature_consistency.py'))],check=True)
        sensitivity=read_json(report/'feature_consistency_evidence.json')
        baseline=[]
        for crop in ('melon','cucumber'):
            cols=manifest['trace_state_columns'][crop]
            for split in ('train','validation'):
                directory=out/'jobs'/f'{crop}__{split}__recorded-one-step__representative_0__42__524288__0'
                result=read_json(directory/'result.json');errors={}
                for ep in result['episode_files']:
                    p=(directory/ep['path']).with_suffix('.jsonl.gz')
                    with gzip.open(p,'rt') as stream:
                        for line in stream:
                            row=json.loads(line)
                            for target,pred in row['prediction_physical'].items():
                                actual=pred-row['one_step_error_physical'][target]
                                hold=row['physical_state'][cols.index(target)]
                                errors.setdefault(target,[]).append((pred-actual,hold-actual))
                for target,pairs in errors.items():
                    arr=np.asarray(pairs);rmse=np.sqrt(np.square(arr).mean(axis=0))
                    baseline.append({'crop':crop,'split':split,'target':target,'count':len(arr),
                        'model_rmse':float(rmse[0]),'persistence_rmse':float(rmse[1]),
                        'model_better_than_persistence':bool(rmse[0]<rmse[1])})
        pd.DataFrame(baseline).to_csv(report/'one_step_persistence_comparison.csv',index=False)
        pairs=pd.read_csv(report/'strawberry_reward_changes.csv')
        episodes=pd.read_csv(report/'strawberry_episode_changes.csv')
        concentration=[]
        for (config,seed),group in episodes.groupby(['config_id','seed']):
            values=np.sort(np.abs(group.reward_sum_delta.to_numpy()))[::-1];total=values.sum()
            concentration.append({'config_id':config,'seed':int(seed),
                'top5_absolute_change_share':float(values[:5].sum()/total) if total else 0.,
                'episode_reward_terminal_correlation':correlation(group.reward_sum_delta,group.terminal_delta)})
        windows=pd.read_csv(report/'training_diagnostics.csv');correlations=[]
        for metric in windows.metric.unique():
            w=windows[(windows.crop=='strawberry')&(windows.metric==metric)&(windows.to_step.isin([524288,1048576]))]
            table=w.pivot(index=['config_id','seed'],columns='to_step',values='median')
            if not {524288,1048576}.issubset(table.columns):continue
            change=(table[1048576]-table[524288]).rename('diagnostic_delta').reset_index()
            merged=pairs.merge(change,on=['config_id','seed'])
            correlations.append({'metric':metric,'n':len(merged),'pearson_r':correlation(merged.reward_delta,merged.diagnostic_delta),
                'interpretation':'Exploratory 15-run association; not evidence of causal PPO instability.'})
        pd.DataFrame(concentration).to_csv(report/'strawberry_episode_concentration.csv',index=False)
        pd.DataFrame(correlations).to_csv(report/'strawberry_update_associations.csv',index=False)
        strawberry={'pairs':len(pairs),'absolute_reward_delta_median':float(pairs.reward_delta.abs().median()),
            'reward_terminal_delta_correlation':correlation(pairs.reward_delta,pairs.delta_terminal),
            'median_top5_episode_absolute_change_share':float(np.median([x['top5_absolute_change_share'] for x in concentration])),
            'update_associations':correlations,
            'classification':'Observed seed/episode variation confirmed; PPO-update causality unproven.'}
        decision={'status':'diagnosis_complete','hpo_decision':'hold','environment_modified':False,'training_started':False,
            'test_accessed':False,'source_code_drift':manifest.get('source_code_drift',[]),
            'melon_cucumber_light_feature_sensitivity':sensitivity['summary'],'strawberry':strawberry,
            'one_step_persistence_comparison':baseline,
            'safety_gate_comparability':{'frozen_gate_basis':'Step17 Train static joint severe coverage','evaluated_event':'state severe OR joint severe','rollout_denominator':'executed calls including terminal OOD call','limitation':'Different event definitions and state visitation distributions; diagnostic scheduled-step rates do not replace official thresholds.'},
            'next_priority':'Align recorded and rollout exogenous derived-feature contracts before designing another long calibration run.',
            'limitations':['Coordinate replacement changes only a diagnostic support query; no repaired policy rollout was executed.',
                'Past source hash differs; matching representative historical metrics does not prove full historical equivalence.',
                'Only three training seeds per configuration; correlations are exploratory.'],
            'evidence_sha256':sha256(report/'evidence.json'),'feature_evidence_sha256':sha256(report/'feature_consistency_evidence.json'),
            'script_sha256':sha256(Path(__file__))}
        write_json(report/'diagnosis_decision.json',decision)
        lines=['# GEAS3.5 Step 19.5 진단 최종 판단','',
            '## 배경과 현재 위치','',
            'Transition 확정(Step 14) → 환경·PPO 기본 검증(Step 15~19) → 학습량 calibration(Step 19.5) → HPO(Step 20) → 최종 학습·평가·인계 순서다. 현재는 Step 19.5 미통과 원인을 진단한 상태이며, HPO는 보류한다.','',
            f"계획된 진단 작업 {evidence['completed_jobs']}/{evidence['expected_jobs']} 완료. 기존 공식 결과와 안전성 기준은 변경하지 않았다.",'',
            '## 멜론·오이','',
            '기록 누적 광량과 rollout에서 재계산되는 누적 광량 사이의 분포 불일치가 확인됐다. 다음 수치는 학습 데이터 기준 support 판정의 좌표 민감도이며, 수정된 정책 성능을 뜻하지 않는다.']
        for crop,value in sensitivity['summary'].items():
            ev=evidence['crop_evidence'][crop];recorded=ev['validation/recorded-rollout']
            direct=ev['validation/recorded-one-step']['state_severe_rate'];train_direct=ev['train/recorded-one-step']['state_severe_rate']
            lines.append(f"- {crop}: 실제 기록 state-only OOD 비율 Train 대조 {train_direct:.3%}, 전체 Validation {direct:.3%}. 이는 중도 종료되는 rollout의 사건률과 표본·분모가 다르다.")
            lines.append(f"- {crop}: Validation OOD 종료 {value['events']}건 중 누적 광량만 기록값으로 대체한 진단 점수가 기준 이하인 경우 {value['light_replacement_below_threshold']}건. 기록 행동 rollout 종료 사유: {recorded['termination_reasons']}.")
        lines += ['- 건수는 설정·seed·에피소드 조합별 관측 수이며 독립 표본 수가 아니다.', '- **확인됨:** 외생 파생변수의 기록/rollout 정의·분포 불일치 및 해당 좌표의 OOD 임계값 초과 기여.',
            '- **유력:** 이 계약 불일치가 여러 설정에 공통인 조기 종료의 주요 기전. 기록 행동에서도 종료가 발생하면 PPO만의 문제로 설명할 수 없다.',
            '- **추가 검토:** 공식 상한은 Train의 정적 joint OOD coverage에서 유도됐지만 평가 사건은 state 또는 joint OOD이며 rollout의 실행 step을 분모로 사용한다. 정의·방문 분포 차이를 검토하되 공식 기준은 유지한다.',
            '- **미확인:** 원천 누적 광량 0 처리의 데이터 의미, 수정 후 rollout 안전성·성능. 수정 적용은 이번 진단 범위 밖이다.','',
            '## 딸기','',
            f"- 같은 평가기의 30개 체크포인트 재평가에서 최근 보상 변화 절대값 중앙값: {strawberry['absolute_reward_delta_median']:.8f}.",
            f"- 설정·seed별 보상 변화와 종료 비용 변화의 상관계수: {strawberry['reward_terminal_delta_correlation']}.",
            f"- 변화량 상위 5개 에피소드가 차지하는 절대 보상 변화 비중의 중앙값: {strawberry['median_top5_episode_absolute_change_share']:.2%}.",
            '- **확인됨:** 평가 재현성 결과, seed/에피소드별 보상 변동 및 종료 비용 기여는 CSV로 추적 가능하다.',
            '- **근거 부족:** PPO 업데이트가 단독 원인이라는 인과 판단. 업데이트 지표와의 상관은 15개 쌍의 탐색적 결과이며 원인 확정에 사용하지 않는다.','',
            '## 권장 다음 조치','',
            '1. 누적 광량·DLI 등 외생 파생변수의 기록 데이터와 시뮬레이터 정의·단위·일 경계·초기 이력을 먼저 일치시키는 수정안을 설계한다. 원천 0/결측 처리와 재계산 경로를 확인해 수정 대상을 선택한다.',
            '2. 수정 시 영향받는 데이터/scaler/support/모델 의존성을 새 버전으로 분리하고, 기록 상태 및 기록 행동 rollout에서 최초 OOD·실행 길이·물리 단위 오차를 재검증한다. 1-step persistence 비교는 대상 변수별로 해석한다.',
            '3. 딸기는 에피소드별 종료 비용·행동 비용·seed 편차를 근거로 정책 학습 개선 가설을 정한다. 환경 계약 문제가 영향을 주는지 먼저 분리한다.',
            '4. 필요한 Step 15~19 검증 후 새 Step 19.5 프로토콜을 사전 정의한다. 현재 진단으로 기존 미통과 판정을 바꾸거나 HPO를 시작하지 않는다.','',
            '상세 근거: [전체 진단](diagnosis_report_ko.md), [광량 일관성](feature_consistency_review_ko.md), [기계 판독 결정](diagnosis_decision.json), [1-step 기준 비교](one_step_persistence_comparison.csv), [딸기 업데이트 연관](strawberry_update_associations.csv).']
        (report/'final_decision_ko.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(report/'final_decision_ko.md')


if __name__=='__main__':main()
