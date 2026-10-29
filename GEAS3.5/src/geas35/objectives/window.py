"""Bounded read-only observer for the current P0 schema (no explicit action link).

This schema exposes sent event IDs only after sending, no preannounced action ID
or explicit subsequent sensor foreign key. Every candidate is therefore null.
Do not manufacture a pre-action binding to increase the valid count.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from .binding import timestamp,canonical
from .live import LivePath,receipt


def atomic(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def within(source_time,received_at,activation):
    start=timestamp(activation['effective_from']);end=timestamp(activation['effective_until'])
    return start<=source_time<end and received_at<end


def summarize(path,activation,completed=False,failure=None):
    reasons=Counter();count=0
    for raw, in path.db.execute("SELECT payload FROM events WHERE key LIKE 'candidate:%'"):
        r=json.loads(raw);count+=1;reasons.update(x['reason'] for x in r['result']['reasons'])
    return dict(status='TECHNICAL_FAILURE' if failure else ('COMPLETED' if completed else 'SCHEDULED_OR_OBSERVING'),verdict='CUCUMBER72 PROFILE-A SHADOW EVALUATION — TECHNICAL FAILURE' if failure else ('CUCUMBER72 PROFILE-A SHADOW EVALUATION — NO SCORABLE ACTION' if completed else None),activation=activation,updated_at=datetime.now(timezone.utc).isoformat(),candidate_actions=count,candidate_unit='individual sent-command event, not aggregated control cycle',valid_attributed_actions=0,UNSCORABLE_actions=count,reason_counts=dict(reasons),valid_rewards=0,reward_statistics=dict(min=None,median=None,max=None),temperature_quality_statistics=dict(min=None,median=None,max=None),rh_quality_statistics=dict(min=None,median=None,max=None),target_parity_violations=0,source_clock_integrity_failures=1 if failure else 0,failure=failure,objective_bound_evaluation_started=False,objective_bound_evaluation_start=None,adapter_capability='P0 schema has no preannounced action ID or explicit outcome link; no nearest join; no fabricated bindings',no_send=True,PPO=False)


def report(directory,summary):
    directory=Path(directory);directory.mkdir(exist_ok=True)
    atomic(directory/'profile_a_shadow_evaluation.json',summary)
    a=summary['activation'];verdict=summary['verdict'] or 'SCHEDULED — WINDOW NOT YET COMPLETE'
    text=f'''# Cucumber72 Profile-A Bounded SHADOW Evaluation

**{verdict}**

- 예정 구간: {a['effective_from']} ≤ action < {a['effective_until']}
- Profile: {a['profile_id']} / {a['profile_version']}
- Profile hash: `{a['profile_hash']}`
- Activation hash: `{a['activation_hash']}`
- Context: `{a['activation_context']}`, EXPLICIT_PROJECT_EVALUATION_CONTEXT
- 실제 objective-bound 시작: {summary['objective_bound_evaluation_started']}
- 후보 전송 이벤트: {summary['candidate_actions']}, valid attributed action: 0, valid reward: 0
- UNSCORABLE: {summary['UNSCORABLE_actions']}, 사유: {summary['reason_counts']}
- source/clock/integrity failure: {summary['source_clock_integrity_failures']}

현재 P0 schema에는 사전 확정 future action ID와 명시적 outcome 연결이 없다. 전송이 이미 발생한 후 알게 된 ID로 binding을 소급 작성하지 않는다. 각 후보는 ACTION_ATTRIBUTION_UNRESOLVED로 보존한다. 시간 근접 연결, DAT/센서값 기반 stage 추론, target/reward 변경은 없다. 09–10시는 이 프로젝트 테스트 구간이며 일반 농업 오전 규칙이 아니다.

프로필의 약30°C/약90%는 독립 인터뷰 reference의 프로젝트 조합이다. 공동/보편 최적값, 정책 인과효과, PPO 성능으로 해석하지 않는다. reward가 없으므로 min/median/max 및 Q_T/Q_RH 통계는 null이다. 실제 start marker를 만들지 않았다.

만료 이후 prepare/action/outcome 및 reward execution은 거부한다. P0 수집기는 중지하지 않는다. 원본 P0와 DB를 수정하지 않으며 제어 전송·PPO·HPO·Transition fit은 없다.

검사/예약 증거는 prospective runtime 저장소의 preflight.json, activation.json, evaluation.sqlite, checkpoint.json에 보존한다. 이 보고서는 작업자가 구간 종료 시 갱신한다. 구간 종료 전에는 PASS/NO SCORABLE ACTION 최종 판정을 부여하지 않는다.

다음 조치: 종료 결과가 NO SCORABLE ACTION이면 명시 action-attribution/live-link만 해결한 뒤 같은 target/reward로 새 미래 구간을 선언한다. 현재 미지원 연결을 자동 생성하지 않는다.

업데이트: {summary['updated_at']}
'''
    dest=directory/'final_profile_a_shadow_evaluation.md';tmp=dest.with_suffix('.tmp');tmp.write_text(text);os.replace(tmp,dest)


def run(config_path):
    # Reuse tested kernel network/exec denial; no source DB is opened here.
    from geas35.collection.receive_only import lock_network
    config=json.loads(Path(config_path).read_text());runtime=Path(config['runtime']);source=Path(config['p0'])
    path=LivePath(runtime/'journal',config['contract'],config['session_id'])
    activation=json.loads((runtime/'activation.json').read_text())
    protection=lock_network()
    atomic(runtime/'worker.json',dict(pid=os.getpid(),no_send=protection,started_at=path.now()))
    cpfile=runtime/'checkpoint.json';cp=json.loads(cpfile.read_text()) if cpfile.exists() else dict(offset=config['initial_sent_offset'])
    try:
        while True:
            now=timestamp(path.now());end=timestamp(activation['effective_until'])
            if not Path('/proc/'+str(config['collector_pid'])).exists():raise ValueError('BASE_COLLECTOR_NOT_ALIVE')
            with (source/'sent_commands.jsonl').open('rb') as f:
                f.seek(cp['offset'])
                for _ in range(10000):
                    offset=f.tell();line=f.readline()
                    if not line or not line.endswith(b'\n'):f.seek(offset);break
                    row=json.loads(line);raw,sent,received=receipt(row,'sent_commands',config['session_id'])
                    if received>now:raise ValueError('RECEIPT_CLOCK_FUTURE')
                    if within(sent,received,activation):
                        key='candidate:'+row['entry_hash']
                        if not path.get(key):
                            result=path.null('ACTION_ATTRIBUTION_UNRESOLVED')
                            path.append(key,dict(action_source_id=row['source_row_id'],action_time=sent.isoformat(),sent_receipt=row,activation_hash=activation['activation_hash'],result=result,reason_detail='No preannounced future action identity or explicit action/outcome link in frozen P0 schema'))
                cp['offset']=f.tell()
                drained=not f.read(1)
            cp.update(updated_at=path.now(),collector_checkpoint=json.loads((source/'checkpoint.json').read_text())['payload']['sequence'])
            atomic(cpfile,cp)
            completed=now>=end and drained
            report(config['artifacts'],summarize(path,activation,completed))
            if completed:break
            time.sleep(10)
    except BaseException as exc:
        report(config['artifacts'],summarize(path,activation,failure=type(exc).__name__+': '+str(exc)))
        raise
    finally:path.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');run(parser.parse_args().config)
