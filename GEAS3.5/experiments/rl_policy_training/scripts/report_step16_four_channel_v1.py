"""저장된 Step16 trace 집계 전용. rollout/모델/threshold 변경 없음."""
from pathlib import Path
import importlib.util,inspect,argparse,json
p=Path(__file__).with_name('validate_step16_four_channel_v1.py')
s=importlib.util.spec_from_file_location('frozen_step16',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main(crop):
 m.check();out=m.D/crop
 assert m.read(out/'replay_validation.json')['PASS']
 source=inspect.getsource(m.summarize)
 source=source.replace('ok[~ok.path_action_changed]','ok[~ok.path_action_changed.astype(bool)]').replace('ok[ok.path_action_changed]','ok[ok.path_action_changed.astype(bool)]')
 exec(compile(source,str(Path(__file__)),'exec'),m.__dict__)
 rows=m.pd.read_parquet(out/'step_trace.parquet').to_dict('records');ep=m.pd.read_csv(out/'episode_summary.csv').to_dict('records');checks=m.read(out/'replay_validation.json');m.summarize(crop,rows,ep,checks)
 m.write(out/'report_correction.json',{'reason_ko':'미실행 종료행의 결측 때문에 object dtype이 된 bool 열의 ~ 연산을 명시적 bool로 집계','rollout_reexecuted':False,'threshold_model_actions_modified':False,'report_script_sha256':m.sha(Path(__file__)),'trace_sha256':m.sha(out/'step_trace.parquet')})
 print(crop,m.read(out/'result.json')['status'],flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--crop',choices=list(m.C['selected']),required=True);a=ap.parse_args();main(a.crop)
