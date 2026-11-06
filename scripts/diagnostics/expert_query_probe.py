"""Replay the original expert, then repeat frozen planning queries without execution."""
from pathlib import Path
import sys,os,json,argparse,importlib.util,copy,time,traceback
import numpy as np
import sapien
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root));sys.path.insert(1,str(root/'scripts'));sys.path.insert(0,str(Path(__file__).resolve().parent));from lift_pot_observer import Observer,plain
parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--config',type=Path,required=True);options=parser.parse_args();r=options.run_dir.resolve();seed=options.seed;record=r/'expert-queries'/str(seed);record.mkdir(parents=True,exist_ok=False)
result={'seed':seed,'status':'running','pid':os.getpid(),'query_only_after_original_expert':True,'original_task_unmodified':True,'experiments':[]}
def save():(record/'summary.json').write_text(json.dumps(result,indent=2))
def event(kind,**fields):
 with (record/'events.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'time':time.time(),**plain(fields)})+'\n')
def frozen(value):
 if isinstance(value,sapien.Pose):return sapien.Pose(value.p.copy(),value.q.copy())
 if isinstance(value,np.ndarray):return value.copy()
 if isinstance(value,list):return [frozen(v) for v in value]
 if isinstance(value,tuple):return tuple(frozen(v) for v in value)
 if isinstance(value,dict):return {k:frozen(v) for k,v in value.items()}
 return copy.deepcopy(value)
def flat_pose(value):
 if isinstance(value,sapien.Pose):return np.r_[value.p,value.q]
 return np.asarray(value,dtype=float)
spec=importlib.util.spec_from_file_location('native_expert_query',root/'scripts/eval_policy_xpolicylab.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
config=json.loads(options.config.read_text());args,_=native.load_task_args(config['native_args']);args.update(eval_mode=True,eval_video_save_dir=str(record/'videos'));environment=native.class_decorator('lift_pot');observer=Observer(event);captured=[];phase={'label':'original_expert'}
save()
try:
 environment.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args);result['model_id']=int(environment.model_id);observer.on_setup(environment,'expert',seed)
 for arm in ('left','right'):
  owner=getattr(environment.robot,f'{arm}_planner')
  for name in ('plan_path','plan_batch'):
   original=getattr(owner,name)
   def observe(*a,_original=original,_arm=arm,_name=name,**kw):
    inputs=(frozen(a),frozen(kw));start=time.perf_counter();value=_original(*a,**kw)
    if phase['label']=='original_expert':captured.append({'arm':_arm,'method':_name,'args':inputs[0],'kwargs':inputs[1],'result_status':plain(value.get('status')),'original':_original,'index':len(captured)})
    event('frozen_query_result',label=phase['label'],arm=_arm,method=_name,seconds=time.perf_counter()-start,status=value.get('status'));return value
   setattr(owner,name,observe)
 environment.play_once();result['original_expert']={'plan_success':bool(environment.plan_success),'task_success':bool(environment.check_success()),'metrics':observer.metrics(environment)};observer.on_expert_end(environment);save()
 failed=[v for v in captured if v['method']=='plan_path' and v['result_status']!='Success'];queries=failed[:1] or [v for v in captured if v['method']=='plan_path' and v['arm']=='right'][:1]
 for query in queries:
  arm=query['arm'];target=flat_pose(query['args'][1]);batch=None;match=None
  for candidate in captured[:query['index']]:
   if candidate['arm']!=arm or candidate['method']!='plan_batch':continue
   for i,pose in enumerate(candidate['args'][1]):
    if np.allclose(flat_pose(pose),target,atol=1e-6):batch=candidate;match=i
  experiment={'arm':arm,'original_single_status':query['result_status'],'frozen_single_args':plain(query['args']),'frozen_single_kwargs':plain(query['kwargs']),'batch_match_index':match,'original_batch_match_status':batch['result_status'][match] if batch is not None else None,'batch_vs_single_start_max_error':float(np.abs(np.asarray(batch['args'][0])-np.asarray(query['args'][0])).max()) if batch is not None else None,'single_repeats':[]}
  for index in range(3):
   phase['label']=f'frozen_single_repeat_{index+1}';start=time.perf_counter();value=query['original'](*frozen(query['args']),**frozen(query['kwargs']));experiment['single_repeats'].append({'status':plain(value['status']),'seconds':time.perf_counter()-start})
  if batch is not None:
   phase['label']='frozen_original_batch_repeat';start=time.perf_counter();value=batch['original'](*frozen(batch['args']),**frozen(batch['kwargs']));experiment['batch_repeat']={'status':plain(value['status']),'matched_status':plain(value['status'][match]),'seconds':time.perf_counter()-start}
   phase['label']='frozen_single_after_batch';value=query['original'](*frozen(query['args']),**frozen(query['kwargs']));experiment['single_after_batch_status']=plain(value['status'])
  result['experiments'].append(experiment);save()
 result['status']='complete';result['physical_execution_note']='Only original play_once executed actions. Repeated frozen planner queries returned paths/statuses; no returned query path was executed and no demonstration was accepted.'
except BaseException:
 result.update(status='failed',error=traceback.format_exc());raise
finally:
 save();environment.close_env()
print(json.dumps(result,indent=2))
