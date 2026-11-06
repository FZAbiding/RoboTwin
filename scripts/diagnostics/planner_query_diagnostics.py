#!/usr/bin/env python3
"""Audit native batch endpoints and frozen single-goal timing refinement queries."""
from pathlib import Path
import argparse,importlib.util,json,sys,time,traceback,os
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'scripts'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from lift_pot_observer import Observer,plain


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--record-dir',type=Path,required=True);p.add_argument('--seed',type=int,default=100001);a=p.parse_args();r=a.record_dir.resolve();r.mkdir(parents=True,exist_ok=False)
    result={'status':'running','pid':os.getpid(),'seed':a.seed,'query_only_after_original_expert':True,'task_predicate_and_planner_tolerances_unchanged':True,'queries':[]}
    def save():(r/'summary.json').write_text(json.dumps(result,indent=2))
    def event(kind,**fields):
        with (r/'events.jsonl').open('a') as f:f.write(json.dumps({'event':kind,'time':time.time(),**plain(fields)})+'\n')
    spec=importlib.util.spec_from_file_location('native_planner_query',ROOT/'scripts/eval_policy_xpolicylab.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    cfg=json.loads(a.config.read_text());args,_=native.load_task_args(cfg['native_args']);args.update(eval_mode=True,eval_video_save_dir=str(r/'videos'));env=native.class_decorator('lift_pot');observer=Observer(event);captures=[];phase={'name':'original'};env.setup_demo(now_ep_num=0,seed=a.seed,is_test=True,**args);observer.on_setup(env,'expert',a.seed)
    from curobo.types.robot import JointState
    def capture(mg,arm,mode,original):
        def observed(start,goal,config,*pos,**kw):
            frozen_start=start.clone();frozen_goal=goal.clone();frozen_config=config.clone();value=original(start,goal,config,*pos,**kw)
            if phase['name']=='original':
                paths=[]
                if value.interpolated_plan is not None:
                    if value.path_buffer_last_tstep is not None:
                        paths=[value.interpolated_plan[i].trim_trajectory(0,int(n)).clone() for i,n in enumerate(value.path_buffer_last_tstep)]
                    else:paths=[value.interpolated_plan[i].clone() for i in range(len(value.interpolated_plan))]
                captures.append({'arm':arm,'mode':mode,'mg':mg,'original':original,'start':frozen_start,'goal':frozen_goal,'config':frozen_config,'success':plain(value.success),'status':str(value.status),'paths':paths,'padded_shape':list(value.interpolated_plan.position.shape) if value.interpolated_plan is not None else None,'last_steps':plain(value.path_buffer_last_tstep),'position_error':plain(value.position_error),'rotation_error':plain(value.rotation_error)})
            return value
        return observed
    for arm in ('left','right'):
        planner=getattr(env.robot,f'{arm}_planner')
        for attr,method,mode in [('motion_gen','plan_single','single'),('motion_gen_batch','plan_batch','batch')]:
            mg=getattr(planner,attr);original=getattr(mg,method);setattr(mg,method,capture(mg,arm,mode,original))
    try:
        env.play_once();result['model_id']=int(env.model_id);result['original_expert']={'plan_success':bool(env.plan_success),'task_success':bool(env.check_success()),'metrics':observer.metrics(env)};phase['name']='query'
        failed=[v for v in captures if v['mode']=='single' and not np.asarray(v['success']).any()]
        if not failed:raise RuntimeError('Original expert produced no failed single query to diagnose')
        one=failed[0];mg=one['mg'];target_p=one['goal'].position.detach().cpu().numpy()[0];target_q=one['goal'].quaternion.detach().cpu().numpy()[0];matches=[]
        for batch in captures:
            if batch['mode']!='batch' or batch['arm']!=one['arm']:continue
            gp=batch['goal'].position.detach().cpu().numpy();gq=batch['goal'].quaternion.detach().cpu().numpy()
            for i in range(len(gp)):
                dq=min(float(np.abs(gq[i]-target_q).max()),float(np.abs(gq[i]+target_q).max()))
                if np.abs(gp[i]-target_p).max()<1e-5 and dq<1e-5:matches.append((batch,i,dq))
        result['frozen_query']={'arm':one['arm'],'start_joint_position':plain(one['start'].position),'goal_position':plain(one['goal'].position),'goal_quaternion':plain(one['goal'].quaternion),'original_status':one['status'],'original_config':{key:plain(getattr(one['config'],key,None)) for key in ['finetune_dt_scale','finetune_dt_decay','finetune_attempts','max_attempts','enable_finetune_trajopt','parallel_finetune','timeout']}}
        endpoint_results=[]
        for batch,index,dq in matches:
            entry={'matched_index':index,'matched_success':bool(np.asarray(batch['success'])[index]),'batch_status':batch['status'],'start_max_difference':float(torch.abs(batch['start'].position[index]-one['start'].position[0]).max().item()),'goal_max_position_difference':float(np.abs(batch['goal'].position[index].cpu().numpy()-target_p).max()),'goal_max_quaternion_difference':dq,'raw_padded_shape':batch['padded_shape'],'valid_path_steps':batch['last_steps'],'all_trimmed_path_lengths':[int(v.position.shape[-2]) for v in batch['paths']]}
            if batch['paths'] and entry['matched_success']:
                path=batch['paths'][index];endpoint=path.position.reshape(-1,path.position.shape[-1])[-1:];fk=batch['mg'].compute_kinematics(JointState.from_position(endpoint,joint_names=batch['start'].joint_names)).ee_pose
                difference=torch.linalg.vector_norm(fk.position-batch['goal'].position[index:index+1],dim=-1);q=fk.quaternion/torch.linalg.vector_norm(fk.quaternion,dim=-1,keepdim=True);g=batch['goal'].quaternion[index:index+1];g=g/torch.linalg.vector_norm(g,dim=-1,keepdim=True);angle=2*torch.acos(torch.clamp(torch.abs(torch.sum(q*g,dim=-1)),max=1));entry.update(endpoint_position_error_m=plain(difference),endpoint_orientation_geodesic_rad=plain(angle),path_finite=bool(torch.isfinite(path.position).all().item()))
            endpoint_results.append(entry)
        result['batch_endpoint_audit']=endpoint_results;save()
        for repetition in range(2):
            for scale in [.85,.9,.95,1.0,1.05]:
                torch.manual_seed(a.seed);mg.reset(reset_seed=True);plan=one['config'].clone();plan.finetune_dt_scale=scale;start=time.perf_counter();value=one['original'](one['start'].clone(),one['goal'].clone(),plan)
                row={'repetition':repetition,'finetune_dt_scale':scale,'reset_before_query':True,'torch_seed':a.seed,'success':bool(value.success.any().item()),'status':str(value.status),'seconds':time.perf_counter()-start,'position_error':plain(value.position_error),'rotation_error':plain(value.rotation_error),'valid_query':plain(value.valid_query),'optimized_dt':plain(value.optimized_dt)}
                result['queries'].append(row);save();print(json.dumps(row),flush=True)
        result['status']='complete';result['scope']='Frozen planner query ablation only. No returned query trajectory was executed or accepted as a demonstration; successful task validation is still required before adoption.'
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());raise
    finally:save();env.close_env()


if __name__=='__main__':main()
