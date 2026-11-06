#!/usr/bin/env python3
"""Follow the preregistered learning study through real paired evaluation.

Wait for the separately supervised native training, verify weights and data, then
reuse the official server and native evaluation for every comparison checkpoint.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    r=p.parse_args().run_dir.resolve();record=r/'evaluation-pipeline';record.mkdir(exist_ok=False)
    os.chdir(ROOT)
    cfg=json.loads((r/'configs/supervisor.json').read_text())
    if Path(sys.prefix)!=Path(cfg['python_prefix']):raise RuntimeError(sys.executable)
    state={'status':'waiting_for_training','pid':os.getpid(),'jobs':[],
           'started':datetime.datetime.now().isoformat(timespec='seconds')}
    source=(ROOT/'scripts/run_learning_evaluation.py').read_bytes()
    state['entry_sha256']=hashlib.sha256(source).hexdigest()
    process=None;used=0.0;last_report=0.0
    report_src=Path('/home/xense/src/xensim-bench/outputs/research/robotwin-overview-20261104/robotwin.src.html')
    report_cmd=[sys.executable,'scripts/diagnostics/update_learning_report.py','--run-dir',str(r),
                '--report-source',str(report_src),'--renderer','/home/xense/.agents/skills/visual-explainer/kit/render.mjs',
                '--report-root','/home/xense/src/xensim-bench']
    def save():write_json(record/'status.json',state)
    def report(force=False):
        nonlocal last_report
        if force or time.monotonic()-last_report>60:
            with (record/'report.log').open('a') as log:
                try:
                    result=subprocess.run(report_cmd,stdout=log,stderr=subprocess.STDOUT,timeout=30)
                    state['last_report_exit_code']=result.returncode
                except subprocess.TimeoutExpired:state['last_report_exit_code']='timeout'
            last_report=time.monotonic();save()
    def stop():
        if process is None or process.poll() is not None:return
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    def terminate(*_):raise KeyboardInterrupt('Study shutdown requested')
    signal.signal(signal.SIGTERM,terminate)

    def run(name,arguments,timeout,charge=False):
        nonlocal process,used
        if charge and used>=cfg['gpu_budget_seconds']:raise TimeoutError('Authorized GPU budget exhausted')
        command=[sys.executable,'-X','faulthandler','-u',*arguments]
        with (record/(name+'.log')).open('xb') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            job={'name':name,'command':command,'pid':process.pid,'started':datetime.datetime.now().isoformat(timespec='seconds')}
            state['jobs'].append(job);save();started=time.monotonic()
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed>timeout or (charge and used+elapsed>=cfg['gpu_budget_seconds']):
                    raise TimeoutError(name+' exceeded wall-time budget')
                report();time.sleep(2)
            job.update(exit_code=process.returncode,seconds=time.monotonic()-started)
            if charge:used+=job['seconds']
            state['gpu_wall_seconds_including_prior_stages']=used;save()
            if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}')
            process=None

    def evaluate(cohort,seed,updates):
        name=f'{cohort}-s{seed}-u{updates}';audit=r/'checks'/f'{name}-audit.json'
        # A new supervisor and official server per checkpoint keeps deployment
        # independent; every policy sees all seeds from the same frozen proof.
        training=json.loads((r/'training'/f'seed-{seed}'/'status.json').read_text())
        weights=Path(training['checkpoint']) if updates==18000 else next(Path(v['path']) for v in training['update_checkpoints'] if v['updates']==updates)
        deploy=r/'deploy'/f's{seed}-u{updates}'
        if not deploy.exists():
            deploy.mkdir(parents=True,exist_ok=False)
            (deploy/'policy_last.ckpt').symlink_to(weights)
            (deploy/'dataset_stats.pkl').symlink_to(weights.parent/'dataset_stats.pkl')
        eval_cfg=json.loads((r/'configs/frozen-smoke-1.json').read_text())
        import yaml
        tc=json.loads((r/'configs'/f'{cohort}.json').read_text())['task_config']
        eval_tc=tc+'_eval';config_path=ROOT/'env_cfg/task_config'/f'{eval_tc}.yml'
        if not config_path.exists():
            task=yaml.safe_load((ROOT/'env_cfg/task_config'/f'{tc}.yml').read_text())
            task['eval_video_log']=True;config_path.write_text(yaml.safe_dump(task,sort_keys=False))
        eval_cfg.update(record_dir=str(r/'eval'/name),frozen_expert_manifest=str(r/'manifests'/cohort/'manifest.json'),max_candidate_seeds=100)
        eval_cfg['native_args'].update(test_num=100,task_config=eval_tc,ckpt_setting=name,port=19142)
        eval_path=r/'configs'/f'{name}.json';write_json(eval_path,eval_cfg)
        supervisor=json.loads((r/'configs/supervisor-frozen-smoke.json').read_text())
        supervisor.update(run_dir=str(r/'supervisors'/name),checkpoint=str(deploy),port=19142,
                          gpu_budget_seconds=cfg['gpu_budget_seconds']-used,
                          experiments=[{'name':name,'config':str(eval_path),'timeout_seconds':7200}])
        supervisor_path=r/'configs'/f'supervisor-{name}.json';write_json(supervisor_path,supervisor)
        run(name,['scripts/run_policy_experiments.py','--config',str(supervisor_path)],7500,charge=True)
        run(name+'-audit',['scripts/diagnostics/audit_policy_record.py','--config',str(eval_path),'--output',str(audit),
                          '--log',str(r/'supervisors'/name/f'{name}.log')],300)
        result=json.loads(audit.read_text());assert result['passed'] and result['learner_trials']==100
        return {'name':name,'training_seed':seed,'updates':updates,'successes':result['successes'],
                'trials':result['trials'],'checkpoint_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                'frozen_manifest_sha256':hashlib.sha256((r/'manifests'/cohort/'manifest.json').read_bytes()).hexdigest()}

    save()
    try:
        while True:
            training_state=json.loads((r/'supervisor/status.json').read_text())
            state['training_supervisor_status']=training_state['status'];save();report()
            if training_state['status']=='failed':raise RuntimeError('Native collection/training supervisor failed; inspect its preserved logs')
            if training_state['status']=='complete':break
            if training_state['gpu_wall_seconds_including_prior_stages']>=cfg['gpu_budget_seconds']:
                raise TimeoutError('Authorized GPU budget exhausted during training')
            time.sleep(10)
        used=training_state['gpu_wall_seconds_including_prior_stages']
        state['status']='validating_training';save()
        run('training-audit',['scripts/diagnostics/validate_study_training.py','--run-dir',str(r)],600)
        # Both cohorts were frozen before learner evaluation or model selection.
        manifests=[json.loads((r/'manifests'/name/'manifest.json').read_text()) for name in ['comparison','confirmation']]
        assert all(v['status']=='frozen' and len(v['accepted'])==100 for v in manifests)
        assert not {v['seed'] for v in manifests[0]['accepted']}&{v['seed'] for v in manifests[1]['accepted']}
        state['status']='comparing_checkpoints';save();comparison=[]
        for updates in [3000,9000,18000]:
            for seed in [0,1]:
                comparison.append(evaluate('comparison',seed,updates))
                write_json(r/'checks/comparison-results.json',{'status':'running','models':comparison})
                report(force=True)
        sums={u:sum(v['successes'] for v in comparison if v['updates']==u) for u in [3000,9000,18000]}
        budget=max(sums,key=lambda u:(sums[u],-u))
        selected=max((v for v in comparison if v['updates']==budget),key=lambda v:(v['successes'],-v['training_seed']))
        selection={'status':'complete','models':comparison,'selected_budget':budget,
                   'selected_training_seed':selected['training_seed'],'selection_rule':'Highest mean comparison success across seeds; tie lower budget. Within budget higher success; tie lower seed.'}
        write_json(r/'checks/comparison-results.json',selection)
        state.update(status='confirming',selection={'updates':budget,'training_seed':selected['training_seed']});save();report(force=True)
        confirmation=evaluate('confirmation',selected['training_seed'],budget)
        baseline=confirmation if budget==3000 else evaluate('confirmation',selected['training_seed'],3000)
        a={v['seed']:v['success'] for v in confirmation['trials']};b={v['seed']:v['success'] for v in baseline['trials']};assert a.keys()==b.keys()
        wins=sum(a[s] and not b[s] for s in a);losses=sum(b[s] and not a[s] for s in a);n=wins+losses
        exact_p=min(1.0,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.0
        outcome={'status':'complete','selected':confirmation,'same_seed_3000_update_baseline':baseline,
                 'paired_wins':wins,'paired_losses':losses,'success_difference_percentage_points':wins-losses,
                 'paired_exact_two_sided_p':exact_p,'same_checkpoint':budget==3000,
                 'scope':'100 expert-conditioned frozen confirmation seeds. Report negative results and all comparison models; no unconditional task success or base1 coverage claim.'}
        write_json(r/'checks/confirmation-results.json',outcome)
        state.update(status='complete',gpu_wall_seconds_including_prior_stages=used)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        stop();state['updated']=datetime.datetime.now().isoformat(timespec='seconds');save();report(force=True)


if __name__=='__main__':main()
