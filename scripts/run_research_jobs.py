#!/usr/bin/env python3
"""Serial, bounded local jobs with process ownership and a GPU wall-time ledger."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)); temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    config=json.loads(parser.parse_args().config.read_text())
    record=Path(config['record_dir']).resolve(); record.mkdir(parents=True,exist_ok=False)
    if Path(sys.prefix)!=Path(config['python_prefix']): raise RuntimeError(sys.executable)
    env=os.environ.copy(); env.update(config.get('environment',{})); env['PYTHONFAULTHANDLER']='1'
    state={'status':'preparing','pid':os.getpid(),'config':config,'jobs':[],
           'started':time.strftime('%Y-%m-%dT%H:%M:%S')}
    used=float(config.get('previous_gpu_wall_seconds',0)); process=None; active_start=None
    def save():
        state['gpu_wall_seconds_including_prior_stages']=used+(time.monotonic()-active_start if active_start else 0)
        write_json(record/'status.json',state)
    def stop():
        if process is None:return
        # Children are confined to the process group created for this job.
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:return
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.wait()
    def terminate(*_):raise KeyboardInterrupt('Supervisor termination requested')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for job in config['jobs']:
            if shutil.disk_usage(record).free < float(config.get('minimum_free_disk_gib',4))*1024**3:
                raise RuntimeError('Insufficient disk space for the next job')
            free=int(subprocess.check_output(['nvidia-smi','--id='+str(config.get('gpu',0)),
                     '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<int(job.get('minimum_free_gpu_mib',16000)):raise RuntimeError(f'GPU free memory {free}MiB')
            command=[sys.executable,'-X','faulthandler','-u',*job['arguments']]
            row={'name':job['name'],'command':command,'started':time.strftime('%Y-%m-%dT%H:%M:%S')}
            with (record/(job['name']+'.log')).open('xb') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,
                                         cwd=config['cwd'],start_new_session=True)
                active_start=time.monotonic(); last_change=active_start; last_mtime=None; stack_sent=False
                row['pid']=process.pid; state['jobs'].append(row); state['status']='running';save()
                print(json.dumps({'start':row}),flush=True)
                last_print=0
                while process.poll() is None:
                    now=time.monotonic(); elapsed=now-active_start
                    if elapsed>job['timeout_seconds']:raise TimeoutError(job['name']+' exceeded wall-time limit')
                    if used+elapsed>config['gpu_budget_seconds']:raise TimeoutError('Authorized GPU budget exhausted')
                    if shutil.disk_usage(record).free<1.5*1024**3:raise RuntimeError('Disk free space below1.5GiB')
                    heartbeat=Path(job['heartbeat']) if job.get('heartbeat') else None
                    if heartbeat and heartbeat.exists():
                        mtime=heartbeat.stat().st_mtime_ns
                        if mtime!=last_mtime:last_change=now;last_mtime=mtime;stack_sent=False
                    silence=now-last_change
                    if job.get('stack_diagnostics') and silence>job.get('stack_after_seconds',120) and not stack_sent:
                        os.kill(process.pid,signal.SIGUSR1);stack_sent=True
                        row.setdefault('stack_signals',[]).append({'seconds':elapsed,'signal':'SIGUSR1'})
                    if silence>job.get('stall_seconds',300):raise TimeoutError(job['name']+' stopped reporting progress')
                    if now-last_print>30:
                        save();print(json.dumps({'job':job['name'],'seconds':round(elapsed,1),
                              'seconds_since_heartbeat':round(silence,1)}),flush=True);last_print=now
                    time.sleep(2)
                row.update(exit_code=process.returncode,seconds=time.monotonic()-active_start)
                used+=row['seconds'];active_start=None
                (record/(job['name']+'.exit')).write_text(str(process.returncode));save()
                if process.returncode:raise RuntimeError(job['name']+f' exited {process.returncode}')
            result_path=job.get('result_status')
            if result_path:
                result=json.loads(Path(result_path).read_text())
                if result['status'] not in job.get('acceptable_status',['complete']):
                    raise RuntimeError(f"Invalid result state: {result['status']}")
            stop();process=None
            print(json.dumps({'complete':row}),flush=True)
        state['status']='complete'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        stop()
        if process is not None and state['jobs']:
            state['jobs'][-1].setdefault('exit_code',process.returncode)
            if active_start is not None:
                state['jobs'][-1]['seconds']=time.monotonic()-active_start
        if active_start is not None:used+=time.monotonic()-active_start;active_start=None
        state['finished']=time.strftime('%Y-%m-%dT%H:%M:%S');save()


if __name__=='__main__':main()
