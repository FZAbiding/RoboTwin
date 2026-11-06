#!/usr/bin/env python3
"""Read only this study's live trainer/worker CPU and memory diagnostics."""
import argparse
import datetime
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    r=p.parse_args().run_dir.resolve();output=r/'checks/worker-resources.jsonl'
    if output.exists():raise FileExistsError(output)
    status_path=r/'checks/worker-monitor-status.json';starts={};count=0
    def read(path):return json.loads(path.read_text())
    def sample(pid,parent=None):
        proc=Path('/proc')/str(pid);status={}
        for line in (proc/'status').read_text().splitlines():
            if ':' in line:
                k,v=line.split(':',1);status[k]=v.strip()
        if parent is not None and int(status['PPid'])!=parent:return {'pid':pid,'skipped':'no longer owned by this trainer'}
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split();start=stat[19]
        if pid in starts and starts[pid]!=start:return {'pid':pid,'skipped':'PID reused'}
        starts[pid]=start
        return {'pid':pid,'ppid':int(status['PPid']),'start_ticks':start,'last_cpu':int(stat[36]),
                **{k:status.get(k) for k in ['Name','State','VmRSS','VmSize','VmSwap','Threads','Cpus_allowed_list']},
                'open_fds':len(list((proc/'fd').iterdir()))}
    while True:
        supervisor=read(r/'supervisor/status.json');rows=[]
        for job in supervisor['jobs']:
            if 'exit_code' in job:continue
            name=job['name'];training=r/'training'/name.removeprefix('train-')/'status.json'
            if not training.exists():continue
            state=read(training)
            try:rows.append({'role':name,**sample(job['pid'])})
            except FileNotFoundError:continue
            if state['epochs']:
                for role,pids in state['epochs'][-1]['loader_worker_pids'].items():
                    for pid in pids:
                        try:rows.append({'role':name+':'+role,**sample(pid,job['pid'])})
                        except FileNotFoundError:pass
        now=datetime.datetime.now().isoformat(timespec='seconds')
        with output.open('a') as f:f.write(json.dumps({'time':now,'supervisor_status':supervisor['status'],'processes':rows})+'\n')
        count+=1;status_path.write_text(json.dumps({'status':'running' if supervisor['status']=='running' else 'complete',
            'pid':os.getpid(),'samples':count,'updated':now,'training_supervisor_final_state':supervisor['status']},indent=2))
        if supervisor['status'] in ['complete','failed']:break
        time.sleep(5)


if __name__=='__main__':main()
