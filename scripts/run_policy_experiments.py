#!/usr/bin/env python3
"""Run recorded native evaluations serially against one official policy server."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    if Path(sys.prefix) != Path(config['python_prefix']):
        raise RuntimeError(f'Unexpected Python: {sys.executable}')
    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    from XPolicyLab.utils.process_data import get_action_dim
    from scripts.eval_policy_server import server_is_ready
    record = Path(config['run_dir']).resolve(); record.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy(); env.update(config.get('thread_limits', {}))
    env['PYTHONFAULTHANDLER'] = '1'; env['ACT_ACTION_DIM'] = str(get_action_dim(config['env_cfg_type']))
    env['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    state = {'status': 'preparing', 'pid': os.getpid(), 'config': config,
             'python': sys.executable, 'jobs': [], 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
    server = child = None; server_start = None
    def save(): write_json(record / 'status.json', state)
    def stop(process):
        if process is None or process.poll() is not None: return
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()
    def emit(**values): print(json.dumps(values), flush=True)
    def progress_command():
        if not config.get('progress_command'): return
        with (record / 'report.log').open('a') as log:
            try: subprocess.run(config['progress_command'], stdout=log, stderr=subprocess.STDOUT,
                                timeout=20, env=env, cwd=ROOT)
            except Exception as error: emit(report_error=str(error))
    def terminate(_signal, _frame): raise KeyboardInterrupt('Supervisor shutdown requested')
    signal.signal(signal.SIGTERM, terminate)
    try:
        free = int(subprocess.check_output(['nvidia-smi', f'--id={config["gpu"]}',
                       '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
        if free < int(config.get('minimum_free_gpu_mib', 16000)):
            raise RuntimeError(f'Only {free} MiB GPU memory free; no unrelated process was stopped')
        if shutil.disk_usage(record).free < 3 * 1024**3: raise RuntimeError('Less than3GiB disk free')
        with socket.socket() as probe: probe.bind((config['host'], int(config['port'])))
        checkpoint = Path(config['checkpoint']).resolve()
        state['checkpoint_sha256'] = hashlib.sha256((checkpoint / 'policy_last.ckpt').read_bytes()).hexdigest()
        state['root_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        state['source_hashes'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
             ['scripts/eval_act_recorded.py', 'scripts/diagnostics/lift_pot_observer.py',
              'scripts/diagnostics/planner_audit_observer.py', 'envs/robot/robot.py', 'envs/_base_task.py']}
        command = [sys.executable, '-X', 'faulthandler', '-u', 'XPolicyLab/setup_policy_server.py',
                   '--config_path', config['policy_config'], '--overrides',
                   f'port={config["port"]}', f'host={config["host"]}', 'bench_name=RoboTwin',
                   'task_name=lift_pot', f'ckpt_name={checkpoint}', f'ckpt_dir={checkpoint}',
                   f'env_cfg_type={config["env_cfg_type"]}', 'seed=0', 'policy_name=ACT',
                   'action_type=joint', f'action_dim={env["ACT_ACTION_DIM"]}']
        with (record / 'server.log').open('xb') as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                      start_new_session=True)
        server_start = time.monotonic(); state.update(server_pid=server.pid, server_command=command); save()
        for _ in range(180):
            if server.poll() is not None: raise RuntimeError(f'Server exited {server.returncode}')
            if server_is_ready(config['host'], int(config['port'])): break
            time.sleep(1)
        else: raise TimeoutError('Policy server did not become ready')
        for experiment in config['experiments']:
            job_config = json.loads(Path(experiment['config']).read_text())
            if Path(job_config['record_dir']).exists(): raise FileExistsError(job_config['record_dir'])
            command = [sys.executable, '-X', 'faulthandler', '-u', 'scripts/eval_act_recorded.py',
                       '--config', experiment['config']]
            with (record / f'{experiment["name"]}.log').open('xb') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                job = {'name': experiment['name'], 'pid': child.pid, 'command': command,
                       'record_dir': job_config['record_dir'], 'started': time.strftime('%Y-%m-%dT%H:%M:%S')}
                state['jobs'].append(job); state['status'] = 'running'; save(); emit(start=job)
                started = time.monotonic(); last_progress = started; last_report = 0
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed > float(experiment['timeout_seconds']): raise TimeoutError(experiment['name'])
                    if time.monotonic() - server_start > float(config['gpu_budget_seconds']):
                        raise TimeoutError('Authorized GPU budget exhausted')
                    if server.poll() is not None: raise RuntimeError('Policy server exited during evaluation')
                    if shutil.disk_usage(record).free < 1.5 * 1024**3: raise RuntimeError('Disk free space below1.5GiB')
                    if time.monotonic() - last_progress > 30:
                        p = Path(job_config['record_dir']) / 'status.json'
                        try:
                            current = json.loads(p.read_text()); emit(job=experiment['name'],
                                  candidates=len(current['candidates']), trials=len(current['trials']),
                                  successes=sum(bool(t['success']) for t in current['trials']),
                                  actions=current['actions_executed'], seconds=round(elapsed,1))
                        except (FileNotFoundError, json.JSONDecodeError): pass
                        save(); last_progress = time.monotonic()
                    if time.monotonic() - last_report > 60:
                        progress_command(); last_report = time.monotonic()
                    time.sleep(2)
            job.update(exit_code=child.returncode, seconds=time.monotonic()-started)
            (record / f'{experiment["name"]}.exit').write_text(str(child.returncode)); save()
            if child.returncode: raise RuntimeError(f'{experiment["name"]} exit {child.returncode}')
            result = json.loads((Path(job_config['record_dir'])/'status.json').read_text())
            if result['status'] not in ('complete','candidate_audit_complete'):
                raise RuntimeError(f'Unexpected child status: {result["status"]}')
            emit(completed=experiment['name'], status=result['status'], candidates=len(result['candidates']),
                 trials=len(result['trials']), successes=result['successes']); progress_command()
        state['status'] = 'complete'
    except BaseException:
        state.update(status='failed', error=traceback.format_exc()); raise
    finally:
        stop(child); stop(server)
        state.update(server_exit=server.returncode if server else None,
                     gpu_occupied_seconds=time.monotonic()-server_start if server_start else 0,
                     finished=time.strftime('%Y-%m-%dT%H:%M:%S'))
        save(); progress_command()


if __name__ == '__main__': main()
