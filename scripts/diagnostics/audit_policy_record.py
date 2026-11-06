#!/usr/bin/env python3
"""Audit recorded native expert/learner episodes and derived path snapshots."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    options = parser.parse_args(); config = json.loads(options.config.read_text())
    record = Path(config['record_dir']); state = json.loads((record/'status.json').read_text())
    expected = 'candidate_audit_complete' if config.get('candidate_audit') else 'complete'
    assert state['status'] == expected, state['status']
    assert not state['rpc_errors'] and not state['simulation_errors']
    assert state['actions_executed'] > 0
    if config.get('candidate_audit'):
        assert len(state['candidates']) == config['max_candidate_seeds']
        start = 100000 * (1 + int(config['native_args']['seed']))
        assert [v['seed'] for v in state['candidates']] == list(range(start, start+config['max_candidate_seeds']))
    else: assert len(state['trials']) == config['native_args']['test_num']
    actions = collections.Counter(); top_calls = collections.Counter(); actual_top_errors = 0
    final_predicates = {}; snapshots = {}; resets = 0
    with (record/'events.jsonl').open() as file:
        for line in file:
            event = json.loads(line); kind = event['event']; seed = event.get('seed')
            if kind == 'action':
                action = np.asarray(event['values']); assert action.shape == (14,) and np.isfinite(action).all()
                for arm in ('left','right'):
                    measured = np.asarray(event[f'{arm}_measured_after']); assert measured.shape == (6,) and np.isfinite(measured).all()
                actions[seed] += 1
            elif kind in ('native_topp','native_topp_error'):
                top_calls[(seed,event['arm'])] += 1
                if kind == 'native_topp_error': actual_top_errors += 1
                else: assert event['finite']
            elif kind == 'diagnostic_trial_end':
                final_predicates[seed] = event['metrics']['original_predicate_conjunction']
            elif kind == 'expert_path_snapshot':
                path = Path(event['path']); assert digest(path) == event['sha256']; snapshots[seed] = event['sha256']
            elif kind == 'controlled_planner_reset': resets += 1
    assert sum(actions.values()) == state['actions_executed']
    candidates = {c['seed']:c for c in state['candidates']}; trials = []
    for i, trial in enumerate(state['trials']):
        seed = trial['seed']; n = actions[seed]; assert 0 < n <= 400
        if not trial['success']: assert n == 400, (seed,n)
        c = candidates[seed]; assert c['expert_plan_success'] and c['expert_task_success']
        assert final_predicates[seed] == trial['success']
        for arm in ('left','right'): assert top_calls[(seed,arm)] == n
        path = Path(state['upstream_output'])/f'episode{i}.mp4'
        stream = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                   '-show_entries','stream=nb_frames,width,height','-of','json',str(path)],text=True))['streams'][0]
        assert (stream['width'],stream['height']) == (320,240)
        assert int(stream['nb_frames']) == n + int(trial['success'])
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        trials.append({'seed':seed,'success':trial['success'],'actions':n,'video':str(path),'video_sha256':digest(path)})
    assert len(actions) == len(state['trials'])
    if options.log:
        text = options.log.read_text(errors='replace')
        assert 'Policy rollout error' not in text and 'env error before first action' not in text
    result = {'passed':True,'result_kind':state.get('result_kind','native completed evaluation'),
              'candidates':len(state['candidates']),'learner_trials':len(trials),
              'successes':sum(v['success'] for v in trials),'actions':state['actions_executed'],
              'original_success_predicates_match':True,'videos_complete':True,
              'one_topp_call_per_action_per_arm':True,'native_topp_errors':actual_top_errors,
              'controlled_resets':resets,'expert_path_hashes':snapshots,'candidate_results':state['candidates'],
              'trials':trials,'latency_ms':state['latency_ms']}
    options.output.parent.mkdir(parents=True,exist_ok=True)
    options.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('expert_path_hashes','candidate_results','trials','latency_ms')}))


if __name__ == '__main__': main()
