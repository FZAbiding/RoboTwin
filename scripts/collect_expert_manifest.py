#!/usr/bin/env python3
"""Freeze eligibility by observing the original expert collector, without replay.

This manifest is an expert-conditioned evaluation cohort, not training data.
The original collector alone decides acceptance and saves its original paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    if Path(sys.prefix) != Path(config['python_prefix']):
        raise RuntimeError(f'Unexpected Python: {sys.executable}')
    record = Path(config['record_dir']).resolve(); record.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT); sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT/'scripts'))
    import yaml
    import scripts.collect_data as native
    from scripts.diagnostics.lift_pot_observer import Observer, plain
    task_config_path = ROOT/'env_cfg/task_config'/f"{config['task_config']}.yml"
    task_config = yaml.safe_load(task_config_path.read_text())
    if task_config['collect_data'] or task_config['use_seed']:
        raise ValueError('Use a fresh expert-only configuration, without replay or preselected seeds')
    expected = int(task_config['episode_num'])
    source_paths = ['scripts/collect_data.py', 'scripts/collection_io.py',
                    f"envs/{config['task_name']}.py", 'envs/_base_task.py', 'envs/robot/robot.py',
                    'envs/robot/planner.py', 'env_cfg/task_config/_eval_step_limit.yml']
    if config['task_name'] != 'lift_pot':
        raise ValueError('The initial-state audit is scoped to lift_pot')
    source_paths += [str(p.relative_to(ROOT)) for p in
                     sorted((ROOT/'assets/objects/060_kitchenpot').rglob('*')) if p.is_file()]
    state = {'status':'running', 'pid':os.getpid(), 'config':config,
             'task_config':task_config, 'task_config_sha256':digest(task_config_path),
             'source_hashes':{p:digest(ROOT/p) for p in source_paths},
             'candidates':[], 'accepted':[], 'started':time.strftime('%Y-%m-%dT%H:%M:%S'),
             'eligibility':'original collector: plan success AND physical task success AND planned joints legal',
             'scope':'expert-conditioned evaluation cohort; no learner results or training data',
             'reset_variant':False}
    original_factory = native.class_decorator
    metrics = Observer(lambda *a, **k:None)
    started = time.monotonic()

    def event(kind, **fields):
        with (record/'events.jsonl').open('a') as file:
            file.write(json.dumps(plain({'event':kind, 'time':time.time(), **fields}))+'\n')

    def factory(task_name):
        task = original_factory(task_name)
        setup, play, check = task.setup_demo, task.play_once, task.check_success
        legal, save = task.planned_joints_legal, task.save_traj_data
        candidate = None

        def setup_demo(*a, **kw):
            nonlocal candidate
            candidate = {'seed':int(kw['seed']), 'episode_index':int(kw['now_ep_num'])}
            state['candidates'].append(candidate)
            if any(row['seed']==candidate['seed'] for row in state['candidates'][:-1]):
                raise RuntimeError('Unexpected repeated candidate')
            try:
                value = setup(*a, **kw)
                candidate.update(model_id=int(task.model_id), model_name=task.model_name,
                                 initial_metrics=metrics.metrics(task))
                event('expert_initial', **candidate)
                return value
            except BaseException as error:
                candidate['setup_error'] = f'{type(error).__name__}: {error}'; raise
            finally: write_json(record/'status.json', state)

        def play_once(*a, **kw):
            try:
                value = play(*a, **kw)
                candidate.update(expert_plan_success=bool(task.plan_success),
                                 final_metrics=metrics.metrics(task), episode_info=plain(value))
                event('expert_end', **candidate)
                return value
            except BaseException as error:
                candidate['expert_error'] = f'{type(error).__name__}: {error}'
                event('expert_error', **candidate); raise
            finally: write_json(record/'status.json', state)

        def check_success(*a, **kw):
            value = check(*a, **kw); candidate['expert_task_success'] = bool(value)
            return value

        def planned_joints_legal(*a, **kw):
            value = legal(*a, **kw)
            candidate.update(planned_joints_legal=bool(value[0]), joint_limit_excess=float(value[1]))
            return value

        def save_traj_data(index):
            if not all(candidate.get(k) for k in ['expert_plan_success','expert_task_success','planned_joints_legal']):
                raise RuntimeError('Collector accepted an unverified expert candidate')
            value = save(index)
            path = Path(task.save_dir)/'_traj_data'/f'episode{index}.pkl'
            candidate.update(path=str(path.resolve()), path_sha256=digest(path), accepted=True)
            state['accepted'].append(dict(candidate))
            event('accepted', **candidate); write_json(record/'status.json', state)
            return value

        task.setup_demo, task.play_once, task.check_success = setup_demo, play_once, check_success
        task.planned_joints_legal, task.save_traj_data = planned_joints_legal, save_traj_data
        return task

    native.class_decorator = factory
    write_json(record/'status.json', state)
    try:
        native.main(config['task_name'], config['task_config'])
        accepted = state['accepted']; assert len(accepted)==expected
        for row in accepted:
            assert digest(row['path'])==row['path_sha256']
            assert row['final_metrics']['original_predicate_conjunction']
        seed_path = Path(accepted[0]['path']).parent.parent/'seed.txt'
        assert [int(s) for s in seed_path.read_text().split()]==[row['seed'] for row in accepted]
        for p, sha in state['source_hashes'].items(): assert digest(ROOT/p)==sha
        state.update(status='complete', seed_path=str(seed_path), seed_sha256=digest(seed_path))
        manifest = {k:state[k] for k in ['scope','eligibility','task_config','task_config_sha256',
                    'source_hashes','reset_variant','accepted','seed_path','seed_sha256']}
        manifest.update(status='frozen', task_name=config['task_name'], frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
        write_json(record/'manifest.json', manifest)
        state['manifest_sha256'] = digest(record/'manifest.json')
    except BaseException:
        state.update(status='failed', error=traceback.format_exc()); raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        write_json(record/'status.json', state)


if __name__=='__main__': main()
