#!/usr/bin/env python3
"""Verify completed native learning trajectories and unchanged source data."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.training_record import read_training_status


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--training-seed',type=int,choices=[0,1],action='append')
    options=p.parse_args();seeds=options.training_seed or [0,1]
    if len(seeds)!=len(set(seeds)):p.error('Training seeds must be distinct')
    r=options.run_dir.resolve();rows=[];reference=None;norm_reference=None
    for seed in seeds:
        cfg=json.loads((r/'configs'/f'train-seed-{seed}.json').read_text())
        state=read_training_status(Path(cfg['record_dir'])/'status.json')
        assert state['status']=='complete' and state['updates']==18000 and len(state['epochs'])==6000
        split=state['split'];signature={k:split[k] for k in ['train_episode_ids','validation_episode_ids','normalization_episode_ids','stats']}
        if reference is None:reference=signature
        else:assert signature==reference
        stats=pickle.loads((Path(cfg['checkpoint_dir'])/'dataset_stats.pkl').read_bytes())
        norm={k:np.asarray(stats[k]) for k in ['action_mean','action_std','qpos_mean','qpos_std']}
        if norm_reference is None:norm_reference=norm
        else:
            for k,v in norm.items():assert np.array_equal(v,norm_reference[k])
        checkpoints=state['update_checkpoints']+[{'updates':18000,'path':state['checkpoint'],'sha256':state['checkpoint_sha256']}]
        assert sorted(v['updates'] for v in checkpoints)==[3000,9000,18000]
        for row in checkpoints:
            assert digest(row['path'])==row['sha256']
            weights=torch.load(row['path'],map_location='cpu',weights_only=True)
            assert all(bool(torch.isfinite(v).all()) for v in weights.values())
            rows.append({'training_seed':seed,**row,'all_tensors_finite':True});del weights
        for epoch in state['epochs']:
            assert all(np.isfinite(v) for group in ['train','validation'] for v in epoch[group].values())
        assert all(epoch['loader_worker_pids']==state['epochs'][0]['loader_worker_pids'] for epoch in state['epochs'])
    raw=json.loads(Path(cfg['source_manifest']).read_text())
    for row in raw:assert digest(row['hdf5'])==row['hdf5_sha256']
    converted_path=Path(cfg['source_manifest']).parent.parent/'checks/converted-manifest.json'
    converted=json.loads(converted_path.read_text())
    for row in converted:assert digest(row['file'])==row['sha256']
    result={'passed':True,'training_seeds_checked':seeds,'checkpoints':rows,
            'same_training_validation_split':True if len(seeds)>1 else None,
            'same_normalization':True if len(seeds)>1 else None,'worker_pids_stable':True,'losses_finite':True,
            'unchanged_raw_hdf5':len(raw),'unchanged_derived_hdf5':len(converted),
            'scope':'Engineering validation only; policy success requires real closed-loop evaluation.'}
    name='training-audit.json' if len(seeds)>1 else f'training-seed-{seeds[0]}-audit.json'
    (r/'checks'/name).write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
