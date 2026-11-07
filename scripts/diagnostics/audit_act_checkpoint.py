#!/usr/bin/env python3
"""Reload one recorded ACT checkpoint on CPU without interrupting training."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path);a=p.parse_args();cfg=json.loads(a.config.read_text())
    state=read_training_status(Path(cfg['record_dir'])/'status.json')
    if state['status']=='complete' and a.updates==state['updates']:
        entry={'path':state['checkpoint'],'sha256':state['checkpoint_sha256'],
               'epoch_index':len(state['epochs'])-1}
    else:entry=next(v for v in state['update_checkpoints'] if v['updates']==a.updates)
    path=Path(entry['path']);sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==entry['sha256']
    weights=torch.load(path,map_location='cpu',weights_only=True)
    assert weights and all(torch.isfinite(v).all() for v in weights.values())
    assert weights['model.action_head.weight'].shape[0]==cfg['action_dim']==14
    stats=pickle.loads((Path(cfg['checkpoint_dir'])/'dataset_stats.pkl').read_bytes())
    for key in ['qpos_mean','qpos_std','action_mean','action_std']:
        v=np.asarray(stats[key]);assert v.shape==(14,) and np.isfinite(v).all()
        if key.endswith('_std'):assert (v>0).all()
    reference_matches=None
    if a.reference_dir:
        old=torch.load(a.reference_dir/'policy_last.ckpt',map_location='cpu',weights_only=True)
        assert weights.keys()==old.keys()
        assert all(v.shape==old[k].shape and v.dtype==old[k].dtype for k,v in weights.items())
        old_stats=pickle.loads((a.reference_dir/'dataset_stats.pkl').read_bytes())
        assert all(np.array_equal(stats[k],old_stats[k]) for k in ['qpos_mean','qpos_std','action_mean','action_std'])
        reference_matches=True
    result={'passed':True,'updates':a.updates,'epoch_index':entry['epoch_index'],'checkpoint':str(path),
            'checkpoint_bytes':path.stat().st_size,'checkpoint_sha256':sha,'state_dict_tensors':len(weights),
            'all_tensors_finite':True,'state_action_dimension':14,'normalization_finite_positive_std':True,
            'reference_architecture_and_normalization_match':reference_matches,
            'training_status_at_audit':state['status'],
            'scope':'CPU reload and checkpoint contract only. Long trajectory completion and real policy rollouts remain separate validation gates.'}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
