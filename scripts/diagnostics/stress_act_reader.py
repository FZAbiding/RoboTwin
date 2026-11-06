#!/usr/bin/env python3
"""Read real ACT batches repeatedly in an instrumented spawned worker, without CUDA."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--record-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=800);a=p.parse_args()
    r=a.record_dir.resolve();r.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'XPolicyLab/policy/ACT'))
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from utils import load_data
    from scripts.act_worker_diagnostics import initialize_worker
    torch.set_num_threads(1);np.random.seed(1);torch.manual_seed(0)
    train,validation,_,_=load_data(str(a.dataset_dir.resolve()),50,
        ['cam_head','cam_right_wrist','cam_left_wrist'],16,16)
    os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']=str(r/'workers')
    loader=DataLoader(train.dataset,batch_size=train.batch_size,sampler=train.sampler,
        num_workers=1,pin_memory=False,prefetch_factor=1,persistent_workers=True,
        multiprocessing_context='spawn',worker_init_fn=initialize_worker,timeout=60)
    state={'status':'running','pid':os.getpid(),'epochs':0,'batches':0,'samples':0,
           'scope':'Real native dataset read/normalization/worker transport; no GPU training or policy success claim.'}
    started=time.monotonic()
    def save():
        temp=r/'status.json.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(r/'status.json')
    save()
    try:
        for epoch in range(a.epochs):
            for images,qpos,actions,padding in loader:
                assert images.shape[1:]==(3,3,480,640)
                assert qpos.shape[1:]==(14,) and actions.shape[1:]==(127,14)
                assert padding.shape[1:]==(127,) and padding.dtype==torch.bool
                assert torch.isfinite(images).all() and torch.isfinite(qpos).all() and torch.isfinite(actions).all()
                assert images.min()>=0 and images.max()<=1
                state['samples']+=len(images);state['batches']+=1
            state['epochs']=epoch+1;state['elapsed_seconds']=time.monotonic()-started;save()
            if epoch%50==0:print(json.dumps(state),flush=True)
        state['status']='complete'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:state['elapsed_seconds']=time.monotonic()-started;save()


if __name__=='__main__':main()
