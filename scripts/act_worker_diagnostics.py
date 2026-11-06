"""Optional worker-local fault traces without changing native sample values or RNG."""
from __future__ import annotations
import faulthandler
import json
import os
from pathlib import Path
import time

_FAULT_FILES = []


def probe_worker_ipc():
    """Exercise the actual local socket path before a long loader job starts."""
    from multiprocessing.connection import Listener
    with Listener(family='AF_UNIX') as listener:
        return {'tmpdir':os.environ.get('TMPDIR'), 'probe_socket':listener.address,
                'socket_path_bytes':len(os.fsencode(listener.address)), 'passed':True}


def initialize_worker(worker_id):
    """Runs after PyTorch installs its own worker signal handlers."""
    import numpy as np
    import torch
    root=Path(os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']);root.mkdir(parents=True,exist_ok=True)
    pid=os.getpid();trace=(root/f'worker-{pid}-fault.log').open('a',buffering=1)
    _FAULT_FILES.append(trace)
    # Re-register after the framework handler even when Python was launched -X
    # faulthandler; the framework may have replaced the OS signal disposition.
    faulthandler.disable();faulthandler.enable(file=trace,all_threads=True)
    info=torch.utils.data.get_worker_info();dataset=info.dataset
    original=type(dataset).__getitem__
    destination=root/f'worker-{pid}-last-sample.json'

    def observed(self,index):
        rng=np.random.get_state()
        row={'pid':pid,'worker_id':worker_id,'time':time.time(),'index':int(index),
             'episode_id':int(self.episode_ids[index]) if hasattr(self,'episode_ids') else None,
             'phase':'before_native_getitem',
             'numpy_rng_state':[rng[0],rng[1].tolist(),int(rng[2]),int(rng[3]),float(rng[4])]}
        temporary=destination.with_suffix('.tmp');temporary.write_text(json.dumps(row));temporary.replace(destination)
        value=original(self,index)
        row.update(phase='native_getitem_returned',returned_at=time.time())
        temporary.write_text(json.dumps(row));temporary.replace(destination)
        return value

    type(dataset).__getitem__=observed
    (root/f'worker-{pid}-init.json').write_text(json.dumps({'pid':pid,'worker_id':worker_id,
        'dataset_type':type(dataset).__module__+'.'+type(dataset).__name__,
        'python':os.sys.executable,'fault_trace':str(root/f'worker-{pid}-fault.log')},indent=2))
