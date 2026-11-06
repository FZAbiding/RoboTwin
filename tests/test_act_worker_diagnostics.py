"""Prove that a native worker fault leaves its Python frame and last sample."""
import json
import os
from pathlib import Path
import signal
import tempfile
import unittest
import torch
from scripts.act_worker_diagnostics import initialize_worker


class FaultingDataset(torch.utils.data.Dataset):
    def __len__(self):return 1
    def __getitem__(self,index):
        os.kill(os.getpid(),signal.SIGSEGV)


class RandomDataset(torch.utils.data.Dataset):
    def __len__(self):return 16
    def __getitem__(self,index):
        import numpy as np
        return int(np.random.choice(127)),torch.randn(2)


class DiagnosticsTest(unittest.TestCase):
    def test_observer_preserves_worker_random_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=os.environ.get('ROBOTWIN_ACT_WORKER_DIAGNOSTICS')
            os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']=folder
            try:
                batches=[]
                for initializer in [None,initialize_worker]:
                    loader=torch.utils.data.DataLoader(RandomDataset(),batch_size=4,num_workers=1,
                        multiprocessing_context='spawn',worker_init_fn=initializer,
                        generator=torch.Generator().manual_seed(2026))
                    batches.append(list(loader))
                for plain,observed in zip(*batches):
                    for x,y in zip(plain,observed):self.assertTrue(torch.equal(x,y))
            finally:
                if previous is None:os.environ.pop('ROBOTWIN_ACT_WORKER_DIAGNOSTICS',None)
                else:os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']=previous

    def test_real_worker_signal_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=os.environ.get('ROBOTWIN_ACT_WORKER_DIAGNOSTICS')
            os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']=folder
            try:
                loader=torch.utils.data.DataLoader(FaultingDataset(),num_workers=1,
                    multiprocessing_context='spawn',worker_init_fn=initialize_worker,timeout=20)
                with self.assertRaises(RuntimeError):next(iter(loader))
                traces=list(Path(folder).glob('*-fault.log'));self.assertEqual(len(traces),1)
                self.assertIn('test_act_worker_diagnostics.py',traces[0].read_text())
                sample=json.loads(next(Path(folder).glob('*-last-sample.json')).read_text())
                self.assertEqual(sample['phase'],'before_native_getitem')
                self.assertEqual(sample['index'],0)
            finally:
                if previous is None:os.environ.pop('ROBOTWIN_ACT_WORKER_DIAGNOSTICS',None)
                else:os.environ['ROBOTWIN_ACT_WORKER_DIAGNOSTICS']=previous


if __name__=='__main__':unittest.main()
