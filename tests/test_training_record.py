import json
from pathlib import Path
import tempfile
import unittest
from scripts.training_record import compact_training_status,read_training_status


class TrainingRecordTest(unittest.TestCase):
    def fixture(self,root,status='running'):
        epochs=[{'epoch':i,'updates_completed':3*(i+1),'validation':{'loss':1/(i+1)}} for i in range(3)]
        state={'status':status,'epochs':epochs,'updates':9,'split':{'train_episode_ids':[0,1]}}
        (root/'epochs.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in epochs))
        return state

    def test_compact_roundtrip_preserves_history_and_model_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.fixture(root,status='complete');path=root/'status.json'
            compact=compact_training_status(state);self.assertNotIn('epochs',compact)
            path.write_text(json.dumps(compact));loaded=read_training_status(path)
            for key,value in state.items():self.assertEqual(loaded[key],value)
            path.write_text(json.dumps(state));self.assertEqual(read_training_status(path),state)

    def test_torn_final_line_is_not_published_as_an_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.fixture(root);path=root/'status.json'
            path.write_text(json.dumps(compact_training_status(state)))
            with (root/'epochs.jsonl').open('a') as f:f.write('{"epoch":3')
            self.assertEqual(len(read_training_status(path)['epochs']),3)

    def test_missing_completed_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.fixture(root,status='complete');path=root/'status.json'
            path.write_text(json.dumps(compact_training_status(state)))
            (root/'epochs.jsonl').write_text('')
            with self.assertRaises(ValueError):read_training_status(path)

    def test_completed_history_with_partial_extra_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);state=self.fixture(root,status='complete');path=root/'status.json'
            path.write_text(json.dumps(compact_training_status(state)))
            with (root/'epochs.jsonl').open('a') as f:f.write('{"epoch":3')
            with self.assertRaises(ValueError):read_training_status(path)


if __name__=='__main__':unittest.main()
