import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.collection_io import write_seed_list


class SeedCheckpointTest(unittest.TestCase):
    def test_native_bytes_and_empty_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'seed.txt'
            write_seed_list(path, [0, 2, 100037])
            self.assertEqual(path.read_bytes(), b'0 2 100037 ')
            self.assertEqual(list(path.parent.glob('.*.tmp')), [])
            write_seed_list(path, [])
            self.assertEqual(path.read_bytes(), b'')

    def test_failed_replace_preserves_old_checkpoint_and_recovery_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'seed.txt'
            write_seed_list(path, [0, 2])
            with patch('scripts.collection_io.os.replace', side_effect=OSError('injected replacement failure')):
                with self.assertRaises(OSError): write_seed_list(path, [0, 2, 7])
            self.assertEqual(path.read_bytes(), b'0 2 ')
            remaining = list(path.parent.glob('.*.tmp'))
            self.assertEqual(len(remaining), 1)
            self.assertEqual(remaining[0].read_bytes(), b'0 2 7 ')


if __name__ == '__main__': unittest.main()
