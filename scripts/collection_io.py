"""Durable checkpoint writes used by the native collector."""
from pathlib import Path
import os
import tempfile


def write_seed_list(path, seeds):
    """Atomically replace a seed list; preserve the old file if replacement fails.

    A failed temporary file is retained for recovery. Successful bytes keep the
    native collector's existing whitespace format.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False,
            dir=destination.parent, prefix=f'.{destination.name}.', suffix='.tmp') as file:
        for seed in seeds:
            file.write(f'{int(seed)} ')
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    os.replace(temporary, destination)
