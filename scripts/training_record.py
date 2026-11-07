"""Versioned training status with an append-only epoch history.

The compact format keeps growing history out of the per-epoch status rewrite.
Legacy embedded histories remain readable without changing their artifacts.
"""
import json
from pathlib import Path

COMPACT_FORMAT='epochs-jsonl-v1'


def compact_training_status(state):
    epochs=state['epochs']
    return {**{k:v for k,v in state.items() if k!='epochs'},
            'training_record_format':COMPACT_FORMAT,'epochs_file':'epochs.jsonl',
            'epochs_completed':len(epochs),'last_epoch':epochs[-1] if epochs else None}


def read_epoch_history(path,require_complete=False):
    path=Path(path)
    if not path.exists():return []
    raw=path.read_text();lines=raw.splitlines(keepends=True);rows=[]
    for index,line in enumerate(lines):
        if not line.endswith('\n'):
            if index!=len(lines)-1:raise ValueError('Incomplete epoch record inside history')
            if require_complete:raise ValueError('Completed history has an incomplete final record')
            break
        row=json.loads(line)
        if row['epoch']!=len(rows):raise ValueError('Epoch history is not consecutive')
        rows.append(row)
    return rows


def read_training_status(path):
    path=Path(path);state=json.loads(path.read_text())
    if 'epochs' in state:return state
    if state.get('training_record_format')!=COMPACT_FORMAT or state.get('epochs_file')!='epochs.jsonl':
        raise ValueError('Unsupported training record format')
    rows=read_epoch_history(path.parent/'epochs.jsonl',require_complete=state['status']=='complete');count=int(state['epochs_completed'])
    if count<0 or len(rows)<count:raise ValueError('Status refers to missing epoch records')
    if state['status']=='complete' and len(rows)!=count:
        raise ValueError('Completed status and epoch history disagree')
    state['epochs']=rows[:count];state['history_rows_available']=len(rows)
    return state
