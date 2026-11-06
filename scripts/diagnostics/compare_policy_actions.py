#!/usr/bin/env python3
"""Compare actual policy actions and measured joints on shared evaluation seeds."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def actions(record):
    result=collections.defaultdict(list)
    with (record/'events.jsonl').open() as file:
        for line in file:
            row=json.loads(line)
            if row['event']=='action':result[int(row['seed'])].append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['left','right','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();left=actions(a.left);right=actions(a.right);rows=[]
    keys=['values','left_measured_after','right_measured_after']
    for seed in sorted(left.keys()&right.keys()):
        row={'seed':seed,'left_actions':len(left[seed]),'right_actions':len(right[seed]),
             'same_action_count':len(left[seed])==len(right[seed])}
        if row['same_action_count']:
            for key in keys:
                x=np.asarray([v[key] for v in left[seed]]);y=np.asarray([v[key] for v in right[seed]])
                assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
                row[key+'_max_difference']=float(np.abs(x-y).max())
        rows.append(row)
    result={'complete':True,'shared_seeds':len(rows),'only_left':sorted(left.keys()-right.keys()),
            'only_right':sorted(right.keys()-left.keys()),'per_seed':rows,
            'all_shared_action_counts_equal':all(v['same_action_count'] for v in rows),
            'scope':'Repeated common seeds are paired comparisons, not additional independent samples.'}
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='per_seed'}))


if __name__=='__main__':main()
