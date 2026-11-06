#!/usr/bin/env python3
"""Compare fixed candidate cohorts separately from conditional learner results."""
import argparse
import collections
import json
import pickle
from pathlib import Path
import numpy as np


def numerical_path_difference(left, right):
    result={'structure_equal':True,'max_position_difference':0.0,'max_velocity_difference':0.0,
            'compared_position_arrays':0,'compared_velocity_arrays':0,'different_array_shapes':0,
            'scope':'Pointwise differences only for equal-shaped arrays; unequal durations are recorded separately, without resampling.'}
    for arm in ('left_joint_path','right_joint_path'):
        if len(left[arm])!=len(right[arm]):result['structure_equal']=False;continue
        for a,b in zip(left[arm],right[arm]):
            if a.get('status')!=b.get('status'):result['structure_equal']=False
            for key in ('position','velocity'):
                if key not in a and key not in b:continue
                if key not in a or key not in b:result['structure_equal']=False;continue
                x=np.asarray(a[key]);y=np.asarray(b[key])
                if x.shape!=y.shape:
                    result['structure_equal']=False;result['different_array_shapes']+=1;continue
                assert np.isfinite(x).all() and np.isfinite(y).all()
                result[f'compared_{key}_arrays']+=1
                delta=float(np.max(np.abs(x-y))) if x.size else 0.0
                result[f'max_{key}_difference']=max(result[f'max_{key}_difference'],delta)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();r=args.run_dir.resolve()
    names=['baseline-1','reset-1','baseline-2','reset-2'];audits={name:json.loads((r/'checks'/f'{name}-audit.json').read_text()) for name in names}
    assert all(v['passed'] for v in audits.values());candidates={name:{c['seed']:c for c in a['candidate_results']} for name,a in audits.items()};trials={name:{c['seed']:c for c in a['trials']} for name,a in audits.items()}
    seq=set(candidates[names[0]]);assert all(set(candidates[name])==seq for name in names)
    pairs=[]
    for left,right in [('baseline-1','baseline-2'),('reset-1','reset-2'),('baseline-1','reset-1'),('baseline-2','reset-2')]:
        a=candidates[left];b=candidates[right];eligible_a=set(trials[left]);eligible_b=set(trials[right]);shared=eligible_a&eligible_b
        comparable_paths=set(audits[left]['expert_path_hashes'])&set(audits[right]['expert_path_hashes']);different_paths=[int(seed) for seed in comparable_paths if audits[left]['expert_path_hashes'][seed]!=audits[right]['expert_path_hashes'][seed]]
        numerical={}
        for seed in sorted(map(int,comparable_paths)):
            files=[r/'eval'/name/'expert_paths'/f'seed_{seed}.pkl' for name in (left,right)]
            numerical[seed]=numerical_path_difference(*[pickle.loads(p.read_bytes()) for p in files])
        pairs.append({'left':left,'right':right,'eligible_changes':sorted(eligible_a^eligible_b),'shared_eligible_count':len(shared),'learner_success_changes_on_shared':sorted(seed for seed in shared if trials[left][seed]['success']!=trials[right][seed]['success']),'learner_action_count_changes_on_shared':sorted(seed for seed in shared if trials[left][seed]['actions']!=trials[right][seed]['actions']),'path_hashes_compared':len(comparable_paths),'path_hash_changes':sorted(different_paths),'numerical_path_differences':numerical})
    summary={}
    for name,a in audits.items():
        counts=collections.Counter((c.get('model_id'),bool(c.get('admitted_by_native_expert_gate'))) for c in a['candidate_results'])
        summary[name]={'candidates':a['candidates'],'learner_trials':a['learner_trials'],'successes':a['successes'],'actions':a['actions'],'resets':a['controlled_resets'],'topp_errors':a['native_topp_errors'],'by_model':{f'base{model}:{"admitted" if admitted else "rejected"}':n for (model,admitted),n in counts.items()}}
    baseline=pairs[0];reset=pairs[1]
    result={'status':'complete','passed':True,'candidate_sequences_identical':True,'per_condition':summary,'pairwise':pairs,'baseline_replica_eligibility_changes':len(baseline['eligible_changes']),'reset_replica_eligibility_changes':len(reset['eligible_changes']),'reset_improved_observed_eligibility_repeatability':len(reset['eligible_changes'])<len(baseline['eligible_changes']),'note':'This is a bounded fixed-candidate audit with a fixed learner. Cohort changes must not be interpreted as a learning improvement; two repeats do not establish general determinism. Byte path hashes can differ for distinct valid solutions.'}
    (r/'checks/control-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
