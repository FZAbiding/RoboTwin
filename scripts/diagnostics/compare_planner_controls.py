#!/usr/bin/env python3
"""Compare fixed candidate cohorts separately from conditional learner results."""
import argparse
import collections
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();r=args.run_dir.resolve()
    names=['baseline-1','reset-1','baseline-2','reset-2'];audits={name:json.loads((r/'checks'/f'{name}-audit.json').read_text()) for name in names}
    assert all(v['passed'] for v in audits.values());candidates={name:{c['seed']:c for c in a['candidate_results']} for name,a in audits.items()};trials={name:{c['seed']:c for c in a['trials']} for name,a in audits.items()}
    seq=set(candidates[names[0]]);assert all(set(candidates[name])==seq for name in names)
    pairs=[]
    for left,right in [('baseline-1','baseline-2'),('reset-1','reset-2'),('baseline-1','reset-1'),('baseline-2','reset-2')]:
        a=candidates[left];b=candidates[right];eligible_a=set(trials[left]);eligible_b=set(trials[right]);shared=eligible_a&eligible_b
        comparable_paths=set(audits[left]['expert_path_hashes'])&set(audits[right]['expert_path_hashes']);different_paths=[int(seed) for seed in comparable_paths if audits[left]['expert_path_hashes'][seed]!=audits[right]['expert_path_hashes'][seed]]
        pairs.append({'left':left,'right':right,'eligible_changes':sorted(eligible_a^eligible_b),'shared_eligible_count':len(shared),'learner_success_changes_on_shared':sorted(seed for seed in shared if trials[left][seed]['success']!=trials[right][seed]['success']),'learner_action_count_changes_on_shared':sorted(seed for seed in shared if trials[left][seed]['actions']!=trials[right][seed]['actions']),'path_hashes_compared':len(comparable_paths),'path_hash_changes':sorted(different_paths)})
    summary={}
    for name,a in audits.items():
        counts=collections.Counter((c.get('model_id'),bool(c.get('admitted_by_native_expert_gate'))) for c in a['candidate_results'])
        summary[name]={'candidates':a['candidates'],'learner_trials':a['learner_trials'],'successes':a['successes'],'actions':a['actions'],'resets':a['controlled_resets'],'topp_errors':a['native_topp_errors'],'by_model':{f'base{model}:{"admitted" if admitted else "rejected"}':n for (model,admitted),n in counts.items()}}
    baseline=pairs[0];reset=pairs[1]
    result={'status':'complete','passed':True,'candidate_sequences_identical':True,'per_condition':summary,'pairwise':pairs,'baseline_replica_eligibility_changes':len(baseline['eligible_changes']),'reset_replica_eligibility_changes':len(reset['eligible_changes']),'reset_improved_observed_eligibility_repeatability':len(reset['eligible_changes'])<len(baseline['eligible_changes']),'note':'This is a bounded fixed-candidate audit with a fixed learner. Cohort changes must not be interpreted as a learning improvement; two repeats do not establish general determinism. Byte path hashes can differ for distinct valid solutions.'}
    (r/'checks/control-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
