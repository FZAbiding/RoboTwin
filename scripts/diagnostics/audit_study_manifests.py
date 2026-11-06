#!/usr/bin/env python3
"""Audit every frozen expert proof and its future evaluation configuration."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.frozen_expert_cohort import FrozenCohort


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);args=parser.parse_args();r=args.run_dir.resolve()
    training=json.loads((r/'configs/train-seed-0.json').read_text())
    source=json.loads(Path(training['source_manifest']).read_text());data_seeds={int(v['seed']) for v in source}
    rows=[];seen=set()
    for name in ['comparison','confirmation']:
        cohort=FrozenCohort(r/'manifests'/name/'manifest.json',ROOT,data_seeds)
        assert len(cohort.rows)==100 and not seen&cohort.by_seed.keys();seen.update(cohort.by_seed)
        cfg=json.loads((r/'configs'/f'{name}.json').read_text())
        eval_yaml=ROOT/'env_cfg/task_config'/f"{cfg['task_config']}_eval.yml"
        task=yaml.safe_load(eval_yaml.read_text());expected=cohort.manifest['task_config']
        for key in ['embodiment','domain_randomization','camera','data_type','eval_mode']:
            assert task.get(key)==expected.get(key),(name,key)
        assert task['eval_video_log'] is True
        arrays=0;waypoints=0
        for proof in cohort.rows:
            assert proof['final_metrics']['original_predicate_conjunction']
            paths=pickle.loads(Path(proof['path']).read_bytes())
            for arm in ['left_joint_path','right_joint_path']:
                assert paths[arm]
                for segment in paths[arm]:
                    assert segment['status']=='Success'
                    for key in ['position','velocity']:
                        values=np.asarray(segment[key]);assert values.ndim==2 and values.shape[1]==6
                        assert len(values)>0 and np.isfinite(values).all();arrays+=1
                        if key=='position':waypoints+=len(values)
        rows.append({'cohort':name,'manifest_sha256':cohort.sha256,'accepted':100,
                     'first_seed':cohort.rows[0]['seed'],'last_seed':cohort.rows[-1]['seed'],
                     'by_asset':dict(collections.Counter(v['model_id'] for v in cohort.rows)),
                     'finite_path_arrays_checked':arrays,'stored_arm_waypoints_checked':waypoints,
                     'evaluation_yaml':str(eval_yaml),'evaluation_yaml_sha256':hashlib.sha256(eval_yaml.read_bytes()).hexdigest()})
    result={'passed':True,'training_seed_count':len(data_seeds),'cohort_seed_count':len(seen),
            'train_comparison_confirmation_disjoint':True,'expert_proofs_and_paths_unchanged':True,
            'finite_native_expert_paths':True,'future_evaluation_scene_configs_match':True,
            'cohorts':rows,'scope':'Proof and configuration audit, not a replacement for future learner physics rollouts.'}
    (r/'checks/manifests-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
