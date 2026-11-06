"""Validate a frozen expert cohort and reuse the native evaluator one seed at a time."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


class FrozenCohortViolation(BaseException):
    """Abort native retry logic if a frozen evaluation contract is violated."""


class FrozenCohort:
    def __init__(self, path, root, data_seeds):
        self.path=Path(path).resolve();self.manifest=json.loads(self.path.read_text())
        if self.manifest['status']!='frozen':raise ValueError('Expert cohort is not frozen')
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.rows=self.manifest['accepted'];self.by_seed={int(r['seed']):r for r in self.rows}
        if len(self.rows)!=len(self.by_seed) or not self.rows:raise ValueError('Invalid cohort seed list')
        if set(self.by_seed)&data_seeds:raise ValueError('Frozen evaluation seeds overlap training data')
        for p,sha in self.manifest['source_hashes'].items():
            if hashlib.sha256((root/p).read_bytes()).hexdigest()!=sha:
                raise ValueError(f'Expert source or asset changed: {p}')
        seed_file=Path(self.manifest['seed_path'])
        if hashlib.sha256(seed_file.read_bytes()).hexdigest()!=self.manifest['seed_sha256']:
            raise ValueError('Original expert seed file changed')
        if [int(s) for s in seed_file.read_text().split()]!=list(self.by_seed):
            raise ValueError('Expert seed list differs from the frozen cohort')
        for row in self.rows:
            if not all(row.get(k) for k in ['expert_plan_success','expert_task_success','planned_joints_legal','accepted']):
                raise ValueError('Unverified expert accepted into cohort')
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['path_sha256']:
                raise ValueError('Frozen expert path changed')
        self.expected_seed=None
        self.cache_schedule=[]

    def validate_setup(self,environment,seed,metrics):
        import numpy as np
        if seed!=self.expected_seed or seed not in self.by_seed:
            raise FrozenCohortViolation(f'Native evaluator attempted an unexpected seed: {seed}')
        row=self.by_seed[seed]
        if int(environment.model_id)!=row['model_id'] or environment.model_name!=row['model_name']:
            raise FrozenCohortViolation('Frozen scene asset differs')
        current=metrics(environment);reference=row['initial_metrics'];maximum=0.0
        for key in ['pot_pose','contact_world_xyz','tcp_world_xyz','gripper_commands']:
            pairs=((current[key]['p'],reference[key]['p']),(current[key]['q'],reference[key]['q'])) if key=='pot_pose' else [(current[key],reference[key])]
            for a,b in pairs:
                difference=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
                if not np.isfinite(difference) or difference>1e-8:
                    raise FrozenCohortViolation(f'Frozen initial state changed: {key} difference={difference}')
                maximum=max(maximum,difference)
        return maximum

    def install(self,native,total_trials):
        if total_trials!=len(self.rows):raise ValueError('Evaluate every seed in the frozen cohort exactly once')
        original=native.eval_remote_policy;instruction=native.build_instruction

        def build_instruction(args,episode_info,instruction_type,test_num,seed):
            return instruction(args,self.by_seed[int(seed)]['episode_info'],instruction_type,total_trials,seed)

        def evaluate(task_name,task_env,args,model_client,st_seed,*,usr_args,test_num,
                     video_size=None,instruction_type=None):
            if task_name!=self.manifest['task_name']:
                raise ValueError('Frozen cohort belongs to a different task')
            if usr_args.get('expert_check') is not False:
                raise ValueError('Frozen evaluation uses its preverified expert proof, without a new expert gate')
            expected=self.manifest['task_config']
            for key in ['embodiment','domain_randomization','camera','data_type','eval_mode']:
                if args.get(key)!=expected.get(key):raise ValueError(f'Frozen scene configuration differs: {key}')
            base=Path(args['eval_video_save_dir']);successes=0
            for ordinal,row in enumerate(self.rows):
                seed=int(row['seed']);self.expected_seed=seed
                one=dict(args);directory=base/f'seed_{seed}';directory.mkdir(exist_ok=False)
                one['eval_video_save_dir']=directory
                # The original loop clears at (completed_trials + 1) % period.
                # A one-trial call resets that counter; preserve the global
                # cadence while retaining forced cleanup on an incomplete trial.
                clear_after_trial=(ordinal+2)%int(args['clear_cache_freq'])==0
                original_close=task_env.close_env
                def close_env(clear_cache=False):
                    ended=bool(getattr(task_env,'eval_success',False)) or (
                        getattr(task_env,'take_action_cnt',-1)>=getattr(task_env,'step_lim',float('inf')))
                    actual_clear=clear_after_trial if ended else clear_cache
                    self.cache_schedule.append({'seed':seed,'ordinal':ordinal,
                        'episode_ended':ended,'native_single_trial_clear_request':bool(clear_cache),
                        'clear_cache':bool(actual_clear)})
                    return original_close(clear_cache=actual_clear)
                task_env.close_env=close_env
                try:
                    next_seed,count=original(task_name,task_env,one,model_client,seed,usr_args=usr_args,
                        test_num=1,video_size=video_size,instruction_type=instruction_type)
                finally:task_env.close_env=original_close
                if next_seed!=seed+1 or task_env.test_num!=1:
                    raise FrozenCohortViolation(f'Native evaluation skipped or retried frozen seed {seed}')
                successes+=count
            self.expected_seed=None
            return int(self.rows[-1]['seed'])+1,successes

        native.build_instruction=build_instruction;native.eval_remote_policy=evaluate
