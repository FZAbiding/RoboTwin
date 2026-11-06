"""Resource-lifecycle contract only; synthetic episodes are not policy results."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.frozen_expert_cohort import FrozenCohort


class CacheScheduleTest(unittest.TestCase):
    def exercise(self,period,emergency=False,early_success=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'expert.pkl';path.write_bytes(b'synthetic unit fixture')
            seeds=list(range(300000,300020));seed_file=root/'seed.txt';seed_file.write_text(' '.join(map(str,seeds)))
            (root/'videos').mkdir()
            task_args={'embodiment':['unit-fixture'],'domain_randomization':{},'camera':{},
                       'data_type':{},'eval_mode':True,'clear_cache_freq':period,
                       'eval_video_save_dir':root/'videos'}
            rows=[{'seed':s,'path':str(path),'path_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'accepted':True,'expert_plan_success':True,'expert_task_success':True,
                   'planned_joints_legal':True,'episode_info':{}} for s in seeds]
            manifest={'status':'frozen','task_name':'lift_pot','accepted':rows,'source_hashes':{},
                      'seed_path':str(seed_file),'seed_sha256':hashlib.sha256(seed_file.read_bytes()).hexdigest(),
                      'task_config':{k:v for k,v in task_args.items() if k!='eval_video_save_dir'}}
            proof=root/'manifest.json';proof.write_text(json.dumps(manifest))
            cohort=FrozenCohort(proof,root,set());clears=[];calls=[]
            task=SimpleNamespace(eval_success=False,take_action_cnt=0,step_lim=400,test_num=0)
            def close(clear_cache=False):clears.append(clear_cache)
            task.close_env=close
            def single(task_name,task_env,args,client,seed,**kw):
                calls.append(seed);task_env.eval_success=early_success
                task_env.take_action_cnt=0 if emergency else (7 if early_success else 400)
                task_env.test_num=1
                task_env.close_env(clear_cache=True if emergency else 2%period==0)
                return seed+1,0
            native=SimpleNamespace(eval_remote_policy=single,build_instruction=lambda *a:None)
            cohort.install(native,len(rows))
            native.eval_remote_policy('lift_pot',task,task_args,None,0,
                usr_args={'expert_check':False},test_num=len(rows))
            self.assertEqual(calls,seeds)
            self.assertEqual(clears,[True]*len(rows) if emergency else [(i+2)%period==0 for i in range(len(rows))])
            self.assertIs(task.close_env,close)

    def test_global_native_cadence(self):
        for period in [1,2,5,7]:
            with self.subTest(period=period):self.exercise(period)

    def test_forced_incomplete_episode_cleanup_is_preserved(self):
        self.exercise(5,emergency=True)

    def test_success_before_step_limit_has_global_cleanup_cadence(self):
        self.exercise(5,early_success=True)


if __name__=='__main__':unittest.main()
