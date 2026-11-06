"""Controlled reset variant plus original expert-path snapshots.

The native expert filter, learner loop, task and action values are unchanged.
"""
from pathlib import Path
import sys
import pickle
import hashlib
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lift_pot_observer import Observer as LiftPotObserver


class Observer(LiftPotObserver):
    def __init__(self, event, options):
        super().__init__(event)
        self.reset_planners = bool(options.get("reset_planners", False))
        self.path_dir = Path(options["expert_path_dir"]).resolve()
        self.path_dir.mkdir(parents=True, exist_ok=False)
        self.setup_count = 0

    def on_setup(self, environment, phase, seed):
        super().on_setup(environment, phase, seed)
        if self.reset_planners and self.setup_count > 0:
            if environment.robot.communication_flag:
                raise RuntimeError("This controlled variant supports direct CuRobo planners only")
            started = time.perf_counter()
            for arm in ("left", "right"):
                planner = getattr(environment.robot, f"{arm}_planner")
                planner.motion_gen.reset(reset_seed=True)
                planner.motion_gen_batch.reset(reset_seed=True)
            self.emit("controlled_planner_reset", reset_seed=True,
                      modules=["left.single", "left.batch", "right.single", "right.batch"],
                      seconds=time.perf_counter() - started)
        self.setup_count += 1

    def on_expert_end(self, environment):
        super().on_expert_end(environment)
        # Same path structure as native save_traj_data; derived snapshot only.
        paths = {"left_joint_path": environment.left_joint_path,
                 "right_joint_path": environment.right_joint_path}
        encoded = pickle.dumps(paths, protocol=4)
        path = self.path_dir / f"seed_{self.seed}.pkl"
        with path.open("xb") as file:
            file.write(encoded)
        self.emit("expert_path_snapshot", path=str(path), bytes=len(encoded),
                  sha256=hashlib.sha256(encoded).hexdigest(),
                  plan_success=bool(environment.plan_success),
                  original_predicate_success=self.metrics(environment)["original_predicate_conjunction"])
