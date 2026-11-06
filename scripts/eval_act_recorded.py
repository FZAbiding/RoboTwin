#!/usr/bin/env python3
"""Observe the upstream real simulator evaluation without replacing its loop."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


class CandidateAuditComplete(BaseException):
    """End a bounded native candidate audit between completed episodes."""


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args()
    config = json.loads(options.config.read_text())
    if config.get("candidate_audit", False):
        budget = int(config["max_candidate_seeds"])
        if budget < 1 or int(config["native_args"]["test_num"]) <= budget:
            raise ValueError("Candidate audit requires test_num above the finite candidate budget")
        if config["native_args"].get("expert_check", True) is not True:
            raise ValueError("Candidate audit retains the native expert check")
    record = Path(config["record_dir"]).resolve()
    record.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ROOT / "scripts"))

    import numpy as np

    source_rows = json.loads(Path(config["source_manifest"]).read_text())
    data_seeds = {int(row["seed"]) for row in source_rows}
    args = dict(config["native_args"])
    args.setdefault("xpolicylab_root", str(ROOT / "XPolicyLab"))
    if args.get("protocol", "ws") != "ws" or args.get("eval_batch"):
        raise ValueError("This recorder supports the upstream single-worker WebSocket path")
    spec = importlib.util.spec_from_file_location(
        "robotwin_recorded_eval", ROOT / "scripts" / "eval_policy_xpolicylab.py"
    )
    assert spec is not None and spec.loader is not None
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    factory = native.class_decorator
    build_client = native.build_policy_client
    save_path = native.build_eval_save_dir
    state: dict = {
        "config": config, "pid": os.getpid(), "status": "running",
        "candidates": [], "trials": [], "rpc_errors": [], "simulation_errors": [],
        "actions_executed": 0,
        "execution_diagnostics": {},
    }
    observations: dict = {"seed": None, "phase": None}
    rpc_times: dict[str, list[float]] = {}
    model_times: dict[str, list[float]] = {}
    started = time.perf_counter()

    def event(kind: str, **fields) -> None:
        with (record / "events.jsonl").open("a") as file:
            file.write(json.dumps({"event": kind, "time": time.time(), **fields}) + "\n")

    diagnostic_observer = None
    if config.get("diagnostics_module"):
        module_path = Path(config["diagnostics_module"]).resolve()
        diagnostic_spec = importlib.util.spec_from_file_location("robotwin_readonly_diagnostics", module_path)
        assert diagnostic_spec is not None and diagnostic_spec.loader is not None
        diagnostic_module = importlib.util.module_from_spec(diagnostic_spec)
        diagnostic_spec.loader.exec_module(diagnostic_module)
        observer_options = config.get("diagnostic_options")
        diagnostic_observer = (diagnostic_module.Observer(event) if observer_options is None
                               else diagnostic_module.Observer(event, observer_options))

    def class_decorator(task_name):
        environment = factory(task_name)
        setup, play = environment.setup_demo, environment.play_once
        check, take = environment.check_success, environment.take_action
        seen: dict[int, int] = {}
        candidates: dict[int, dict] = {}

        def setup_demo(*positional, **keywords):
            seed = int(keywords["seed"])
            if seed in data_seeds:
                raise KeyboardInterrupt(f"Evaluation seed overlaps collected data: {seed}")
            if seed not in seen:
                if len(seen) >= int(config.get("max_candidate_seeds", 200)):
                    if config.get("candidate_audit", False):
                        raise CandidateAuditComplete("Requested candidate audit budget completed")
                    raise KeyboardInterrupt("Evaluation candidate seed budget exhausted")
                seen[seed] = 0
                candidates[seed] = {"seed": seed}
                state["candidates"].append(candidates[seed])
            seen[seed] += 1
            phase = "expert" if args.get("expert_check", True) and seen[seed] == 1 else "learner"
            observations.update(seed=seed, phase=phase)
            try:
                result = setup(*positional, **keywords)
            except BaseException as error:
                candidates[seed][f"{phase}_setup_error"] = f"{type(error).__name__}: {error}"
                raise
            if diagnostic_observer is not None:
                diagnostic_observer.on_setup(environment, phase, seed)
            if hasattr(environment, "model_name"):
                candidates[seed]["model_name"] = str(environment.model_name)
            if hasattr(environment, "model_id"):
                candidates[seed]["model_id"] = int(environment.model_id)
            if phase == "learner":
                candidates[seed]["admitted_by_native_expert_gate"] = True
                if config.get("record_execution_detail", False):
                    state["execution_diagnostics"][str(seed)] = {
                        "left": {"topp_calls": 0, "topp_errors": 0, "empty_paths": 0, "nonfinite_paths": 0, "max_measured_step_delta": 0.0},
                        "right": {"topp_calls": 0, "topp_errors": 0, "empty_paths": 0, "nonfinite_paths": 0, "max_measured_step_delta": 0.0},
                    }
                    for arm in ("left", "right"):
                        planner = getattr(environment.robot, f"{arm}_mplib_planner")
                        original_topp = getattr(planner.TOPP, "_robotwin_original_topp", planner.TOPP)

                        def observe_topp(*a, _arm=arm, _original=original_topp, **kw):
                            stats = state["execution_diagnostics"][str(observations["seed"])][_arm]
                            stats["topp_calls"] += 1
                            try:
                                value = _original(*a, **kw)
                            except Exception as error:
                                stats["topp_errors"] += 1
                                event("native_topp_error", seed=observations["seed"], arm=_arm, error=f"{type(error).__name__}: {error}")
                                raise
                            positions = np.asarray(value[1])
                            empty = len(positions) == 0
                            finite = bool(np.isfinite(positions).all())
                            stats["empty_paths"] += int(empty)
                            stats["nonfinite_paths"] += int(not finite)
                            event("native_topp", seed=observations["seed"], arm=_arm, path_steps=len(positions), finite=finite)
                            return value

                        observe_topp._robotwin_original_topp = original_topp
                        planner.TOPP = observe_topp
            event("setup", phase=phase, **candidates[seed])
            write_json(record / "status.json", state)
            return result

        def play_once(*positional, **keywords):
            seed = observations["seed"]
            try:
                result = play(*positional, **keywords)
            except BaseException as error:
                candidates[seed]["expert_error"] = f"{type(error).__name__}: {error}"
                event("expert_error", seed=seed, error=candidates[seed]["expert_error"])
                raise
            if diagnostic_observer is not None:
                diagnostic_observer.on_expert_end(environment)
            candidates[seed]["expert_plan_success"] = bool(environment.plan_success)
            event("expert_play", seed=seed, plan_success=bool(environment.plan_success))
            return result

        def check_success(*positional, **keywords):
            result = check(*positional, **keywords)
            if observations["phase"] == "expert":
                candidates[observations["seed"]]["expert_task_success"] = bool(result)
            return result

        def take_action(*positional, **keywords):
            action = np.asarray(positional[0], dtype=float).reshape(-1)
            if not np.isfinite(action).all():
                raise FloatingPointError("Nonfinite policy action")
            diagnostics = {}
            if config.get("record_execution_detail", False):
                for arm in ("left", "right"):
                    diagnostics[f"{arm}_measured_before"] = np.asarray(getattr(environment.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:-1].tolist()
                    diagnostics[f"{arm}_drive_before"] = np.asarray(getattr(environment.robot, f"get_{arm}_arm_jointState")(), dtype=float).tolist()
            start = time.perf_counter()
            try:
                result = take(*positional, **keywords)
            except Exception as error:
                failure = {"seed": observations["seed"], "error": f"{type(error).__name__}: {error}"}
                state["simulation_errors"].append(failure)
                event("simulation_error", **failure)
                raise
            if config.get("record_execution_detail", False):
                for arm in ("left", "right"):
                    after = np.asarray(getattr(environment.robot, f"get_{arm}_arm_real_jointState")(), dtype=float)[:-1]
                    diagnostics[f"{arm}_measured_after"] = after.tolist()
                    diagnostics[f"{arm}_drive_after"] = np.asarray(getattr(environment.robot, f"get_{arm}_arm_jointState")(), dtype=float).tolist()
                    delta = float(np.max(np.abs(after - np.asarray(diagnostics[f"{arm}_measured_before"]))))
                    stats = state["execution_diagnostics"][str(observations["seed"])][arm]
                    stats["max_measured_step_delta"] = max(stats["max_measured_step_delta"], delta)
            if diagnostic_observer is not None:
                diagnostic_observer.on_policy_action(environment)
            state["actions_executed"] += 1
            event(
                "action", seed=observations["seed"], values=action.tolist(),
                action_type=keywords.get("action_type"),
                simulation_call_ms=(time.perf_counter() - start) * 1000,
                native_eval_success=bool(environment.eval_success),
                **diagnostics,
            )
            return result

        environment.setup_demo = setup_demo
        environment.play_once = play_once
        environment.check_success = check_success
        environment.take_action = take_action
        return environment

    def build_policy_client(native_args):
        start = time.perf_counter()
        client = build_client(native_args)
        state["connection_ms"] = (time.perf_counter() - start) * 1000
        call, run = client.call, client._run
        context = {"method": None}

        def observe_run(coroutine):
            start = time.perf_counter()
            response = run(coroutine)
            method = context["method"]
            if method:
                wall = (time.perf_counter() - start) * 1000
                rpc_times.setdefault(method, []).append(wall)
                payload = getattr(response, "payload", {}) or {}
                latency = payload.get("latency_ms")
                if latency is not None:
                    model_times.setdefault(method, []).append(float(latency))
                event("rpc", method=method, seed=observations["seed"], rpc_ms=wall, server_method_ms=latency)
            return response

        def observed_call(func_name=None, obs=None, **keywords):
            context["method"] = func_name
            try:
                result = call(func_name=func_name, obs=obs, **keywords)
                if func_name == "trial_end":
                    if diagnostic_observer is not None:
                        diagnostic_observer.on_trial_end(diagnostic_observer.environment, obs)
                    state["trials"].append(dict(obs))
                    event("trial_end", **obs)
                    write_json(record / "status.json", state)
                return result
            except Exception as error:
                failure = {"method": func_name, "seed": observations["seed"], "error": f"{type(error).__name__}: {error}"}
                state["rpc_errors"].append(failure)
                event("rpc_error", **failure)
                raise
            finally:
                context["method"] = None

        client._run = observe_run
        client.call = observed_call
        return client

    def build_eval_save_dir(*positional, **keywords):
        result = save_path(*positional, **keywords)
        state["upstream_output"] = str(result.resolve())
        return result

    native.class_decorator = class_decorator
    native.build_policy_client = build_policy_client
    native.build_eval_save_dir = build_eval_save_dir
    write_json(record / "status.json", state)
    try:
        native.main(args)
        trials = state["trials"]
        assert len(trials) == int(args["test_num"]), "Missing native trial-end records"
        assert state["actions_executed"] > 0, "No actual robot actions were executed"
        assert not state["rpc_errors"], "Policy protocol failed; inspect events.jsonl"
        assert not state["simulation_errors"], "Simulator raised during policy execution; inspect events.jsonl"
        state.update(status="complete", successes=sum(bool(row["success"]) for row in trials))
        output = Path(state["upstream_output"])
        assert (output / "_result.txt").is_file()
        (record / "upstream-output").symlink_to(output, target_is_directory=True)
    except CandidateAuditComplete:
        assert config.get("candidate_audit", False)
        assert len(state["candidates"]) == int(config["max_candidate_seeds"])
        assert not state["rpc_errors"] and not state["simulation_errors"]
        state.update(
            status="candidate_audit_complete",
            successes=sum(bool(row["success"]) for row in state["trials"]),
            result_kind="bounded candidate audit; native benchmark test_num was not completed",
        )
        event("candidate_audit_complete", candidates=len(state["candidates"]),
              completed_learner_trials=len(state["trials"]), successes=state["successes"])
        output = Path(state["upstream_output"])
        (record / "upstream-output").symlink_to(output, target_is_directory=True)
    except BaseException:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        state["elapsed_seconds"] = time.perf_counter() - started
        state["latency_ms"] = {}
        for label, groups in [("rpc", rpc_times), ("server_method", model_times)]:
            state["latency_ms"][label] = {
                method: {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
                for method, values in groups.items()
            }
        for candidate in state["candidates"]:
            if not candidate.get("admitted_by_native_expert_gate"):
                candidate["skipped_by_native_gate"] = True
        write_json(record / "status.json", state)


if __name__ == "__main__":
    main()
