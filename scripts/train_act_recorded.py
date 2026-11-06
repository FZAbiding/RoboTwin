#!/usr/bin/env python3
"""Run upstream ACT training with a recorded split, metrics and checkpoint checks.

The training loop, sampling, loss, optimizer and model stay in XPolicyLab.
This entry point adds observations around that loop and refuses existing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ACT = ROOT / "XPolicyLab" / "policy" / "ACT"


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    options = parser.parse_args()
    config = json.loads(options.config.read_text(encoding="utf-8"))
    if config.get("enable_stack_diagnostics", False):
        import faulthandler
        import signal
        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    record_dir = Path(config["record_dir"]).resolve()
    checkpoint_dir = Path(config["checkpoint_dir"]).resolve()
    if checkpoint_dir.exists():
        raise FileExistsError(f"Use a new checkpoint directory: {checkpoint_dir}")
    record_dir.mkdir(parents=True, exist_ok=False)
    if config.get("python_prefix") and Path(sys.prefix) != Path(config["python_prefix"]):
        raise RuntimeError(f"Wrong Python environment: {sys.executable}")

    import numpy as np
    import torch

    args = {
        "bench_name": config["benchmark"],
        "task_name": config["ckpt_name"],
        "ckpt_setting": config["ckpt_setting"],
        "ckpt_dir": str(checkpoint_dir),
        "policy_class": "ACT",
        "onscreen_render": False,
        "batch_size": int(config.get("batch_size", 16)),
        "seed": int(config.get("seed", 0)),
        "num_epochs": int(config["num_epochs"]),
        "lr": float(config.get("lr", 1e-5)),
        "kl_weight": int(config.get("kl_weight", 10)),
        "chunk_size": int(config.get("chunk_size", 50)),
        "hidden_dim": int(config.get("hidden_dim", 512)),
        "dim_feedforward": int(config.get("dim_feedforward", 3200)),
        "save_freq": int(config.get("save_freq", config["num_epochs"])),
        "temporal_agg": bool(config.get("temporal_agg", True)),
    }
    if args["num_epochs"] < 1 or args["batch_size"] < 1:
        raise ValueError("Positive num_epochs and batch_size are required")
    sys.path.insert(0, str(ROOT))
    from XPolicyLab.utils.process_data import get_action_dim

    action_dim = get_action_dim(config["env_cfg_type"])
    if config.get("action_dim", action_dim) != action_dim:
        raise ValueError("Configured action dimension disagrees with the shared robot profile")
    os.environ["ACT_ACTION_DIM"] = str(action_dim)
    os.chdir(ACT)
    sys.path.insert(0, str(ACT))
    sys.path.insert(1, str(ROOT))
    # DETR reads the same native CLI again while constructing the policy.
    native_argv: list[str] = []
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                native_argv.append(f"--{key}")
        else:
            native_argv.extend((f"--{key}", str(value)))
    sys.argv = [str(ACT / "imitate_episodes.py"), *native_argv]
    spec = importlib.util.spec_from_file_location(
        "robotwin_recorded_act_training", ACT / "imitate_episodes.py"
    )
    assert spec is not None and spec.loader is not None
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)

    source_manifest = json.loads(Path(config["source_manifest"]).read_text())
    by_episode = {int(row["episode_index"]): row for row in source_manifest}
    state: dict = {
        "config": config,
        "native_args": args,
        "python": sys.executable,
        "pid": os.getpid(),
        "status": "running",
        "updates": 0,
        "epochs": [],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    original_load = training.load_data
    original_forward = training.forward_pass
    original_policy = training.make_policy
    original_optimizer = training.make_optimizer
    epoch = 0
    previous_mode: bool | None = None
    epoch_start = time.perf_counter()
    step_start = 0.0
    observations: dict[str, list[dict]] = {"train": [], "validation": []}
    head_before = None
    policy_reference = None
    observed_loaders = {}
    update_checkpoints = {int(value) for value in config.get("checkpoint_update_steps", [])}
    if any(value < 1 for value in update_checkpoints):
        raise ValueError("Checkpoint updates must be positive")

    def load_data(*positional, **keywords):
        result = original_load(*positional, **keywords)
        train, validation, stats, is_sim = result
        loader_options = config.get("data_loader", {})
        if loader_options:
            from torch.utils.data import DataLoader
            context = loader_options.get("multiprocessing_context", "spawn")
            persistent = bool(loader_options.get("persistent_workers", True))
            def rebuild(loader):
                if loader.num_workers < 1:
                    raise ValueError("Explicit multiprocessing context requires at least one worker")
                return DataLoader(loader.dataset, batch_size=loader.batch_size,
                    sampler=loader.sampler, num_workers=loader.num_workers,
                    collate_fn=loader.collate_fn, pin_memory=loader.pin_memory,
                    drop_last=loader.drop_last, timeout=loader.timeout,
                    worker_init_fn=loader.worker_init_fn, generator=loader.generator,
                    prefetch_factor=loader.prefetch_factor, persistent_workers=persistent,
                    multiprocessing_context=context)
            train, validation = rebuild(train), rebuild(validation)
            result = (train, validation, stats, is_sim)
        observed_loaders.update(train=train, validation=validation)
        state["expected_updates"] = len(train) * args["num_epochs"]
        if any(value > state["expected_updates"] for value in update_checkpoints):
            raise ValueError("Requested update checkpoint exceeds the native training budget")
        train_ids = [int(i) for i in train.dataset.episode_ids]
        val_ids = [int(i) for i in validation.dataset.episode_ids]
        assert not set(train_ids) & set(val_ids)
        assert set(train_ids) | set(val_ids) == set(by_episode)
        split = {
            "data_loader_options": config.get("data_loader", {}),
            "dataset_rng_seed_in_upstream_main": 1,
            "training_seed": args["seed"],
            "train_episode_ids": train_ids,
            "validation_episode_ids": val_ids,
            "train_seeds": [by_episode[i]["seed"] for i in train_ids],
            "validation_seeds": [by_episode[i]["seed"] for i in val_ids],
            "camera_names": list(train.dataset.camera_names),
            "train_batches_per_epoch": len(train),
            "validation_batches_per_epoch": len(validation),
            "normalization_episode_ids": list(range(len(by_episode))),
            "normalization_source": "upstream all episodes, last-frame padding, std clipped to >=0.01",
            "maximum_action_length": train.dataset.max_action_len,
            "stats": {
                name: np.asarray(value).tolist()
                for name, value in stats.items()
                if name != "example_qpos"
            },
        }
        write_json(record_dir / "split-and-stats.json", split)
        state["split"] = split
        return result

    def finish_epoch() -> None:
        row = {"epoch": epoch, "seconds": time.perf_counter() - epoch_start}
        for kind, items in observations.items():
            if items:
                row[kind] = {
                    name: float(np.mean([item[name] for item in items]))
                    for name in items[0]
                }
        row["updates_completed"] = state["updates"]
        row["loader_worker_pids"] = {
            label: [worker.pid for worker in getattr(getattr(loader, "_iterator", None), "_workers", [])]
            for label, loader in observed_loaders.items()
        }
        state["epochs"].append(row)
        with (record_dir / "epochs.jsonl").open("a") as file:
            file.write(json.dumps(row) + "\n")
        print("[recorded ACT]", json.dumps(row), flush=True)
        write_json(record_dir / "status.json", state)

    def forward_pass(data, policy):
        nonlocal epoch, previous_mode, epoch_start, observations, step_start
        mode = bool(policy.training)
        if not mode and previous_mode is True:
            finish_epoch()
            epoch += 1
            epoch_start = time.perf_counter()
            observations = {"train": [], "validation": []}
        previous_mode = mode
        step_start = time.perf_counter()
        output = original_forward(data, policy)
        values = {name: float(value.detach().item()) for name, value in output.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError(f"Nonfinite loss: {values}")
        observations["train" if mode else "validation"].append(values)
        return output

    def make_policy(*positional, **keywords):
        nonlocal head_before, policy_reference
        policy = original_policy(*positional, **keywords)
        policy_reference = policy
        head_before = policy.state_dict()["model.action_head.weight"].detach().cpu().clone()
        state["parameters"] = sum(parameter.numel() for parameter in policy.parameters())
        return policy

    def make_optimizer(*positional, **keywords):
        optimizer = original_optimizer(*positional, **keywords)

        def after_step(optimizer, _args, _kwargs):
            torch.cuda.synchronize()
            state["updates"] += 1
            state["last_update_seconds"] = time.perf_counter() - step_start
            state["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
            state["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
            if state["updates"] == 1:
                finite = [
                    torch.isfinite(parameter.grad).all()
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if not finite or not bool(torch.stack(finite).all().item()):
                    raise FloatingPointError("Missing or nonfinite first-step gradients")
                state["first_step_gradients_finite"] = True
            if state["updates"] in update_checkpoints:
                assert policy_reference is not None
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                destination = checkpoint_dir / f"policy_update_{state['updates']}_seed_{args['seed']}.ckpt"
                if destination.exists():
                    raise FileExistsError(destination)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                torch.save(policy_reference.state_dict(), temporary)
                temporary.replace(destination)
                entry = {"updates": state["updates"], "epoch_index": epoch,
                         "path": str(destination), "bytes": destination.stat().st_size,
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
                state.setdefault("update_checkpoints", []).append(entry)
                write_json(record_dir / "status.json", state)

        optimizer.register_step_post_hook(after_step)
        return optimizer

    training.load_data = load_data
    training.forward_pass = forward_pass
    training.make_policy = make_policy
    training.make_optimizer = make_optimizer
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    write_json(record_dir / "status.json", state)
    try:
        training.main(args)
        finish_epoch()
        checkpoint = checkpoint_dir / "policy_last.ckpt"
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert all(bool(torch.isfinite(value).all()) for value in weights.values())
        assert head_before is not None
        head_change = (weights["model.action_head.weight"] - head_before).abs().max().item()
        assert head_change > 0, "Optimizer did not change the action head"
        state.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - start,
            action_head_max_change=head_change,
            checkpoint=str(checkpoint),
            checkpoint_bytes=checkpoint.stat().st_size,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_tensors_finite=True,
        )
        assert len(state["epochs"]) == args["num_epochs"]
        assert state["updates"] == state["expected_updates"]
        assert {v["updates"] for v in state.get("update_checkpoints", [])} == update_checkpoints
    except BaseException:
        state.update(status="failed", error=traceback.format_exc(), elapsed_seconds=time.perf_counter() - start)
        raise
    finally:
        write_json(record_dir / "status.json", state)


if __name__ == "__main__":
    main()
