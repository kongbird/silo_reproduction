import argparse
import csv
import gc
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import mani_skill.envs  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a ManiSkill vectorized environment."
    )

    parser.add_argument(
        "--env-id",
        type=str,
        default="PickCube-v1",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="state",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_delta_pos",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="gpu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )

    return parser.parse_args()


def value_is_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())

    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())

    if isinstance(value, dict):
        return all(value_is_finite(v) for v in value.values())

    if isinstance(value, (tuple, list)):
        return all(value_is_finite(v) for v in value)

    try:
        array = np.asarray(value)
        return bool(np.isfinite(array).all())
    except (TypeError, ValueError):
        return True


def create_env(args):
    return gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=None,
        sim_backend=args.sim_backend,
    )


def benchmark(args):
    result = {
        "status": "started",
        "env_id": args.env_id,
        "num_envs": args.num_envs,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "seed": args.seed,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "sim_backend": args.sim_backend,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "pid": os.getpid(),
    }

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    result["gpu_name"] = torch.cuda.get_device_name(0)
    result["gpu_capability"] = list(
        torch.cuda.get_device_capability(0)
    )
    result["gpu_total_memory_gib"] = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = None

    try:
        print("=" * 72)
        print("Environment:", args.env_id)
        print("Number of environments:", args.num_envs)
        print("Observation mode:", args.obs_mode)
        print("Control mode:", args.control_mode)
        print("Simulation backend:", args.sim_backend)
        print("GPU:", result["gpu_name"])
        print("Warm-up steps:", args.warmup_steps)
        print("Measured steps:", args.steps)
        print("=" * 72)

        torch.cuda.reset_peak_memory_stats()

        creation_start = time.perf_counter()
        env = create_env(args)
        creation_seconds = time.perf_counter() - creation_start

        result["environment_creation_seconds"] = creation_seconds
        print(
            "Environment creation seconds:",
            f"{creation_seconds:.6f}",
        )

        reset_start = time.perf_counter()
        obs, info = env.reset(seed=args.seed)

        torch.cuda.synchronize()
        reset_seconds = time.perf_counter() - reset_start

        result["reset_seconds"] = reset_seconds
        result["action_space_shape"] = list(env.action_space.shape)

        print("Reset seconds:", f"{reset_seconds:.6f}")
        print("Action-space shape:", env.action_space.shape)

        if not value_is_finite(obs):
            raise RuntimeError(
                "Observation contains NaN or Inf after reset"
            )

        print("Starting warm-up")

        for step in range(args.warmup_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(
                action
            )

            if step % 50 == 0:
                if not value_is_finite(obs):
                    raise RuntimeError(
                        f"Observation contains NaN or Inf "
                        f"during warm-up step {step}"
                    )

                if not value_is_finite(reward):
                    raise RuntimeError(
                        f"Reward contains NaN or Inf "
                        f"during warm-up step {step}"
                    )

        torch.cuda.synchronize()

        print("Warm-up completed")
        print("Starting measured benchmark")

        start = time.perf_counter()

        numeric_error_count = 0

        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(
                action
            )

            # Checking every step can affect performance.
            # Check periodically during the timed section.
            if step % 100 == 0:
                if not value_is_finite(obs):
                    numeric_error_count += 1
                    raise RuntimeError(
                        f"Observation contains NaN or Inf at step {step}"
                    )

                if not value_is_finite(reward):
                    numeric_error_count += 1
                    raise RuntimeError(
                        f"Reward contains NaN or Inf at step {step}"
                    )

                if not value_is_finite(action):
                    numeric_error_count += 1
                    raise RuntimeError(
                        f"Action contains NaN or Inf at step {step}"
                    )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        env_step_calls_per_second = args.steps / elapsed
        vector_transitions = args.num_envs * args.steps
        vector_transitions_per_second = vector_transitions / elapsed

        result.update(
            {
                "status": "passed",
                "elapsed_seconds": elapsed,
                "env_step_calls_per_second": (
                    env_step_calls_per_second
                ),
                "vector_transitions": vector_transitions,
                "vector_transitions_per_second": (
                    vector_transitions_per_second
                ),
                "numeric_error_count": numeric_error_count,
                "torch_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                ),
                "torch_peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 1024**3
                ),
            }
        )

        print("=" * 72)
        print("BENCHMARK PASSED")
        print("Elapsed seconds:", f"{elapsed:.6f}")
        print(
            "env.step calls/s:",
            f"{env_step_calls_per_second:.2f}",
        )
        print(
            "Vector transitions/s:",
            f"{vector_transitions_per_second:.2f}",
        )
        print(
            "PyTorch peak allocated memory GiB:",
            f"{result['torch_peak_allocated_gib']:.3f}",
        )
        print(
            "PyTorch peak reserved memory GiB:",
            f"{result['torch_peak_reserved_gib']:.3f}",
        )
        print("Numeric error count:", numeric_error_count)
        print("=" * 72)

    except Exception as exc:
        result["status"] = "failed"
        result["error_type"] = type(exc).__name__
        result["error_message"] = str(exc)
        result["traceback"] = traceback.format_exc()

        print(result["traceback"], file=sys.stderr)

    finally:
        if env is not None:
            env.close()

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def save_result(result, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("Saved result:", path)


def main():
    args = parse_args()
    result = benchmark(args)

    if args.output is None:
        output_path = (
            f"logs/day1/benchmarks/"
            f"{args.env_id}_envs_{args.num_envs}.json"
        )
    else:
        output_path = args.output

    save_result(result, output_path)

    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
