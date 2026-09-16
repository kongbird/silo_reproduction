import csv
import json
from pathlib import Path


def read_benchmark(num_envs):
    path = Path(
        f"logs/day1/benchmarks/pickcube_{num_envs}.json"
    )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_gpu_samples(num_envs):
    path = Path(
        f"logs/day1/gpu_samples/pickcube_{num_envs}.csv"
    )

    rows = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if not rows:
        return {
            "sample_count": 0,
            "mean_gpu_utilization_percent": None,
            "peak_gpu_utilization_percent": None,
            "mean_memory_used_mib": None,
            "peak_memory_used_mib": None,
            "peak_temperature_c": None,
        }

    utilization = [
        float(row["utilization_gpu_percent"])
        for row in rows
    ]
    memory = [
        float(row["memory_used_mib"])
        for row in rows
    ]
    temperature = [
        float(row["temperature_c"])
        for row in rows
    ]

    return {
        "sample_count": len(rows),
        "mean_gpu_utilization_percent": (
            sum(utilization) / len(utilization)
        ),
        "peak_gpu_utilization_percent": max(utilization),
        "mean_memory_used_mib": sum(memory) / len(memory),
        "peak_memory_used_mib": max(memory),
        "peak_temperature_c": max(temperature),
    }


def main():
    summaries = []

    for num_envs in [16, 64, 256]:
        benchmark = read_benchmark(num_envs)
        gpu = read_gpu_samples(num_envs)

        summary = {
            "num_envs": num_envs,
            "status": benchmark.get("status"),
            "elapsed_seconds": benchmark.get(
                "elapsed_seconds"
            ),
            "env_step_calls_per_second": benchmark.get(
                "env_step_calls_per_second"
            ),
            "vector_transitions_per_second": benchmark.get(
                "vector_transitions_per_second"
            ),
            "numeric_error_count": benchmark.get(
                "numeric_error_count"
            ),
            **gpu,
        }

        summaries.append(summary)

    output = Path("logs/day1/day1_summary.json")

    with output.open("w", encoding="utf-8") as f:
        json.dump(
            summaries,
            f,
            indent=2,
            ensure_ascii=False,
        )

    header = (
        f"{'envs':>6} "
        f"{'status':>10} "
        f"{'step/s':>12} "
        f"{'trans/s':>14} "
        f"{'GPU avg%':>10} "
        f"{'GPU peak%':>11} "
        f"{'VRAM peak MiB':>15} "
        f"{'temp C':>8} "
        f"{'errors':>8}"
    )

    print(header)
    print("-" * len(header))

    for item in summaries:
        print(
            f"{item['num_envs']:>6} "
            f"{str(item['status']):>10} "
            f"{(item['env_step_calls_per_second'] or 0):>12.2f} "
            f"{(item['vector_transitions_per_second'] or 0):>14.2f} "
            f"{(item['mean_gpu_utilization_percent'] or 0):>10.2f} "
            f"{(item['peak_gpu_utilization_percent'] or 0):>11.2f} "
            f"{(item['peak_memory_used_mib'] or 0):>15.2f} "
            f"{(item['peak_temperature_c'] or 0):>8.2f} "
            f"{str(item['numeric_error_count']):>8}"
        )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
