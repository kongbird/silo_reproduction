#!/usr/bin/env bash

set -u

OUTPUT_FILE="${1:-logs/day1/gpu_samples/gpu_samples.csv}"
INTERVAL_SECONDS="${2:-1}"

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "timestamp,index,name,utilization_gpu_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w" \
    > "$OUTPUT_FILE"

while true; do
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits \
        >> "$OUTPUT_FILE"

    sleep "$INTERVAL_SECONDS"
done
