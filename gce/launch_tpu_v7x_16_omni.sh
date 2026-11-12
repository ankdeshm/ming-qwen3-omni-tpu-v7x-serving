#!/usr/bin/env bash
# Direct GCE Cloud TPU v7x-16 (2x2x4 = 4 hosts = 16 chips = 32 TensorCores) Launcher
# Stages inclusionAI/Ming-flash-omni-2.0 + Qwen/Qwen3-Omni-30B-A3B-{Thinking,Instruct} into /dev/shm
# and runs Suite A (vLLM TPxDPxEP) + Suite B (32-TC Compiled Duplex Multimodal Pipeline).
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-YOUR_GCP_PROJECT_ID}"
ZONE="${ZONE:-us-central1-c}"
TPU_NAME="${TPU_NAME:-tpu-v7x-16-omni}"
GCS_MODELS_URI="${GCS_MODELS_URI:-gs://YOUR_GCS_BUCKET/models}"

echo "[1/3] Staging Ming-flash-omni-2.0 (219.4 GiB) + Qwen3-Omni-30B-A3B (124.8 GiB) to /dev/shm across all 4 TPU v7x hosts..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" --worker=all \
  --command="mkdir -p /dev/shm/omni_models && gcloud storage cp -r '${GCS_MODELS_URI}/inclusionAI' '${GCS_MODELS_URI}/Qwen' /dev/shm/omni_models/"

echo "[2/3] Launching persistent Omni Omni Host Agents (8 TensorCores/host = 32 TensorCores)..."
for WORKER in 0 1 2 3; do
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --project="${PROJECT_ID}" --zone="${ZONE}" --worker="${WORKER}" \
    --command="nohup python3 ~/scripts/run_ming_qwen3_omni_v7x_bench.py --mode=host-agent --host-rank=${WORKER} --agent-port=19120 --shm-root=/dev/shm/omni_models > /dev/shm/omni_agent_${WORKER}.log 2>&1 &" &
done
wait
sleep 5

echo "[3/3] Running Omni Orchestrator on Worker 0 across all 4 hosts..."
HOST_IPS=$(gcloud compute tpus tpu-vm describe "${TPU_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" --format="value(networkEndpoints[].ipAddress)" | tr ';' ',')
ENDPOINTS=$(python3 -c "import sys; print(','.join(f'http://{ip}:19120' for ip in sys.argv[1].split(',') if ip))" "${HOST_IPS}")

gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" --zone="${ZONE}" --worker=0 \
  --command="python3 ~/scripts/run_ming_qwen3_omni_v7x_bench.py --mode=orchestrator --host-endpoints='${ENDPOINTS}' --shm-root=/dev/shm/omni_models --out-dir=~/results"
