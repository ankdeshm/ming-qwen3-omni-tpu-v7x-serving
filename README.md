# Qwen3-Omni-30B-A3B & Ming-flash-omni-2.0 Serving on Google Cloud TPU v7x-16

Multi-host `vLLM` serving, sharding (`TP × DP × EP`), and multimodal preprocessing benchmarks for `Qwen3-Omni-30B-A3B` (`Instruct` and `Thinking`) and `Ming-flash-omni-2.0` on a 16-chip (`32`-TensorCore) Google Cloud TPU v7x slice.

**Summary:** On `Qwen/Qwen3-Omni-30B-A3B-Instruct` (`bfloat16` in `vllm-torchtpu-omni`), sharding across a single 8-TensorCore `v7x` host with expert parallelism (`TP=8, EP=True`) reaches **4.54 ms TPOT `p50` and 157.27 ms TTFT `p50` (`2.46×` faster TPOT and `1.63×` lower TTFT than `TP=2 × DP=4`)**, while scaling `TP=2 × DP=16` across all 4 hosts (`32` TensorCores) reaches **1,565.6 output tok/s (`3.46×` higher than a single host at `452.7 output tok/s`)**, and `Qwen3-Omni-30B-A3B-Thinking` reaches **3,870.8 output tok/s (`53,262.5 total tok/s`)** across 4 hosts (`TP=4 × DP=8`). Native `vLLM` serving for `inclusionAI/Ming-flash-omni-2.0` did not complete because `BailingMM2NativeForConditionalGeneration` is unregistered in `vllm-torchtpu-omni` `v0.32.0` (block-level `torch_tpu` kernel timings and `vLLM` diagnostic logs are in [`APPENDIX.md`](./APPENDIX.md)).

---

## 1. Model, Framework, and Hardware

- **Models:**
  - **`Qwen/Qwen3-Omni-30B-A3B-Instruct` (`30B` total / `~3.5B` active params, `65.68 GiB` `bfloat16`):** 32-layer audio encoder (`d = 1280`) + 27-layer vision ViT (`d = 1152`) + 48-layer sparse MoE Thinker (`d = 2048`, `128` experts top-8) + 20-layer MoE Talker and 24 kHz `Code2Wav` speech synthesizer.
  - **`Qwen/Qwen3-Omni-30B-A3B-Thinking` (`30B` total / `~3.5B` active params, `59.08 GiB` `bfloat16`):** Same audio/vision + 48-layer 128-expert MoE Thinker trained for `<think>...</think>` chain-of-thought reasoning (no speech Talker).
  - **`inclusionAI/Ming-flash-omni-2.0` (`~72B` total / `~14B` active params, `219.36 GiB` `f32`/`bfloat16`):** `BailingMM2NativeForConditionalGeneration` combining 32-layer Whisper audio, 27-layer ViT + 28-layer Qwen2 connector, 32-layer 256-expert top-8 MoE Thinker (`d = 4096`), Audio-DiT, and Image-Gen DiT (evaluated only in block-level `torch_tpu` microbenchmarks due to missing `vLLM` `v0.32.0` model registration).
- **Framework:** `vllm-torchtpu-omni:v0.32.0` (`PyTorch` + `torch_tpu`, `USE_MOE_FUSED_EP_KERNEL=1`, least-inflight HTTP Keep-Alive router across multi-host replicas) and C3 Intel Xeon Platinum 8481C (`44` vCPUs) CPU preprocessing offload (`multimodal-cpu-offload-svc:8090`).
- **Hardware:** Google Cloud TPU v7x-16 (Ironwood) slice on GKE (`2x2x4` 3D torus: `4` hosts × `4` physical chips = `16` chips / `32` hardware TensorCores, `192 GiB` HBM3e per chip = `3,072 GiB` total HBM3e).

---

## 2. Main Results

### A. End-to-End `vLLM` Sharding & Multi-Host Scaling (`Qwen3-Omni-30B-A3B-Instruct`, `bfloat16`, `4K` Context, `128` Max Output Tokens)

| Sharding Topology | Hosts (`TensorCores`) | Concurrency | TTFT `p50` (ms) | TPOT `p50` (ms) | Output Tok/s | Comparison |
| :--- | :---: | ---: | ---: | ---: | ---: | :--- |
| `TP=2, DP=4` (`EP=False`) | `1` (`8 TC`) | 1 / 16 | 255.99 / 579.91 | 11.17 / 19.56 | 65.4 / 452.7 | Single-host throughput baseline (`1.0×`) |
| `TP=4, DP=2` (`EP=False`) | `1` (`8 TC`) | 1 / 16 | 265.93 / 773.33 | 10.12 / 49.35 | 48.0 / 241.4 | `0.53×` peak tok/s vs. 1-host `TP=2, DP=4` |
| **`TP=8, DP=1` (`EP=True`)** | **`1` (`8 TC`)** | **1 / 8** | **157.27** / 282.67 | **4.54** / 16.18 | **91.2** / 332.1 | **2.46× faster TPOT** and **1.63× lower TTFT** at `CON=1` vs. `TP=2, DP=4` |
| `TP=2, Cluster DP=8` (`EP=False`) | `2` (`16 TC`) | 32 | 567.51 | 21.20 | 871.0 | **1.92× higher** output tok/s than 1-host `TP=2, DP=4` |
| **`TP=2, Cluster DP=16` (`EP=False`)** | **`4` (`32 TC`)** | **1 / 64** | 253.50 / 585.99 | 11.01 / 20.95 | 68.5 / **1,565.6** | **3.46× higher** peak output tok/s than 1-host `TP=2, DP=4` (`1.29×` over 4-host `TP=4`) |
| `TP=4, Cluster DP=8` (`EP=False`) | `4` (`32 TC`) | 8 / 64 | 273.92 / 753.98 | 8.14 / 31.81 | 387.7 / 1,212.8 | **2.68× higher** peak output tok/s than 1-host `TP=2, DP=4` |
| **`TP=8, Cluster DP=4` (`EP=True`)** | **`4` (`32 TC`)** | **1 / 16** | 161.12 / 336.19 | **4.66 / 9.23** | 69.2 / **971.1** | **2.92× higher** tok/s than 1-host `TP=8+EP` while keeping TPOT `<10 ms` |

*Notes:* Measured on running `vllm-torchtpu-omni` servers (`server_boot_s` `290–503 s` cold startup excluded). Raw outputs are in [`results/suite_a_vllm_omni_serving.json`](./results/suite_a_vllm_omni_serving.json).

### B. Chain-of-Thought Reasoning (`Qwen3-Omni-30B-A3B-Thinking`, `4` Hosts = `32 TC`, `TP=4 × DP=8`, `4K` Context, `320` Output Tokens) & Context Scaling

| Workload | Concurrency | TTFT `p50` (ms) | TPOT `p50` (ms) | Output Tok/s (`Total Tok/s`) | Comparison |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `Thinking` (`4` hosts, `TP=4 × DP=8`) | 1 | 279.39 | — | 76.6 (`4,194 ms` turn) | Single-stream CoT baseline (`100%` `<think>` tags) |
| `Thinking` (`4` hosts, `TP=4 × DP=8`) | 16 | 490.95 | 6.20 | 1,999.8 (`27,518.2`) | **26.1×** output tok/s vs. `CON=1` |
| **`Thinking` (`4` hosts, `TP=4 × DP=8`)** | **64** | 978.66 | 12.27 | **3,870.8 (`53,262.5`)** | **50.5×** output tok/s vs. `CON=1`; **3.19×** vs. `Instruct` (`128`-tok turns) on same `TP=4 × DP=8` |
| `Instruct` Context `4K` (`3,584` tok, `1` host `TP=8`) | 1 / 4 | 148.33 / 351.93 | — | — (`8,650.2` / **`17,917.2`**) | `4K` `bfloat16` prefill baseline |
| `Instruct` Context `16K` (`15,872` tok, `1` host `TP=8`) | 1 / 4 | 838.52 / 1,972.71 | — | — (`13,552.6` / **`17,752.3`**) | `4.43×` longer context preserves `99.1%` of `CON=4` total tok/s (`64K+` exceeded 1-host `bf16` KV cap) |

### C. Disaggregated CPU Multimodal Offload Microbenchmark (`Intel Xeon Platinum 8481C`, `44` vCPUs)

| Stage | Local TPU-Host CPU | Remote Server Compute | Remote HTTP RTT | Comparison (Remote RTT vs. Local CPU) |
| :--- | ---: | ---: | ---: | :--- |
| `1024×1024` PNG frame compression | 161.26 ms | 140.49 ms | **142.36 ms** | **1.13× faster** end-to-end (`1.15×` server compute) |
| Dual-camera `640×480 → 256×256` resize/normalize | **7.02 ms** | 5.61 ms | 32.76 ms | `1.25×` faster server compute, but **4.67× slower** synchronous HTTP RTT |
| `4 s` `24 kHz` WAV audio encode | **0.264 ms** | 0.205 ms | 0.782 ms | Sub-millisecond on both (`2.96×` slower over HTTP RTT; useful only under CPU contention) |

---

## 3. Reproducing the Main Results

### GKE Deployment (`4`-Host `v7x-16` Slice + Disaggregated CPU Service)
Manifests: [`k8s/multimodal-cpu-offload-service.yaml`](./k8s/multimodal-cpu-offload-service.yaml) and [`k8s/tpu-v7x-16-omni-serving.yaml`](./k8s/tpu-v7x-16-omni-serving.yaml):

```bash
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml
# Edit YOUR_ARTIFACT_REGISTRY and omni-models-pvc in k8s/tpu-v7x-16-omni-serving.yaml, then apply:
kubectl apply -f k8s/tpu-v7x-16-omni-serving.yaml
```

### Direct GCE TPU v7x-16 Slice Execution
Scripts: [`gce/launch_tpu_v7x_16_omni.sh`](./gce/launch_tpu_v7x_16_omni.sh) and [`scripts/run_ming_qwen3_omni_v7x_bench.py`](./scripts/run_ming_qwen3_omni_v7x_bench.py):

```bash
export PROJECT_ID=YOUR_GCP_PROJECT_ID
export ZONE=us-central1-c
export TPU_NAME=tpu-v7x-16-omni
export GCS_MODELS_URI=gs://YOUR_GCS_BUCKET/models
bash gce/launch_tpu_v7x_16_omni.sh
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Full 34-point `vLLM` sharding table, open-loop Poisson (`0.5–8.0 RPS`) sweep, `vLLM` diagnostic failures (`Ming-flash-omni-2.0` unregistered architecture, `64K` `bfloat16` single-host KV cap, `fp8` KV register-padding error, `128K` YaRN schema validation error), and stage-level `torch_tpu` kernel microbenchmark breakdowns (`Suite B`) with methodology caveats.
- **[`results/`](./results/):** Raw JSON/JSONL outputs (`suite_a_vllm_omni_serving.json`, `suite_b_duplex_multimodal_pipeline.json`, `omni_v7x16_summary.json`, `diagnostic_matrix.json`, and `checkpoint_manifests.json`).
