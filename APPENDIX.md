# Appendix: Detailed Sweeps, `vLLM` Diagnostics, and Kernel Microbenchmarks (Omni on TPU v7x-16)

This document records the complete `vLLM` sharding and Poisson tables (`Suite A`), all diagnostic blockers encountered in `vllm-torchtpu-omni:v0.32.0`, disaggregated CPU offload ablations, and the stage-level `torch_tpu` kernel microbenchmark tables (`Suite B`) along with their exact measurement scope. For headline end-to-end serving numbers and reproduction commands, see [`README.md`](./README.md).

---

## 1. What Did Not Work / Diagnostic Blockers in `vLLM` (`v0.32.0`)

During `Suite A` evaluation on `vllm-torchtpu-omni:v0.32.0` ([`results/diagnostic_matrix.json`](./results/diagnostic_matrix.json) and [`results/suite_a_vllm_omni_serving.json`](./results/suite_a_vllm_omni_serving.json)), five configurations failed or hit engine/memory limits:

| Target Configuration | Checkpoint | Observed Status / Blocker | Root Cause |
| :--- | :--- | :--- | :--- |
| Native `vllm serve` (`--nnodes=4`) | `inclusionAI/Ming-flash-omni-2.0` (`219.36 GiB`) | `DIAGNOSTIC_UNREGISTERED_ARCH_IN_VLLM_0_32` | `BailingMM2NativeForConditionalGeneration` is not registered in `vllm` `0.32.0`, and the checkpoint omits a standard HuggingFace `preprocessor_config.json` (`OSError: Can't load image processor`). |
| Single-host `64K` (`65,024` tokens) `bfloat16` KV | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `DIAGNOSTIC_EXCEEDS_SINGLE_HOST_BF16_KV_CAP` | Holding `65.68 GiB` model weights + XLA workspace + `64K` `bfloat16` KV cache on 1 host (`TP=8`) exceeded single-host KV allocation headroom. |
| `--kv-cache-dtype=fp8` at `60K` context | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `DIAGNOSTIC_ATTN_LAYOUT_REGISTER_PADDING` | TPU Pallas/XLA `fp8` KV attention kernel rejected the head/block register padding layout for `Qwen3OmniMoe`. |
| `128K` context with YaRN RoPE override | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `DIAGNOSTIC_PYDANTIC_ARGSKWARGS_ROPE_SCHEMA` | `vllm` `0.32.0` Pydantic `ArgsKwargs` validation rejected `--rope-scaling` override fields on `Qwen3OmniMoeConfig`. |
| Dynamic `fp8` weight quantization in `vllm` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `DIAGNOSTIC_CPU_FP8_CALIBRATION_OVERHEAD` | On-CPU dynamic `fp8` weight calibration across 48×128 MoE experts exceeded startup timeout limits. |

In addition, in the disaggregated CPU microbenchmark ([`results/omni_v7x16_summary.json`](./results/omni_v7x16_summary.json)):
- **Synchronous HTTP offload for small inputs (`2× 640×480` frames or `4 s` WAV):** Although remote Xeon compute was slightly faster (`5.61 ms` vs. `7.02 ms` on 2-camera resize; `0.205 ms` vs. `0.264 ms` on WAV encode), HTTP payload framing + network RTT made synchronous remote calls slower (`32.76 ms` vs. `7.02 ms` on 2-camera resize; `0.782 ms` vs. `0.264 ms` on WAV). Synchronous offload only beat local CPU on heavier `1024×1024` PNG compression (`142.36 ms` vs. `161.26 ms`, `1.13×` faster).

---

## 2. Full `Suite A`: Native `vLLM` Serving Tables (`Qwen3-Omni-30B-A3B`)

Raw data: [`results/suite_a_vllm_omni_serving.json`](./results/suite_a_vllm_omni_serving.json).

### A.1 Complete Sharding Topology × Multi-Host Closed-Loop Sweep (`Qwen3-Omni-30B-A3B-Instruct`, `bfloat16`, `4,096` Context, `128` Max Output Tokens)

| Hosts | TensorCores | `TP` | `DP` / Host | Cluster `DP` | `EP` | Boot (s) | Concurrency | TTFT `p50` (ms) | TPOT `p50` (ms) | Output Tok/s |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 2 | 4 | 4 | False | 503.0 | 1 | 255.99 | 11.17 | 65.40 |
| 1 | 8 | 2 | 4 | 4 | False | 503.0 | 8 | 429.34 | 16.22 | 270.80 |
| 1 | 8 | 2 | 4 | 4 | False | 503.0 | 16 | 579.91 | 19.56 | 452.70 |
| 2 | 16 | 2 | 4 | 8 | False | 503.0 | 8 | 313.85 | 14.42 | 233.00 |
| 2 | 16 | 2 | 4 | 8 | False | 503.0 | 16 | 480.59 | 17.23 | 474.90 |
| 2 | 16 | 2 | 4 | 8 | False | 503.0 | 32 | 567.51 | 21.20 | 871.00 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 1 | 253.50 | 11.01 | 68.50 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 4 | 256.56 | 11.56 | 202.00 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 8 | 256.87 | 11.60 | 338.20 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 16 | 259.80 | 14.43 | 560.90 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 32 | 486.46 | 17.67 | 872.80 |
| 4 | 32 | 2 | 4 | 16 | False | 503.0 | 64 | 585.99 | 20.95 | **1,565.60** |
| 1 | 8 | 4 | 2 | 2 | False | 360.0 | 1 | 265.93 | 10.12 | 48.00 |
| 1 | 8 | 4 | 2 | 2 | False | 360.0 | 8 | 765.01 | 16.90 | 231.40 |
| 1 | 8 | 4 | 2 | 2 | False | 360.0 | 16 | 773.33 | 49.35 | 241.40 |
| 2 | 16 | 4 | 2 | 4 | False | 360.0 | 8 | 395.06 | 10.47 | 331.80 |
| 2 | 16 | 4 | 2 | 4 | False | 360.0 | 16 | 573.12 | 18.75 | 327.10 |
| 2 | 16 | 4 | 2 | 4 | False | 360.0 | 32 | 917.30 | 30.78 | 506.90 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 1 | 826.04 | 9.88 | 39.40 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 4 | 264.95 | 9.92 | 239.90 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 8 | 273.92 | 8.14 | 387.70 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 16 | 393.57 | 9.98 | 599.10 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 32 | 573.33 | 15.26 | 1,002.10 |
| 4 | 32 | 4 | 2 | 8 | False | 360.0 | 64 | 753.98 | 31.81 | 1,212.80 |
| 1 | 8 | 8 | 1 | 1 | True | 290.0 | 1 | **157.27** | **4.54** | 91.20 |
| 1 | 8 | 8 | 1 | 1 | True | 290.0 | 8 | 282.67 | 16.18 | 332.10 |
| 1 | 8 | 8 | 1 | 1 | True | 290.0 | 16 | 636.88 | 42.86 | 241.40 |
| 2 | 16 | 8 | 1 | 2 | True | 290.0 | 8 | 391.45 | 9.14 | 347.40 |
| 2 | 16 | 8 | 1 | 2 | True | 290.0 | 16 | 380.44 | 18.42 | 476.00 |
| 2 | 16 | 8 | 1 | 2 | True | 290.0 | 32 | 508.39 | 34.13 | 643.90 |
| 4 | 32 | 8 | 1 | 4 | True | 290.0 | 1 | 161.12 | 4.66 | 69.20 |
| 4 | 32 | 8 | 1 | 4 | True | 290.0 | 8 | 162.32 | 5.95 | 653.60 |
| 4 | 32 | 8 | 1 | 4 | True | 290.0 | 16 | 336.19 | 9.23 | **971.10** |
| 4 | 32 | 8 | 1 | 4 | True | 290.0 | 32 | 362.54 | 17.00 | 853.50 |

*(Note: The `826.04 ms` TTFT on 4-host `TP=4` at `CON=1` included an initial bucket warmup on one replica; at `CON=4` TTFT settled to `264.95 ms`.)*

### A.2 Open-Loop Poisson Arrival Sweep (`Qwen3-Omni-30B-A3B-Instruct`, `4` Hosts = `32 TC`, `TP=4 × Cluster DP=8`)

| Target RPS | Achieved RPS | TTFT `p50` (ms) | TTFT `p99` (ms) | Output Tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.85 | 265.27 | 268.59 | 48.90 |
| 1.0 | 0.51 | 269.16 | 270.70 | 24.10 |
| 2.0 | 1.48 | 265.37 | 272.45 | 72.60 |
| 4.0 | 3.60 | 264.72 | 307.20 | 176.90 |
| 8.0 | **6.38** | **264.73** | **384.59** | **317.20** |

### A.3 Chain-of-Thought `<think>` Sweep (`Qwen3-Omni-30B-A3B-Thinking`, `4` Hosts = `32 TC`, `TP=4 × Cluster DP=8`, `320` Max Output Tokens)

| Concurrency | TTFT `p50` / `p95` (ms) | TPOT `p50` / `p95` (ms) | E2E Turn `p50` / `p95` (ms) | Output Tok/s | Total Tok/s | `<think>` Tag Fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 279.39 / — | — | 4,194.40 / — | 76.60 | — | 1.00 |
| 4 | 270.69 / 277.23 | 12.18 / 12.35 | 4,170.32 / 4,221.23 | 305.00 | 4,197.4 | 1.00 |
| 8 | 333.67 / 510.96 | 12.11 / 12.85 | 4,364.48 / 4,384.08 | 586.30 | 8,068.1 | 1.00 |
| 16 | 490.95 / 751.13 | 6.20 / 7.79 | 2,499.50 / 2,786.60 | 1,999.80 | 27,518.2 | 1.00 |
| 32 | 739.89 / 1,109.05 | 8.08 / 9.79 | 3,261.78 / 3,544.28 | 3,040.10 | 41,831.5 | 1.00 |
| 64 | 978.66 / 1,843.51 | 12.27 / 15.56 | 4,901.40 / 5,780.36 | **3,870.80** | **53,262.5** | 1.00 |

---

## 3. `Suite B` Block-Level `torch_tpu` Kernel Microbenchmark Harness & Caveats

Because `inclusionAI/Ming-flash-omni-2.0` (`219.36 GiB`) is unregistered in `vLLM` `0.32.0`, [`scripts/run_ming_qwen3_omni_v7x_bench.py`](./scripts/run_ming_qwen3_omni_v7x_bench.py) (`run_tpu_subworker`) also includes a standalone `torch_tpu` kernel microbenchmark (`Suite B`, [`results/suite_b_duplex_multimodal_pipeline.json`](./results/suite_b_duplex_multimodal_pipeline.json)).

**Important measurement caveat (why `Suite B` is excluded from `README.md` headline serving numbers):**
- Each worker streams the checkpoint's `.safetensors` shards into HBM3e (`48–64 GiB/host`) to occupy realistic HBM capacity and calibrates `tpu_dist` ICI `all_reduce`, then times representative architectural sub-blocks (`8` audio layers, `9` vision layers + `7` connector layers, `4` MoE prefill blocks, `8` KV-cached decode calls, `10` Talker layers, and `1` Image-Gen DiT block) and scales those block timings by layer-count multipliers in software.
- Consequently, `Suite B` is useful strictly for **relative kernel ablations on TPU v7x** (`eager` vs. `torch.compile(dynamic=False)` vs. `torch._scaled_mm` `fp8`), NOT as an end-to-end full-model serving benchmark (for end-to-end `Qwen3-Omni-30B-A3B` serving latencies and throughput, use `Suite A` above).

### B.1 Kernel Compilation & `FP8` (`torch._scaled_mm`) Block Timing Comparison (`1` Host = `8 TC`, `4K` Context)

| Model Architecture | Precision | `torch.compile` | Audio Block Scaled (ms) | Vision Block Scaled (ms) | MoE Prefill Scaled (ms) | Decode Step Scaled (ms) | Notes |
| :--- | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| `Ming-flash-omni-2.0` (`d=4096`, `256E` top-8) | `BF16` | False | 915.34 | 105.95 | 539.57 | 121.13 | Uncompiled eager dispatch bottleneck |
| `Ming-flash-omni-2.0` (`d=4096`, `256E` top-8) | `BF16` | True | **23.70** | 14.31 | **33.59** | **2.69** | `16–45×` faster block execution vs. eager |
| `Ming-flash-omni-2.0` (`d=4096`, `256E` top-8) | `FP8` (`SCALAR_SCALE`) | True | 24.88 | **12.41** | 376.29 *(31.31 warm)* | 2.83 *(2.26 warm)* | First `EP=4` prefill included XLA compile (`376.29 ms` vs. `31.31 ms` warm in `B1_cpu_offload_ablation`) |
| `Qwen3-Omni-30B-A3B-Thinking` (`d=2048`, `128E` top-8) | `BF16` | False | 32.80 | 436.21 | 829.84 | 20.89 | Eager baseline |
| `Qwen3-Omni-30B-A3B-Thinking` (`d=2048`, `128E` top-8) | `BF16` | True | 22.71 | 12.24 | 49.02 | **3.76** | `5.6–16.9×` faster MoE blocks vs. eager |
| `Qwen3-Omni-30B-A3B-Thinking` (`d=2048`, `128E` top-8) | `FP8` (`SCALAR_SCALE`) | True | **18.92** | **11.36** | **45.87** | 3.86 | `6.4%` faster prefill, ~equal decode vs. `BF16` |
| `Qwen3-Omni-30B-A3B-Instruct` (`d=2048`, `128E` + Talker) | `BF16` | False | 27.08 | 20.39 | 75.52 | 6.35 | Eager baseline |
| `Qwen3-Omni-30B-A3B-Instruct` (`d=2048`, `128E` + Talker) | `BF16` | True | 22.04 | 10.78 | **43.83** | **3.29** | `1.93×` faster decode block vs. eager |
| `Qwen3-Omni-30B-A3B-Instruct` (`d=2048`, `128E` + Talker) | `FP8` (`SCALAR_SCALE`) | True | **20.14** | **10.57** | 45.80 | 3.53 | Halves resident shard HBM (`~0.55×` `BF16`) |
| `Ming-flash-omni-2.0` Image-Gen DiT branch (`d=2560`) | `FP8` (`SCALAR_SCALE`) | True | — | — | — | — | `31,344.5 ms` cold XLA compile → **`404.5 ms` warm** + `141.5 ms` PNG encode |
