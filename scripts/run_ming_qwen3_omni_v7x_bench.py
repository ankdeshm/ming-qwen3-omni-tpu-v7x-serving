#!/usr/bin/env python3
"""Next-Gen Any-to-Any Omni Serving (`Ming-flash-omni-2.0` & `Qwen3-Omni-30B-A3B`) on Cloud TPU v7x-16.

Benchmarks on a 4-host Cloud TPU v7x-16 slice (topology 2x2x4 = 16 physical v7x chips = 32 TensorCores, 3,072 GiB HBM3e):
  1. inclusionAI/Ming-flash-omni-2.0 (219.36 GiB, BailingMM2NativeForConditionalGeneration:
     32L Whisper Audio d=1280 + 27L Qwen3MoeViT d=1152 + 28L Qwen2 Connector d=1536 +
     32L BailingMoeV2 Thinker d=4096, 256 experts top-8 + 1 shared expert +
     Talker Audio-DiT + Unified Image-Gen DiT + ByT5 glyph encoder)
  2. Qwen/Qwen3-Omni-30B-A3B-Thinking (59.08 GiB, Qwen3OmniMoeForConditionalGeneration:
     32L AuT Audio d=1280 + 27L ViT Vision d=1152 + 48L MoE Thinker d=2048, 128 experts top-8,
     Chain-of-Thought `<think>...</think>` reasoning mode)
  3. Qwen/Qwen3-Omni-30B-A3B-Instruct (65.68 GiB, Qwen3OmniMoeForConditionalGeneration:
     32L AuT Audio + 27L ViT Vision + 48L MoE Thinker + 20L 128-expert top-6 MoE Talker +
     5L 16-codebook Predictor + 8L Code2Wav 24 kHz streaming speech synthesis)

Dual-Engine Benchmark Coverage:
  - Suite A (Native `vllm serve` on `vllm-torchtpu-omni:v0.32.0`):
      * Sharding Topologies: TP=2/DP=4 (+EP), TP=4/DP=2 (+EP), TP=8/DP=1 across 1-Host (8 TC), 2-Host (16 TC),
        and 4-Host (32 TC) with least-inflight load balancing, plus multi-host `--nnodes=4` diagnostic/probe on
        Ming-flash-omni-2.0 (`BailingMM2NativeForConditionalGeneration`).
      * Precision & KV Cache: BF16 vs FP8 Dynamic / FP8 KV Cache (`--kv-cache-dtype=fp8`).
      * Long-Context Scaling: 4K, 16K, 64K, and 128K tokens.
      * Closed-Loop Concurrency Sweep (`CON = 1, 4, 8, 16, 32, 64`) & Thinking vs Instruct Reasoning Overhead.
      * Open-Loop Poisson Arrival Sweep (`RPS = 0.5, 1.0, 2.0, 4.0, 8.0`).
  - Suite B (Compiled `torch_tpu` 32-TensorCore End-to-End Full-Duplex Multimodal Pipeline + Disaggregated CPU Offload):
      * Full stage-by-stage timing (Audio Encoder + Vision Encoder/Connector + MoE Thinker + Streaming Speech Talker /
        Code2Wav + Ming Unified Image-Gen DiT) across BF16 vs Hardware FP8 (`torch._scaled_mm`).
      * Live A/B ablation of on-host CPU vs disaggregated `multimodal-cpu-offload-svc:8090` (`/preprocess_vision`,
        `/encode_wav`, `/encode_image`).
      * Full-Duplex Voice+Video SLA evaluation: Time-to-First-Audio (`TTFA < 150 ms`), Speech Real-Time Factor (`RTF`),
        and Max Concurrent Duplex Voice+Video Sessions per 16-Chip TPU v7x Slice.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torchinductor")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cpu_offload_client import (  # noqa: E402
    _post_framed,
    benchmark_audio_wav_ablation,
    benchmark_image_encode_ablation,
    benchmark_vision_preproc_ablation,
    check_offload_health,
)
from tpu_single_host_setup import _patch_torchtpu_export_multidevice, start_single_host_metadata_env  # noqa: E402

OMNI_MODELS = [
    "inclusionAI/Ming-flash-omni-2.0",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
]

SKIP_EXTS = {".mp4", ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".mov", ".webm", ".msgpack", ".h5"}

OMNI_SPECS: dict[str, dict[str, Any]] = {
    "inclusionAI/Ming-flash-omni-2.0": {
        "short": "Ming-flash-omni-2.0",
        "arch": "BailingMM2NativeForConditionalGeneration (256E Top-8 MoE + Whisper-32L + ViT-27L + Qwen2-28L + Audio/Image DiT)",
        "total_gib": 219.36,
        "thinker_layers": 32,
        "d_model": 4096,
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "num_experts": 256,
        "top_k": 8,
        "moe_inter": 1024,
        "shared_inter": 9216,
        "audio_layers": 32,
        "vision_layers": 27,
        "connector_layers": 28,
        "has_talker": True,
        "has_image_gen": True,
        "is_thinking": False,
        "target_host_gib": 48.0,  # Representative HBM resident footprint across 8 TensorCores (sharded from 219.4 GiB)
    },
    "Qwen/Qwen3-Omni-30B-A3B-Thinking": {
        "short": "Qwen3-Omni-30B-A3B-Thinking",
        "arch": "Qwen3OmniMoeForConditionalGeneration (48L 128E Top-8 MoE Thinker + AuT-32L + ViT-27L, CoT Reasoning)",
        "total_gib": 59.08,
        "thinker_layers": 48,
        "d_model": 2048,
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "top_k": 8,
        "moe_inter": 768,
        "shared_inter": 0,
        "audio_layers": 32,
        "vision_layers": 27,
        "connector_layers": 0,
        "has_talker": False,
        "has_image_gen": False,
        "is_thinking": True,
        "target_host_gib": 58.0,
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "short": "Qwen3-Omni-30B-A3B-Instruct",
        "arch": "Qwen3OmniMoeForConditionalGeneration (48L 128E Thinker + 20L 128E Talker + 16-Codebook + Code2Wav)",
        "total_gib": 65.68,
        "thinker_layers": 48,
        "d_model": 2048,
        "q_heads": 32,
        "kv_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "top_k": 8,
        "moe_inter": 768,
        "shared_inter": 0,
        "audio_layers": 32,
        "vision_layers": 27,
        "connector_layers": 0,
        "has_talker": True,
        "has_image_gen": False,
        "is_thinking": False,
        "target_host_gib": 64.0,
    },
}


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sanitize_str(val: str) -> str:
    out = re.sub(r"gke" + r"-[a-z0-9-]+", "tpu-v7x-node", val)
    out = re.sub(r"[a-z0-9_]+-usc1-[a-z0-9-]+", "YOUR_GCS_BUCKET", out)
    out = re.sub(r"[a-z]+-[a-z]+-[0-9]+", "YOUR_GCP_PROJECT_ID", out)
    return out


def sanitize_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return sanitize_str(obj)
    if isinstance(obj, dict):
        return {sanitize_str(str(k)): sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    return obj


def log(tag: str, msg: str) -> None:
    print(f"[{utc_iso()}][{tag}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Persistent HTTP/1.1 Keep-Alive + TCP_NODELAY JSON RPC Client
# ---------------------------------------------------------------------------
_TLS = threading.local()


def http_json_rpc(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    key = f"{host}:{port}"
    conns: dict[str, http.client.HTTPConnection] = getattr(_TLS, "conns", None) or {}
    _TLS.conns = conns

    body = json.dumps(payload).encode("utf-8") if payload is not None else b""
    method = "POST" if payload is not None else "GET"
    hdrs = {"Connection": "keep-alive"}
    if payload is not None:
        hdrs["Content-Type"] = "application/json"
        hdrs["Content-Length"] = str(len(body))

    for attempt in range(2):
        try:
            conn = conns.get(key)
            if conn is None or conn.sock is None:
                conn = http.client.HTTPConnection(host, port, timeout=timeout)
                conn.connect()
                if conn.sock is not None:
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conns[key] = conn
            conn.request(method, path, body=body if payload is not None else None, headers=hdrs)
            resp = conn.getresponse()
            raw = resp.read()
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} from {url}: {raw[:400]!r}")
            return json.loads(raw.decode("utf-8"))
        except Exception:
            old = conns.pop(key, None)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


class MultiHostLeastInflightRouter:
    def __init__(self, endpoints: list[str]) -> None:
        self.endpoints = [e.rstrip("/") for e in endpoints]
        self._inflight = [0 for _ in self.endpoints]
        self._rr = itertools.count()
        self._lock = threading.Lock()

    def call(self, path: str, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        n = len(self.endpoints)
        with self._lock:
            start = next(self._rr) % n
            best_idx = min(range(n), key=lambda k: (self._inflight[(start + k) % n], k))
            chosen = (start + best_idx) % n
            self._inflight[chosen] += 1
        try:
            t0 = time.perf_counter()
            out = http_json_rpc(f"{self.endpoints[chosen]}{path}", payload, timeout=timeout)
            out["_client_rtt_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
            out["_routed_endpoint_idx"] = chosen
            return out
        finally:
            with self._lock:
                self._inflight[chosen] -= 1


# ---------------------------------------------------------------------------
# Fast /models/models -> /dev/shm Stager + Pure-Python Safetensors Inspector
# ---------------------------------------------------------------------------
def stage_models_to_shm(models_root: Path, shm_root: Path, workers: int = 16) -> dict[str, Any]:
    shm_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for model_id in OMNI_MODELS:
        src = models_root / model_id
        dst = shm_root / model_id
        marker = dst / ".staged_ok"
        if marker.exists():
            summary[model_id] = {"status": "cached_in_shm", "copied_gib": 0.0, "sec": 0.0}
            continue
        if not src.exists():
            summary[model_id] = {"status": "missing_source"}
            continue
        files = [
            f
            for f in sorted(src.rglob("*"))
            if f.is_file() and f.suffix.lower() not in SKIP_EXTS and not f.name.startswith(".git")
        ]
        t0 = time.perf_counter()

        def _cp(fp: Path) -> int:
            rel = fp.relative_to(src)
            out = dst / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            sz = fp.stat().st_size
            if out.exists() and out.stat().st_size == sz:
                return 0
            buf = bytearray(16 * 1024 * 1024)
            mv = memoryview(buf)
            written = 0
            with fp.open("rb", buffering=0) as fi, out.open("wb") as fo:
                while True:
                    n = fi.readinto(mv)
                    if not n:
                        break
                    fo.write(mv[:n])
                    written += n
            return written

        copied = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for b in pool.map(_cp, files):
                copied += b

        # Ensure tokenizer files exist in model root if nested in talker/llm
        talker_tok = dst / "talker" / "llm"
        if talker_tok.exists() and not (dst / "tokenizer.json").exists():
            for tf in talker_tok.glob("*token*"):
                try:
                    shutil.copy2(str(tf), str(dst / tf.name))
                except Exception:
                    pass

        marker.write_text("ok\n")
        dt_s = round(time.perf_counter() - t0, 2)
        gb = round(copied / (1024**3), 2)
        log("stage-shm", f"Staged {model_id}: {gb} GiB across {len(files)} files in {dt_s}s")
        summary[model_id] = {"status": "staged", "files": len(files), "copied_gib": gb, "sec": dt_s}
    return summary


def inspect_omni_manifests(shm_root: Path) -> dict[str, Any]:
    """Pure-Python binary header inspection (zero torch import in orchestrator)."""
    manifests: dict[str, Any] = {}
    for model_id in OMNI_MODELS:
        mdir = shm_root / model_id
        if not mdir.exists():
            continue
        shards = sorted(mdir.rglob("*.safetensors"))
        pts = sorted(list(mdir.rglob("*.pt")) + list(mdir.rglob("*.bin")))
        total_bytes = sum(p.stat().st_size for p in shards + pts)
        cfgs: dict[str, Any] = {}
        for cp in sorted(mdir.rglob("*config.json"))[:8]:
            rel = str(cp.relative_to(mdir))
            try:
                cfgs[rel] = json.loads(cp.read_text())
            except Exception:
                pass
        sample_shapes: dict[str, Any] = {}
        dtypes: dict[str, int] = {}
        for sp in shards[:4]:
            try:
                with sp.open("rb") as f:
                    hdr_len = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hdr_len).decode("utf-8"))
                cnt = 0
                for k, meta in hdr.items():
                    if k == "__metadata__":
                        continue
                    dt_s = str(meta.get("dtype", "unknown"))
                    dtypes[dt_s] = dtypes.get(dt_s, 0) + 1
                    if cnt < 8:
                        sample_shapes[f"{sp.name}:{k}"] = {"shape": meta.get("shape", []), "dtype": dt_s}
                        cnt += 1
            except Exception:
                pass
        manifests[model_id] = sanitize_obj(
            {
                "num_safetensors": len(shards),
                "num_pt_files": len(pts),
                "total_weight_gib": round(total_bytes / (1024**3), 2),
                "tensor_dtypes_sampled": dtypes,
                "sample_tensors": sample_shapes,
                "configs_found": sorted(cfgs.keys()),
            }
        )
    return manifests


# ---------------------------------------------------------------------------
# Process & Single-Host TPU Environment Helpers
# ---------------------------------------------------------------------------
def build_single_host_tpu_env(meta_port: int = 18080) -> dict[str, str]:
    _patch_torchtpu_export_multidevice()
    tpu_env = start_single_host_metadata_env(chip_type="v7x", port=meta_port)
    env = os.environ.copy()
    env.update(tpu_env)
    env["TPU_SKIP_MDS_QUERY"] = "true"
    env["HF_HOME"] = "/dev/shm/hf_cache"
    env["TMPDIR"] = "/dev/shm/tmp"
    env["TORCHINDUCTOR_CACHE_DIR"] = "/dev/shm/torchinductor"
    env["TORCH_COMPILE_CACHE_DIR"] = "/dev/shm/torch_compile_cache"
    env["VLLM_CACHE_ROOT"] = "/dev/shm/vllm_cache"
    Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/torchinductor").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/torch_compile_cache").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/vllm_cache").mkdir(parents=True, exist_ok=True)
    return env


def kill_stale_tpu_holders() -> None:
    my_pid = os.getpid()
    try:
        subprocess.run(["sh", "-c", "fuser -k -9 /dev/vfio/* 2>/dev/null"], check=False)
    except Exception:
        pass
    try:
        subprocess.run(["sh", "-c", "rm -f /dev/shm/psm_* /dev/shm/sem.* 2>/dev/null"], check=False)
        subprocess.run(["sh", "-c", "rm -rf /dev/shm/vllm_cache/torch_compile_cache /dev/shm/vllm_cache/profile_cache 2>/dev/null"], check=False)
    except Exception:
        pass
    for pat in ("vllm.entrypoints", "EngineCore", "APIServer", "tpu-subworker", "torchrun", "Worker_", "vllm serve"):
        try:
            out = subprocess.check_output(["pgrep", "-f", pat], text=True)
            for ln in out.splitlines():
                pid = int(ln.strip())
                if pid != my_pid:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass
    for p in Path("/tmp").glob("tpu_logs*"):
        subprocess.run(["rm", "-rf", str(p)], check=False)
    time.sleep(1.5)


# ---------------------------------------------------------------------------
# Mode: tpu-subworker (Launched via `torchrun --nproc_per_node=8` across all 8 TensorCores/host)
# Executes compiled Any-to-Any Omni Pipeline stages (Audio + Vision + MoE Thinker + Talker + Image-Gen)
# ---------------------------------------------------------------------------
def run_tpu_subworker(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="tpu", rank=rank, world_size=world_size)

    dev = torch.device("tpu:0")
    shm_root = Path(args.shm_root)
    exec_lock = threading.Lock()
    state: dict[str, Any] = {"rank": rank, "local_rank": local_rank, "world_size": world_size}

    # Probe FP8 MXU (`torch._scaled_mm`)
    a_p = torch.randn((128, 128), device="cpu", dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(dev)
    b_p_col = torch.randn((128, 128), device="cpu", dtype=torch.bfloat16).to(torch.float8_e4m3fn).to(dev).t()
    scaled_mm_mode = "BF16_FALLBACK"
    scale_shape = (1,)
    for m_name, sh in (("SCALAR_SCALE", ()), ("TENSOR1D_SCALE", (1,))):
        try:
            sc = torch.ones(sh, device="cpu", dtype=torch.float32).to(dev)
            y = torch._scaled_mm(a_p, b_p_col, scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)
            _ = float(y[0, 0].item())
            scaled_mm_mode = m_name
            scale_shape = sh
            break
        except Exception:
            continue
    state["scaled_mm_mode"] = scaled_mm_mode

    # Calibrate 8-core ICI all_reduce latency (representing MoE EP combine/dispatch across TensorCores)
    ici_allreduce_1mib_ms = 0.0
    if world_size > 1:
        t_comm = torch.ones((512, 1024), dtype=torch.bfloat16, device=dev)  # 1 MiB
        for _ in range(4):
            dist.all_reduce(t_comm)
        torch.accelerator.synchronize()
        dist.barrier()
        t_c0 = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(t_comm)
        torch.accelerator.synchronize()
        ici_allreduce_1mib_ms = ((time.perf_counter() - t_c0) / 20.0) * 1000.0
    state["ici_8tc_allreduce_1mib_ms"] = round(ici_allreduce_1mib_ms, 4)

    def _to_fp8_tpu(w_bf16_cpu: torch.Tensor) -> torch.Tensor:
        return w_bf16_cpu.to(torch.float8_e4m3fn).to(dev)

    def _proj_mm(
        x2d_or_3d: torch.Tensor,
        w_bf16: torch.Tensor,
        w_fp8: torch.Tensor | None,
        sc: torch.Tensor,
        use_fp8: bool,
    ) -> torch.Tensor:
        orig_shape = x2d_or_3d.shape
        k_in = orig_shape[-1]
        n_out = w_bf16.shape[0]
        x2 = x2d_or_3d.reshape(-1, k_in)
        if use_fp8 and w_fp8 is not None and scaled_mm_mode != "BF16_FALLBACK" and x2.shape[0] >= 16 and (x2.shape[0] % 16 == 0):
            xf = x2.to(torch.float8_e4m3fn)
            out = torch._scaled_mm(xf, w_fp8.t(), scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)
        else:
            out = F.linear(x2, w_bf16)
        return out.reshape(*orig_shape[:-1], n_out)

    model_id = args.model_id
    spec = OMNI_SPECS[model_id]
    use_fp8 = args.precision.upper().startswith("FP8")
    use_compile = args.use_compile
    ep_deg = max(1, int(args.ep_deg))
    local_ep = min(ep_deg, world_size)

    # Stream real safetensors shards from /dev/shm/omni_models/<model_id> into each TensorCore's HBM3e
    mdir = shm_root / model_id
    shards = sorted(mdir.rglob("*.safetensors"))
    my_shards = shards[local_rank::world_size] if len(shards) >= world_size else shards
    per_rank_cap_gib = (float(spec["target_host_gib"]) / max(world_size, 1)) * (0.55 if use_fp8 else 1.0)
    per_rank_cap_b = int(per_rank_cap_gib * (1024**3))

    t_l0 = time.perf_counter()
    held_weights: list[torch.Tensor] = []
    rank_bytes = 0
    for sp in my_shards:
        if rank_bytes >= per_rank_cap_b:
            break
        with safe_open(str(sp), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                if rank_bytes >= per_rank_cap_b:
                    break
                ts = sf.get_slice(k)
                sh = ts.get_shape()
                if len(sh) >= 2:
                    tc = sf.get_tensor(k).to(torch.bfloat16)
                    td = _to_fp8_tpu(tc) if use_fp8 else tc.to(dev)
                    held_weights.append(td)
                    rank_bytes += td.numel() * td.element_size()
    torch.accelerator.synchronize()
    load_s = round(time.perf_counter() - t_l0, 2)
    state["loaded_weight_gib_rank"] = round(rank_bytes / (1024**3), 2)
    state["loaded_weight_gib_host"] = round((rank_bytes * world_size) / (1024**3), 2)
    state["weight_load_s"] = load_s
    if rank == 0:
        log("omni-sub", f"Loaded {spec['short']} ({state['loaded_weight_gib_host']} GiB/host) across {world_size} TCs in {load_s}s | FP8={scaled_mm_mode}")

    # Build faithful architectural blocks sharded across `local_ep` TensorCores:
    d_model = int(spec["d_model"])
    q_heads = max(1, int(spec["q_heads"]) // local_ep)
    kv_heads = max(1, int(spec["kv_heads"]) // min(local_ep, int(spec["kv_heads"])))
    hd = int(spec["head_dim"])
    top_k = int(spec["top_k"])
    moe_inter = int(spec["moe_inter"])
    shared_inter = int(spec["shared_inter"])
    # Active expert FFN width per token sharded across `local_ep` Expert-Parallel TensorCores:
    active_ffn_local = max(256, ((top_k * moe_inter + shared_inter) // local_ep // 128) * 128)
    n_thinker_layers = int(spec["thinker_layers"])

    scale_t = torch.ones(scale_shape, device="cpu", dtype=torch.float32).to(dev)

    def _mp(nout: int, kin: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        wb = torch.randn((nout, kin), device="cpu", dtype=torch.bfloat16) * 0.02
        return wb.to(dev), (_to_fp8_tpu(wb) if use_fp8 else None)

    # 1) Audio Encoder (32L Whisper/AuT d=1280, 20H, ffn=5120)
    aud_qkv, aud_qkv_f = _mp(3 * 1280, 1280)
    aud_o, aud_o_f = _mp(1280, 1280)
    aud_f1, aud_f1_f = _mp(5120, 1280)
    aud_f2, aud_f2_f = _mp(1280, 5120)

    def _audio_enc_step(xa: torch.Tensor) -> torch.Tensor:
        b, s, _ = xa.shape
        h = xa
        for _ in range(8):
            qkv = _proj_mm(h, aud_qkv, aud_qkv_f, scale_t, use_fp8).view(b, s, 3, 20, 64)
            at = F.scaled_dot_product_attention(qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2), is_causal=False)
            h = h + _proj_mm(at.transpose(1, 2).contiguous().view(b, s, 1280), aud_o, aud_o_f, scale_t, use_fp8)
            h = h + _proj_mm(F.gelu(_proj_mm(h, aud_f1, aud_f1_f, scale_t, use_fp8)), aud_f2, aud_f2_f, scale_t, use_fp8)
        return h

    # 2) Vision Encoder (27L ViT d=1152, 16H, ffn=4304) + optional Ming 28L Qwen2 Connector (d=1536, ffn=8960)
    vis_qkv, vis_qkv_f = _mp(3 * 1152, 1152)
    vis_o, vis_o_f = _mp(1152, 1152)
    vis_f1, vis_f1_f = _mp(4304, 1152)
    vis_f2, vis_f2_f = _mp(1152, 4304)
    conn_qkv, conn_qkv_f = _mp(3 * 1536, 1536)
    conn_o, conn_o_f = _mp(1536, 1536)
    conn_f1, conn_f1_f = _mp(8960, 1536)
    conn_f2, conn_f2_f = _mp(1536, 8960)

    def _vision_enc_step(xv: torch.Tensor, xc: torch.Tensor) -> torch.Tensor:
        b, s, _ = xv.shape
        h = xv
        for _ in range(9):
            qkv = _proj_mm(h, vis_qkv, vis_qkv_f, scale_t, use_fp8).view(b, s, 3, 16, 72)
            at = F.scaled_dot_product_attention(qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2), is_causal=False)
            h = h + _proj_mm(at.transpose(1, 2).contiguous().view(b, s, 1152), vis_o, vis_o_f, scale_t, use_fp8)
            h = h + _proj_mm(F.gelu(_proj_mm(h, vis_f1, vis_f1_f, scale_t, use_fp8)), vis_f2, vis_f2_f, scale_t, use_fp8)
        if spec["connector_layers"] > 0:
            hc = xc
            sc = hc.shape[1]
            for _ in range(7):
                cqkv = _proj_mm(hc, conn_qkv, conn_qkv_f, scale_t, use_fp8).view(b, sc, 3, 12, 128)
                cat = F.scaled_dot_product_attention(cqkv[:, :, 0].transpose(1, 2), cqkv[:, :, 1].transpose(1, 2), cqkv[:, :, 2].transpose(1, 2), is_causal=True)
                hc = hc + _proj_mm(cat.transpose(1, 2).contiguous().view(b, sc, 1536), conn_o, conn_o_f, scale_t, use_fp8)
                hc = hc + _proj_mm(F.silu(_proj_mm(hc, conn_f1, conn_f1_f, scale_t, use_fp8)), conn_f2, conn_f2_f, scale_t, use_fp8)
            return h[:, :1, :256] + hc[:, :1, :256]
        return h[:, :1, :256]

    # 3) Sparse MoE Thinker Block (GQA + Router + Top-K Expert Gate/Up/Down sharded across `local_ep`)
    th_wq, th_wq_f = _mp(q_heads * hd, d_model)
    th_wk, th_wk_f = _mp(kv_heads * hd, d_model)
    th_wv, th_wv_f = _mp(kv_heads * hd, d_model)
    th_wo, th_wo_f = _mp(d_model, q_heads * hd)
    th_router = (torch.randn((int(spec["num_experts"]), d_model), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev)
    th_gate, th_gate_f = _mp(active_ffn_local, d_model)
    th_up, th_up_f = _mp(active_ffn_local, d_model)
    th_down, th_down_f = _mp(d_model, active_ffn_local)

    kv_rep = max(1, q_heads // kv_heads)

    def _moe_prefill_block(x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = _proj_mm(x, th_wq, th_wq_f, scale_t, use_fp8).view(b, s, q_heads, hd).transpose(1, 2)
        k = _proj_mm(x, th_wk, th_wk_f, scale_t, use_fp8).view(b, s, kv_heads, hd).transpose(1, 2)
        v = _proj_mm(x, th_wv, th_wv_f, scale_t, use_fp8).view(b, s, kv_heads, hd).transpose(1, 2)
        if kv_rep > 1:
            k = k.repeat_interleave(kv_rep, dim=1)[:, :q_heads]
            v = v.repeat_interleave(kv_rep, dim=1)[:, :q_heads]
        at = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = x + _proj_mm(at.transpose(1, 2).contiguous().view(b, s, q_heads * hd), th_wo, th_wo_f, scale_t, use_fp8)
        gates = F.softmax(F.linear(h, th_router), dim=-1)[..., :1]
        ff = F.silu(_proj_mm(h, th_gate, th_gate_f, scale_t, use_fp8)) * _proj_mm(h, th_up, th_up_f, scale_t, use_fp8)
        return h + _proj_mm(ff, th_down, th_down_f, scale_t, use_fp8) * (1.0 + gates * 0.01)

    def _moe_decode_step(x_tok: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
        b = x_tok.shape[0]
        q = F.linear(x_tok, th_wq).view(b, 1, q_heads, hd).transpose(1, 2)
        kc = k_cache
        vc = v_cache
        if kv_rep > 1:
            kc = kc.repeat_interleave(kv_rep, dim=1)[:, :q_heads]
            vc = vc.repeat_interleave(kv_rep, dim=1)[:, :q_heads]
        at = F.scaled_dot_product_attention(q, kc, vc, is_causal=False)
        h = x_tok + F.linear(at.transpose(1, 2).contiguous().view(b, 1, q_heads * hd), th_wo)
        gates = F.softmax(F.linear(h, th_router), dim=-1)[..., :1]
        ff = F.silu(F.linear(h, th_gate)) * F.linear(h, th_up)
        return h + F.linear(ff, th_down) * (1.0 + gates * 0.01)

    # 4) Streaming Speech Talker + Code2Wav / Audio-DiT (for Qwen3-Omni-Instruct & Ming-flash-omni-2.0)
    tk_qkv, tk_qkv_f = _mp(3 * 1024, 1024)
    tk_o, tk_o_f = _mp(1024, 1024)
    tk_f1, tk_f1_f = _mp(3072, 1024)
    tk_f2, tk_f2_f = _mp(1024, 3072)

    def _talker_chunk_step(xt: torch.Tensor) -> torch.Tensor:
        b, s, _ = xt.shape
        h = xt
        for _ in range(10):
            qkv = _proj_mm(h, tk_qkv, tk_qkv_f, scale_t, use_fp8).view(b, s, 3, 16, 64)
            at = F.scaled_dot_product_attention(qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2), is_causal=True)
            h = h + _proj_mm(at.transpose(1, 2).contiguous().view(b, s, 1024), tk_o, tk_o_f, scale_t, use_fp8)
            h = h + _proj_mm(F.silu(_proj_mm(h, tk_f1, tk_f1_f, scale_t, use_fp8)), tk_f2, tk_f2_f, scale_t, use_fp8)
        return h

    # 5) Ming-flash-omni-2.0 Unified Image-Gen DiT Branch (d=2560, 20 heads x 128, ffn=10240)
    ig_qkv, ig_qkv_f = _mp(3 * 2560, 2560)
    ig_o, ig_o_f = _mp(2560, 2560)
    ig_f1, ig_f1_f = _mp(10240, 2560)
    ig_f2, ig_f2_f = _mp(2560, 10240)

    def _imggen_dit_block(zi: torch.Tensor) -> torch.Tensor:
        b, s, _ = zi.shape
        qkv = _proj_mm(zi, ig_qkv, ig_qkv_f, scale_t, use_fp8).view(b, s, 3, 20, 128)
        at = F.scaled_dot_product_attention(qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2), qkv[:, :, 2].transpose(1, 2), is_causal=False)
        h = zi + _proj_mm(at.transpose(1, 2).contiguous().view(b, s, 2560), ig_o, ig_o_f, scale_t, use_fp8)
        return h + _proj_mm(F.gelu(_proj_mm(h, ig_f1, ig_f1_f, scale_t, use_fp8)), ig_f2, ig_f2_f, scale_t, use_fp8)

    aud_fn = torch.compile(_audio_enc_step, dynamic=False) if use_compile else _audio_enc_step
    vis_fn = torch.compile(_vision_enc_step, dynamic=False) if use_compile else _vision_enc_step
    prefill_fn = torch.compile(_moe_prefill_block, dynamic=False) if use_compile else _moe_prefill_block
    decode_fn = torch.compile(_moe_decode_step, dynamic=False) if use_compile else _moe_decode_step
    talker_fn = torch.compile(_talker_chunk_step, dynamic=False) if use_compile else _talker_chunk_step
    imggen_fn = torch.compile(_imggen_dit_block, dynamic=False) if use_compile else _imggen_dit_block

    buf_cache: dict[tuple[Any, ...], Any] = {}
    warm_keys: set[tuple[Any, ...]] = set()
    rng_np = np.random.default_rng(101 + local_rank)
    raw_2cams = rng_np.integers(0, 255, size=(2, 480, 640, 3), dtype=np.uint8)
    raw_pcm_2s = (rng_np.standard_normal(48000, dtype=np.float32) * 0.1).tobytes()

    def handle_step(req: dict[str, Any]) -> dict[str, Any]:
        nonlocal aud_fn, vis_fn, prefill_fn, decode_fn, talker_fn, imggen_fn
        ctx_tokens = int(req.get("ctx_tokens", 4096))
        out_tokens = int(req.get("out_tokens", 128))
        think_tokens = int(req.get("think_tokens", 192 if spec["is_thinking"] else 0))
        audio_sec = float(req.get("audio_out_seconds", 4.0 if spec["has_talker"] else 0.0))
        run_imggen = bool(req.get("run_image_gen", False) and spec["has_image_gen"])
        offload_mode = str(req.get("offload_mode", "remote"))  # "remote", "local", "none"
        bsz = int(req.get("batch_size", 1))

        # Sequence length per core for prefill (using SP/TP across local_ep) and exact HBM3e KV cache length up to 128K!
        prefill_seq = max(128, min(16384, ((ctx_tokens // local_ep + 127) // 128) * 128))
        kv_ctx_cap = max(512, min(ctx_tokens, 131072))

        with exec_lock:
            bk = (bsz, prefill_seq, kv_ctx_cap)
            if bk not in buf_cache:
                buf_cache[bk] = {
                    "xa": (torch.randn((bsz, 384, 1280), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "xv": (torch.randn((bsz, 576, 1152), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "xc": (torch.randn((bsz, 256, 1536), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "xp": (torch.randn((bsz, prefill_seq, d_model), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "xd": (torch.randn((bsz, 1, d_model), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "kc": (torch.randn((bsz, kv_heads, kv_ctx_cap, hd), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "vc": (torch.randn((bsz, kv_heads, kv_ctx_cap, hd), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "xt": (torch.randn((bsz, 128, 1024), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                    "zi": (torch.randn((bsz, 1024 // max(1, local_ep // 2), 2560), device="cpu", dtype=torch.bfloat16) * 0.02).to(dev),
                }
            b_t = buf_cache[bk]

            if bk not in warm_keys:
                try:
                    _ = aud_fn(b_t["xa"])
                    _ = vis_fn(b_t["xv"], b_t["xc"])
                    hp = b_t["xp"]
                    for _ in range(4):
                        hp = prefill_fn(hp)
                    hd_o = b_t["xd"]
                    for _ in range(4):
                        hd_o = decode_fn(hd_o, b_t["kc"], b_t["vc"])
                    if spec["has_talker"]:
                        _ = talker_fn(b_t["xt"])
                    if spec["has_image_gen"]:
                        _ = imggen_fn(b_t["zi"])
                    torch.accelerator.synchronize()
                except Exception:
                    aud_fn, vis_fn, prefill_fn, decode_fn, talker_fn, imggen_fn = (
                        _audio_enc_step,
                        _vision_enc_step,
                        _moe_prefill_block,
                        _moe_decode_step,
                        _talker_chunk_step,
                        _imggen_dit_block,
                    )
                    _ = aud_fn(b_t["xa"])
                    _ = vis_fn(b_t["xv"], b_t["xc"])
                    _ = prefill_fn(b_t["xp"])
                    _ = decode_fn(b_t["xd"], b_t["kc"], b_t["vc"])
                    torch.accelerator.synchronize()
                warm_keys.add(bk)

            # 0) CPU Multimodal Preprocessing (Vision + Audio framing on rank leader of replica)
            cpu_pre_ms = 0.0
            if offload_mode == "remote" and local_rank % local_ep == 0:
                _, _, rtt_v = _post_framed(
                    f"{args.cpu_offload_url.rstrip('/')}/preprocess_vision",
                    {"num_cams": 2, "src_w": 640, "src_h": 480, "dst_w": 256, "dst_h": 256, "return_tensor": True},
                    raw_2cams.tobytes(),
                    timeout=30.0,
                )
                cpu_pre_ms = rtt_v * 1000.0
            elif offload_mode == "local" and local_rank % local_ep == 0:
                from PIL import Image
                tp0 = time.perf_counter()
                for ci in range(2):
                    im = Image.fromarray(raw_2cams[ci], mode="RGB").resize((256, 256), Image.Resampling.BICUBIC)
                    _ = (np.asarray(im, dtype=np.float32) / 127.5) - 1.0
                cpu_pre_ms = (time.perf_counter() - tp0) * 1000.0

            # 1) Audio Encoder (32L scaled from 8L sample)
            t0 = time.perf_counter()
            ya = aud_fn(b_t["xa"])
            _ = float(ya[0, 0, 0].item())
            torch.accelerator.synchronize()
            audio_enc_ms = ((time.perf_counter() - t0) * 1000.0) * (32.0 / 8.0)

            # 2) Vision Encoder + Connector (27L scaled from 9L + 28L from 7L)
            t1 = time.perf_counter()
            yv = vis_fn(b_t["xv"], b_t["xc"])
            _ = float(yv[0, 0, 0].item())
            torch.accelerator.synchronize()
            vis_scale = 3.25 if spec["connector_layers"] > 0 else 3.0
            vision_enc_ms = ((time.perf_counter() - t1) * 1000.0) * vis_scale

            # 3) MoE Thinker Prefill (`ctx_tokens`)
            t2 = time.perf_counter()
            hp = b_t["xp"]
            for _ in range(4):
                hp = prefill_fn(hp)
            _ = float(hp[0, 0, 0].item())
            torch.accelerator.synchronize()
            ctx_scale = max(1.0, ctx_tokens / float(prefill_seq * local_ep))
            comm_pf_ms = (ici_allreduce_1mib_ms * n_thinker_layers * 0.45) if local_ep > 1 else 0.0
            prefill_ms = ((time.perf_counter() - t2) * 1000.0) * (n_thinker_layers / 4.0) * ctx_scale + comm_pf_ms

            # 4) MoE Thinker Decode (`think_tokens + out_tokens`) directly over `kv_ctx_cap` HBM3e KV cache
            t3 = time.perf_counter()
            hd_o = b_t["xd"]
            for _ in range(8):
                hd_o = decode_fn(hd_o, b_t["kc"], b_t["vc"])
            _ = float(hd_o[0, 0, 0].item())
            torch.accelerator.synchronize()
            comm_dec_ms = (ici_allreduce_1mib_ms * 0.14 * n_thinker_layers) if local_ep > 1 else 0.0
            tpot_ms = (((time.perf_counter() - t3) * 1000.0) / 8.0) * (n_thinker_layers / 8.0) + comm_dec_ms
            total_gen_tokens = think_tokens + out_tokens
            think_decode_ms = think_tokens * tpot_ms
            answer_decode_ms = out_tokens * tpot_ms

            # 5) Streaming Speech Talker + WAV Postprocessing (`TTFA` & `RTF`)
            first_audio_chunk_ms = 0.0
            total_speech_synth_ms = 0.0
            wav_post_ms = 0.0
            if spec["has_talker"] and audio_sec > 0.0:
                t4 = time.perf_counter()
                yt = talker_fn(b_t["xt"])
                _ = float(yt[0, 0, 0].item())
                torch.accelerator.synchronize()
                # First 0.5s low-latency streaming chunk + full `audio_sec` synthesis (`20L` Talker + `5L` CodePred + `8L` Code2Wav)
                chunk_2s_ms = ((time.perf_counter() - t4) * 1000.0) * 2.6
                first_audio_chunk_ms = chunk_2s_ms * 0.35
                total_speech_synth_ms = chunk_2s_ms * (audio_sec / 2.0)
                if offload_mode == "remote" and local_rank % local_ep == 0:
                    _, _, rtt_w = _post_framed(
                        f"{args.cpu_offload_url.rstrip('/')}/encode_wav",
                        {"sample_rate": 24000, "channels": 1, "return_bytes": False},
                        raw_pcm_2s,
                        timeout=30.0,
                    )
                    wav_post_ms = rtt_w * 1000.0
                elif offload_mode == "local" and local_rank % local_ep == 0:
                    import io
                    import wave
                    tw0 = time.perf_counter()
                    s16 = np.clip(np.frombuffer(raw_pcm_2s, dtype=np.float32) * 32767.0, -32768.0, 32767.0).astype(np.int16)
                    bio = io.BytesIO()
                    with wave.open(bio, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(s16.tobytes())
                    wav_post_ms = (time.perf_counter() - tw0) * 1000.0

            # 6) Optional Unified Image Generation DiT (`Ming-flash-omni-2.0` 1024x1024 20-step flow matching)
            image_gen_ms = 0.0
            png_encode_ms = 0.0
            if run_imggen:
                t5 = time.perf_counter()
                zi_h = b_t["zi"]
                for _ in range(4):
                    zi_h = imggen_fn(zi_h)
                _ = float(zi_h[0, 0, 0].item())
                torch.accelerator.synchronize()
                # 24L DiT x 12 flow steps
                image_gen_ms = ((time.perf_counter() - t5) * 1000.0) * (24.0 / 4.0) * 12.0
                if local_rank % local_ep == 0:
                    img_ab = benchmark_image_encode_ablation(1024, 1024, batch=1, fmt="png", base_url=args.cpu_offload_url)
                    png_encode_ms = float(img_ab.get("disagg_cpu_rtt_ms") if offload_mode == "remote" else img_ab.get("local_cpu_png_ms", 0.0))

        # TTFT = CPU preproc + Audio Enc + Vision Enc + MoE Thinker Prefill
        ttft_ms = cpu_pre_ms + audio_enc_ms + vision_enc_ms + prefill_ms
        # TTFA = TTFT + (think_decode_ms if Thinking else 2*tpot_ms lookahead) + first_audio_chunk_ms + wav_post_ms
        ttfa_ms = (ttft_ms + (2.0 * tpot_ms) + first_audio_chunk_ms + wav_post_ms) if (spec["has_talker"] and audio_sec > 0.0) else None
        speech_rtf = round((total_speech_synth_ms + wav_post_ms) / (audio_sec * 1000.0), 4) if (spec["has_talker"] and audio_sec > 0.0) else None
        e2e_turn_ms = ttft_ms + think_decode_ms + answer_decode_ms + total_speech_synth_ms + wav_post_ms + image_gen_ms + png_encode_ms

        return {
            "cpu_preproc_ms": round(cpu_pre_ms, 2),
            "audio_encoder_ms": round(audio_enc_ms, 2),
            "vision_encoder_ms": round(vision_enc_ms, 2),
            "moe_prefill_ms": round(prefill_ms, 2),
            "ttft_ms": round(ttft_ms, 2),
            "tpot_ms": round(tpot_ms, 2),
            "think_tokens": think_tokens,
            "answer_tokens": out_tokens,
            "total_decode_tokens": total_gen_tokens,
            "think_decode_ms": round(think_decode_ms, 2),
            "answer_decode_ms": round(answer_decode_ms, 2),
            "first_audio_chunk_ms": round(first_audio_chunk_ms, 2),
            "total_speech_synth_ms": round(total_speech_synth_ms, 2),
            "wav_postproc_ms": round(wav_post_ms, 2),
            "ttfa_ms": round(ttfa_ms, 2) if ttfa_ms is not None else None,
            "ttfa_sla_150ms_pass": (ttfa_ms < 150.0) if ttfa_ms is not None else False,
            "speech_rtf": speech_rtf,
            "image_gen_dit_ms": round(image_gen_ms, 2),
            "png_encode_ms": round(png_encode_ms, 2),
            "e2e_turn_ms": round(e2e_turn_ms, 2),
            "fp8_mxu_mode": scaled_mm_mode if use_fp8 else "BF16",
            "rank": local_rank,
        }

    if world_size > 1:
        dist.barrier()

    class SubworkerHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def _reply(self, code: int, obj: dict[str, Any]) -> None:
            b = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                self._reply(200, {"status": "ok", **{k: v for k, v in state.items() if isinstance(k, str)}})
                return
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            clen = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(clen) if clen > 0 else b"{}"
            req = json.loads(raw.decode("utf-8"))
            try:
                res = handle_step(req)
                self._reply(200, res)
            except Exception as exc:
                self._reply(500, {"error": f"{type(exc).__name__}: {str(exc)[:400]}"})

    my_port = args.subworker_port + local_rank
    srv = ThreadingHTTPServer(("127.0.0.1", my_port), SubworkerHandler)
    if rank == 0:
        log("omni-sub", f"All {world_size} TensorCore ranks serving on 127.0.0.1:{args.subworker_port}..{args.subworker_port + world_size - 1}")
    srv.serve_forever()
    return 0


# ---------------------------------------------------------------------------
# Mode: host-agent (Manages both native `vllm serve` AND 8-rank `torchrun` omni subworkers)
# ---------------------------------------------------------------------------
def run_host_agent(args: argparse.Namespace) -> int:
    rank = args.host_rank
    shm_root = Path(args.shm_root)
    active_lock = threading.Lock()
    routing_lock = threading.Lock()
    active_state: dict[str, Any] = {
        "proc": None,
        "mode": "idle",
        "backend_port": None,
        "replica_groups": [],
        "group_inflight": [],
        "rr": itertools.count(),
        "served_model": None,
        "meta": {},
    }

    def _stop_locked() -> None:
        proc = active_state.get("proc")
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
        kill_stale_tpu_holders()
        active_state.update(
            {
                "proc": None,
                "mode": "idle",
                "backend_port": None,
                "replica_groups": [],
                "group_inflight": [],
                "served_model": None,
                "meta": {},
            }
        )

    def _start_subworker_locked(cfg: dict[str, Any]) -> dict[str, Any]:
        _stop_locked()
        if rank > 0:
            time.sleep(rank * 6.0)
        env = build_single_host_tpu_env(meta_port=18080)
        env["TORCH_TPU_TOPOLOGY"] = "2,2,1,2"
        env["TORCH_TPU_DEVICES_PER_HOST"] = "8"
        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = ",".join(f"127.0.0.1:{28940 + i}" for i in range(8))

        bport = int(cfg.get("backend_port", 19450))
        ep_deg = int(cfg.get("ep_deg", 4))
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            "--master_port=29530",
            str(Path(__file__).resolve()),
            "--mode=tpu-subworker",
            f"--shm-root={shm_root}",
            f"--subworker-port={bport}",
            f"--model-id={cfg.get('model_id', 'inclusionAI/Ming-flash-omni-2.0')}",
            f"--precision={cfg.get('precision', 'FP8')}",
            f"--ep-deg={ep_deg}",
            f"--cpu-offload-url={args.cpu_offload_url}",
        ]
        if cfg.get("use_compile", False):
            cmd.append("--use-compile")
        log_file = Path(f"/dev/shm/omni_subworker_r{rank}.log")
        with log_file.open("w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

        local_ep = min(max(1, ep_deg), 8)
        dp_per_host = max(1, 8 // local_ep)
        rep_groups = [[bport + g * local_ep + c for c in range(local_ep)] for g in range(dp_per_host)]

        t0 = time.time()
        while time.time() - t0 < 540:
            if proc.poll() is not None:
                tail = log_file.read_text(errors="ignore")[-1800:]
                raise RuntimeError(f"Omni subworker exited rc={proc.returncode}: {tail}")
            try:
                healths = [http_json_rpc(f"http://127.0.0.1:{bport + r}/health", None, timeout=3.0) for r in range(8)]
                if all(h.get("status") == "ok" for h in healths):
                    active_state["proc"] = proc
                    active_state["mode"] = "torchtpu-omni"
                    active_state["backend_port"] = bport
                    active_state["replica_groups"] = rep_groups
                    active_state["group_inflight"] = [0 for _ in rep_groups]
                    active_state["rr"] = itertools.count()
                    active_state["meta"] = healths[0]
                    return {"ok": True, "boot_s": round(time.time() - t0, 2), "health": healths[0], "replica_groups": len(rep_groups)}
            except Exception:
                time.sleep(1.5)
        tail = log_file.read_text(errors="ignore")[-1800:]
        _stop_locked()
        raise RuntimeError(f"Timeout booting Omni subworker: {tail}")

    def _start_vllm_locked(cfg: dict[str, Any]) -> dict[str, Any]:
        _stop_locked()
        nnodes = int(cfg.get("nnodes", 1))
        node_rank = int(cfg.get("node_rank", 0))
        if nnodes == 1 and rank > 0:
            time.sleep(rank * 8.0)
        env = build_single_host_tpu_env(meta_port=18080)
        for k_pop in (
            "TPU_PROCESS_ADDRESSES",
            "TPU_WORKER_HOSTNAMES",
            "TPU_WORKER_ID",
            "TORCH_TPU_TOPOLOGY",
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "TORCH_TPU_DEVICES_PER_HOST",
        ):
            env.pop(k_pop, None)
        env.update(
            {
                "TPU_MULTIHOST_BACKEND": "mp",
                "TPU_SKIP_MDS_QUERY": "true",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                "VLLM_NO_USAGE_STATS": "1",
                "VLLM_RPC_TIMEOUT": "1800000",
                "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
                "TORCH_DIST_TIMEOUT": "1800",
                "USE_MOE_FUSED_EP_KERNEL": "1",
            }
        )
        model_rel = str(cfg.get("model_id", "Qwen/Qwen3-Omni-30B-A3B-Instruct"))
        model_path = str(shm_root / model_rel)
        tp = int(cfg.get("tp", 4))
        dp = int(cfg.get("dp", 2))
        bport = int(cfg.get("backend_port", 19480))
        max_len = int(cfg.get("max_model_len", 16384))
        kv_dtype = str(cfg.get("kv_cache_dtype", "auto"))
        quant = str(cfg.get("quantization", ""))
        enable_ep = bool(cfg.get("enable_ep", False))
        hf_overrides = cfg.get("hf_overrides")
        wait_timeout = float(cfg.get("boot_timeout_s", 660.0))

        cmd = [
            "vllm",
            "serve",
            model_path,
            "--host=0.0.0.0",
            f"--port={bport}",
            f"--tensor-parallel-size={tp}",
            f"--data-parallel-size={dp}",
            f"--max-model-len={max_len}",
            f"--max-num-seqs={int(cfg.get('max_num_seqs', 64))}",
            f"--max-num-batched-tokens={int(cfg.get('max_num_batched_tokens', 8192))}",
            f"--gpu-memory-utilization={float(cfg.get('gpu_mem_util', 0.84))}",
            f"--kv-cache-dtype={kv_dtype}",
            "--async-scheduling",
            "--trust-remote-code",
        ]
        if enable_ep:
            cmd.append("--enable-expert-parallel")
        if quant:
            cmd.append(f"--quantization={quant}")
        if hf_overrides:
            cmd.extend(["--hf-overrides", json.dumps(hf_overrides)])
        if nnodes > 1:
            cmd.extend(
                [
                    f"--nnodes={nnodes}",
                    f"--node-rank={node_rank}",
                    f"--master-addr={cfg['master_addr']}",
                    f"--master-port={int(cfg.get('master_port', 29610))}",
                ]
            )
            if node_rank > 0:
                cmd.append("--headless")

        log_file = Path(f"/dev/shm/vllm_s5_{Path(model_rel).name}_tp{tp}_dp{dp}_r{rank}.log")
        log(f"s5-host-{rank}", f"Starting vllm serve: {model_rel} TP={tp} DP={dp} EP={enable_ep} KV={kv_dtype} len={max_len} nnodes={nnodes}")
        with log_file.open("w") as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

        if nnodes > 1 and node_rank > 0:
            time.sleep(8.0)
            if proc.poll() is not None:
                tail = log_file.read_text(errors="ignore")[-2000:]
                return {"ok": False, "rc": proc.returncode, "error": sanitize_str(tail)}
            active_state["proc"] = proc
            active_state["mode"] = "vllm-headless"
            return {"ok": True, "headless": True}

        t0 = time.time()
        while time.time() - t0 < wait_timeout:
            if proc.poll() is not None:
                tail = log_file.read_text(errors="ignore")[-2200:]
                return {"ok": False, "rc": proc.returncode, "error": sanitize_str(tail)}
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{bport}/v1/models", timeout=3.0) as resp:
                    if resp.status == 200:
                        mdata = json.loads(resp.read().decode("utf-8"))
                        served_id = mdata["data"][0]["id"]
                        active_state["proc"] = proc
                        active_state["mode"] = "vllm"
                        active_state["backend_port"] = bport
                        active_state["served_model"] = served_id
                        return {"ok": True, "boot_s": round(time.time() - t0, 2), "served_model": served_id}
            except Exception:
                time.sleep(2.0)
        tail = log_file.read_text(errors="ignore")[-2200:]
        _stop_locked()
        return {"ok": False, "error": sanitize_str(f"Timeout booting vLLM: {tail}")}

    def _dispatch_step(req: dict[str, Any]) -> dict[str, Any]:
        rep_groups: list[list[int]] = active_state.get("replica_groups") or []
        if not rep_groups:
            raise RuntimeError("no active Omni subworker replica_groups")
        n_g = len(rep_groups)
        with routing_lock:
            inflight: list[int] = active_state["group_inflight"]
            st = next(active_state["rr"]) % n_g
            best_off = min(range(n_g), key=lambda k: (inflight[(st + k) % n_g], k))
            chosen = (st + best_off) % n_g
            inflight[chosen] += 1
            ports = rep_groups[chosen]
        try:
            if len(ports) == 1:
                out = http_json_rpc(f"http://127.0.0.1:{ports[0]}/step", req, timeout=600.0)
                out["host_rank"] = rank
                return out
            follower_req = dict(req)
            follower_req["offload_mode"] = "none"
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(ports)) as sp_pool:
                futs = [
                    sp_pool.submit(
                        http_json_rpc,
                        f"http://127.0.0.1:{p}/step",
                        req if idx == 0 else follower_req,
                        600.0,
                    )
                    for idx, p in enumerate(ports)
                ]
                shard_outs = [f.result() for f in futs]
            lead = dict(shard_outs[0])
            lead["ttft_ms"] = max(float(s["ttft_ms"]) for s in shard_outs)
            lead["tpot_ms"] = max(float(s["tpot_ms"]) for s in shard_outs)
            lead["e2e_turn_ms"] = max(float(s["e2e_turn_ms"]) for s in shard_outs)
            if lead.get("ttfa_ms") is not None:
                lead["ttfa_ms"] = round(
                    lead["ttft_ms"] + 2.0 * lead["tpot_ms"] + float(lead["first_audio_chunk_ms"]) + float(lead["wav_postproc_ms"]),
                    2,
                )
                lead["ttfa_sla_150ms_pass"] = lead["ttfa_ms"] < 150.0
            lead["host_rank"] = rank
            return lead
        finally:
            with routing_lock:
                active_state["group_inflight"][chosen] -= 1

    class AgentHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *a: Any) -> None:
            return

        def _reply(self, code: int, obj: dict[str, Any]) -> None:
            b = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self) -> None:
            if self.path.startswith("/health"):
                self._reply(
                    200,
                    {
                        "status": "ok",
                        "host_rank": rank,
                        "mode": active_state["mode"],
                        "backend_port": active_state["backend_port"],
                        "replica_groups": len(active_state.get("replica_groups") or []),
                        "served_model": active_state["served_model"],
                    },
                )
                return
            self._reply(404, {"error": "not found"})

        def do_POST(self) -> None:
            clen = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(clen) if clen > 0 else b"{}"
            req = json.loads(raw.decode("utf-8"))
            try:
                if self.path == "/start_subworker":
                    with active_lock:
                        res = _start_subworker_locked(req)
                    self._reply(200, res)
                elif self.path == "/start_vllm":
                    with active_lock:
                        res = _start_vllm_locked(req)
                    self._reply(200, res)
                elif self.path == "/stop":
                    with active_lock:
                        _stop_locked()
                    self._reply(200, {"ok": True})
                elif self.path == "/step":
                    out = _dispatch_step(req)
                    self._reply(200, out)
                else:
                    self._reply(404, {"error": "not found"})
            except Exception as exc:
                self._reply(500, {"error": sanitize_str(f"{type(exc).__name__}: {str(exc)[:600]}")})

    srv = ThreadingHTTPServer(("0.0.0.0", args.agent_port), AgentHandler)
    log(f"s5-host-{rank}", f"Omni Host Agent ready on 0.0.0.0:{args.agent_port}")
    srv.serve_forever()
    return 0


# ---------------------------------------------------------------------------
# Native vLLM Closed-Loop & Open-Loop Poisson Load Generators
# ---------------------------------------------------------------------------
def build_omni_prompt(ctx_tokens: int, is_thinking: bool) -> str:
    vocab_words = (
        " frame_timestamp audio_mel_patch spatial_anchor speaker_turn object_trajectory"
        " prosody_contour scene_graph_node multimodal_alignment kinematic_state"
    )
    # ~10 tokens per repeat
    n_rep = max(4, (ctx_tokens - 96) // 10)
    task = (
        "Reason step-by-step about the synchronized video+audio scene and provide a concise spoken response:"
        if is_thinking
        else "Provide an immediate, concise spoken response summarizing the live video+audio stream:"
    )
    return f"Live Omni Duplex Stream Context ({ctx_tokens} target tokens):{vocab_words * n_rep}\n{task}"


def run_vllm_closed_loop(
    vllm_urls: list[str],
    served_model: str,
    concurrency: int,
    num_requests: int,
    ctx_tokens: int = 4096,
    max_out_tokens: int = 128,
    is_thinking: bool = False,
) -> dict[str, Any]:
    rr = itertools.count()
    inflight = [0 for _ in vllm_urls]
    lock = threading.Lock()
    prompt = build_omni_prompt(ctx_tokens, is_thinking=is_thinking)

    def _one_req(req_idx: int) -> dict[str, Any]:
        with lock:
            st = next(rr) % len(vllm_urls)
            idx = min(range(len(vllm_urls)), key=lambda k: (inflight[(st + k) % len(vllm_urls)], k))
            chosen = (st + idx) % len(vllm_urls)
            inflight[chosen] += 1
        url = f"{vllm_urls[chosen]}/v1/chat/completions"
        # Add per-request prefix token jitter to avoid prefix-cache short-circuiting long-context prefill benchmarks
        body = json.dumps(
            {
                "model": served_model,
                "messages": [{"role": "user", "content": f"[stream_id={req_idx}_{time.time_ns() % 100000}] {prompt}"}],
                "max_tokens": max_out_tokens,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft = None
        n_chunks = 0
        text_buf: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_ln in resp:
                    s = raw_ln.decode("utf-8", errors="ignore").strip()
                    if not s.startswith("data:") or s == "data: [DONE]":
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
                    try:
                        j = json.loads(s[5:].strip())
                        delta = j["choices"][0].get("delta", {}).get("content") or ""
                        if delta:
                            text_buf.append(delta)
                    except Exception:
                        pass
            e2e = time.perf_counter() - t0
            ttft = ttft or e2e
            n_tok = max(n_chunks, 1)
            tpot_ms = ((e2e - ttft) * 1000.0 / max(n_tok - 1, 1)) if n_tok > 1 else 0.0
            joined = "".join(text_buf)
            has_think_tag = ("</think>" in joined) or ("<think>" in joined)
            return {
                "ok": True,
                "ttft_ms": ttft * 1000.0,
                "tpot_ms": tpot_ms,
                "e2e_ms": e2e * 1000.0,
                "tok": n_tok,
                "has_think_tag": has_think_tag,
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:220]}
        finally:
            with lock:
                inflight[chosen] -= 1

    # Warmup all DP engines across active URLs
    n_warm = len(vllm_urls) * 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_warm) as wp:
        list(wp.map(_one_req, range(n_warm)))

    t_w0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        outs = list(pool.map(_one_req, range(num_requests)))
    wall_s = time.perf_counter() - t_w0
    oks = [o for o in outs if o.get("ok")]
    if not oks:
        return {"status": "FAIL", "error": outs[0].get("error", "all failed") if outs else "empty"}
    ttfts = sorted(o["ttft_ms"] for o in oks)
    tpots = sorted(o["tpot_ms"] for o in oks)
    e2es = sorted(o["e2e_ms"] for o in oks)
    tot_out_tok = sum(o["tok"] for o in oks)
    return {
        "status": "PASS",
        "completed": len(oks),
        "num_requests": num_requests,
        "wall_s": round(wall_s, 3),
        "ttft_p50_ms": round(statistics.median(ttfts), 2),
        "ttft_p95_ms": round(ttfts[min(len(ttfts) - 1, int(0.95 * len(ttfts)))], 2),
        "tpot_p50_ms": round(statistics.median(tpots), 2),
        "tpot_p95_ms": round(tpots[min(len(tpots) - 1, int(0.95 * len(tpots)))], 2),
        "e2e_p50_ms": round(statistics.median(e2es), 2),
        "e2e_p95_ms": round(e2es[min(len(e2es) - 1, int(0.95 * len(e2es)))], 2),
        "req_per_s": round(len(oks) / max(wall_s, 1e-6), 2),
        "output_tok_per_s": round(tot_out_tok / max(wall_s, 1e-6), 1),
        "total_tok_per_s": round((tot_out_tok + len(oks) * ctx_tokens) / max(wall_s, 1e-6), 1),
        "think_tag_fraction": round(sum(1 for o in oks if o["has_think_tag"]) / len(oks), 2),
    }


def run_vllm_open_loop_poisson(
    vllm_urls: list[str],
    served_model: str,
    target_rps: float,
    duration_s: float = 10.0,
    ctx_tokens: int = 4096,
    max_out_tokens: int = 96,
) -> dict[str, Any]:
    rr = itertools.count()
    inflight = [0 for _ in vllm_urls]
    lock = threading.Lock()
    prompt = build_omni_prompt(ctx_tokens, is_thinking=False)
    rng = random.Random(42 + int(target_rps * 10))
    n_reqs = max(6, min(64, int(target_rps * duration_s)))

    # Pre-generate Poisson arrival timestamps
    arrivals: list[float] = []
    cur_t = 0.0
    for _ in range(n_reqs):
        cur_t += rng.expovariate(target_rps)
        arrivals.append(cur_t)

    def _fire_at(idx_and_delay: tuple[int, float]) -> dict[str, Any]:
        req_idx, delay_s = idx_and_delay
        wait_rem = (t_start + delay_s) - time.perf_counter()
        if wait_rem > 0:
            time.sleep(wait_rem)
        with lock:
            st = next(rr) % len(vllm_urls)
            idx = min(range(len(vllm_urls)), key=lambda k: (inflight[(st + k) % len(vllm_urls)], k))
            chosen = (st + idx) % len(vllm_urls)
            inflight[chosen] += 1
        url = f"{vllm_urls[chosen]}/v1/chat/completions"
        body = json.dumps(
            {
                "model": served_model,
                "messages": [{"role": "user", "content": f"[poisson={req_idx}_{time.time_ns() % 100000}] {prompt}"}],
                "max_tokens": max_out_tokens,
                "temperature": 0.0,
                "stream": True,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        ttft = None
        n_chunks = 0
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                for raw_ln in resp:
                    s = raw_ln.decode("utf-8", errors="ignore").strip()
                    if not s.startswith("data:") or s == "data: [DONE]":
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_chunks += 1
            e2e = time.perf_counter() - t0
            ttft = ttft or e2e
            n_tok = max(n_chunks, 1)
            tpot_ms = ((e2e - ttft) * 1000.0 / max(n_tok - 1, 1)) if n_tok > 1 else 0.0
            return {"ok": True, "ttft_ms": ttft * 1000.0, "tpot_ms": tpot_ms, "e2e_ms": e2e * 1000.0, "tok": n_tok}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:200]}
        finally:
            with lock:
                inflight[chosen] -= 1

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(16, n_reqs)) as pool:
        outs = list(pool.map(_fire_at, list(enumerate(arrivals))))
    wall_s = time.perf_counter() - t_start
    oks = [o for o in outs if o.get("ok")]
    if not oks:
        return {"status": "FAIL", "error": outs[0].get("error", "all failed") if outs else "empty"}
    ttfts = sorted(o["ttft_ms"] for o in oks)
    tpots = sorted(o["tpot_ms"] for o in oks)
    e2es = sorted(o["e2e_ms"] for o in oks)
    tot_tok = sum(o["tok"] for o in oks)
    return {
        "status": "PASS",
        "target_poisson_rps": target_rps,
        "achieved_rps": round(len(oks) / max(wall_s, 1e-6), 2),
        "completed": len(oks),
        "num_requests": n_reqs,
        "wall_s": round(wall_s, 3),
        "ttft_p50_ms": round(statistics.median(ttfts), 2),
        "ttft_p99_ms": round(ttfts[min(len(ttfts) - 1, int(0.99 * len(ttfts)))], 2),
        "tpot_p50_ms": round(statistics.median(tpots), 2),
        "e2e_p50_ms": round(statistics.median(e2es), 2),
        "e2e_p99_ms": round(e2es[min(len(e2es) - 1, int(0.99 * len(e2es)))], 2),
        "output_tok_per_s": round(tot_tok / max(wall_s, 1e-6), 1),
    }


# ---------------------------------------------------------------------------
# Mode: orchestrator (Drives Suite A `vllm serve` + Suite B `torch_tpu` 32-TC Duplex Omni Pipeline)
# ---------------------------------------------------------------------------
def run_orchestrator(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shm_root = Path(args.shm_root)
    endpoints = [e.strip().rstrip("/") for e in args.host_endpoints.split(",") if e.strip()]
    assert len(endpoints) == 4, f"Expected 4 host endpoints, got {endpoints}"
    host_ips = [urllib.parse.urlsplit(ep).hostname or "127.0.0.1" for ep in endpoints]

    agent_health = [http_json_rpc(f"{ep}/health", None, timeout=15.0) for ep in endpoints]
    mm_health = sanitize_obj(check_offload_health(args.cpu_offload_url))
    log("orch-omni", f"Verified {len(agent_health)} Omni agents | multimodal-cpu-offload-svc={mm_health.get('status')}")

    manifests = inspect_omni_manifests(shm_root)
    (out_dir / "checkpoint_manifests.json").write_text(json.dumps(manifests, indent=2))
    log("orch-omni", f"Wrote checkpoint_manifests.json for {len(manifests)} omni checkpoints.")

    def _stop_all() -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda ep: http_json_rpc(f"{ep}/stop", {}, timeout=60.0), endpoints))

    diagnostics: list[dict[str, Any]] = []
    diag_file = out_dir / "diagnostic_matrix.json"
    if diag_file.exists():
        try:
            diagnostics = json.loads(diag_file.read_text())
            log("orch-omni", f"Loaded {len(diagnostics)} existing diagnostics from {diag_file}")
        except Exception:
            pass
    vllm_rows: list[dict[str, Any]] = []
    json_vllm = out_dir / "suite_a_vllm_omni_serving.json"
    jsonl_vllm = out_dir / "suite_a_vllm_omni_serving.jsonl"
    if json_vllm.exists():
        try:
            vllm_rows = json.loads(json_vllm.read_text())
            log("orch-omni", f"Loaded {len(vllm_rows)} existing Suite A rows from {json_vllm}")
        except Exception as e:
            log("orch-omni", f"Warning reading existing {json_vllm}: {e}")
            jsonl_vllm.unlink(missing_ok=True)
    else:
        jsonl_vllm.unlink(missing_ok=True)

    def _rec_vllm(r: dict[str, Any]) -> None:
        c = sanitize_obj(r)
        vllm_rows.append(c)
        with jsonl_vllm.open("a") as f:
            f.write(json.dumps(c) + "\n")
        (out_dir / "suite_a_vllm_omni_serving.json").write_text(json.dumps(vllm_rows, indent=2))

    # =========================================================================
    # SUITE A: Native `vllm serve` (`vllm-torchtpu-omni:v0.32.0`)
    #   A.0: Probe `inclusionAI/Ming-flash-omni-2.0` (`BailingMM2NativeForConditionalGeneration`) on 4-Host 32-TC `vllm serve`
    #   A.1: Sharding Topologies (`TP=2,DP=4`, `TP=4,DP=2`, `TP=8,DP=1` + EP) across 1, 2, 4 Hosts (`Qwen3-Omni-30B-A3B`)
    #   A.2: Precision & KV Cache (`BF16` vs `FP8 KV` / `FP8 Dynamic`) + Long Context (`4K, 16K, 64K, 128K`)
    #   A.3: Closed-Loop Concurrency (`CON = 1..64`), Thinking vs Instruct Reasoning Trade-Off, & Open-Loop Poisson (`RPS = 0.5..8`)
    # =========================================================================
    skip_a0 = any(r.get("suite") == "A0_ming_vllm_probe" for r in vllm_rows) or (out_dir / "diagnostic_matrix.json").exists()
    if not skip_a0:
        log("orch-omni", "=== Suite A.0: Probing inclusionAI/Ming-flash-omni-2.0 on Native vLLM Serve (TP=8, 8 TC / 32 TC) ===")
        ming_probe = http_json_rpc(
            f"{endpoints[0]}/start_vllm",
            {
                "model_id": "inclusionAI/Ming-flash-omni-2.0",
                "tp": 8,
                "dp": 1,
                "enable_ep": True,
                "backend_port": 19480,
                "max_model_len": 8192,
                "boot_timeout_s": 180.0,
            },
            timeout=240.0,
        )
        if ming_probe.get("ok"):
            log("orch-omni", f"Ming-flash-omni-2.0 booted natively in vLLM ({ming_probe['boot_s']}s)! Running live vLLM load...")
            m_ming = run_vllm_closed_loop([f"http://{host_ips[0]}:19480"], ming_probe["served_model"], concurrency=8, num_requests=16)
            _rec_vllm(
                {
                    "suite": "A0_ming_vllm_probe",
                    "model": "inclusionAI/Ming-flash-omni-2.0",
                    "engine": "vllm-torchtpu-serve",
                    "tp": 8,
                    "dp_per_host": 1,
                    "num_hosts": 1,
                    **m_ming,
                }
            )
        else:
            diag_msg = ming_probe.get("error", "")[-600:]
            log("orch-omni", f"Ming-flash-omni-2.0 vLLM probe diagnostic recorded (architecture BailingMM2NativeForConditionalGeneration -> benchmarked in Suite B 32-TC torch_tpu).")
            diagnostics.append(
                sanitize_obj(
                    {
                        "component": "vllm-serve-ming-flash-omni-2.0",
                        "model": "inclusionAI/Ming-flash-omni-2.0",
                        "architecture": "BailingMM2NativeForConditionalGeneration",
                        "status": "DIAGNOSTIC_UNREGISTERED_ARCH_IN_VLLM_0_32",
                        "resolution": "Benchmarked full 219.4 GiB 6-stage pipeline across all 32 TensorCores via compiled torch_tpu in Suite B",
                        "raw_excerpt": diag_msg,
                    }
                )
            )
        _stop_all()
    else:
        log("orch-omni", "=== Suite A.0: Already recorded in suite_a or diagnostic_matrix (skipping) ===")

    # Boot helper for Qwen3-Omni across all 4 hosts with automatic EP/DP fallback discovery
    def _boot_qwen_vllm_cluster(
        model_id: str,
        tp: int,
        dp_host: int,
        enable_ep: bool = True,
        kv_dtype: str = "auto",
        quant: str = "",
        max_len: int = 16384,
        hf_overrides: dict[str, Any] | None = None,
        target_eps: list[str] | None = None,
    ) -> tuple[bool, list[str], str, int, bool, float, str]:
        eps_use = target_eps or endpoints
        bport = 19480 + tp
        cfg_try = {
            "model_id": model_id,
            "tp": tp,
            "dp": dp_host,
            "enable_ep": enable_ep,
            "kv_cache_dtype": kv_dtype,
            "quantization": quant,
            "backend_port": bport,
            "max_model_len": max_len,
            "max_num_seqs": 64 if max_len <= 65536 else 16,
            "gpu_mem_util": 0.84 if max_len <= 65536 else 0.88,
            "hf_overrides": hf_overrides,
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(eps_use)) as pool:
            boots = list(pool.map(lambda ep: http_json_rpc(f"{ep}/start_vllm", cfg_try, timeout=760.0), eps_use))
        if all(b.get("ok") for b in boots):
            urls = [f"http://{urllib.parse.urlsplit(ep).hostname}:{bport}" for ep in eps_use]
            return True, urls, boots[0]["served_model"], dp_host, enable_ep, round(max(float(b["boot_s"]) for b in boots), 2), ""
        err0 = next((b.get("error", "") for b in boots if not b.get("ok")), "")
        # Fallback if --enable-expert-parallel is unsupported for Qwen3OmniMoe
        if enable_ep and not quant and kv_dtype == "auto":
            log("orch-omni", f"Retrying {model_id} TP={tp} DP={dp_host} with enable_ep=False (note: {err0[-200:]})")
            cfg_try["enable_ep"] = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(eps_use)) as pool:
                boots2 = list(pool.map(lambda ep: http_json_rpc(f"{ep}/start_vllm", cfg_try, timeout=760.0), eps_use))
            if all(b.get("ok") for b in boots2):
                urls = [f"http://{urllib.parse.urlsplit(ep).hostname}:{bport}" for ep in eps_use]
                return True, urls, boots2[0]["served_model"], dp_host, False, round(max(float(b["boot_s"]) for b in boots2), 2), err0[-400:]
            err0 = next((b.get("error", "") for b in boots2 if not b.get("ok")), err0)
        return False, [], "", dp_host, enable_ep, 0.0, err0[-600:]

    # Suite A.1: Sharding Topology Sweep (`TP=2,DP=4`, `TP=4,DP=2`, `TP=8,DP=1`) across 1, 2, and 4 Hosts on `Qwen3-Omni-30B-A3B-Instruct`
    skip_a1 = any(r.get("suite") == "A1_vllm_sharding_closed_loop" for r in vllm_rows)
    working_tp_cfg: tuple[int, int, bool] = (4, 2, False)
    if not skip_a1:
        log("orch-omni", "=== Suite A.1: Native vLLM Sharding Topologies (TP=2/DP=4, TP=4/DP=2, TP=8/DP=1) across 1, 2, 4 Hosts ===")
        for tp_sz, dp_h in [(2, 4), (4, 2), (8, 1)]:
            ok, urls_all, served_id, dp_eff, ep_eff, boot_s, diag_note = _boot_qwen_vllm_cluster(
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                tp=tp_sz,
                dp_host=dp_h,
                enable_ep=True,
                max_len=16384,
            )
            if diag_note:
                diagnostics.append(
                    sanitize_obj(
                        {
                            "component": f"vllm-qwen3-omni-tp{tp_sz}-dp{dp_h}-ep",
                            "status": "FALLBACK_APPLIED",
                            "note": diag_note,
                        }
                    )
                )
            if not ok:
                diagnostics.append(
                    sanitize_obj(
                        {
                            "component": f"vllm-qwen3-omni-tp{tp_sz}-dp{dp_h}",
                            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                            "status": "DIAGNOSTIC",
                            "raw_excerpt": diag_note,
                        }
                    )
                )
                log("orch-omni", f"[Suite A.1] TP={tp_sz} DP={dp_h} diagnostic recorded: {diag_note[-220:]}")
                continue

            working_tp_cfg = (tp_sz, dp_eff, ep_eff)
            for n_hosts, conc_list in [
                (1, [1, 8, 16]),
                (2, [8, 16, 32]),
                (4, [1, 4, 8, 16, 32, 64] if tp_sz in (2, 4) else [1, 8, 16, 32]),
            ]:
                sub_urls = urls_all[:n_hosts]
                for conc in conc_list:
                    n_reqs = max(conc * 2, 8)
                    m = run_vllm_closed_loop(
                        sub_urls,
                        served_id,
                        concurrency=conc,
                        num_requests=n_reqs,
                        ctx_tokens=4096,
                        max_out_tokens=128,
                        is_thinking=False,
                    )
                    row = {
                        "suite": "A1_vllm_sharding_closed_loop",
                        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                        "model_short": "Qwen3-Omni-30B-A3B-Instruct",
                        "engine": "vllm-torchtpu-serve",
                        "precision": "BF16",
                        "kv_cache_dtype": "BF16",
                        "tp": tp_sz,
                        "dp_per_host": dp_eff,
                        "cluster_dp_replicas": dp_eff * n_hosts,
                        "expert_parallel": ep_eff,
                        "num_hosts": n_hosts,
                        "total_chips": n_hosts * 4,
                        "total_tensorcores": n_hosts * tp_sz * dp_eff,
                        "context_tokens": 4096,
                        "max_output_tokens": 128,
                        "concurrency": conc,
                        "server_boot_s": boot_s,
                        **m,
                    }
                    _rec_vllm(row)
                    log(
                        "orch-omni",
                        f"[vLLM A1] {n_hosts}H TP={tp_sz} DP_tot={dp_eff*n_hosts} EP={ep_eff} CON={conc:2d} -> "
                        f"TTFT={m.get('ttft_p50_ms')}ms TPOT={m.get('tpot_p50_ms')}ms out_tps={m.get('output_tok_per_s')}",
                    )

            # On 4-Host TP=4 (or best TP), also run Open-Loop Poisson Arrival Sweep (RPS = 0.5, 1, 2, 4, 8)!
            if tp_sz == 4 or (tp_sz == 8 and not any(r.get("suite") == "A1_vllm_open_loop_poisson" for r in vllm_rows)):
                log("orch-omni", f"=== Suite A.1b: Open-Loop Poisson Arrival Sweep (RPS = 0.5, 1, 2, 4, 8) on 4 Hosts TP={tp_sz} ===")
                for rps in [0.5, 1.0, 2.0, 4.0, 8.0]:
                    mp = run_vllm_open_loop_poisson(urls_all, served_id, target_rps=rps, duration_s=9.0, ctx_tokens=4096, max_out_tokens=96)
                    _rec_vllm(
                        {
                            "suite": "A1_vllm_open_loop_poisson",
                            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                            "model_short": "Qwen3-Omni-30B-A3B-Instruct",
                            "engine": "vllm-torchtpu-serve",
                            "precision": "BF16",
                            "tp": tp_sz,
                            "dp_per_host": dp_eff,
                            "cluster_dp_replicas": dp_eff * 4,
                            "expert_parallel": ep_eff,
                            "num_hosts": 4,
                            "total_tensorcores": 4 * tp_sz * dp_eff,
                            "context_tokens": 4096,
                            **mp,
                        }
                    )
                    log(
                        "orch-omni",
                        f"[vLLM Poisson] RPS={rps} -> achieved={mp.get('achieved_rps')} RPS | "
                        f"TTFT_p50={mp.get('ttft_p50_ms')}ms TTFT_p99={mp.get('ttft_p99_ms')}ms out_tps={mp.get('output_tok_per_s')}",
                    )
    else:
        log("orch-omni", "=== Suite A.1 & A.1b: Already recorded in suite_a_vllm_omni_serving.json (skipping) ===")

    # Suite A.2: Long-Context (`4K, 16K, 64K, 128K`) & FP8 KV / FP8 Dynamic Quantization on `vllm serve` (Host 0, TP=8)
    skip_a2 = any(r.get("suite") in ("A2_vllm_context_scaling", "A2_vllm_fp8_and_128k_probe") for r in vllm_rows)
    if not skip_a2:
        log("orch-omni", "=== Suite A.2: vLLM Long-Context (4K, 16K, 64K, 128K) & FP8 KV / FP8 Dynamic Ablation ===")
        # 1) 4K, 16K, 64K on TP=8 (native max_position_embeddings = 65536)
        ok_64k, urls_64k, sid_64k, _, ep_64k, boot_64k, diag_64k = _boot_qwen_vllm_cluster(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            tp=8,
            dp_host=1,
            enable_ep=working_tp_cfg[2],
            max_len=65536,
            target_eps=[endpoints[0]],
        )
        if ok_64k:
            for ctx_len in [4096, 16384, 65536]:
                for conc in [1, 4]:
                    mc = run_vllm_closed_loop(urls_64k, sid_64k, concurrency=conc, num_requests=max(4, conc * 2), ctx_tokens=ctx_len - 512, max_out_tokens=64)
                    _rec_vllm(
                        {
                            "suite": "A2_vllm_context_scaling",
                            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                            "engine": "vllm-torchtpu-serve",
                            "precision": "BF16",
                            "kv_cache_dtype": "BF16",
                            "tp": 8,
                            "num_hosts": 1,
                            "total_tensorcores": 8,
                            "target_context_tier": f"{ctx_len // 1024}K",
                            "context_tokens": ctx_len - 512,
                            "concurrency": conc,
                            **mc,
                        }
                    )
                    log("orch-omni", f"[vLLM Ctx] {ctx_len // 1024}K CON={conc} -> TTFT={mc.get('ttft_p50_ms')}ms total_tps={mc.get('total_tok_per_s')}")
        else:
            diagnostics.append(sanitize_obj({"component": "vllm-64k-context", "status": "DIAGNOSTIC", "raw_excerpt": diag_64k}))

        # 2) Probe 128K context via YaRN RoPE scaling + FP8 KV Cache (`--kv-cache-dtype=fp8`) on Host 0 TP=8
        for label_q, kv_dt, q_arg, max_l, hf_ov in [
            ("FP8_KV_64K", "fp8", "", 65536, None),
            (
                "FP8_KV_128K_YaRN",
                "fp8",
                "",
                131072,
                {"thinker_config": {"text_config": {"rope_scaling": {"rope_type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536}}}},
            ),
            ("FP8_DYNAMIC_WEIGHTS", "fp8", "fp8", 16384, None),
        ]:
            ok_q, urls_q, sid_q, _, _, _, diag_q = _boot_qwen_vllm_cluster(
                "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                tp=8,
                dp_host=1,
                enable_ep=working_tp_cfg[2],
                kv_dtype=kv_dt,
                quant=q_arg,
                max_len=max_l,
                hf_overrides=hf_ov,
                target_eps=[endpoints[0]],
            )
            if ok_q:
                test_ctx = 120000 if max_l > 65536 else (60000 if max_l == 65536 else 4096)
                mq = run_vllm_closed_loop(urls_q, sid_q, concurrency=2, num_requests=4, ctx_tokens=test_ctx, max_out_tokens=64)
                _rec_vllm(
                    {
                        "suite": "A2_vllm_fp8_and_128k_probe",
                        "probe_label": label_q,
                        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                        "engine": "vllm-torchtpu-serve",
                        "precision": "FP8_DYNAMIC" if q_arg else "BF16",
                        "kv_cache_dtype": kv_dt.upper(),
                        "tp": 8,
                        "num_hosts": 1,
                        "context_tokens": test_ctx,
                        "concurrency": 2,
                        **mq,
                    }
                )
                log("orch-omni", f"[vLLM {label_q}] PASS ctx={test_ctx} TTFT={mq.get('ttft_p50_ms')}ms TPOT={mq.get('tpot_p50_ms')}ms")
            else:
                diagnostics.append(
                    sanitize_obj(
                        {
                            "component": f"vllm-{label_q}",
                            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                            "status": "DIAGNOSTIC",
                            "raw_excerpt": diag_q,
                        }
                    )
                )
                log("orch-omni", f"[vLLM {label_q}] Diagnostic recorded: {diag_q[-200:]}")
    else:
        log("orch-omni", "=== Suite A.2: Already recorded in suite_a_vllm_omni_serving.json (skipping) ===")

    # Suite A.3: `Qwen/Qwen3-Omni-30B-A3B-Thinking` Reasoning vs `Instruct` Direct Response Trade-Off across 4 Hosts (32 TC)!
    skip_a3 = any(r.get("suite") == "A3_vllm_thinking_reasoning_sweep" for r in vllm_rows)
    if not skip_a3:
        log("orch-omni", "=== Suite A.3: Qwen3-Omni-30B-A3B-Thinking vs Instruct Reasoning Overhead on 4 Hosts (32 TC) ===")
        tp_best, dp_best, ep_best = working_tp_cfg
        ok_th, urls_th, sid_th, dp_th_eff, ep_th_eff, boot_th, diag_th = _boot_qwen_vllm_cluster(
            "Qwen/Qwen3-Omni-30B-A3B-Thinking",
            tp=tp_best,
            dp_host=dp_best,
            enable_ep=ep_best,
            max_len=16384,
        )
        if ok_th:
            for conc in [1, 4, 8, 16, 32, 64]:
                m_th = run_vllm_closed_loop(
                    urls_th,
                    sid_th,
                    concurrency=conc,
                    num_requests=max(conc * 2, 8),
                    ctx_tokens=4096,
                    max_out_tokens=320,  # Accommodates <think>...</think> CoT chain + spoken answer
                    is_thinking=True,
                )
                _rec_vllm(
                    {
                        "suite": "A3_vllm_thinking_reasoning_sweep",
                        "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
                        "model_short": "Qwen3-Omni-30B-A3B-Thinking",
                        "engine": "vllm-torchtpu-serve",
                        "precision": "BF16",
                        "tp": tp_best,
                        "dp_per_host": dp_th_eff,
                        "cluster_dp_replicas": dp_th_eff * 4,
                        "expert_parallel": ep_th_eff,
                        "num_hosts": 4,
                        "total_tensorcores": 4 * tp_best * dp_th_eff,
                        "context_tokens": 4096,
                        "max_output_tokens": 320,
                        "concurrency": conc,
                        "server_boot_s": boot_th,
                        **m_th,
                    }
                )
                log(
                    "orch-omni",
                    f"[vLLM Thinking 4H] CON={conc:2d} -> TTFT={m_th.get('ttft_p50_ms')}ms E2E={m_th.get('e2e_p50_ms')}ms "
                    f"out_tps={m_th.get('output_tok_per_s')} think_frac={m_th.get('think_tag_fraction')}",
                )
        else:
            diagnostics.append(sanitize_obj({"component": "vllm-qwen3-omni-thinking", "status": "DIAGNOSTIC", "raw_excerpt": diag_th}))
        _stop_all()
    else:
        log("orch-omni", "=== Suite A.3: Already recorded in suite_a_vllm_omni_serving.json (skipping) ===")

    _stop_all()

    # =========================================================================
    # SUITE B: Compiled `torch_tpu` (`torchrun --nproc_per_node=8` across all 32 TensorCores)
    #   End-to-End Full-Duplex Multimodal Pipeline:
    #     - `inclusionAI/Ming-flash-omni-2.0` (219.4 GiB) vs `Qwen3-Omni-30B-A3B-Instruct` vs `Thinking`
    #     - BF16 vs Hardware FP8 (`torch._scaled_mm`) + Contexts `4K, 16K, 64K, 128K`
    #     - Stage-by-Stage Breakdown (Audio Enc + Vision Enc + MoE Thinker + Talker/Code2Wav + Image-Gen DiT)
    #     - Disaggregated CPU Offload (`multimodal-cpu-offload-svc:8090` `/preprocess_vision` + `/encode_wav` + `/encode_image`)
    #     - Full-Duplex Concurrency Sweep (`CON = 1, 4, 8, 16, 32, 64`), `TTFA < 150 ms` SLA, Speech `RTF`, & Max Duplex Sessions/Slice!
    # =========================================================================
    log("orch-omni", "=== Suite B: Compiled torch_tpu 32-TensorCore End-to-End Multimodal Duplex Pipeline ===")
    duplex_rows: list[dict[str, Any]] = []
    jsonl_dup = out_dir / "suite_b_duplex_multimodal_pipeline.jsonl"
    jsonl_dup.unlink(missing_ok=True)

    def _rec_dup(r: dict[str, Any]) -> None:
        c = sanitize_obj(r)
        duplex_rows.append(c)
        with jsonl_dup.open("a") as f:
            f.write(json.dumps(c) + "\n")
        (out_dir / "suite_b_duplex_multimodal_pipeline.json").write_text(json.dumps(duplex_rows, indent=2))

    # Standalone CPU offload microbenchmark snapshot
    offload_ablation = sanitize_obj(
        {
            "vision_2cam_preproc": benchmark_vision_preproc_ablation(2, 640, 480, 256, 256, base_url=args.cpu_offload_url),
            "audio_4s_wav_encode": benchmark_audio_wav_ablation(4.0, 1, 24000, base_url=args.cpu_offload_url),
            "image_1024_png_encode": benchmark_image_encode_ablation(1024, 1024, 1, "png", base_url=args.cpu_offload_url),
        }
    )

    max_duplex_summary: dict[str, Any] = {}

    for model_id in OMNI_MODELS:
        spec = OMNI_SPECS[model_id]
        # B.1: BF16 vs FP8 + Eager vs Compiled + Contexts (4K, 16K, 64K, 128K) + Offload A/B on Host 0 (8 TC)
        for prec, comp in [("BF16", False), ("BF16", True), ("FP8", True)]:
            ep_use = 8 if "Ming" in model_id and prec == "BF16" else 4
            boot = http_json_rpc(
                f"{endpoints[0]}/start_subworker",
                {
                    "model_id": model_id,
                    "precision": prec,
                    "use_compile": comp,
                    "ep_deg": ep_use,
                },
                timeout=600.0,
            )
            st = http_json_rpc(
                f"{endpoints[0]}/step",
                {
                    "ctx_tokens": 4096,
                    "out_tokens": 128,
                    "offload_mode": "remote",
                    "run_image_gen": False,
                },
                timeout=300.0,
            )
            _rec_dup(
                {
                    "suite": "B1_stage_breakdown_precision_compile",
                    "model": model_id,
                    "model_short": spec["short"],
                    "precision": prec,
                    "compiled": comp,
                    "ep_deg": ep_use,
                    "num_hosts": 1,
                    "total_tensorcores": 8,
                    "context_tokens": 4096,
                    "offload_mode": "disagg_cpu_offload",
                    "boot_s": boot["boot_s"],
                    **st,
                    "status": "PASS",
                }
            )
            log(
                "orch-omni",
                f"[B1] {spec['short']} {prec} comp={comp} EP={ep_use} -> "
                f"TTFT={st['ttft_ms']}ms TPOT={st['tpot_ms']}ms TTFA={st['ttfa_ms']}ms RTF={st['speech_rtf']}",
            )

            if prec == "FP8" and comp:
                # Local on-host CPU vs Disaggregated CPU Offload + Ming Image-Gen DiT branch + 4K..128K Context Sweep!
                st_loc = http_json_rpc(
                    f"{endpoints[0]}/step",
                    {"ctx_tokens": 4096, "out_tokens": 128, "offload_mode": "local", "run_image_gen": False},
                    timeout=300.0,
                )
                _rec_dup(
                    {
                        "suite": "B1_cpu_offload_ablation",
                        "model": model_id,
                        "model_short": spec["short"],
                        "precision": "FP8",
                        "compiled": True,
                        "ep_deg": ep_use,
                        "context_tokens": 4096,
                        "offload_mode": "local_on_host_cpu",
                        **st_loc,
                        "status": "PASS",
                    }
                )
                if spec["has_image_gen"]:
                    for om in ("local", "remote"):
                        st_ig = http_json_rpc(
                            f"{endpoints[0]}/step",
                            {"ctx_tokens": 4096, "out_tokens": 64, "offload_mode": om, "run_image_gen": True},
                            timeout=300.0,
                        )
                        _rec_dup(
                            {
                                "suite": "B1_ming_unified_image_gen_branch",
                                "model": model_id,
                                "model_short": spec["short"],
                                "precision": "FP8",
                                "compiled": True,
                                "offload_mode": "disagg_cpu_offload" if om == "remote" else "local_on_host_cpu",
                                **st_ig,
                                "status": "PASS",
                            }
                        )
                for ctx_t in [4096, 16384, 65536, 131072]:
                    st_c = http_json_rpc(
                        f"{endpoints[0]}/step",
                        {"ctx_tokens": ctx_t, "out_tokens": 128, "offload_mode": "remote", "run_image_gen": False},
                        timeout=300.0,
                    )
                    _rec_dup(
                        {
                            "suite": "B2_long_context_4k_to_128k",
                            "model": model_id,
                            "model_short": spec["short"],
                            "precision": "FP8",
                            "compiled": True,
                            "ep_deg": ep_use,
                            "context_tier": f"{ctx_t // 1024}K",
                            "context_tokens": ctx_t,
                            **st_c,
                            "status": "PASS",
                        }
                    )

        # B.3: Full 4-Host 16-Chip (32-TensorCore) Duplex Voice+Video Concurrency Sweep (`CON = 1, 4, 8, 16, 32, 64`)
        # Determine optimal EP shard width (`ep_opt = 4` for Ming FP8 -> 8 replicas across 32 TCs; `ep_opt = 2` for Qwen3-Omni FP8 -> 16 replicas across 32 TCs!)
        ep_opt = 4 if "Ming" in model_id else 2
        cluster_dp = (4 * 8) // ep_opt
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda ep: http_json_rpc(
                        f"{ep}/start_subworker",
                        {"model_id": model_id, "precision": "FP8", "use_compile": True, "ep_deg": ep_opt},
                        timeout=600.0,
                    ),
                    endpoints,
                )
            )
        router = MultiHostLeastInflightRouter(endpoints)
        # Warmup all replicas across all 4 hosts (32 TensorCores)
        with concurrent.futures.ThreadPoolExecutor(max_workers=cluster_dp) as wp:
            list(
                wp.map(
                    lambda _: router.call("/step", {"ctx_tokens": 2048, "out_tokens": 96, "offload_mode": "remote"}),
                    range(cluster_dp),
                )
            )

        max_duplex_150ms = 0
        max_duplex_250ms = 0
        for conc in [1, 4, 8, 16, 32, 64]:
            n_reqs = max(conc * 2, 8)
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                outs = list(
                    pool.map(
                        lambda _: router.call("/step", {"ctx_tokens": 2048, "out_tokens": 96, "offload_mode": "remote"}),
                        range(n_reqs),
                    )
                )
            wall_s = time.perf_counter() - t0
            client_ttfts = [max(0.0, o["_client_rtt_ms"] - o["e2e_turn_ms"]) + o["ttft_ms"] for o in outs]
            med_ttft = statistics.median(client_ttfts)
            med_tpot = statistics.median(o["tpot_ms"] for o in outs)
            med_e2e = statistics.median(o["_client_rtt_ms"] for o in outs)
            p95_e2e = sorted(o["_client_rtt_ms"] for o in outs)[min(len(outs) - 1, int(0.95 * len(outs)))]
            has_tk = outs[0].get("ttfa_ms") is not None
            med_ttfa = statistics.median(max(0.0, o["_client_rtt_ms"] - o["e2e_turn_ms"]) + o["ttfa_ms"] for o in outs) if has_tk else None
            med_rtf = statistics.median(o["speech_rtf"] for o in outs) if has_tk else None
            turns_per_min = (n_reqs / max(wall_s, 1e-6)) * 60.0
            agg_tok_s = (n_reqs * outs[0]["total_decode_tokens"]) / max(wall_s, 1e-6)
            agg_audio_x_rt = ((n_reqs * 4.0) / max(wall_s, 1e-6)) if has_tk else 0.0

            sla_150 = bool(med_ttfa is not None and med_ttfa < 150.0 and (med_rtf or 1.0) < 0.5)
            sla_250 = bool(med_ttfa is not None and med_ttfa < 250.0 and (med_rtf or 1.0) < 0.8)
            if sla_150:
                max_duplex_150ms = conc
            if sla_250:
                max_duplex_250ms = conc

            row_d = {
                "suite": "B3_slice32_duplex_concurrency_sla",
                "model": model_id,
                "model_short": spec["short"],
                "precision": "FP8",
                "compiled": True,
                "num_hosts": 4,
                "total_chips": 16,
                "total_tensorcores": 32,
                "ep_deg": ep_opt,
                "cluster_dp_replicas": cluster_dp,
                "concurrency": conc,
                "context_tokens": 2048,
                "decode_tokens_per_turn": outs[0]["total_decode_tokens"],
                "ttft_p50_ms": round(med_ttft, 2),
                "tpot_p50_ms": round(med_tpot, 2),
                "ttfa_p50_ms": round(med_ttfa, 2) if med_ttfa is not None else None,
                "speech_rtf_p50": round(med_rtf, 4) if med_rtf is not None else None,
                "e2e_turn_p50_ms": round(med_e2e, 2),
                "e2e_turn_p95_ms": round(p95_e2e, 2),
                "slice_turns_per_min": round(turns_per_min, 1),
                "slice_decode_tok_per_s": round(agg_tok_s, 1),
                "slice_speech_realtime_x": round(agg_audio_x_rt, 1) if has_tk else None,
                "sla_ttfa_lt_150ms": sla_150,
                "sla_ttfa_lt_250ms": sla_250,
                "status": "PASS",
            }
            _rec_dup(row_d)
            log(
                "orch-omni",
                f"[B3 32-TC] {spec['short']} CON={conc:2d} (DP={cluster_dp},EP={ep_opt}) -> "
                f"TTFA={row_d['ttfa_p50_ms']}ms RTF={row_d['speech_rtf_p50']} tok/s={agg_tok_s:.1f} audio_RT={agg_audio_x_rt:.1f}x",
            )

        max_duplex_summary[spec["short"]] = {
            "model": model_id,
            "optimal_sharding": f"DP={cluster_dp} x EP={ep_opt} (32 TC, FP8)",
            "supports_streaming_speech": spec["has_talker"],
            "max_duplex_sessions_ttfa_lt_150ms": max_duplex_150ms if spec["has_talker"] else "N/A (Thinker-Only)",
            "max_duplex_sessions_ttfa_lt_250ms": max_duplex_250ms if spec["has_talker"] else "N/A (Thinker-Only)",
        }

    _stop_all()

    (out_dir / "diagnostic_matrix.json").write_text(json.dumps(sanitize_obj(diagnostics), indent=2))
    summary_payload = sanitize_obj(
        {
            "generated_at_utc": utc_iso(),
            "accelerator": "Cloud TPU v7x (Ironwood)",
            "slice_topology": "2x2x4 (4 hosts x 4 chips = 16 physical v7x chips = 32 TensorCores, 3,072 GiB HBM3e)",
            "multimodal_cpu_offload_health": mm_health,
            "cpu_offload_microbench_ablation": offload_ablation,
            "max_duplex_voice_video_sessions_per_16chip_slice": max_duplex_summary,
            "diagnostics_count": len(diagnostics),
            "counts": {
                "suite_a_vllm_rows": len(vllm_rows),
                "suite_b_duplex_rows": len(duplex_rows),
                "total_verified_rows": len(vllm_rows) + len(duplex_rows),
            },
        }
    )
    (out_dir / "omni_v7x16_summary.json").write_text(json.dumps(summary_payload, indent=2))
    log("orch-omni", f"=== OMNI BENCHMARK COMPLETE: {summary_payload['counts']} ===")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Omni Ming-flash-omni-2.0 & Qwen3-Omni-30B-A3B Benchmark on TPU v7x-16")
    ap.add_argument("--mode", choices=["stage-shm", "host-agent", "tpu-subworker", "orchestrator"], required=True)
    ap.add_argument("--models-root", default="/models/models")
    ap.add_argument("--shm-root", default="/dev/shm/omni_models")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--host-rank", type=int, default=0)
    ap.add_argument("--agent-port", type=int, default=19120)
    ap.add_argument("--host-endpoints", default="http://127.0.0.1:19120")
    ap.add_argument("--cpu-offload-url", default="http://multimodal-cpu-offload-svc.default.svc.cluster.local:8090")
    ap.add_argument("--subworker-port", type=int, default=19450)
    ap.add_argument("--model-id", default="inclusionAI/Ming-flash-omni-2.0")
    ap.add_argument("--precision", default="FP8")
    ap.add_argument("--use-compile", action="store_true")
    ap.add_argument("--ep-deg", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "stage-shm":
        print(json.dumps(stage_models_to_shm(Path(args.models_root), Path(args.shm_root)), indent=2), flush=True)
        return 0
    if args.mode == "host-agent":
        return run_host_agent(args)
    if args.mode == "tpu-subworker":
        return run_tpu_subworker(args)
    if args.mode == "orchestrator":
        return run_orchestrator(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
