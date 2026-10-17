"""Few-shot episode evaluation and efficiency measurements."""
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from .train import amp_ctx


@torch.no_grad()
def embed(model, ds, idx, device, bs=256, res=None, amp=True):
    """L2-normalised 512-d embeddings of ds[idx] (CPU float32)."""
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        x = ds.fetch(idx[s:s + bs], device, train=False, res=res)
        with amp_ctx(device, amp):
            f = model.features(x)
        out.append(F.normalize(f.float(), dim=1).cpu())
    return torch.cat(out)


def make_episodes(local_labels, n_ep, way, shot, query, seed):
    """[n_ep, way, shot+query] indices into the embedding set; the same for every method (paired)."""
    rng = np.random.default_rng(seed)
    K = int(local_labels.max()) + 1
    pools = [np.where(local_labels == k)[0] for k in range(K)]
    eps = np.empty((n_ep, way, shot + query), dtype=np.int64)
    for e in range(n_ep):
        cs = rng.choice(K, way, replace=False)
        for j, c in enumerate(cs):
            eps[e, j] = rng.choice(pools[c], shot + query, replace=False)
    return eps


def episode_accuracy(feats, eps, shot, chunk=500):
    """Per-episode accuracy of a cosine nearest-prototype classifier. feats: [N,D] normalised."""
    n_ep, way, per = eps.shape
    query = per - shot
    lab = torch.arange(way).repeat_interleave(query)
    out = []
    for s in range(0, n_ep, chunk):
        e = torch.from_numpy(eps[s:s + chunk])
        f = feats[e]                                                   # [c, way, per, D]
        proto = F.normalize(f[:, :, :shot].mean(2), dim=-1)
        q = f[:, :, shot:].reshape(e.shape[0], way * query, -1)
        pred = torch.einsum("eqd,ewd->eqw", q, proto).argmax(-1)
        out.append((pred == lab).float().mean(-1))
    return torch.cat(out).numpy()


def host_info(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "device": str(device)}
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
            info["cpu"] = out.stdout.strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info["cpu"] = platform.processor()
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
        info["cuda"] = torch.version.cuda
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        pass
    return info


@torch.no_grad()
def latency_ms(model, in_ch, res, device, batch=1, threads=None, warm=20, iters=100, amp=False):
    """Median / quartiles of features() latency in milliseconds (fp32 unless amp on CUDA)."""
    model.eval()
    if device.type == "cpu" and threads:
        prev = torch.get_num_threads()
        torch.set_num_threads(threads)
    x = torch.randn(batch, in_ch, res, res, device=device)
    times = []
    for i in range(warm + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with amp_ctx(device, amp):
            model.features(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            times.append((time.perf_counter() - t) * 1000)
    if device.type == "cpu" and threads:
        torch.set_num_threads(prev)
    q = statistics.quantiles(times, n=4)
    return {"median": statistics.median(times), "q1": q[0], "q3": q[2]}


@torch.no_grad()
def peak_mem_mb(model, in_ch, res, device, batch=1):
    if device.type != "cuda":
        return None
    model.eval()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch, in_ch, res, res, device=device)
    model.features(x)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20
