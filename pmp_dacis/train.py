"""Supervised and episodic training stages."""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def amp_ctx(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp) and device.type == "cuda")


def set_bn_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def _opt(model, lr, wd, include_fc=True):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not include_fc and n.startswith("fc."):
            continue
        (decay if p.ndim > 1 else no_decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=0.9, nesterov=True)


def _cosine(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def train_ce(model, ds, classes, epochs, lr, device, rng, bs=128, wd=5e-4, ls=0.1, amp=True, new_head=False):
    """Cross-entropy training on the images of `classes` (labels remapped to 0..len-1) through model.fc."""
    if new_head or model.fc is None:
        model.fc = nn.Linear(getattr(model, "feat_dim", 512), len(classes)).to(device)
    lut = torch.full((len(ds.class_names) + 1,), -1, dtype=torch.long)
    for i, c in enumerate(classes):
        lut[c] = i
    by = ds.idx_by_class()
    pool = torch.cat([by[c] for c in classes])
    steps_per_epoch = len(pool) // bs
    total = max(1, steps_per_epoch * epochs)
    opt = _opt(model, lr, wd)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp) and device.type == "cuda")
    model.train()
    t0, step, last = time.time(), 0, float("nan")
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(len(pool)))
        for s in range(steps_per_epoch):
            idx = pool[perm[s * bs:(s + 1) * bs]]
            x = ds.fetch(idx, device, train=True)
            y = lut[ds.y[idx]].to(device)
            for g in opt.param_groups:
                g["lr"] = lr * _cosine(step, total, max(1, int(0.03 * total)))
            with amp_ctx(device, amp):
                loss = F.cross_entropy(model(x).float(), y, label_smoothing=ls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 20 == 0 or step == total:
                last = float(loss.detach())
    return {"steps": step, "images": step * bs, "seconds": time.time() - t0, "loss": last}


def _sample_episode(by, classes, way, per_class, rng):
    cs = rng.choice(classes, size=min(way, len(classes)), replace=False)
    return torch.stack([by[int(c)][torch.from_numpy(rng.choice(len(by[int(c)]), per_class, replace=False))] for c in cs])


def train_episodic(model, ds, classes, n_episodes, lr, device, rng, way=5, shot=5, query=15, wd=5e-4,
                   amp=True, meta_batch=2):
    """Prototypical (nearest-class-mean, cosine) episodic training. BN statistics stay frozen.

    Also accumulates, per block, the mean L2 norm of d(episode loss)/d(conv1 filter) over the run:
    the 'meta-gradient' that stage 3 of PMP uses.
    """
    by = ds.idx_by_class()
    model.train()
    set_bn_eval(model)
    opt = _opt(model, lr, wd, include_fc=False)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(amp) and device.type == "cuda")
    blocks = model.blocks
    gsum = [torch.zeros(b.conv1.out_channels, device=device) for b in blocks]
    steps = max(1, n_episodes // meta_batch)
    t0, losses = time.time(), []
    w = min(way, len(classes))
    for step in range(steps):
        idx = torch.stack([_sample_episode(by, classes, w, shot + query, rng) for _ in range(meta_batch)])
        x = ds.fetch(idx.flatten(), device, train=True)
        for g in opt.param_groups:
            g["lr"] = lr * _cosine(step, steps, max(1, int(0.03 * steps)))
        with amp_ctx(device, amp):
            f = model.features(x)
        f = F.normalize(f.float(), dim=1).view(meta_batch, w, shot + query, -1)
        proto = F.normalize(f[:, :, :shot].mean(2), dim=-1)                       # [mb, w, D]
        q = f[:, :, shot:].reshape(meta_batch, w * query, -1)
        logits = model.log_tau.exp() * torch.einsum("eqd,ewd->eqw", q, proto)
        y = torch.arange(w, device=device).repeat_interleave(query).unsqueeze(0).expand(meta_batch, -1)
        loss = F.cross_entropy(logits.reshape(-1, w), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for gi, b in enumerate(blocks):
            gsum[gi] += b.conv1.weight.grad.flatten(1).norm(dim=1)
        scaler.step(opt)
        scaler.update()
        if step % 10 == 0:
            losses.append(float(loss.detach()))
    return {"steps": steps, "images": steps * meta_batch * w * (shot + query), "seconds": time.time() - t0,
            "loss": float(np.mean(losses[-10:])) if losses else float("nan"),
            "gmeta": [(g / steps).cpu() for g in gsum]}
