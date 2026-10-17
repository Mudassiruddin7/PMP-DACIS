"""Command line entry point:  python -m pmp_dacis.cli --profile paper --out results"""
import argparse
import os
import sys
import time

import torch

from . import experiments as E


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="paper", choices=list(E.PROFILES))
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--variants", default="plantvillage,eurosat13,eurosat3")
    ap.add_argument("--tiers", default="1,2,3")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args(argv)
    cfg = dict(E.PROFILES[a.profile])
    if a.seeds:
        cfg["seeds"] = [int(s) for s in a.seeds.split(",")]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False  # autotuning every pruned shape costs more than it saves
    os.makedirs(a.out, exist_ok=True)
    if a.download:
        from . import data as D
        D.download(a.data)
    from . import evaluate as EV
    import json
    hp = os.path.join(a.out, "host.json")
    if not os.path.exists(hp):
        json.dump({"profile": a.profile, "host": EV.host_info(dev), "cfg": cfg}, open(hp, "w"), indent=1, default=str)
    fh = open(os.path.join(a.out, "log.txt"), "a")
    E.log(f"profile={a.profile} device={dev} variants={a.variants} tiers={a.tiers}", fh)
    t0 = time.time()
    E.run_all(cfg, dev, a.data, a.cache, a.out, variants=tuple(a.variants.split(",")),
              tiers=tuple(int(t) for t in a.tiers.split(",")), fh=fh)
    E.log(f"done in {(time.time() - t0) / 60:.1f} min", fh)


if __name__ == "__main__":
    main()
