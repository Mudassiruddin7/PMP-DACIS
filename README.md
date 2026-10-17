# PMP-DACIS

Channel pruning of a ResNet-18 for few-shot recognition of crop diseases and Sentinel-2 land cover, with a one-click
notebook that reruns every experiment and rebuilds the paper tables, figures and PDFs from the saved results.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/Mudassiruddin7/PMP-DACIS/blob/main/notebooks/pmp_dacis_cv4eo.py)

## Reproduce in one click

1. Open `notebooks/pmp_dacis_cv4eo.py` in [molab](https://molab.marimo.io) (button above) on a GPU machine.
2. Press **Run all**. Nothing has to be selected or typed.
3. When the last cell finishes, press **Results**. You get `results.zip` with every run (JSON and per-episode arrays),
   all tables and figures, the logs, and the compiled PDFs (`paper_review.pdf`, anonymous, and `paper_final.pdf`, with authors).

The notebook downloads the three datasets from the Hugging Face Hub (about 4 GB), trains the models, and writes finished runs
to disk, so *Run all* can be repeated after an interruption and continues where it stopped. Without a GPU it falls back to a
tiny `smoke` profile that only checks the code path; those numbers are not results.
It also runs as a plain script: `python notebooks/pmp_dacis_cv4eo.py`.

The library lives in `pmp_dacis/` (readable sources; the notebook embeds a copy so that one file is enough).
`python tests/test_core.py` checks the pruning surgery, the budget solver and the episode sampler.

## What is compared

* **Where to cut.** Output channels of the first convolution of each ResNet-18 basic block (1,920 channels). Block widths
  of the residual stream never change. The budget is the share of the 11.18 M trunk parameters that remain.
* **Which channels to keep.** Random, filter l1 norm, BatchNorm scale, first-order Taylor, and DACIS (gradient norm,
  activation variance and Fisher ratio, each as a within-block rank, weighted 0.3 / 0.2 / 0.5).
* **When to prune.** P (prune once, then recover), P-M, M-P, P-M-P (the PMP schedule), P-M-P-M, and P-M with twice the episodes.
* **Where to test.** 5-way 1/5/10-shot episodes on PlantVillage classes held out from training, on PlantDoc field
  photographs after PlantVillage training, and on held-out classes of EuroSAT with its 13 Sentinel-2 bands.

## Results

The numbers of the reported run will be added here, generated from `results/` by `pmp_dacis/report.py` (no number in this
repository is typed by hand). Until then there are no results in this README on purpose: the tables that used to be here
could not be regenerated (see below).

## About earlier numbers

Earlier versions of this README, and the first arXiv versions of the paper, reported different results. The scripts
that were in this repository then (now in `legacy/`) have no working dataset loader, so those numbers cannot be
regenerated with this code and should not be cited.

## Layout

```
pmp_dacis/        library: data, ResNet-18 surgery, scores, training stages, evaluation, report, paper build
notebooks/        pmp_dacis_cv4eo.py, the single-file marimo notebook
paper/            LaTeX sources (WACV 2027 style) and the built PDFs
results/          saved runs of the reported experiment, generated tables and figures
tests/            unit tests for the parts where a silent bug would change the science
tools/            builds the notebook from the sources
legacy/           the first prototype, kept for history (see legacy/README.md)
```

## License

MIT, see `LICENSE`.
