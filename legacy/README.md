# Legacy prototype (kept for history)

These are the first scripts of the project (`train.py`, `evaluate.py`, `models/`, `utils/`, `configs/`).
They are not used by anything in this repository any more.

What they do and do not do:

- `train.py` always raises `NotImplementedError` at the point where a dataset should be loaded, catches it, and falls
  back to a `DummyDataset` that returns random tensors. `evaluate.py` always builds its test set from a
  `DummyDataset`, whatever `--dataset` is passed.
- The self-test at the bottom of `utils/metrics.py` draws episode accuracies from a normal distribution centred on 0.89;
  that block is a demonstration of the metric code, not a measurement.
- Because of that, the result tables that appeared in earlier versions of the top-level README (and the numbers in the
  first arXiv versions of the paper) cannot be regenerated with this code. They should not be cited.
- The current implementation, with real data loading, is in `pmp_dacis/`. All numbers in the paper and in the
  top-level README come from `results/`, produced by `notebooks/pmp_dacis_cv4eo.py`.
