# Gesture models

- `gesture_student_v2.npz`: bundled IMU classifier used before a new training run.
- `gesture_sim_v1.npz`: earlier checkpoint retained for comparison.
- `*.metrics.json`: training settings and evaluation results.
- `*.blind_test.json`: independent scoring of saved student predictions.
- `*.history.csv`: per-epoch training and validation loss.
- `runs/`: local outputs from `loop train`, excluded from Git.

Network code lives in `paceloop/learning/model.py`, and training code in
`paceloop/training/train.py`. The `.npz` files contain learned parameters and
metadata. Loss records and reports use the same filename stem as their model.

`loop show` and `loop twin` select the last successfully trained model recorded in
`logs/last_trained_model.json`, or the bundled checkpoint when no record is available.
Use `--model PATH.npz` to choose a model explicitly.
