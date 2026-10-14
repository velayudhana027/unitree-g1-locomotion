# Training record: flat-terrain velocity policy

Run `2026-03-09_21-42-00`, task `Mjlab-Velocity-Flat-Unitree-G1`, mjlab 1.2.0 (commit `18b1c8b`),
4,096 environments, 30,000 PPO iterations, one A100 on Google Colab.

| File | Contents |
| --- | --- |
| `train_colab.ipynb` | Colab notebook used for training, video export and the rough-terrain fine-tuning attempt (outputs cleared) |
| `config.yaml` | Full environment and agent configuration of the run, as logged to Weights & Biases |
| `wandb-summary.json` | Metrics at the final iteration |
| `metrics.csv` | Every 10th iteration of the console log: reward, episode length, tracking errors, curriculum, terminations |
| `plot_training.py` | Regenerates `docs/media/training_curves.png` from `metrics.csv` |
| `requirements-frozen.txt` | Exact package versions of the training environment |

The resulting checkpoints are in [`checkpoints/mjlab_velocity_flat`](../../checkpoints/mjlab_velocity_flat):
`model_29999.pt` (rsl_rl checkpoint) and `policy.onnx` (exported actor with joint names, default pose,
action scale and PD gains stored as metadata).
