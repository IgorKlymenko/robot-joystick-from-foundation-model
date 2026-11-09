# Media

| File | What it shows | Source |
|---|---|---|
| `bridge_episode_{stove,table,counter}.gif` | Bridge dataset episodes (input frames) | Octo example notebook, run 2025-12-05 |
| `octo_zeroshot_bridge_{stove,table}.jpg` | Pretrained Octo-small, zero-shot actions vs ground truth (upstream diffusion head, not the teleop head) | Octo example notebook, run 2025-12-05 |
| `teleop_finetune_loss.jpg` | NLL of `TeleoperationActionHead` over a 1000-step finetune on ALOHA sim | `scripts/teleop_finetune.py`, TensorBoard, 2025-12-16 |
| `traj_runA_train_h1_pred_h5.jpg` | Sampled actions vs ground truth, model trained at horizon 1, evaluated at 5 | `scripts/visualize_trajectories.py`, run A, 2025-12-22 |
| `traj_runB_train_h1_pred_h5.jpg` | Same pairing, separate run | run B, 2025-12-22 |
| `traj_runB_train_h5_pred_h5.jpg` | Trained and evaluated at horizon 5 | run B, 2025-12-22 |
| `traj_train_h5_pred_h5_segment3.jpg` | 160-step segment, trained and evaluated at horizon 5 (used in the main README) | 2025-12-22, run not recorded |

Trajectory plots: ALOHA sim, two 400-step episodes per plot. Only the first 7 of the 14 action dims are shown. The axis names (x, y, z, roll, …) are the script's default `--action_dim_labels`, not ALOHA's joint names. Runs A and B used the same code. Run A was shared at 20:28 and run B at 21:00. The sampled actions are noisy around zero because the head is zero-mean by design.
