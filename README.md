# octo_teleoperation

Finetunes the Octo generalist robot policy to predict an action covariance. That covariance becomes a mapping from a 2-DOF joystick to a 14-DOF arm. It's research code for assistive teleoperation.

Write-up: [Video Language Teleoperation Models (PDF)](docs/OctoTeleop_Report.pdf), with Michael Przystupa and Matthew E. Taylor, University of Alberta.

![demo](docs/demo.gif)
<!--
Record ~15s of terminal, then the plot it produces:
  1. python scripts/teleoperationUnitTest.py           (prints mean/cov shapes)
  2. python scripts/visualize_trajectories.py --results_dir=./results/multi_horizon \
       --output_dir=./results/multi_horizon/plots --max_trajectories=1 --show_plots=true
  End on one predicted-vs-ground-truth trajectory plot.
  If you'd rather not record anything, swap this line for docs/recon_heatmap.png.
-->

## Why I built this

Assistive teleoperation maps a 2D or 3D joystick onto a high-DOF robot. Existing systems learn that mapping per task, and collecting data for each task is expensive. I wanted to check whether a pretrained generalist policy could produce the mapping once from the scene and the instruction, so the controller doesn't need task-specific action labels.

## What I changed vs. upstream

This is a fork of [Octo](https://github.com/octo-models/octo). Everything not listed here is upstream code (original README in `README.upstream.md`).

- `octo/model/components/action_heads.py`: `TeleoperationActionHead` (the last ~290 lines)
- `scripts/teleop_finetune.py`: replaces the action head on a pretrained Octo model and finetunes it
- `scripts/multi_horizon_finetune.py`, `multi_horizon_predict.py`, `run_multi_horizon_experiment.py`: train at several action horizons and cross-evaluate every train/predict pair
- `scripts/visualize_trajectories.py`: predicted-vs-ground-truth plots
- `notebooks/`: the Colab notebook that produced the report's results
- `scripts/MULTI_HORIZON_README.md`: full flag reference

## Run it locally

Python 3.10. Training needs a GPU.

```bash
conda create -n octo python=3.10 && conda activate octo
pip install -e .
pip install -r requirements.txt
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# smoke test the head (no data needed)
python scripts/teleoperationUnitTest.py

# simulated ALOHA cube-handover data
wget https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip && unzip example_sim_data.zip

# finetune at horizons 1/5/15, then evaluate all 9 pairs (settings from the report)
python scripts/run_multi_horizon_experiment.py \
  --pretrained_path=hf://rail-berkeley/octo-small-1.5 \
  --data_dir=./example_sim_data \
  --base_save_dir=./checkpoints/multi_horizon \
  --output_dir=./results/multi_horizon \
  --action_horizons=1,5,15 --num_steps=2000 --num_trajectories=3
```

There are no env vars. Everything is set with absl flags.

## How it works

```mermaid
flowchart LR
  A[image + proprio + instruction] --> B[pretrained Octo transformer]
  B --> C[MAP pooling over readout tokens]
  C --> D["Dense → lower-triangular L"]
  D --> E["Σ = L·Lᵀ + εI"]
  E --> F["top-k eigenvectors: W = U_k·Λ_k^½ (14×2)"]
  F --> G["robot action = W · joystick input"]
```

Octo's tokenizers and transformer stay as they are. Only the output head changes. The head predicts a zero-mean Gaussian over actions. Training minimizes the negative log-likelihood of the expert action chunk under `N(0, Σ)`. The top-k eigenvectors of Σ, scaled by √λ, become the joystick-to-robot mapping. The eigenvectors set the directions and the eigenvalues set the gains.

## The hard part

**Predicting a control mapping instead of an action, and then showing it works.**

*Parameterization.* The first version predicted a separate Σ for every step in the chunk. That's H × 105 outputs for 14-dim actions, or 5,250 at horizon 50, and it gives you a mapping that keeps shifting under the user's hand. In `03c4c8c` I switched to one Cholesky factor per timestep, shared across the horizon. That's 105 outputs, and `L·Lᵀ + εI` keeps Σ positive-definite without optimizer constraints. The tradeoff: the model can't express uncertainty that grows later in the chunk. In return the operator gets a mapping that holds steady.

*Evaluation.* With a zero mean there's no prediction to score, so MSE tells you nothing. `multi_horizon_predict.py` draws 128 samples per timestep, builds the rank-2 basis B from their covariance, and solves `min_z ‖a* − Bz‖²` for each expert action. Reconstruction error answers one question: could a 2-DOF joystick have produced the action the expert took?

Result: on held-out trajectories, training at horizon 5 gave the best mapping at every evaluation horizon (MSE 0.113 at horizon 15). That's about 3× lower than training at horizon 1 or 15. Horizon 1 doesn't capture the task's temporal structure, and horizon 15 overfits.

![reconstruction error](docs/recon_heatmap.png)

## Stack

Python, JAX/Flax, TensorFlow Probability (JAX substrate), Optax, TF/RLDS data pipeline (from Octo), matplotlib.

## Status

- Simulation only (ALOHA sim). No user study or real-robot run yet. That's the next step in the report.
- The report's numbers come from the notebook in `notebooks/`. The scripts are the cleaned-up version and haven't been re-run end to end.
- `teleop_finetune.py` hardcodes the dataset and step count. Use the multi-horizon scripts instead.

## License

MIT, inherited from upstream Octo (see `LICENSE`).
