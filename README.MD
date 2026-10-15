<div align="center">

# Model-Based Online Decision Making via Generative Trajectory Planning

**NeurIPS 2026**

[Haldun Balim](https://haldunbalim.github.io) · [Yilun Du](https://yilundu.github.io)<sup>†</sup> · [Na Li](https://nali.seas.harvard.edu)<sup>†</sup>

Harvard University

<sup>†</sup>Equal advising

[Paper](https://arxiv.org/abs/2610.18677) · [Project page](https://haldunbalim.github.io/GTPlanning/) · [Models](https://huggingface.co/haldunbalim/GTPlanning) · [BibTeX](#citation)

<a href="https://haldunbalim.github.io/GTPlanning/"><img src="assets/gtp-overview.gif" alt="GTP overview slides: a self-improving online loop, how GTP plans, and results against SAC, DPMD and TD-MPC2 on locomotion, manipulation and exploration" width="100%"></a>

</div>

Generative Trajectory Planning (GTP) is a model-based framework for online decision making. Instead
of optimizing action sequences from scratch at every step, GTP learns a diffusion prior over action
sequences and uses it as a proposal distribution: it samples candidate sequences, refines them with
beam search, and scores them with learned dynamics, reward and value models. This gives
trajectory-level reasoning without solving a high-dimensional optimization problem at every
decision. Across continuous control and long-horizon manipulation, GTP matches or exceeds strong
model-free, diffusion-policy and model-based baselines.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Training

Each domain has its own algorithm config in `configs/algo/`. To reproduce the results in the paper:

```bash
# MuJoCo locomotion (also Hopper-v4, Walker2d-v4, Ant-v4)
python scripts/train.py algo=gtp-loco env_name=HalfCheetah-v4

# Fetch manipulation (also FetchPickAndPlaceDense-v4, FetchSlideDense-v4)
python scripts/train.py algo=gtp-manip env_name=FetchPushDense-v4

# PushT from pixels
python scripts/train.py algo=gtp-vision env_name=PushT-v0 use_image_observations=true
```

Settings in `configs/config.yaml` can be overridden on the command line in the same way, for
example `seed=0`, `total_steps=500000` or `use_wandb=true` to log to Weights & Biases.

### Elite bootstrapping (optional, faster)

GTP normally plans a fresh pool of candidate trajectories for every row of every update batch.
With elite bootstrapping, the pool the rollout planner produces anyway (diffusion beam search plus
Langevin refinement) is cached with each transition as action sequences, and the update rescores
those cached elites with the current models instead of planning again. Every `elite_refresh_every`
environment steps, the elites of `elite_refresh_batch_size` buffer rows are replanned so they don't
go stale, and elite planning runs in bfloat16 while updates stay in float32. It is off by default
(`n_elites: null`); the `*-boot` configs turn it on:

```bash
python scripts/train.py algo=gtp-manip-boot env_name=FetchPushDense-v4 n_train_envs=4
python scripts/train.py algo=gtp-vision-boot env_name=PushT-v0 use_image_observations=true n_train_envs=4
```

## Trained models

Trained GTP models are on the Hugging Face Hub at
[haldunbalim/GTPlanning](https://huggingface.co/haldunbalim/GTPlanning): `hopper`, `halfcheetah`,
`walker2d`, `ant`, `fetchpush`, `fetchpickandplace`, `fetchslide` and `pusht` (from pixels), each
with a checkpoint every 100k environment steps from 100k to 1M. `scripts/eval.py` downloads a
model's config and the requested checkpoints, then reports the average return and, for the Fetch
tasks and PushT, the success rate:

```bash
python scripts/eval.py --model hopper                                # 1M checkpoint, 20 episodes
python scripts/eval.py --model fetchslide --step 500k 1M --episodes 50
```

The [model card](https://huggingface.co/haldunbalim/GTPlanning) lists the scores of each model.

## Repository structure

```
configs/          Hydra configs: config.yaml (training) and algo/ (GTP and baselines)
gtp/algo/         GTP, its vision variant, and the SAC and TD3 baselines
gtp/trainer/      off-policy training loop and replay buffer
gtp/utils/        environment wrappers, the PushT environment, plotting
scripts/train.py  training entry point
scripts/eval.py   evaluation of the trained models from the Hugging Face Hub
```

## Citation

```bibtex
@inproceedings{balim2026modelbasedonline,
  title     = {Model-Based Online Decision Making via Generative Trajectory Planning},
  author    = {Balim, Haldun and Du, Yilun and Li, Na},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
