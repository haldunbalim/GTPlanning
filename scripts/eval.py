"""Evaluate released GTP checkpoints.

Downloads a model from the Hugging Face Hub (or reads it from a local folder) and reports the
average return over evaluation episodes and, for environments that report it, the success rate
(an episode counts as a success if it succeeds at any step). Each model has checkpoints every
100k environment steps, from 100k to 1M.

    python scripts/eval.py --model hopper                      # 1M checkpoint, 20 episodes
    python scripts/eval.py --model fetchpush --step 200k 1M --episodes 50
    python scripts/eval.py --model pusht --model_dir models    # local copy instead of the Hub

Models: hopper, halfcheetah, walker2d, ant, fetchpush, fetchpickandplace, fetchslide, pusht.
"""
import argparse
import os.path as op

import gymnasium as gym
import gymnasium_robotics  # noqa: F401  (registers the Fetch environments)
import jax
import numpy as np
from flax.serialization import msgpack_restore
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gtp.trainer.off_policy import OffPolicyTrainer, scale_action

HF_REPO = 'haldunbalim/GTPlanning'
MODELS = ['hopper', 'halfcheetah', 'walker2d', 'ant', 'fetchpush', 'fetchpickandplace', 'fetchslide', 'pusht']
CHECKPOINT_EVERY = 100_000


def parse_step(text):
    """'500k', '1M' or '500000' -> 500000; checkpoints exist at multiples of 100k."""
    text = text.strip().lower()
    scale = {'k': 1_000, 'm': 1_000_000}.get(text[-1], 1)
    step = int(float(text.rstrip('km')) * scale)
    if step <= 0 or step > 1_000_000 or step % CHECKPOINT_EVERY:
        raise argparse.ArgumentTypeError(f'checkpoints exist every 100k steps up to 1M, got {text}')
    return step


def fetch(model, filename, model_dir=None, repo=HF_REPO):
    """Path to <model>/<filename>, downloaded from the Hub unless a local model folder is given."""
    if model_dir:
        return op.join(model_dir, model, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=f'{model}/{filename}')


def build_algo(cfg):
    """Instantiate the algorithm the same way scripts/train.py does."""
    if 'PushT' in cfg.env_name:
        import gtp.utils.pusht.pusht_env  # noqa: F401
    use_images = bool(cfg.get('use_image_observations', False))
    env = gym.make(cfg.env_name, **({'render_mode': 'rgb_array'} if use_images else {}))
    overrides = {'act_dim': env.action_space.shape[-1] * cfg.get('action_chunk_size', 1)}
    if use_images:
        size = cfg.image_observation_size
        height, width = (size, size) if isinstance(size, int) else tuple(size)
        overrides['image_shape'] = (height, width, 3)
    else:
        space = env.observation_space
        if isinstance(space, gym.spaces.Dict):
            overrides['obs_dim'] = sum(space[k].shape[-1] for k in cfg.obs_keys)
            goal_dim = space['desired_goal'].shape[-1]
        else:
            overrides['obs_dim'] = space.shape[-1]
            goal_dim = 0
        if 'goal_dim' in cfg.algo:
            overrides['goal_dim'] = goal_dim
    env.close()
    return instantiate(OmegaConf.merge(cfg.algo, overrides)), use_images


def load_state(algo, cfg, checkpoint_path, step):
    with open(checkpoint_path, 'rb') as f:
        params = msgpack_restore(f.read())  # network weights and observation-normalization statistics
    state = algo.init_train_state(cfg).replace(**params)
    # the planning horizon grew during training; plan with the horizon this checkpoint reached
    algo.current_curriculum_horizon = int(cfg.planning_horizon[step])
    return state


class _ImageObs:
    """The trainer's rendering and resizing of image observations, without building a trainer."""
    _get_vector_envs = OffPolicyTrainer._get_vector_envs
    _resize_frame = OffPolicyTrainer._resize_frame
    render_obs = OffPolicyTrainer.render_obs

    def __init__(self, size):
        self.image_observation_size = (size, size) if isinstance(size, int) else tuple(size)


def evaluate(algo, state, cfg, use_images, episodes, seed, batch_size=20):
    policy = jax.jit(algo.get_eval_action)
    key = jax.random.PRNGKey(seed)
    image_obs = _ImageObs(cfg.get('image_observation_size', 96)) if use_images else None
    returns, successes = [], []
    for start in range(0, episodes, batch_size):
        n = min(batch_size, episodes - start)
        env = gym.make_vec(cfg.env_name, num_envs=n, vectorization_mode='sync',
                           **({'render_mode': 'rgb_array'} if use_images else {}))

        def policy_obs(obs):
            if use_images:
                return image_obs.render_obs(env)
            if isinstance(obs, dict):
                return np.concatenate([obs[k] for k in cfg.obs_keys], axis=-1)
            return obs

        obs = policy_obs(env.reset(seed=seed + start)[0])
        alive = np.ones(n, dtype=bool)
        ep_return = np.zeros(n)
        ep_success = np.zeros(n, dtype=bool)
        has_success = False
        while alive.any():
            key, k = jax.random.split(key)
            action = np.asarray(jax.device_get(policy(state, obs, k)), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(scale_action(action, env.action_space))
            obs = policy_obs(obs)
            ep_return += np.where(alive, reward, 0.0)
            if 'is_success' in info:
                has_success = True
                ep_success |= alive & np.asarray(info['is_success']).astype(bool)
            alive &= ~(terminated | truncated)
        env.close()
        returns += ep_return.tolist()
        if has_success:
            successes += ep_success.tolist()
    return np.array(returns), (np.array(successes) if successes else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help=f'one of {", ".join(MODELS)}')
    parser.add_argument('--step', type=parse_step, nargs='+', default=[1_000_000],
                        help='checkpoint(s) to evaluate, multiples of 100k such as 500k or 1M (default: 1M)')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0, help='episode i resets the environment with seed + i')
    parser.add_argument('--model_dir', default=None, help='local folder with <model>/config.yaml and checkpoints/')
    parser.add_argument('--repo', default=HF_REPO, help='Hugging Face model repo to download from')
    args = parser.parse_args()
    if args.model_dir is None and args.repo == HF_REPO and args.model not in MODELS:
        parser.error(f'unknown model {args.model}; choose from {", ".join(MODELS)}')

    cfg = OmegaConf.load(fetch(args.model, 'config.yaml', args.model_dir, args.repo))
    algo, use_images = build_algo(cfg)
    print(f'{args.model}: {cfg.env_name}, {args.episodes} episodes')
    for step in args.step:
        ckpt = fetch(args.model, f'checkpoints/step_{step:07d}.msgpack', args.model_dir, args.repo)
        state = load_state(algo, cfg, ckpt, step)
        jax.clear_caches()  # the planning horizon is fixed when the policy is traced
        returns, successes = evaluate(algo, state, cfg, use_images, args.episodes, args.seed)
        line = f'  {step // 1000:>5}k  return {returns.mean():9.2f} +/- {returns.std():7.2f}'
        if successes is not None:
            line += f'   success rate {successes.mean() * 100:5.1f}%'
        print(line, flush=True)


if __name__ == '__main__':
    main()
