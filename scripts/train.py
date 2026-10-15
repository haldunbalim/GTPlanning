from gtp import *
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import gymnasium as gym
import gymnasium_robotics


def normalize_image_observation_size(size):
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"image_observation_size must be an int or [height, width], got {size}")
    return tuple(int(v) for v in size)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Allow for v2 env registration
    if 'PushT' in cfg.env_name:
        import  gtp.utils.pusht.pusht_env
    set_seed(cfg.seed)
    use_image_observations = bool(cfg.get("use_image_observations", False))
    if use_image_observations and "VISION" not in cfg.algo._target_.upper():
        raise ValueError("use_image_observations requires a vision-capable algorithm target.")

    env_kwargs = {"render_mode": "rgb_array"} if use_image_observations else {}
    env = gym.make(cfg.env_name, **env_kwargs)
    algo_overrides = {"act_dim": env.action_space.shape[-1] * cfg.action_chunk_size}
    if use_image_observations:
        height, width = normalize_image_observation_size(cfg.image_observation_size)
        algo_overrides["image_shape"] = (height, width, 3)
    else:
        if isinstance(env.observation_space, gym.spaces.Dict):
            obs_dim = sum([env.observation_space[key].shape[-1] for key in cfg.obs_keys])
            goal_dim = env.observation_space['desired_goal'].shape[-1]
        else:
            obs_dim = env.observation_space.shape[-1]
            goal_dim = 0
        algo_overrides["obs_dim"] = obs_dim
        if "goal_dim" in cfg.algo:
            algo_overrides["goal_dim"] = goal_dim
    algo_cfg = OmegaConf.merge(cfg.algo, algo_overrides)
    algo = instantiate(algo_cfg)
    
    trainer = OffPolicyTrainer(algo, cfg)
    try:
        trainer.train()
    except JAXCompileTimeoutError as e:
        print(f"[FATAL] compile timeout: {e}", flush=True)
        raise SystemExit(42)


if __name__ == "__main__":
    main()
