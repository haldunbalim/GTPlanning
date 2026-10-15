from typing import Optional
import jax
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from tensorboardX import SummaryWriter
from flax.serialization import to_state_dict
from gtp.utils import hash_config, get_seed
import wandb
import os
import os.path as op
import orbax.checkpoint as ocp
from hydra.utils import get_class
from omegaconf import OmegaConf
import jax.numpy as jnp
from .misc import EpisodeTracker
from gtp.trainer.buffer import Experience, EpisodeReplayBuffer
import logging
import yaml
from gymnasium.vector.vector_env import AutoresetMode
import glob
from gtp.utils import accumulate_metrics, ActionChunkVectorWrapper, run_with_timeout, get_seed, merge_dicts
from gymnasium import Env
from gtp.algo.common import Algorithm, TrainState
import zipfile
import time
import coloredlogs
from functools import partial
import cv2


logger = logging.getLogger(__name__)
class OffPolicyTrainer:
    def __init__(self, algo: Algorithm, cfg: OmegaConf, env: Optional[Env]=None, eval_env: Optional[Env]=None, verbose=1):
        self.algo = algo
        self.cfg = cfg
        self.use_image_observations = bool(cfg.get('use_image_observations', False))
        self.image_observation_size = normalize_image_observation_size(cfg.get('image_observation_size', 64))
        if env is None:
            env = gym.make_vec(cfg.env_name, num_envs=cfg.n_train_envs, vectorization_mode="sync", vector_kwargs={
                               'autoreset_mode': AutoresetMode.DISABLED}, render_mode='rgb_array')
            env = ActionChunkVectorWrapper(env, n=cfg.action_chunk_size, repeat=cfg.repeat_action)
        self.env = env
        if eval_env is None:
            eval_env = gym.make_vec(cfg.env_name, num_envs=cfg.n_eval_envs,
                                    vectorization_mode="sync", render_mode='rgb_array')
            eval_env = ActionChunkVectorWrapper(eval_env, n=cfg.action_chunk_size, repeat=cfg.repeat_action)
        self.eval_env = eval_env
        self.obs_shape, self.obs_dtype = self.get_obs_spec()
        self.verbose = verbose

    def get_obs_spec(self):
        if self.use_image_observations:
            height, width = self.image_observation_size
            return (height, width, 3), np.uint8

        single_obs_space = getattr(self.env, 'single_observation_space', self.env.observation_space)
        if isinstance(single_obs_space, gym.spaces.Dict):
            obs_dim = sum([single_obs_space[key].shape[-1] for key in self.cfg.obs_keys])
            return (obs_dim,), np.float32
        return tuple(single_obs_space.shape), np.float32

    def _get_vector_envs(self, env):
        current = env
        visited = set()
        while current is not None and id(current) not in visited:
            if hasattr(current, 'envs'):
                return current.envs
            visited.add(id(current))
            current = getattr(current, 'env', None)

        unwrapped = getattr(env, 'unwrapped', None)
        if unwrapped is not None and hasattr(unwrapped, 'envs'):
            return unwrapped.envs
        raise AttributeError('Unable to access sub-environments for image observations.')

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame)
        if frame.ndim != 3:
            raise ValueError(f'Expected rendered frame with shape [H, W, C], got {frame.shape}')

        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                max_value = float(frame.max()) if frame.size > 0 else 0.0
                scale = 255.0 if max_value <= 1.0 else 1.0
                frame = np.clip(frame * scale, 0.0, 255.0).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        height, width = self.image_observation_size
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame)

    def render_obs(self, env) -> np.ndarray:
        frames = [self._resize_frame(sub_env.render()) for sub_env in self._get_vector_envs(env)]
        return np.stack(frames, axis=0)

    def flatten_obs(self, obs, env=None):
        if self.use_image_observations:
            if env is None:
                raise ValueError('env is required when use_image_observations=True')
            return self.render_obs(env)
        if isinstance(obs, dict):
            obs = np.concatenate([obs[k] for k in self.cfg.obs_keys], axis=-1)
        return obs.astype(np.float32, copy=False)

    def setup_train(self, resume_path: Optional[str] = None):
        if resume_path is not None:
            self.output_dir = resume_path
        else:
            self.output_dir = self.get_output_dir()
            os.makedirs(self.output_dir, exist_ok=True)
            # Write source code to output dir
            self.write_file_contents(self.output_dir)

        self.ep_tracker = EpisodeTracker(self.cfg.n_train_envs)
        if self.cfg.save_every_n_step > 0:
            folder = op.join(self.output_dir, 'checkpoints')
            self.ckptr = self.get_ckpt_manager(folder)
    

        if self.cfg.use_wandb:
            self.setup_wandb()
        self.logger = SummaryWriter(self.output_dir)

        # Log messages to file
        if self.verbose:
            coloredlogs.install(
                datefmt='%d/%m %H:%M:%S.%f',
                fmt='%(asctime)s %(levelname)s %(message)s',
                level='INFO',
            )
            root_logger = logging.getLogger()
            num = 1
            msg_file = op.join(self.output_dir, 'messages.log')
            while os.path.exists(msg_file):
                msg_file = op.join(self.output_dir, f'messages_{num}.log')
                num += 1
            file_handler = logging.FileHandler(msg_file)
            file_handler.setFormatter(root_logger.handlers[0].formatter)
            for handler in root_logger.handlers[1:]:  # all except stdout
                root_logger.removeHandler(handler)
            root_logger.addHandler(file_handler)
        
        logger.info(f'Available devices:\t{jax.devices()}')

    def init_buffer(self) -> EpisodeReplayBuffer:
        return EpisodeReplayBuffer(
            obs_shape=self.obs_shape,
            act_dim=self.algo.act_dim,
            horizon=self.cfg.algo.get('horizon', 1),
            num_envs=self.cfg.n_train_envs,
            max_episode_length=self.env.spec.max_episode_steps,
            size=self.cfg.replay_buffer_size,
            n_elites=self.n_elites,
            seed=self.cfg.seed,
            obs_dtype=self.obs_dtype,
        )

    def random_elites(self, batch_size: int) -> np.ndarray:
        """Random elite sequences, used for the warmup transitions (no policy available yet)."""
        shape = (batch_size, self.buffer.n_elites, self.buffer.horizon, self.algo.act_dim)
        return np.random.rand(*shape).astype(np.float32) * 2 - 1

    def warmup(self):
        obs, _ = self.env.reset()
        obs = self.flatten_obs(obs, env=self.env)
        while self.step < self.cfg.warmup_steps:
            action = np.random.rand(*self.env.action_space.shape) * 2 - 1
            next_obs, reward, terminated, truncated, info = self.env.step(scale_action(action, self.env.action_space))
            next_obs = self.flatten_obs(next_obs, env=self.env)
            experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
            self.buffer.add_batch(experience, elites=self.random_elites(self.cfg.n_train_envs),
                                  elite_horizon=self.elite_horizon)
            dones = terminated | truncated
            if dones.any():
                next_obs, _ = self.env.reset(options={'reset_mask': dones})
                next_obs = self.flatten_obs(next_obs, env=self.env)
            obs = next_obs
            self.step += self.cfg.n_train_envs
        logger.info(f'Warmup done, buffer size: {len(self.buffer)}')
        return obs
    
    @property
    def n_elites(self) -> int:
        """Elites cached per transition: 0 unless the algo bootstraps (`GTP` with `n_elites`)."""
        return getattr(self.algo, 'n_elites', None) or 0

    @property
    def bootstrap_elites(self) -> bool:
        """Whether the rollout policy hands back the elites it picked the action from."""
        return self.n_elites > 0

    def sample(self, policy_fn, state: TrainState, obs: np.ndarray, sample_key: jax.Array, policy_aux: dict):
        out = policy_fn(state, obs, sample_key, self.step, **policy_aux)
        elites = None
        if self.bootstrap_elites:
            # The rollout policy returns the elites it selected the action from, they are cached in
            # the buffer and reused as training targets (see `GTP.get_rollout_action`).
            action, elites = jax.device_get(out)
        else:
            action, policy_aux = out if isinstance(out, tuple) else (out, {})
            action = jax.device_get(action)
        action = np.asarray(action).astype(np.float32)
        next_obs, reward, terminated, truncated, info = self.env.step(
            scale_action(action, self.env.action_space))
        next_obs = self.flatten_obs(next_obs, env=self.env)

        experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
        self.buffer.add_batch(experience, elites=elites, elite_horizon=self.elite_horizon)
        dones = terminated | truncated
        is_success = info.get('is_success', None)
        self.ep_tracker.step(reward, dones, None if is_success is None else np.asarray(is_success).astype(bool))
        if dones.any():
            if hasattr(self.algo, 'reset_policy_aux'):
                sample_key, r = jax.random.split(sample_key)
                fresh_policy_aux = self.algo.reset_policy_aux(r, self.cfg.n_train_envs)
                policy_aux = merge_dicts(fresh_policy_aux, policy_aux, dones)
            next_obs, _ = self.env.reset(options={'reset_mask': dones})
            next_obs = self.flatten_obs(next_obs, env=self.env)
        return next_obs, policy_aux
    
    def compile_algo(self):
        def _compile_algo():
            dummy_experience = self.buffer.sample(self.cfg.batch_size, obs_frames=self.update_obs_frames)
            dummy_obs_roll = jnp.asarray(np.zeros((self.cfg.n_train_envs, *self.obs_shape), dtype=self.obs_dtype))
            dummy_obs_eval = jnp.asarray(np.zeros((self.cfg.n_eval_envs, *self.obs_shape), dtype=self.obs_dtype))
            dummy_obs_refresh = jnp.asarray(
                np.zeros((self.elite_refresh_batch_size, *self.obs_shape), dtype=self.obs_dtype))
            dummy_key = jax.random.PRNGKey(0)

            jax.clear_caches()
            update_fn = jax.jit(self.algo.update)
            update_fn = update_fn.lower(self.state, dummy_experience, dummy_key, 0).compile()
            dummy_roll_policy_aux = self.algo.reset_policy_aux(
                dummy_key, self.cfg.n_train_envs) if hasattr(self.algo, 'reset_policy_aux') else {}
            dummy_eval_policy_aux = self.algo.reset_policy_aux(
                dummy_key, self.cfg.n_eval_envs) if hasattr(self.algo, 'reset_policy_aux') else {}
            policy_roll = jax.jit(self.algo.get_rollout_action)
            policy_roll = policy_roll.lower(
                self.state, dummy_obs_roll, dummy_key, 0, **dummy_roll_policy_aux).compile()
            policy_eval = jax.jit(self.algo.get_eval_action)
            policy_eval = policy_eval.lower(
                self.state, dummy_obs_eval, dummy_key, 0, **dummy_eval_policy_aux).compile()
            refresh_fn = None
            if self.elite_refresh_every > 0:
                refresh_fn = jax.jit(self.algo.refresh_elites)
                refresh_fn = refresh_fn.lower(self.state, dummy_obs_refresh, dummy_key, 0).compile()
            logger.info('Algorithm compiled')
            return update_fn, policy_roll, policy_eval, refresh_fn
        # single-CPU cluster jobs compile far slower than a workstation; let them raise the limit
        return run_with_timeout(partial(_compile_algo), timeout_s=self.cfg.get('compile_timeout_s', 360),
                                name="Compile Algo")

    @property
    def elite_horizon(self) -> int:
        """Planning horizon the elites are currently being optimized for (curriculum horizon)."""
        return getattr(self.algo, 'current_curriculum_horizon', 0)

    @property
    def update_obs_frames(self) -> Optional[int]:
        """Observations per sampled segment the update actually reads (None = all H+1)."""
        return getattr(self.algo, 'update_obs_frames', None)

    @property
    def elite_refresh_every(self) -> int:
        """Environment steps between two elite-refresh passes (0 disables refreshing)."""
        return getattr(self.algo, 'elite_refresh_every', 0) if self.bootstrap_elites else 0

    @property
    def elite_refresh_batch_size(self) -> int:
        """Number of buffer entries whose elites are recomputed per refresh pass."""
        return getattr(self.algo, 'elite_refresh_batch_size', 0)

    def refresh_elites(self, refresh_fn, key: jax.Array):
        """Re-run diffusion + langevin for a batch of buffer entries and overwrite their elites.

        This keeps the cached elites from becoming arbitrarily stale as the models improve, and
        gradually replaces elites planned with a short curriculum horizon by current-horizon ones.
        """
        # only the first observation of each row is planned from
        trajectory, indices = self.buffer.sample_with_indices(self.elite_refresh_batch_size, obs_frames=1)
        elites = jax.device_get(refresh_fn(self.state, trajectory.obs[:, 0], key, self.step))
        self.buffer.replace_elites(indices, elites, elite_horizon=self.elite_horizon)

    def train(self, resume_path: Optional[str] = None, resume_step: Optional[int] = None):
        update_metrics = []
        def post_step(key, force_log=False):
            nonlocal update_metrics
            # Evaluate
            if self.cfg.eval_every_n_step > 0 and (self.step) % self.cfg.eval_every_n_step == 0:
                key, eval_key = jax.random.split(key)
                self.evaluate(policy_eval, self.state, eval_key, log=True)

            # Log
            if force_log or (self.cfg.log_every_n_step > 0 and self.step % self.cfg.log_every_n_step == 0):
                train_env_dct = self.ep_tracker.reset()
                self.add_scalar({f'train/{k}': v for k, v in train_env_dct.items()})
                if len(update_metrics) > 0:
                    self.add_scalar({f'update/{k}': v for k, v in accumulate_metrics(update_metrics).items()})
                update_metrics = []

            # Save checkpoint
            if self.cfg.save_every_n_step > 0 and self.step % self.cfg.save_every_n_step == 0:
                self.save_checkpoint()

        # Initialize model and optimizer
        self.setup_train(resume_path)
        key = jax.random.PRNGKey(get_seed())
        self.state = self.algo.init_train_state(self.cfg)
        self.buffer = self.init_buffer()
        if resume_path is None:
            assert resume_step is None, "resume_step should be None when resume_path is None"
            self.step = 0
        else:
            if resume_step is None:
                ckpts = glob.glob(op.join(resume_path, 'checkpoints', 'step_*'))
                ckpts = [int(op.basename(ckpt)[5:]) for ckpt in ckpts]
                if len(ckpts) == 0:
                    raise ValueError(f"No checkpoints found in {op.join(resume_path, 'checkpoints')}")
                resume_step = max(ckpts) if len(ckpts) > 0 else None
            self.buffer = self.buffer.__class__.load(
                op.join(resume_path, 'buffers', f'replay_buffer_{resume_step}.pkl'))
            state_dict = self.ckptr.restore(resume_step)
            self.state = self.state.replace(**state_dict)
            self.step = resume_step

        if curriculum_info := self.algo.update_curriculum(self.step):
            self.add_scalar({f'curriculum/{k}': v for k, v in curriculum_info.items()})

        if self.step < self.cfg.warmup_steps:
            self.warmup()
            # TODO: ep_tracker should be resetted after warmup
            self.buffer.reset_ongoing()

        pbar = tqdm(
            total=self.cfg.total_steps,
            initial=self.step,
            desc="Steps",
            position=0,
            leave=True
        )

        update_fn, policy_roll, policy_eval, refresh_fn = self.compile_algo()
        obs, _ = self.env.reset()
        obs = self.flatten_obs(obs, env=self.env)
        steps_since_refresh = 0
        if hasattr(self.algo, 'reset_policy_aux'):
            key, r = jax.random.split(key)
            policy_aux = self.algo.reset_policy_aux(r, self.cfg.n_train_envs)
        else:
            policy_aux = {}
        while self.step < self.cfg.total_steps:
            key, sample_key, update_key, refresh_key, post_key = jax.random.split(key, 5)

            # Sample
            for _ in range(self.cfg.sample_per_iteration):
                sample_key, r = jax.random.split(sample_key)
                obs, policy_aux = self.sample(policy_roll, self.state, obs, r, policy_aux)

            # Refresh the elites of a batch of (potentially old) buffer entries
            if self.elite_refresh_every > 0:
                steps_since_refresh += self.cfg.n_train_envs
                if steps_since_refresh >= self.elite_refresh_every:
                    steps_since_refresh = 0
                    self.refresh_elites(refresh_fn, refresh_key)

            # Update
            for _ in range(self.cfg.update_per_iteration):
                update_key, r = jax.random.split(update_key)
                experience = self.buffer.sample(self.cfg.batch_size, obs_frames=self.update_obs_frames)
                self.state, _update_metrics = update_fn(self.state, experience, r, self.step)
                if _update_metrics:
                    update_metrics.append(_update_metrics)

            if curriculum_info := self.algo.update_curriculum(self.step):
                # Recompile algo if curriculum changed
                update_fn, policy_roll, policy_eval, refresh_fn = self.compile_algo()
                self.add_scalar({f'curriculum/{k}': v for k, v in curriculum_info.items()})
            
            # Post step (eval, log, save)
            post_step(post_key)
            
            # Advance step
            self.step += self.cfg.n_train_envs
            pbar.update(self.cfg.n_train_envs)

        # Finish training 
        post_step(key, force_log=True)
        # Wait for checkpoints to be saved
        self.ckptr.wait_until_finished()
        if self.cfg.use_wandb:
            wandb.finish()
        pbar.close()

    def evaluate(self, policy_fn, state, key, log=True, record_all=False):
        frames = []
        env = self.eval_env
        obs, _ = env.reset()
        obs = self.flatten_obs(obs, env=env)
        if hasattr(self.algo, 'reset_policy_aux'):
            key, r = jax.random.split(key)
            policy_aux = self.algo.reset_policy_aux(r, self.cfg.n_eval_envs)
        else:
            policy_aux = {}
        if self.cfg.record_eval_videos:
            frame = env.unwrapped.envs[0].render() if not record_all and hasattr(
                env.unwrapped, 'envs') else env.render()
            frames.append(frame)

        rewards, dones, suc = [], [], []
        term = np.array([False] * self.cfg.n_eval_envs)

        while not term.all():
            key, r = jax.random.split(key)
            action = policy_fn(state, obs, r, self.step, **policy_aux)
            action, policy_aux = action if isinstance(action, tuple) else (action, {})
            action = jax.device_get(action).astype(np.float32)
            obs, rew, terminated, truncated, info = env.step(scale_action(action, env.action_space))
            obs = self.flatten_obs(obs, env=env)
            if self.cfg.record_eval_videos:
                frame = env.unwrapped.envs[0].render() if not record_all and hasattr(
                    env.unwrapped, 'envs') else env.render()
                frames.append(frame)

            done = terminated | truncated
            dones.append(done)
            rewards.append(rew)
            if 'is_success' in info:
                # cast: an env reporting numpy.bool_ gets an object array from the vector env, and
                # .mean() of an object bool array is a logical OR -- "any env succeeded"
                suc.append(np.asarray(info['is_success']).astype(bool))

            term = term | done

        dones = np.array(dones).T
        rewards = np.array(rewards).T
        
        dead_idx = dones.argmax(axis=1) + (~dones.any(axis=1)) * dones.shape[1] + 1
        rewards = np.where(np.arange(rewards.shape[1]) < dead_idx[:, None], rewards, 0)
        returns = rewards.sum(axis=1)
        dct = {'eval/ep_returns': returns.mean(), 'eval/ep_lens': dead_idx.mean(), 
               'eval/ep_return_min': returns.min(), 'eval/ep_return_max': returns.max()}
        if len(suc) > 0:
            suc = np.array(suc).T
            suc = np.where(np.arange(suc.shape[1]) < dead_idx[:, None], suc, 0)
            dct['eval/success_rate'] = suc.any(axis=1).mean()
        if log:
            self.add_scalar(dct)
        if self.cfg.record_eval_videos:
            frames = np.array(frames)
            if record_all:
                frames = [frames[:d, i] for i, d in enumerate(dead_idx)]
            else:
                frames = frames[:dead_idx[0]]
            self.save_videos(frames)
        return dct
    
    def save_videos(self, frames):
        video_folder = op.join(self.output_dir, 'videos')
        os.makedirs(video_folder, exist_ok=True)
        def _save_video(path, _frames):
            video_path = op.join(video_folder, f'{path}.mp4')
            height, width, _ = _frames[0].shape
            max_sz = max(height, width)
            resize_factor = 1
            if max_sz > 120:
                resize_factor = 1 / (max_sz // 120)
            height, width = int(height*resize_factor), int(width*resize_factor)
            fps = self.eval_env.metadata.get('render_fps', 30)    
            video_writer = cv2.VideoWriter(
                video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            for frame in _frames:
                # Resize to reduce video size
                frame = cv2.resize(frame, (width, height))
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video_writer.release()
            if self.cfg.use_wandb:
                wandb.run.log({'video_0': wandb.Video(video_path, format="mp4"), 'global_step': self.step})
        if isinstance(frames, list):
            for i, _frames in enumerate(frames):
                _save_video(f'step_{self.step}_{i}', _frames)
        else:
            _save_video(f'step_{self.step}', frames)

    def save_checkpoint(self):
        ckpt = to_state_dict(self.state)
        try:
            self.ckptr.save(self.step, ckpt)
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {self.step}: {e}")
            time.sleep(2)  # Wait a bit before retrying
            try:
                self.ckptr.save(self.step, ckpt)
            except Exception as e:
                logger.error(f"Retry failed to save checkpoint at step {self.step}: {e}")
        # image buffers are never saved; with elite bootstrapping a 1M buffer is ~0.75 GB per
        # save, which `save_replay_buffer=False` skips
        if self.use_image_observations or not self.cfg.get('save_replay_buffer', True):
            return
        os.makedirs(op.join(self.output_dir, 'buffers'), exist_ok=True)
        try:
            self.buffer.save(
                op.join(self.output_dir, 'buffers', f'replay_buffer_{self.step}.pkl'))
        except Exception as e:
            logger.error(f"Failed to save replay buffer at step {self.step}: {e}")
            time.sleep(2)  # Wait a bit before retrying
            try:
                self.buffer.save(op.join(self.output_dir, 'buffers', f'replay_buffer_{self.step}.pkl'))
            except Exception as e:
                logger.error(f"Retry failed to save replay buffer at step {self.step}: {e}")

    def get_output_dir(self):
        identifier = output_dir = self.get_model_identifier()
        num = 1
        while os.path.exists(output_dir):
            output_dir = f"{identifier}-{num}"
            num += 1
        return output_dir

    def add_scalar(self, dct: dict):
        for tag, value in dct.items():
            self.logger.add_scalar(tag, value, global_step=self.step)
        self.logger.flush()

    def setup_wandb(self):
            # Start wandb
        tags = None if self.cfg.wandb_tag == '' else [self.cfg.wandb_tag]
        wandb.init(project=self.cfg.wandb_project_name, sync_tensorboard=True, tags=tags)
        # log code
        fp = os.path.abspath(__file__)
        folder = os.path.dirname(fp)
        project_folder = os.path.dirname(folder)
        wandb.run.log_code(project_folder)
        # log hypm
        wandb.config.update(OmegaConf.to_container(
            self.cfg, resolve=True, throw_on_missing=True))
        wandb.config.update({'log_dir': self.output_dir})

    @staticmethod
    # todo: too verbose
    def get_ckpt_manager(folder, verbose=0):
        from absl import logging

        class StandardLogger:
            def log_entry(self, msg, *args, **kwargs):
                step, dir = msg['step'], msg['directory']
                logger.info(f'Saved ckpt for epoch={step} to {dir}')
        # logging.set_verbosity(logging.DEBUG)
        folder = op.abspath(folder)
        return ocp.CheckpointManager(folder, ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
                                     logger=StandardLogger(),
                                     options=ocp.CheckpointManagerOptions(step_prefix='step',
                                                                          step_format_fixed_length=7,
                                                                          enable_async_checkpointing=True))

    def write_file_contents(self, target_base_dir):
        """Write cached config file contents to target directory."""
        assert op.isdir(target_base_dir)

        # write combined config yaml
        fpath = op.relpath(op.join(target_base_dir, 'combined.yaml'))
        with open(fpath, 'w') as f:
            f.write(yaml.dump(OmegaConf.to_container(
                self.cfg, resolve=True, throw_on_missing=True)))
        logger.info('Written %s' % fpath)

        # Copy source folder contents over
        target_path = op.relpath(target_base_dir + '/src.zip')
        source_path = op.relpath(op.dirname(__file__) + '/../')
        def filter_(x): return x.endswith('.py') or x.endswith('.yaml')  # noqa
        with zipfile.ZipFile(target_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(source_path):
                for file_or_dir in files + dirs:
                    full_path = op.join(root, file_or_dir)
                    if op.isfile(full_path) and filter_(full_path):
                        zip_file.write(
                            op.join(root, file_or_dir),
                            op.relpath(op.join(root, file_or_dir),
                                       op.join(source_path, op.pardir)))
        logger.info('Written source folder to %s' % op.relpath(target_path))

    ### ---- PROJECT-SPECIFIC ---- ###
    def get_model_identifier(self):
        algo_config = OmegaConf.to_container(self.cfg.algo, resolve=True)

        algo_config['seed'] = self.cfg.seed

        algo = get_class(self.cfg.algo._target_).__name__
        return os.path.join('outputs', self.cfg.env_name, algo.title(), hash_config(algo_config))


def scale_action(scaled_action, action_space):
    """Convert action from [-1, 1] to [low, high]."""
    low = action_space.low
    high = action_space.high
    action = low + (0.5 * (scaled_action + 1.0) * (high - low))
    return np.clip(action, low, high)


def normalize_image_observation_size(size) -> tuple[int, int]:
    if isinstance(size, int):
        return (size, size)
    if len(size) != 2:
        raise ValueError(f"image_observation_size must be an int or [height, width], got {size}")
    return tuple(int(v) for v in size)
