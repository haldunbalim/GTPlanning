import pickle
from typing import NamedTuple, Optional, Tuple
import numpy as np

# Elites are only used as diffusion/critic targets, so half precision is plenty and
# keeps the (potentially very large) elite storage affordable.
ELITE_DTYPE = np.float16


def normalize_obs_shape(obs_shape) -> tuple[int, ...]:
    if isinstance(obs_shape, int):
        return (obs_shape,)
    return tuple(obs_shape)


class Trajectory(NamedTuple):
    """
    Data class for finite horizon trajectories.

    Args:
        obs: Observations with shape [H+1, *obs_shape] for single trajectory
             or [batch_size, H+1, *obs_shape] for batch
        action: Actions with shape [H, act_dim] for single trajectory or [batch_size, H, act_dim] for batch
        reward: Rewards with shape [H,] for single trajectory or [batch_size, H] for batch
        done: Done flags with shape [H,] for single trajectory or [batch_size, H] for batch
        elites: Elite action sequences cached at the first timestep of the segment with shape
            [n_elites, H, act_dim] for single trajectory or [batch_size, n_elites, H, act_dim] for batch
        elite_horizon: Planning horizon the elites were optimized for (curriculum horizon at the
            time they were computed), scalar for single trajectory or [batch_size] for batch
    """
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    segment_length: np.ndarray
    elites: np.ndarray
    elite_horizon: np.ndarray

    def __repr__(self):
        return f"Trajectory(batch_size={self.batch_size()}, H={self.horizon})"

    @property
    def horizon(self) -> int:
        return self.action.shape[-2]

    def batch_size(self) -> int:
        """Return the batch size, or 0 if single trajectory."""
        if self.action.ndim == 2:  # Single trajectory: [H, act_dim]
            return 0
        return self.action.shape[0]

    def __getitem__(self, idx):
        if self.batch_size() == 0:
            raise IndexError("Single trajectory has no batch dimension.")
        return Trajectory(
            obs=self.obs[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            done=self.done[idx],
            segment_length=self.segment_length[idx],
            elites=self.elites[idx],
            elite_horizon=self.elite_horizon[idx]
        )


def probe_batch_size(reward: np.ndarray) -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    next_obs: np.ndarray

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        obs_shape = normalize_obs_shape(obs_dim)
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, *obs_shape), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, *obs_shape), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        return Experience(obs=obs, action=action, reward=reward, done=terminated, next_obs=next_obs)

class ReplayBuffer:
    def add_batch(self, samples, elites: Optional[np.ndarray] = None, elite_horizon: int = 0) -> None:
        raise NotImplementedError

    def reset_ongoing(self) -> None:
        raise NotImplementedError

    def sample(self, size: int):
        raise NotImplementedError

    def sample_with_indices(self, size: int) -> Tuple[object, np.ndarray]:
        raise NotImplementedError

    def replace(self, indices: np.ndarray, samples) -> None:
        raise NotImplementedError

    def replace_elites(self, indices: np.ndarray, elites: np.ndarray, elite_horizon: int) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError


class OngoingEpisodes:
    """
    Helper class to manage ongoing episodes for parallel environments.
    Stores incomplete episodes with one slot per environment.
    """

    def __init__(self, obs_shape, act_dim: int, horizon: int, max_episode_length: int, num_envs: int,
                 obs_dtype=np.float32, n_elites: int = 0):
        self.obs_shape = normalize_obs_shape(obs_shape)
        self.obs_dtype = np.dtype(obs_dtype)
        self.act_dim = act_dim
        self.max_episode_length = max_episode_length
        self.num_envs = num_envs
        self.horizon = horizon
        self.n_elites = n_elites

        # Storage arrays:
        buf_len = max_episode_length + horizon - 1
        self.obs = np.zeros((num_envs, buf_len + 1, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((num_envs, buf_len, act_dim), dtype=np.float32)
        self.rewards = np.zeros((num_envs, buf_len), dtype=np.float32)
        self.terminated = np.zeros((num_envs, buf_len), dtype=np.bool_)
        self.truncated = np.zeros((num_envs, buf_len), dtype=np.bool_)
        # Elite action sequences that produced each transition (diffusion + langevin) and the
        # curriculum horizon they were optimized for
        self.elites = np.zeros((num_envs, buf_len, n_elites, horizon, act_dim), dtype=ELITE_DTYPE)
        self.elite_horizon = np.zeros((num_envs, buf_len), dtype=np.int32)

        # Track which environments have active episodes and their current step
        self.active = np.zeros(num_envs, dtype=np.bool_)
        self.steps = np.zeros(num_envs, dtype=np.int32)

    def add_batch_transition(self, experience: Experience, elites: Optional[np.ndarray] = None,
                             elite_horizon: int = 0) -> list:
        """
        Add transitions for all environments at once.
        Automatically starts new episodes for environments that just finished.

        Args:
            experience: batched transition, one entry per environment
            elites: elite action sequences [num_envs, n_elites, horizon, act_dim] used to pick the action
            elite_horizon: planning horizon the elites were optimized for

        Returns:
            List of tuples for completed episodes: (env_idx, episode_data)
        """
        completed_episodes = []

        # Start new episodes for inactive environments
        inactive_mask = ~self.active
        if np.any(inactive_mask):
            self.obs[inactive_mask, 0] = experience.obs[inactive_mask]
            self.steps[inactive_mask] = 0
            self.active[inactive_mask] = True

        # All environments are now active, get current steps
        current_steps = self.steps

        # Check for truncation
        truncated = (current_steps + 1 >= self.max_episode_length)

        # Store transitions for all environments
        env_indices = np.arange(self.num_envs)
        self.actions[env_indices, current_steps] = experience.action
        self.rewards[env_indices, current_steps] = experience.reward
        self.terminated[env_indices, current_steps] = experience.done
        self.truncated[env_indices, current_steps] = truncated
        self.obs[env_indices, current_steps + 1] = experience.next_obs
        if self.n_elites > 0:
            assert elites is not None, "elites must be provided when n_elites > 0"
            self.elites[env_indices, current_steps] = elites.astype(ELITE_DTYPE)
            self.elite_horizon[env_indices, current_steps] = elite_horizon

        # Update step counters
        self.steps += 1

        # Find completed episodes
        done_mask = experience.done | truncated
        done_envs = np.where(done_mask)[0]

        for env_idx in done_envs:
            episode_length = self.steps[env_idx]
            completed_episode = (
                self.obs[env_idx, :episode_length + 1].copy(),
                self.actions[env_idx, :episode_length].copy(),
                self.rewards[env_idx, :episode_length].copy(),
                self.terminated[env_idx, :episode_length].copy(),
                self.truncated[env_idx, :episode_length].copy(),
                self.elites[env_idx, :episode_length].copy(),
                self.elite_horizon[env_idx, :episode_length].copy(),
                episode_length
            )
            completed_episodes.append((env_idx, completed_episode))

            # Reset this environment's slot
            self.active[env_idx] = False
            self.steps[env_idx] = 0

        return completed_episodes
    
    def reset(self) -> None:
        """
        Drop all current ongoing episodes and reset all environments to inactive state.
        """
        self.active[:] = False
        self.steps[:] = 0

    def sample_counts(self) -> np.ndarray:
        return self.steps * self.active.astype(np.int32)

    def sample_count(self) -> int:
        return int(self.sample_counts().sum())

    def sample_index_to_coords(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = self.sample_counts()
        cumulative = np.zeros(self.num_envs + 1, dtype=np.int32)
        cumulative[1:] = np.cumsum(counts, dtype=np.int32)
        env_idx = np.searchsorted(cumulative, queries, side='right') - 1
        env_idx = np.clip(env_idx, 0, self.num_envs - 1)
        time_idx = queries - cumulative[env_idx]
        return env_idx, time_idx

    def sample_trajectory(self, env_indices: np.ndarray, time_indices: np.ndarray,
                          obs_frames: Optional[int] = None) -> Trajectory:
        obs_offset = np.arange(0, self.horizon + 1 if obs_frames is None else obs_frames)
        act_offset = np.arange(0, self.horizon)
        observations = self.obs[env_indices[:, None], time_indices[:, None] + obs_offset]
        actions = self.actions[env_indices[:, None], time_indices[:, None] + act_offset]
        rewards = self.rewards[env_indices[:, None], time_indices[:, None] + act_offset]
        done_flags = self.terminated[env_indices[:, None], time_indices[:, None] + act_offset]
        segment_lengths = self.horizon - np.maximum(time_indices + self.horizon - self.steps[env_indices], 0)

        return Trajectory(
            obs=observations,
            action=actions,
            reward=rewards,
            done=done_flags,
            segment_length=segment_lengths,
            elites=self.elites[env_indices, time_indices],
            elite_horizon=self.elite_horizon[env_indices, time_indices]
        )

    def replace_elites(self, env_indices: np.ndarray, time_indices: np.ndarray, elites: np.ndarray,
                       elite_horizon: int) -> None:
        self.elites[env_indices, time_indices] = elites.astype(ELITE_DTYPE)
        self.elite_horizon[env_indices, time_indices] = elite_horizon


class EpisodeReplayBuffer(ReplayBuffer):
    """
    A replay buffer that stores complete episodes from parallel environments.

    Args:
        obs_dim: Dimension of observations
        act_dim: Dimension of actions
        size: Maximum number of transitions to store (N)
        max_episode_length: Maximum length of each episode (T)
        num_envs: Number of parallel environments (k)
        horizon: Length of trajectories to sample (H)
        n_elites: Number of elite action sequences cached per transition (0 disables elite storage)
        seed: Random seed for sampling
    """

    def __init__(self, obs_shape, act_dim: int, horizon: int, num_envs: int=1,
                 size: int=1_000_000, max_episode_length: int=1000, n_elites: int = 0,
                 seed: int = 0, obs_dtype=np.float32) -> None:
        self.obs_shape = normalize_obs_shape(obs_shape)
        self.obs_dtype = np.dtype(obs_dtype)
        self.obs_dim = self.obs_shape[0] if len(self.obs_shape) == 1 else None
        self.act_dim = act_dim
        num_episodes = size // max_episode_length
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.num_envs = num_envs
        self.horizon = horizon
        self.n_elites = n_elites

        # Storage arrays for completed episodes: [N, T+1, d] for obs, [N, T, d] for others
        buf_len = max_episode_length + horizon - 1
        self.obs = np.zeros((num_episodes, buf_len + 1, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.zeros((num_episodes, buf_len, act_dim), dtype=np.float32)
        self.rewards = np.zeros((num_episodes, buf_len), dtype=np.float32)
        self.terminated = np.zeros((num_episodes, buf_len), dtype=np.bool_)
        self.truncated = np.zeros((num_episodes, buf_len), dtype=np.bool_)
        # Elite action sequences (diffusion + langevin) attached to each stored transition and the
        # curriculum horizon they were optimized for
        self.elites = np.zeros((num_episodes, buf_len, n_elites, horizon, act_dim), dtype=ELITE_DTYPE)
        self.elite_horizon = np.zeros((num_episodes, buf_len), dtype=np.int32)

        # Episode metadata
        self.episode_lengths = np.zeros(num_episodes, dtype=np.int32)
        self.indices = np.zeros(num_episodes + 1, dtype=np.int32)
        self.num_stored_episodes = 0
        self.episode_ptr = 0

        # Helper for managing ongoing episodes
        self.ongoing = OngoingEpisodes(
            self.obs_shape,
            act_dim,
            horizon,
            max_episode_length,
            num_envs,
            obs_dtype=self.obs_dtype,
            n_elites=n_elites,
        )
        self.rng = np.random.default_rng(seed)

    def _completed_sample_count(self) -> int:
        return int(self.indices[-1])

    def _total_sample_count(self) -> int:
        return self._completed_sample_count() + self.ongoing.sample_count()

    def _store_episode(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                      terminated: np.ndarray, truncated: np.ndarray, elites: np.ndarray,
                      elite_horizon: np.ndarray, episode_length: int) -> None:
        """Store a completed episode in the buffer."""
        episode_idx = self.episode_ptr

        # Store episode data
        self.obs[episode_idx, :episode_length + 1] = obs
        self.actions[episode_idx, :episode_length] = actions
        self.rewards[episode_idx, :episode_length] = rewards
        self.terminated[episode_idx, :episode_length] = terminated
        self.truncated[episode_idx, :episode_length] = truncated
        if self.n_elites > 0:
            self.elites[episode_idx, :episode_length] = elites
            self.elite_horizon[episode_idx, :episode_length] = elite_horizon

        # Update metadata
        old_len = self.episode_lengths[episode_idx]
        self.episode_lengths[episode_idx] = episode_length

        # Update indices for trajectory sampling
        delta = episode_length - old_len

        if delta != 0:
            self.indices[episode_idx + 1:] += delta

        # Advance pointer and update count
        self.episode_ptr = (self.episode_ptr + 1) % self.num_episodes
        self.num_stored_episodes = min(self.num_stored_episodes + 1, self.num_episodes)

    def add_batch(self, experience: Experience, elites: Optional[np.ndarray] = None,
                  elite_horizon: int = 0) -> None:
        """
        Add a batch of transitions from all parallel environments efficiently.

        Args:
            experience: Experience object containing obs, action, reward, done, next_obs
                       with batch dimension [num_envs, ...] for each field
            elites: elite action sequences [num_envs, n_elites, horizon, act_dim] the action was picked from
            elite_horizon: planning horizon the elites were optimized for
        """
        batch_size = experience.batch_size()
        if batch_size != self.num_envs:
            raise ValueError(f"Experience batch size {batch_size} must match num_envs {self.num_envs}")

        # Add batch transition and get completed episodes
        # OngoingEpisodes will automatically start new episodes for inactive environments
        completed_episodes = self.ongoing.add_batch_transition(
            experience, elites=elites, elite_horizon=elite_horizon)

        # Store completed episodes in the main buffer
        for _, episode_data in completed_episodes:
            self._store_episode(*episode_data)

    def reset_ongoing(self) -> None:
        """
        Drop all current ongoing episodes and reset all environments to inactive state.
        """
        self.ongoing.reset()

    def _split_indices(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split flat indices into a boolean mask for completed episodes and validate bounds."""
        indices = np.atleast_1d(np.asarray(indices))
        assert np.all(indices >= 0) and np.all(indices < self._total_sample_count()), "Indices out of bounds"
        return indices, indices < self._completed_sample_count()

    def _get_completed_idx(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ep_idx = np.searchsorted(self.indices, queries, side='right') - 1
        ep_idx = np.clip(ep_idx, 0, len(self.indices) - 2)
        return ep_idx, queries - self.indices[ep_idx]

    def __getitem__(self, indices: np.ndarray) -> Trajectory:
        return self.get(indices)

    def get(self, indices: np.ndarray, obs_frames: Optional[int] = None) -> Trajectory:
        """Trajectories starting at `indices`. `obs_frames` keeps only the first observations of
        each segment (all H+1 when None) -- an image algo whose update reads two frames should not
        gather, copy and transfer nine."""
        indices, completed_mask = self._split_indices(indices)
        batch_size = len(indices)
        completed_total = self._completed_sample_count()
        n_obs = self.horizon + 1 if obs_frames is None else obs_frames

        observations = np.empty((batch_size, n_obs, *self.obs_shape), dtype=self.obs_dtype)
        actions = np.empty((batch_size, self.horizon, self.act_dim), dtype=np.float32)
        rewards = np.empty((batch_size, self.horizon), dtype=np.float32)
        done_flags = np.empty((batch_size, self.horizon), dtype=np.bool_)
        segment_lengths = np.empty(batch_size, dtype=np.int32)
        elites = np.empty((batch_size, self.n_elites, self.horizon, self.act_dim), dtype=ELITE_DTYPE)
        elite_horizons = np.empty(batch_size, dtype=np.int32)

        if np.any(completed_mask):
            completed_indices = indices[completed_mask]
            ep_idx, time_idx = self._get_completed_idx(completed_indices)
            obs_offset = np.arange(0, n_obs)
            act_offset = np.arange(0, self.horizon)
            observations[completed_mask] = self.obs[ep_idx[:, None], time_idx[:, None] + obs_offset]
            actions[completed_mask] = self.actions[ep_idx[:, None], time_idx[:, None] + act_offset]
            rewards[completed_mask] = self.rewards[ep_idx[:, None], time_idx[:, None] + act_offset]
            done_flags[completed_mask] = self.terminated[ep_idx[:, None], time_idx[:, None] + act_offset]
            segment_lengths[completed_mask] = self.horizon - np.maximum(
                time_idx + self.horizon - self.episode_lengths[ep_idx], 0
            )
            elites[completed_mask] = self.elites[ep_idx, time_idx]
            elite_horizons[completed_mask] = self.elite_horizon[ep_idx, time_idx]

        if np.any(~completed_mask):
            ongoing_indices = indices[~completed_mask] - completed_total
            env_idx, time_idx = self.ongoing.sample_index_to_coords(ongoing_indices)
            ongoing_trajectory = self.ongoing.sample_trajectory(env_idx, time_idx, obs_frames=obs_frames)
            observations[~completed_mask] = ongoing_trajectory.obs
            actions[~completed_mask] = ongoing_trajectory.action
            rewards[~completed_mask] = ongoing_trajectory.reward
            done_flags[~completed_mask] = ongoing_trajectory.done
            segment_lengths[~completed_mask] = ongoing_trajectory.segment_length
            elites[~completed_mask] = ongoing_trajectory.elites
            elite_horizons[~completed_mask] = ongoing_trajectory.elite_horizon

        return Trajectory(
            obs=observations,
            action=actions,
            reward=rewards,
            done=done_flags,
            segment_length=segment_lengths,
            elites=elites,
            elite_horizon=elite_horizons
        )

    def replace_elites(self, indices: np.ndarray, elites: np.ndarray, elite_horizon: int) -> None:
        """
        Overwrite the cached elite sequences at the given flat indices.

        Args:
            indices: flat sample indices as returned by `sample_with_indices`
            elites: new elite sequences [len(indices), n_elites, horizon, act_dim]
            elite_horizon: planning horizon the new elites were optimized for
        """
        if self.n_elites == 0:
            return
        indices, completed_mask = self._split_indices(indices)
        elites = np.asarray(elites).astype(ELITE_DTYPE)
        assert len(elites) == len(indices), "elites must have one entry per index"

        if np.any(completed_mask):
            ep_idx, time_idx = self._get_completed_idx(indices[completed_mask])
            self.elites[ep_idx, time_idx] = elites[completed_mask]
            self.elite_horizon[ep_idx, time_idx] = elite_horizon

        if np.any(~completed_mask):
            ongoing_indices = indices[~completed_mask] - self._completed_sample_count()
            env_idx, time_idx = self.ongoing.sample_index_to_coords(ongoing_indices)
            self.ongoing.replace_elites(env_idx, time_idx, elites[~completed_mask], elite_horizon)


    def sample_with_indices(self, batch_size: int,
                            obs_frames: Optional[int] = None) -> tuple[Trajectory, np.ndarray]:
        """
        Sample finite horizon trajectories and return with their starting flat indices.

        Args:
            batch_size: Number of trajectories to sample
            obs_frames: keep only the first `obs_frames` observations of each segment (see `get`)

        Returns:
            tuple of (Trajectory object, starting_flat_indices)
        """
        total = self._total_sample_count()
        if total <= 0:
            raise ValueError("Cannot sample from an empty episode replay buffer.")

        indices = self.rng.integers(0, total, size=batch_size)
        return self.get(indices, obs_frames=obs_frames), indices

    def sample(self, batch_size: int, obs_frames: Optional[int] = None) -> Trajectory:
        """
        Sample finite horizon trajectories from the buffer.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            Trajectory object containing sampled trajectories
        """
        return self.sample_with_indices(batch_size, obs_frames=obs_frames)[0]

    def save(self, path: str) -> None:
        """Save only completed episodes to disk. Ongoing episodes are discarded."""
        data = {
            'obs': self.obs,
            'actions': self.actions,
            'rewards': self.rewards,
            'terminated': self.terminated,
            'truncated': self.truncated,
            'elites': self.elites,
            'elite_horizon': self.elite_horizon,
            'episode_lengths': self.episode_lengths,
            'indices': self.indices,
            'num_stored_episodes': self.num_stored_episodes,
            'episode_ptr': self.episode_ptr,
            'metadata': {
                'obs_dim': self.obs_dim,
                'obs_shape': self.obs_shape,
                'obs_dtype': self.obs_dtype.str,
                'act_dim': self.act_dim,
                'num_episodes': self.num_episodes,
                'max_episode_length': self.max_episode_length,
                'num_envs': self.num_envs,
                'horizon': self.horizon,
                'n_elites': self.n_elites
            }
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> 'EpisodeReplayBuffer':
        """Load an episode buffer from disk. Ongoing episodes helper is reinitialized."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        metadata = data['metadata']
        obs_shape = metadata.get('obs_shape')
        if obs_shape is None:
            obs_shape = (metadata['obs_dim'],)
        obs_dtype = np.dtype(metadata.get('obs_dtype', data['obs'].dtype))
        buffer = cls(
            obs_shape=obs_shape,
            act_dim=metadata['act_dim'],
            horizon=metadata['horizon'],
            num_envs=metadata['num_envs'],
            size=metadata['num_episodes']*metadata['max_episode_length'],
            max_episode_length=metadata['max_episode_length'],
            n_elites=metadata.get('n_elites', 0),
            seed=seed,
            obs_dtype=obs_dtype,
        )

        # Restore completed episode data
        buffer.obs = data['obs']
        buffer.actions = data['actions']
        buffer.rewards = data['rewards']
        buffer.terminated = data['terminated']
        buffer.truncated = data['truncated']
        if 'elites' in data:
            buffer.elites = data['elites']
            buffer.elite_horizon = data['elite_horizon']
        buffer.episode_lengths = data['episode_lengths']
        buffer.indices = data.get('indices', np.zeros(metadata['num_episodes'] + 1, dtype=np.int32))
        buffer.num_stored_episodes = data['num_stored_episodes']
        buffer.episode_ptr = data['episode_ptr']

        # OngoingEpisodes is already initialized in __init__, no need to restore

        return buffer

    def __len__(self) -> int:
        """Return the number of trajectories that can be sampled."""
        return self._total_sample_count()

    def __repr__(self) -> str:
        active_envs = np.sum(self.ongoing.active)
        return (f"EpisodeReplayBuffer(episodes={self.num_stored_episodes}/{self.num_episodes}, "
                f"max_len={self.max_episode_length}, envs={self.num_envs}, "
                f"active_envs={active_envs}, ptr={self.episode_ptr})")
    
