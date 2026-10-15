import numpy as np
import jax.numpy as jnp

class EpisodeTracker:
    """Tracks episode returns and lengths for vectorized environments."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.episode_returns = np.zeros(num_envs)
        self.episode_lens = np.zeros(num_envs)
        self.is_success = np.zeros(num_envs, dtype=bool)
        self.completed_episodes = {'ep_returns': [], 'ep_lens': [], 'is_success': []}

    def step(self, rewards: np.ndarray, dones: np.ndarray, is_success: np.ndarray = None):
        """Update tracker with step results."""
        self.episode_returns += rewards
        self.episode_lens += 1
        if is_success is not None:
            self.is_success = np.logical_or(self.is_success, is_success)

        # Record completed episodes
        if np.any(dones):
            for i in np.where(dones)[0]:
                self.completed_episodes['ep_returns'].append(self.episode_returns[i])
                self.completed_episodes['ep_lens'].append(self.episode_lens[i])
                if is_success is not None:
                    self.completed_episodes['is_success'].append(self.is_success[i])

                self.episode_returns[i] = 0 # Reset return for next episode, first reward will be 0
                self.episode_lens[i] = 0 # Reset length for next episode
                self.is_success[i] = False # Reset success for next episode

    def reset(self):
        """Reset completed episodes."""
        dct = {}
        if len(self.completed_episodes['ep_returns']) > 0:
            dct['ep_returns'] = np.mean(self.completed_episodes['ep_returns'])
            dct['ep_returns_min'] = np.min(self.completed_episodes['ep_returns'])
            dct['ep_returns_max'] = np.max(self.completed_episodes['ep_returns'])
            dct['ep_lens'] = np.mean(self.completed_episodes['ep_lens'])
            if len(self.completed_episodes['is_success']) > 0:
                dct['success_rate'] = np.mean(self.completed_episodes['is_success'])
        self.completed_episodes = {'ep_returns': [], 'ep_lens': [], 'is_success': []}
        return dct