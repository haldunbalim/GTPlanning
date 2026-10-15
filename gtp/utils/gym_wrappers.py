import gymnasium
import numpy as np
from gymnasium.vector import VectorWrapper, VectorEnv
from gymnasium.vector.utils import batch_space


class ActionChunkVectorWrapper(VectorWrapper):
    def __init__(self, env: VectorEnv, n: int, repeat=False):
        super().__init__(env)
        assert n >= 1
        self.repeat = repeat
        self.n = n

        orig = env.single_action_space
        assert isinstance(orig, gymnasium.spaces.Box)

        self._orig_action_dim = orig.shape[0]

        # Single env sees (n * action_dim,), batched across num_envs
        n_tile = n if repeat else 1
        self.single_action_space = gymnasium.spaces.Box(
            low=np.tile(orig.low, n_tile),
            high=np.tile(orig.high, n_tile),
            dtype=orig.dtype,
        )
        self.action_space = batch_space(self.single_action_space, env.num_envs)

    def step(self, actions: np.ndarray):
        # actions: (num_envs, n * action_dim)
        if self.repeat:
            actions = actions.repeat(self.n, axis=1)  # (num_envs, n * action_dim)
        chunks = actions.reshape(self.env.num_envs, self.n, self._orig_action_dim)

        total_rewards = np.zeros(self.env.num_envs, dtype=np.float32)
        terminated = np.zeros(self.env.num_envs, dtype=bool)
        truncated = np.zeros(self.env.num_envs, dtype=bool)
        obs, info = None, {}

        for i in range(self.n):
            # Only step envs that haven't ended yet
            active = ~(terminated | truncated)
            if not np.any(active):
                break

            obs, rewards, term, trunc, info = self.env.step(chunks[:, i, :])
            total_rewards[active] += rewards[active]
            terminated |= term
            truncated |= trunc

        return obs, total_rewards, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
