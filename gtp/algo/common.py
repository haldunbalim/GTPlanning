from typing import Callable, Optional, Tuple
import jax
import flax.linen as nn
import optax
from flax.struct import dataclass
import jax.numpy as jnp
from gtp.algo.networks import MLP

@dataclass
class TrainState:
    variables: jax.tree
    ema_variables: jax.tree
    opt_state: optax.OptState


class Algorithm:
    def init_train_state(self, cfg) -> TrainState:
        raise NotImplementedError('base class')
    
    def get_action(self, state, obs, key, deterministic=False) -> jax.Array:
        raise NotImplementedError('base class')

    def get_rollout_action(self, state, obs, key, step=0, **policy_aux) -> jax.Array:
        return self.get_action(state, obs, key, deterministic=False, **policy_aux)

    def get_eval_action(self, state, obs, key, step=0, **policy_aux) -> jax.Array:
        return self.get_action(state, obs, key, deterministic=True, **policy_aux)
    
    def update(self, state, experience, key, step) -> Tuple[TrainState, dict]:
        raise NotImplementedError('base class')
    
    def update_curriculum(self, step) -> Optional[dict]:
        return None
    
class Critic(nn.Module):
    """Q-function network."""
    hidden_dims: Tuple[int]
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = MLP(self.hidden_dims + (1,),
                hidden_act_class=self.activation, use_layer_norm=self.use_layer_norm, squeeze_output=True, dtype=self.dtype)(x)
        return x

class DoubleCritic(nn.Module):
    """Twin Q-networks."""
    hidden_dims: list
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action):
        q1 = Critic(self.hidden_dims, activation=self.activation,
                    use_layer_norm=self.use_layer_norm, dtype=self.dtype)(obs, action)
        q2 = Critic(self.hidden_dims, activation=self.activation,
                    use_layer_norm=self.use_layer_norm, dtype=self.dtype)(obs, action)
        return q1, q2

    def compute_target(self, obs, action):
        return jnp.minimum(*self(obs, action))

class MultiCritic(nn.Module):
    """Multi Q-networks."""
    hidden_dims: list
    num_critics: int = 2
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, action):
        # [B, num_critics, obs_dim]
        obs_tiled = jnp.tile(obs[..., None], (1,) *
                             (obs.ndim - 1) + (self.num_critics,))
        action_tiled = jnp.tile(action[..., None], (1,) *
                                 (action.ndim - 1) + (self.num_critics,))
        vmapped_mlp = nn.vmap(
            Critic,  # Pass the class, not an instance
            in_axes=(-1, -1),  # Map over last dimension of input
            out_axes=-1,  # Stack outputs along last dimension
            variable_axes={'params': 0},  # Create separate params for each
            split_rngs={'params': True}  # Split RNG for initialization
        )(hidden_dims=self.hidden_dims, activation=self.activation, use_layer_norm=self.use_layer_norm, dtype=self.dtype)
        return vmapped_mlp(obs_tiled, action_tiled)

    def compute_target(self, obs, action):
        return self(obs, action).min(axis=-1)
    

class ValueCritic(nn.Module):
    """Value function network."""
    hidden_dims: Tuple[int]
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden_dims + (1,),
                hidden_act_class=self.activation, use_layer_norm=self.use_layer_norm, squeeze_output=True, dtype=self.dtype)(obs)
        return x
    
    def compute_target(self, obs):
        return self(obs)

class MultiValueCritic(nn.Module):
    """Multi V-networks."""
    hidden_dims: list
    num_critics: int = 1
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs):
        obs_tiled = jnp.tile(obs[..., None], (1,) * (obs.ndim - 1) + (self.num_critics,))  # [B, num_critics, obs_dim]
        vmapped_mlp = nn.vmap(
            ValueCritic,  # Pass the class, not an instance
            in_axes=(-1),  # Map over last dimension of input
            out_axes=-1,  # Stack outputs along last dimension
            variable_axes={'params': 0},  # Create separate params for each
            split_rngs={'params': True}  # Split RNG for initialization
        )(hidden_dims=self.hidden_dims, activation=self.activation, use_layer_norm=self.use_layer_norm, dtype=self.dtype)
        return vmapped_mlp(obs_tiled)
    
    def compute_target(self, obs):
        return self(obs).min(axis=-1)
    

"""
Dual-dtype wrapper for Flax neural networks.

This wrapper creates two instances of a network with different dtypes
(typically one for inference and one for training) and automatically
routes method calls to the appropriate instance.
"""


class DualDtypeWrapper:
    """
    Wrapper that creates two instances of a network with different dtypes.
    
    By default, all method calls are routed to the inference instance.
    When use_train=True is passed to any method, it routes to the training instance.
    
    Args:
        network_class: The Flax module class to instantiate
        infer_dtype: dtype for the inference network (default: 'bfloat16')
        train_dtype: dtype for the training network (default: 'float32')
        **kwargs: Arguments to pass to both network instances (except dtype)
    
    Example:
        >>> policy = DualDtypeWrapper(
        ...     MPCPolicy,
        ...     infer_dtype='bfloat16',
        ...     train_dtype='float32',
        ...     nx=obs_dim,
        ...     nu=act_dim,
        ...     horizon=16
        ... )
        >>> # Uses inference (bfloat16) instance
        >>> output = policy.apply(params, x, t, obs)
        >>> # Uses training (float32) instance
        >>> output = policy.apply(params, x, t, obs, use_train=True)
    """

    def __init__(
        self,
        network_class: type,
        infer_dtype: jnp.dtype = 'bfloat16',
        train_dtype: jnp.dtype = 'float32',
        **kwargs
    ):
        self.network_class = network_class
        self.infer_dtype = infer_dtype
        self.train_dtype = train_dtype

        # Create inference instance
        self._infer = network_class(dtype=infer_dtype, **kwargs)

        # Create training instance
        self._train = network_class(dtype=train_dtype, **kwargs)

        # Store kwargs for reference
        self._kwargs = kwargs

    def _route_call(self, method_name: str, use_train: bool = False, *args, **kwargs):
        """
        Route method call to appropriate network instance.
        
        Args:
            method_name: Name of the method to call
            use_train: If True, use training instance; otherwise use inference instance
            *args, **kwargs: Arguments to pass to the method
        """
        # Remove use_train from kwargs to avoid passing it to the underlying method
        kwargs.pop('use_train', None)

        # Select the appropriate instance
        if use_train:
            instance = self._train
        else:
            instance = self._infer

        # Special handling for 'apply' method with 'method' argument
        if method_name == 'apply' and 'method' in kwargs:
            method_arg = kwargs['method']
            # Check if method_arg is a bound method from either _infer or _train
            # If it's from the wrong instance, get the equivalent from the correct instance
            if hasattr(method_arg, '__self__'):
                # It's a bound method - get the function name
                method_func_name = method_arg.__name__
                # Replace with the method from the correct instance
                kwargs['method'] = getattr(instance, method_func_name)
            # Note: if method_arg is an unbound function, it will work as-is

        # Get and call the method
        method = getattr(instance, method_name)
        return method(*args, **kwargs)

    def __getattr__(self, name: str) -> Callable:
        """
        Intercept attribute access and return a wrapper function that routes calls.
        
        This allows syntax like: wrapper.apply(params, x, use_train=True)
        """
        # Avoid infinite recursion for private attributes
        if name.startswith('_'):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Check if the attribute exists on the inference instance
        if not hasattr(self._infer, name):
            raise AttributeError(
                f"'{self.network_class.__name__}' object has no attribute '{name}'"
            )

        # If it's not callable, just return the attribute from inference instance
        attr = getattr(self._infer, name)
        if not callable(attr):
            return attr

        # Return a wrapper function that handles use_train routing
        def wrapper(*args, use_train: bool = False, **kwargs):
            return self._route_call(name, use_train, *args, **kwargs)

        return wrapper

    @property
    def infer(self):
        """Access the inference dtype network directly."""
        return self._infer

    @property
    def train(self):
        """Access the training dtype network directly."""
        return self._train

    def __repr__(self) -> str:
        return (
            f"DualDtypeWrapper(\n"
            f"  network_class={self.network_class.__name__},\n"
            f"  infer_dtype={self.infer_dtype},\n"
            f"  train_dtype={self.train_dtype},\n"
            f"  kwargs={self._kwargs}\n"
            f")"
        )
    
class Standardizer(nn.Module):
    dim: int

    def setup(self):
        self.mean = self.param('mean', nn.initializers.zeros, (self.dim,))
        self.std = self.param('std', nn.initializers.ones, (self.dim,))
        self.count = self.param('count', lambda *_: jnp.array(0.0))

    def __call__(self, x):
        return self.normalize(x)
    
    def normalize(self, x):
        mean = self.mean.astype(x.dtype)
        std = self.std.astype(x.dtype)
        return (x - mean) / (std + 1e-8)
    
    def denormalize(self, x_norm):
        mean = self.mean.astype(x_norm.dtype)
        std = self.std.astype(x_norm.dtype)
        return x_norm * (std + 1e-8) + mean
    
    def update_stats(self, x, eps: float = 1e-8):
        """
        Update running mean/var with counts using a numerically stable parallel update.
        norm_stats: {params: {'mean': [D], 'std': [D], 'count': scalar}}
        x:  [..., D] or [B, T, D] etc.
        """
        x = x.astype(jnp.float32)

        # Flatten everything except last dim (obs_dim)
        obs_flat = x.reshape((-1, self.dim))
        batch_count = obs_flat.shape[0]

        mean = self.mean
        # Store variance internally; reconstruct from std
        var = jnp.square(self.std)
        count = self.count
        batch_mean = jnp.mean(obs_flat, axis=0)

        # population variance (ddof=0) is what you want for stable running stats
        batch_var = jnp.var(obs_flat, axis=0)
        total_count = jnp.maximum(count + batch_count, 1.0)
        delta = batch_mean - mean
        new_mean = mean + delta * (batch_count / total_count)

        # Parallel variance merge (M2 merge)
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * (count * batch_count / total_count)

        new_var = m2 / total_count
        new_std = jnp.sqrt(jnp.maximum(new_var, eps))

        return {'params': {'mean': new_mean, 'std': new_std, 'count': total_count}}

class MinMaxScaler(nn.Module):
    dim: int
    abs_max_init: float = 100

    def setup(self):
        self.min = self.param('min', lambda *_: self.abs_max_init * jnp.ones((self.dim,)))
        self.max = self.param('max', lambda *_: -self.abs_max_init * jnp.ones((self.dim,)))

    def __call__(self, x):
        return self.normalize(x)
    
    def normalize(self, x):
        min_val = self.min.astype(x.dtype)
        max_val = self.max.astype(x.dtype)
        return (x - min_val) / (max_val - min_val + 1e-8) * 2.0 - 1.0
    
    def denormalize(self, x_norm):
        min_val = self.min.astype(x_norm.dtype)
        max_val = self.max.astype(x_norm.dtype)
        return (x_norm + 1.0) / 2.0 * (max_val - min_val) + min_val
    
    def update_stats(self, x):
        obs_flat = x.reshape((-1, x.shape[-1]))
        batch_min = jnp.min(obs_flat, axis=0)
        batch_max = jnp.max(obs_flat, axis=0)

        new_min = jnp.minimum(self.min, batch_min)
        new_max = jnp.maximum(self.max, batch_max)

        return {'params': {'min': new_min, 'max': new_max}}
    

def clamped_huber_loss(true, pred, delta=5.0, cap_mult=2.0):
    if cap_mult is not None:
        err = true - pred
        true_clamped = jax.lax.stop_gradient(jnp.clip(err, -cap_mult*delta, cap_mult*delta) + pred)
    else:
        true_clamped = true

    err = pred - true_clamped
    abs_err = jnp.abs(err)
    huber = jnp.where(
        abs_err <= delta,
        0.5 * err**2,
        delta * (abs_err - 0.5 * delta)
    )
    return huber.mean()
