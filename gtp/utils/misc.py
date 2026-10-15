from flax import struct
from typing import Any
import dataclasses
from dataclasses import make_dataclass, field
import jax
import json
import jax.numpy as jnp
import re
import threading
import hashlib
from functools import reduce
import numpy as np
import wandb
import os.path as op
import tempfile
import random
from functools import partial
import contextlib
import io
import logging
import math
import optax

logger = logging.getLogger(__name__)


def compose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), funcs, x)


def get_seed():
    return np.random.randint(0, 2**32 - 1)


def repr_tree(d, prefix=''):
    """Recursively prints a dictionary as a tree structure."""
    st = ''
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, print the key and recurse
            st += f"{prefix}├── {key}\n"
            st += repr_tree(value, prefix + "│   ")
        else:
            # If the value is not a dictionary, print the key-value pair
            st += f"{prefix}├── {key}: size = {value}\n"
    return st


def wandb_log_model_arch(params):
    shapes = jax.tree.map(lambda x: x.shape, params)
    str_model = repr_tree(shapes)
    artifact = wandb.Artifact(name="model_arch", type="arch_str")
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = op.join(tmp_dir, 'arch.txt')
        with open(filename, 'w') as f:
            f.write(str_model)
        artifact.add_file(filename)
    wandb.log_artifact(artifact)


def flatten_dict(data, parent_key='', sep='-'):
    """
    Flattens a nested dictionary into a single-level dictionary.

    Parameters:
    - data (dict): The nested dictionary structure.
    - parent_key (str): The base key used for recursion (default: '').
    - sep (str): The separator for concatenated keys (default: '-').

    Returns:
    - dict: A flattened dictionary with concatenated keys.
    """
    flat_dict = {}
    for key, value in data.items():
        # Create new key by appending current key to the parent key
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            flat_dict.update(flatten_dict(value, new_key, sep=sep))
        else:
            # Add to flat dictionary
            flat_dict[new_key] = value
    return flat_dict


def accumulate_metrics(metrics):
    metrics = {k: np.array([metric[k] for metric in metrics if not np.isnan(metric[k])]) for k in metrics[0]}
    return {k: (metrics[k].max() if 'max' in k else metrics[k].mean()) for k in metrics}


def hash_config(cfg):
    config_str = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def log_info(st):
    if type(st) != str:
        st = str(st)
    return logger.info(st)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class HiddenPrints(contextlib.redirect_stdout):
    def __init__(self):
        super().__init__(io.StringIO())


def create_iter_dataset(dataset, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    while True:
        batch_indices = rng.permutation(len(dataset))  # shuffle
        batch_indices = batch_indices[:len(
            dataset) - (len(dataset) % batch_size)]  # drop last
        batch_indices = batch_indices.reshape(-1, batch_size)  # batch
        for batch_idx in batch_indices:
            yield dataset[batch_idx]


def get_optimizer(lr0=2e-4, weight_decay=0.0, gradient_clip_val=1.0):
    optimizer = optax.adamw(lr0, weight_decay=weight_decay)
    if gradient_clip_val > 0.0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clip_val), optimizer)
    return optimizer


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def update_params(loss_fn, params, opt_state, optimizer, has_aux=True):
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)
    if has_aux:
        (_, aux), grads = gradient_fn(params)
    else:
        _, grads = gradient_fn(params)

    #grads = jax.lax.pmean(grads, axis_name='devices')
    param_updates, opt_state = optimizer.update(
        grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    if has_aux:
        return params, opt_state, aux
    else:
        return params, opt_state


def delay_target_update(step, vars, target_vars, tau, update_every=1):
    return jax.lax.cond(
        step % update_every == 0,
        lambda target_params: optax.incremental_update(
            vars, target_params, tau),
        lambda target_params: target_params,
        target_vars
    )


def merge_dicts(d1, d2, mask):
    return {
        k: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (d1[k].ndim - 1)),
            d1[k],
            d2[k],
        )
        for k in d1
    }

def cosine_between(y_min, y_max, n):
    x = np.linspace(np.pi, 2*np.pi, n, endpoint=True)
    mid = 0.5 * (y_max + y_min)
    amp = 0.5 * (y_max - y_min)
    y = mid + amp * np.cos(x)
    return y


def normal_log_prob(value, loc, scale):
    normalize_term = jnp.log(jnp.sqrt(2 * jnp.pi) * scale)
    value_scaled = (value - loc) / scale
    return -0.5 * value_scaled**2 - normalize_term


def schedule(schdl, step):
    """
    Outputs values following a scalar schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    lib = jnp if hasattr(step, "dtype") else np
    try:
        return float(schdl)
    except (ValueError, TypeError):
        match = re.match(r'linear\((.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration, skip = [float(g) for g in match.groups()]
            assert (final - init) % skip == 0, "final must be reachable from init by discrete steps of size skip"
            # Behaves like range(init, final, skip) over `duration` steps
            # Number of discrete steps possible in range
            num_steps = int(abs(final - init) / skip)
            if duration == 0 or num_steps == 0:
                return final
            # Which discrete step are we on?
            mix = lib.clip(step / duration, 0.0, 1.0)
            current_step = lib.round(mix * num_steps)
            return init + current_step * skip * (1 if final > init else -1)

        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            if duration == 0:
                return final
            mix = lib.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final

        match = re.match(r'cosine\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            if duration == 0:
                return final
            frac = step / duration
            return cosine_schedule(init, final, frac)

        raise NotImplementedError(schdl)


"""Generate flax struct.dataclass from dict - copy/paste ready."""
def make_dataclass_from_dict(name: str, data: dict):
    """
    Dynamically create a flax dataclass from a dict.
    
    Usage:
        StateClass = make_dataclass_from_dict('State', ckptr.restore(step))
        state = StateClass(**ckptr.restore(step))
    """
    # Build field annotations
    annotations = {key: Any for key in data.keys()}

    # Create class with struct.dataclass decorator
    cls = type(name, (), {
        '__annotations__': annotations,
        **{key: struct.field(default=None) for key in data.keys()}
    })

    return struct.dataclass(cls)


def flax_namespace(data: dict):
    """
    Convert dict to flax dataclass instance with .replace() method.
    
    Usage:
        state = flax_namespace(ckptr.restore(step))
        print(state.params)
        new_state = state.replace(step=state.step + 1)
    """
    # Create the dataclass type
    StateClass = make_dataclass_from_dict('State', data)

    # Return instance
    return StateClass(**data)


def cosine_schedule(init: float,
                    final: float,
                    frac: float) -> float:

    # Clamp step to valid range
    lib = jnp if isinstance(frac, jnp.ndarray) else np
    frac = lib.clip(frac, 0, 1)

    cosine = 0.5 * (1 + lib.cos(math.pi * frac))
    return final + (init - final) * cosine


def generate_colored_noise(key, H, D, shp=(), sigma=0.1, rho=0.9):
    """
    AR(1) colored noise:
      eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * eta_t
    where eta_t ~ N(0, sigma^2 I)

    Returns: eps with shape (*shp, H, D)
    """
    rho_other = jnp.sqrt(1.0 - rho**2)
    step_shape = (*shp, D)

    def step(carry, _):
        prev, key = carry
        key, sub = jax.random.split(key)
        eps = rho * prev + rho_other * sigma * \
            jax.random.normal(sub, step_shape)
        return (eps, key), eps  # carry, output at this timestep

    key, sub = jax.random.split(key)
    init_noise = sigma * jax.random.normal(sub, step_shape)
    init = (init_noise, key)
    _, eps = jax.lax.scan(step, init, xs=None, length=H-1)
    # add initial noise at the beginning
    eps = jnp.concatenate([init_noise[None], eps], axis=0)
    # rearrange to (*shp, H, D)
    eps = jnp.transpose(eps, (*range(1, eps.ndim-1), 0, eps.ndim-1))
    return eps


class JAXCompileTimeoutError(RuntimeError):
    pass


def run_with_timeout(fn, timeout_s, name):
    result = {}

    def target():
        try:
            result["value"] = fn()
        except Exception as e:
            result["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        raise JAXCompileTimeoutError(
            f"[{name}] exceeded timeout of {timeout_s}s"
        )

    if "error" in result:
        raise result["error"]

    return result["value"]
