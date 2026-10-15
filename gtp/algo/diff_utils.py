import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import chex
from flax import linen as nn
from dataclasses import dataclass
from typing import Optional, Callable   


@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(
            np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * \
            np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * \
            np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 50.
        b_min = 1e-6
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min)
                       * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas

    @staticmethod
    def linear_beta_schedule(timesteps: int):
        start = 0.9999
        end = 0.01
        alpha = np.linspace(start, end, timesteps)
        betas = 1 - alpha
        betas = np.clip(betas, 0, 0.999)
        return betas
    
    @property
    def snr(self):
        return self.alphas_cumprod / (1 - self.alphas_cumprod)
    
    @property
    def n_steps(self):
        return self.betas.shape[0]


@dataclass(frozen=True)
class GaussianDiffusion:
    n_steps: int
    predict_noise: bool

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            betas = BetaScheduleCoefficients.cosine_beta_schedule(
                self.n_steps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, x: jax.Array, t: int, noise_pred: jax.Array, clip_val: float = 1.0):
        B = self.beta_schedule()
        if self.predict_noise:
            x_recon = x * B.sqrt_recip_alphas_cumprod[t].astype(x.dtype) - \
                noise_pred * \
                B.sqrt_recipm1_alphas_cumprod[t].astype(x.dtype)
        else:
            x_recon = noise_pred

        x_recon = jnp.clip(x_recon, -clip_val, clip_val)

        model_mean = B.posterior_mean_coef1[t].astype(x.dtype) * \
            x_recon + B.posterior_mean_coef2[t].astype(x.dtype) * x
        model_log_variance = B.posterior_log_variance_clipped[t].astype(
            x.dtype)
        return model_mean, model_log_variance

    def q_sample(self, x_start: jax.Array, t: int, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t].astype(x_start.dtype) * x_start + B.sqrt_one_minus_alphas_cumprod[t].astype(x_start.dtype) * noise

    def p_sample_loop(self, x, sample_fn, key: jax.Array, temperature: float = 1.0, replan_n_steps: int = -1,
                      clip_val: float = 1.0, return_path: bool = False, use_p_mean_variance: bool = True,
                      condition_fn: Optional[Callable] = None):
        def body_fn(carry, t):
            rng_key, x = carry
            info = {}
            rng_key, r = jax.random.split(rng_key)
            t_vec = jnp.ones((x.shape[0],), dtype=jnp.int32) * t

            sampled = sample_fn(x, t_vec)
            noise_pred, _info = sampled if isinstance(
                sampled, tuple) else (sampled, {})
            info.update(_info)

            if use_p_mean_variance:
                x, log_var = jax.vmap(
                    partial(self.p_mean_variance, clip_val=clip_val))(x, t_vec, noise_pred)
                log_var = log_var[0]  # all samples have the same log_var
            else:
                x = noise_pred
                log_var = self.beta_schedule().posterior_log_variance_clipped[t].astype(x.dtype)

            std = jnp.exp(0.5 * log_var)
            x += (t > 0) * temperature * std * \
                jax.random.normal(r, x.shape, dtype=x.dtype)  # add noise

            if condition_fn is not None:
                x = condition_fn(x)

            return (rng_key, x), (x, info)

        end = self.n_steps if replan_n_steps < 0 else replan_n_steps
        if condition_fn is not None:
            x = condition_fn(x)
        _, (x_sampled, infos) = jax.lax.scan(body_fn, (key, x), xs=jnp.arange(0, end)[::-1])
        if len(infos) == 0:
            return x_sampled if return_path else x_sampled[-1]
        else:
            if not return_path:
                x_sampled = x_sampled[-1]
                infos = {k: v[-1] for k, v in infos.items()}
            return x_sampled, infos


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape=None) -> jax.Array:
    assert dim % 2 == 0

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb


class GuidedSampler:
    def __init__(self, model_fn, diffusion, scale: float, n_guide_steps: float, t_stopgrad: int = 0, guide=None, condition_fn=None):
        self.model_fn = model_fn
        self.diffusion = diffusion
        self.scale = scale
        self.t_stopgrad = t_stopgrad
        self.n_guide_steps = n_guide_steps
        self.guide = guide
        self.condition_fn = condition_fn

    def sample_step(self, x, t):
        log_var = self.diffusion.beta_schedule(
        ).posterior_log_variance_clipped[t]
        var = jnp.exp(log_var)[:, None, None]

        if self.guide is None:
            value = jnp.zeros((x.shape[0], ), dtype=jnp.float32)
        else:
            for _ in range(self.n_guide_steps):
                g = jax.grad(lambda x: self.guide(x, t).sum())(x)
                g = jnp.where((t < self.t_stopgrad)[:, None, None], 0, g)

                x += self.scale * var * g
                if self.condition_fn is not None:
                    x = self.condition_fn(x)  # conditioning
            value = self.guide(x, t)

        # p_sample
        noise_pred = self.model_fn(x, t)
        return noise_pred, {'values': value}