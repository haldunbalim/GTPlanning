from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from gtp.algo.common import DualDtypeWrapper
from gtp.utils import get_optimizer, get_seed, update_params
from .gtp import GTP, TrainState


def normalize_image(x, dtype: jnp.dtype):
    if jnp.issubdtype(x.dtype, jnp.integer):
        return x.astype(dtype) / 255.0 - 0.5
    return x.astype(dtype) - 0.5


class EncBlock(nn.Module):
    out_channels: int
    stride: int = 1
    dtype: jnp.dtype = jnp.float32
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        x = nn.GroupNorm(
            num_groups=min(self.n_groups, self.out_channels)
        )(x)
        x = nn.swish(x)
        return x


class ImageEncoder(nn.Module):
    latent_dim: int
    base_channels: int = 32
    dtype: jnp.dtype = jnp.float32
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = normalize_image(x, self.dtype)  # assumed to map to [-0.5, 0.5]

        # 96 -> 48
        x = EncBlock(
            out_channels=self.base_channels,
            stride=2,
            dtype=self.dtype,
            n_groups=self.n_groups,
        )(x)

        # 48 -> 24
        x = EncBlock(
            out_channels=self.base_channels,
            stride=2,
            dtype=self.dtype,
            n_groups=self.n_groups,
        )(x)

        # 24 -> 12
        x = EncBlock(
            out_channels=self.base_channels * 2,
            stride=2,
            dtype=self.dtype,
            n_groups=self.n_groups,
        )(x)

        # Keep resolution, add depth/capacity
        x = EncBlock(
            out_channels=self.base_channels * 2,
            stride=1,
            dtype=self.dtype,
            n_groups=self.n_groups,
        )(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.latent_dim, dtype=self.dtype)(x)
        x = nn.LayerNorm()(x)
        return x

class DecBlock(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
        )(x)
        x = nn.GroupNorm(
            num_groups=min(self.n_groups, self.out_channels)
        )(x)
        x = nn.swish(x)
        return x

class ImageDecoder(nn.Module):
    image_shape: Tuple[int, int, int]
    base_channels: int = 64
    seed_hw: Tuple[int, int] = (12, 12)
    dtype: jnp.dtype = jnp.float32
    n_groups: int = 8

    @nn.compact
    def __call__(self, z):
        height, width, out_channels = self.image_shape
        seed_h, seed_w = self.seed_hw

        z = z.astype(self.dtype)

        # Project latent to a small spatial feature map.
        x = nn.Dense(
            seed_h * seed_w * self.base_channels * 2,
            dtype=self.dtype,
        )(z)
        x = nn.swish(x)
        x = x.reshape((z.shape[0], seed_h, seed_w, self.base_channels * 2))

        x = DecBlock(
            out_channels=self.base_channels * 2,
            dtype=self.dtype,
            n_groups=self.n_groups,
        )(x)

        # Progressive upsampling.
        channel_schedule = [
            self.base_channels * 2,
            self.base_channels,
            self.base_channels,
        ]

        target_shapes = [
            (max(seed_h * 2, min(height, seed_h * 2)),
             max(seed_w * 2, min(width, seed_w * 2))),
            (max(seed_h * 4, min(height, seed_h * 4)),
             max(seed_w * 4, min(width, seed_w * 4))),
            (height, width),
        ]

        for (th, tw), ch in zip(target_shapes, channel_schedule):
            x = jax.image.resize(
                x,
                shape=(x.shape[0], th, tw, x.shape[-1]),
                method="bilinear",
            )
            x = DecBlock(
                out_channels=ch,
                dtype=self.dtype,
                n_groups=self.n_groups,
            )(x)

        x = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            dtype=self.dtype,
        )(x)

        # Images are scaled to [-0.5, 0.5]
        x = 0.5 * jnp.tanh(x)
        return x

class GTPVision(GTP):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        image_shape: Tuple[int, int, int],
        base_channels: int = 32,
        reconstruction_loss_weight: float = 1.0,
        consistency_loss_weight: Optional[float] = None,
        dynamics_loss_weight: Optional[float] = None,
        reward_loss_weight: float = 0.1,
        alive_loss_weight: float = 1.0,
        goal_dim: int = 0,
        horizon: int = 16,
        n_steps: int = 20,
        unet_dim: int = 32,
        x0_embed_dim: int = 32,
        unet_dim_mults: Tuple[int, ...] = (1, 2, 4),
        reward_hidden_dims: Tuple[int, ...] = (256, 256),
        dynamics_hidden_dims: Tuple[int, ...] = (256, 256),
        n_dynamics: int = 5,
        critics_hidden_dims: Tuple[int, ...] = (256, 256),
        num_critics: int = 2,
        topk_train: int = 4,
        num_samples: int = 32,
        beam_k: int = 8,
        num_beam: int = 2,
        lambda0: float = 1.0,
        update_noise_schedule: str = "0.2",
        update_noise_typ: str = "langevin",
        lr: float = 3e-4,
        tau: float = 0.005,
        gamma: float = 0.99,
        normalize_obs: bool = False,
        resample_temp: float = 1.0,
        resample_steps: int = 5,
        delay_update: int = 2,
        horizon_curriculum: Optional[str] = None,
        infer_dtype: jnp.dtype = 'float32',
        train_dtype: jnp.dtype = 'float32',
        use_ema_policy: bool = False,
        target_ess: Optional[float] = 0.8,
        up_agg: str = 'min',
        rol_agg: str = 'median',
        chunk_size: int = 1,
        langevin_steps: int = 4,
        n_elites: Optional[int] = None,
        elite_refresh_every: int = 50,
        elite_refresh_batch_size: int = 256,
    ):
        del normalize_obs
        super().__init__(
            obs_dim=obs_dim,
            act_dim=act_dim,
            goal_dim=goal_dim,
            horizon=horizon,
            n_steps=n_steps,
            unet_dim=unet_dim,
            x0_embed_dim=x0_embed_dim,
            unet_dim_mults=unet_dim_mults,
            reward_hidden_dims=reward_hidden_dims,
            dynamics_hidden_dims=dynamics_hidden_dims,
            n_dynamics=n_dynamics,
            critics_hidden_dims=critics_hidden_dims,
            num_critics=num_critics,
            topk_train=topk_train,
            num_samples=num_samples,
            beam_k=beam_k,
            num_beam=num_beam,
            lambda0=lambda0,
            update_noise_schedule=update_noise_schedule,
            update_noise_typ=update_noise_typ,
            lr=lr,
            tau=tau,
            gamma=gamma,
            normalize_obs=False,
            resample_temp=resample_temp,
            resample_steps=resample_steps,
            delay_update=delay_update,
            horizon_curriculum=horizon_curriculum,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            use_ema_policy=use_ema_policy,
            target_ess=target_ess,
            up_agg=up_agg,
            rol_agg=rol_agg,
            chunk_size=chunk_size,
            langevin_steps=langevin_steps,
            n_elites=n_elites,
            elite_refresh_every=elite_refresh_every,
            elite_refresh_batch_size=elite_refresh_batch_size,
        )
        self.image_shape = tuple(image_shape)
        # The bootstrapped update reads two frames per segment: frame 0 (planning start state,
        # policy and critic input) and frame 1 (the one-step model loss in _joint_model_loss), so
        # the trainer gathers, transfers and encodes only those instead of all horizon + 1.
        self.update_obs_frames = 2 if self.bootstrap else None
        self.base_channels = base_channels
        if consistency_loss_weight is None:
            consistency_loss_weight = 20.0 if dynamics_loss_weight is None else dynamics_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.alive_loss_weight = alive_loss_weight
        self.encoder = DualDtypeWrapper(
            ImageEncoder,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            latent_dim=obs_dim,
            base_channels=self.base_channels, 
        )
        self.decoder = DualDtypeWrapper(
            ImageDecoder,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            image_shape=self.image_shape,
            base_channels=self.base_channels,
        )
        self.model_opt = get_optimizer(lr)

    def encode_with_params(self, encoder_params, obs, use_train: bool = False):
        obs = jnp.asarray(obs)
        if tuple(obs.shape[-3:]) != self.image_shape:
            raise ValueError(f'Expected trailing image shape {self.image_shape}, got {obs.shape[-3:]}')
        flat_obs = obs.reshape((-1, *self.image_shape))
        latents = self.encoder.apply(encoder_params, flat_obs, use_train=use_train)
        return latents.reshape((*obs.shape[:-3], self.obs_dim))

    def encode(self, state, obs, use_train: bool = False):
        return self.encode_with_params(state.variables['encoder'], obs, use_train=use_train)

    def decode_with_params(self, decoder_params, latents, use_train: bool = False):
        latents = jnp.asarray(latents)
        flat_latents = latents.reshape((-1, latents.shape[-1]))
        recon = self.decoder.apply(decoder_params, flat_latents, use_train=use_train)
        return recon.reshape((*latents.shape[:-1], *self.image_shape))

    def decode(self, state, latents, use_train: bool = False):
        return self.decode_with_params(state.variables['decoder'], latents, use_train=use_train)

    def init_train_state(self, cfg) -> TrainState:
        state = super().init_train_state(cfg)
        key, dec_key = jax.random.split(jax.random.PRNGKey(get_seed()))
        dummy_obs = jnp.zeros((1, *self.image_shape), dtype=jnp.uint8)
        dummy_latent = jnp.zeros((1, self.obs_dim), dtype=self.train_dtype)
        encoder_vars = self.encoder.init(key, dummy_obs)
        decoder_vars = self.decoder.init(dec_key, dummy_latent)
        model_params = {
            'encoder': encoder_vars,
            'decoder': decoder_vars,
            'dynamics': state.variables['dynamics'],
            'reward': state.variables['reward'],
        }
        return state.replace(
            variables={**state.variables, 'encoder': encoder_vars, 'decoder': decoder_vars},
            opt_state={**state.opt_state, 'model': self.model_opt.init(model_params)},
        )

    def _reconstruction_loss(self, decoder_params, latents, obs):
        obs = jnp.asarray(obs)
        target_obs = normalize_image(obs, self.train_dtype)
        pred_obs = self.decode_with_params(decoder_params, latents, use_train=True)
        loss = jnp.mean(jnp.square(pred_obs - target_obs))
        return loss, {'reconstruction_loss': loss}

    def _joint_model_loss(self, model_params, transition, key, segment_len_mask):
        first_step_obs = transition.obs[:, :2]
        first_step_action = transition.action[:, :1]
        first_step_reward = transition.reward[:, :1]
        first_step_done = transition.done[:, :1]
        first_step_mask = segment_len_mask[:, :1]

        encoded_obs = self.encode_with_params(model_params['encoder'], first_step_obs, use_train=True)

        reconstruction_loss, reconstruction_info = self._reconstruction_loss(
            model_params['decoder'],
            encoded_obs[:, :1],
            first_step_obs[:, :1],
        )
        curr_latents = encoded_obs[:, :-1]
        next_latents = jax.lax.stop_gradient(encoded_obs[:, 1:])

        consistency_loss, dynamics_info = self._dynamics_loss(
            model_params['dynamics'],
            key,
            curr_latents,
            first_step_action,
            next_latents,
            first_step_mask,
        )
        _, reward_info = self._reward_loss(
            model_params['reward'],
            curr_latents,
            first_step_action,
            first_step_reward,
            1 - first_step_done.astype(jnp.float32),
            first_step_mask,
        )

        loss = (
            self.reconstruction_loss_weight * reconstruction_loss
            + self.consistency_loss_weight * consistency_loss
            + self.reward_loss_weight * reward_info['reward_reward_loss']
            + self.alive_loss_weight * reward_info['reward_alive_loss']
        )
        info = {
            **reconstruction_info,
            **dynamics_info,
            **reward_info,
            'model_loss': loss,
            'consistency_loss': consistency_loss,
            'dynamics_loss': consistency_loss,
            'weighted_reconstruction_loss': self.reconstruction_loss_weight * reconstruction_loss,
            'weighted_consistency_loss': self.consistency_loss_weight * consistency_loss,
            'weighted_dynamics_loss': self.consistency_loss_weight * consistency_loss,
            'weighted_reward_loss': self.reward_loss_weight * reward_info['reward_reward_loss'],
            'weighted_alive_loss': self.alive_loss_weight * reward_info['reward_alive_loss'],
        }
        return loss, info

    def update_model_modules(self, state: TrainState, transition, model_obs: jnp.ndarray,
                             encoded_obs: jnp.ndarray, segment_len_mask: jnp.ndarray,
                             key: jax.Array):
        del model_obs, encoded_obs
        model_params = {
            'encoder': state.variables['encoder'],
            'decoder': state.variables['decoder'],
            'dynamics': state.variables['dynamics'],
            'reward': state.variables['reward'],
        }
        model_loss_fn = partial(
            self._joint_model_loss,
            transition=transition,
            key=key,
            segment_len_mask=segment_len_mask,
        )
        model_params, model_opt_state, model_info = update_params(
            model_loss_fn,
            model_params,
            state.opt_state['model'],
            self.model_opt,
            has_aux=True,
        )
        variables = {
            'encoder': model_params['encoder'],
            'decoder': model_params['decoder'],
            'dynamics': model_params['dynamics'],
            'reward': model_params['reward'],
        }
        opt_state = {'model': model_opt_state}
        return variables, opt_state, model_info
