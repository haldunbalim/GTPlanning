from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from gtp.utils import get_optimizer, update_params, delay_target_update, schedule, get_seed
from gtp.algo.common import Algorithm, DualDtypeWrapper, Standardizer
from optax import sigmoid_binary_cross_entropy as bce

from .networks import TemporalUnetFiLM, MLP
from .diff_utils import GaussianDiffusion

from flax.struct import dataclass


@dataclass
class TrainState:
    variables: jax.tree
    ema_variables: jax.tree
    opt_state: optax.OptState
    obs_norm_stats: jax.tree


class MPCPolicy(nn.Module):
    nx: int
    nu: int
    horizon: int = 16
    unet_dim: int = 32
    x0_embed_dim: int = 32
    unet_dim_mults: Tuple[int] = (1, 2, 4)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.network = TemporalUnetFiLM(dim=self.unet_dim, cond_embed_dim=self.x0_embed_dim,
                                        dim_mults=self.unet_dim_mults, out_dim=self.nu, dtype=self.dtype)

    @nn.compact
    def __call__(self, us, t, x0):
        return self.network(us, t, x0)


class DynamicsModel(nn.Module):
    nx: int
    nu: int
    goal_dim: int = 0
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, u):
        nx_w_g = self.nx - self.goal_dim
        xu = jnp.concatenate([x[..., :nx_w_g], u], axis=-1)
        out = MLP(self.hidden_dims + (nx_w_g,), hidden_act_class=self.activation,
                  use_layer_norm=self.use_layer_norm, dtype=self.dtype)(xu)
        pred = x[..., :nx_w_g] + out
        # re-append goal to the state without change
        return jnp.concatenate([pred, x[..., nx_w_g:]], axis=-1)


class EnsembleDynamicsModel(nn.Module):
    nx: int
    nu: int
    n_dynamics: int
    goal_dim: int = 0
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, obs, inp, tile=False):
        if tile:
            # [n_dynamics, *, state_dim]
            obs = obs[None].repeat(self.n_dynamics, axis=0)
            # [n_dynamics, *, action_dim]
            inp = inp[None].repeat(self.n_dynamics, axis=0)

        vmapped_mlp = nn.vmap(
            DynamicsModel,  # Pass the class, not an instance
            in_axes=0,  # Map over first dimension of input
            out_axes=0,  # Stack outputs along first dimension
            variable_axes={'params': 0},  # Create separate params for each
            split_rngs={'params': True}  # Split RNG for initialization
        )(nx=self.nx, nu=self.nu, goal_dim=self.goal_dim, 
          hidden_dims=self.hidden_dims, activation=self.activation,
          use_layer_norm=self.use_layer_norm, dtype=self.dtype)
        return vmapped_mlp(obs, inp)

    def compute_rollout(self, obs0, inps, tile=True):
        def step_fn(carry, u):
            x = carry
            # [ensemble_size, batch_size, state_dim]
            next_xs = self(x, u, tile=False)
            return next_xs, next_xs
        inps = inps.transpose(inps.ndim-2, *range(inps.ndim-2), inps.ndim-1)
        if tile:
            # [n_dynamics, *, state_dim]
            obs0 = obs0[None].repeat(self.n_dynamics, axis=0)
            # [horizon, n_dynamics, *, action_dim]
            inps = inps[:, None].repeat(self.n_dynamics, axis=1)
        # xs: [horizon, ensemble_size, *, state_dim]
        _, xs = jax.lax.scan(step_fn, obs0, inps)
        # xs: [ensemble_size, *, horizon, state_dim]
        xs = xs.transpose(*range(1, xs.ndim-1), 0, xs.ndim-1)
        return xs

class RewardModel(nn.Module):
    nx: int
    nu: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: Callable = jax.nn.swish
    use_layer_norm: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, u, return_alive_logit=False):
        xu = jnp.concatenate([x, u], axis=-1)
        out = MLP(self.hidden_dims + (2,), hidden_act_class=self.activation,
                  use_layer_norm=self.use_layer_norm, dtype=self.dtype)(xu)
        r, a =  out[..., 0], out[..., 1]  # reward, alive_logit
        if not return_alive_logit:
            a = nn.sigmoid(a)
        return r, a


class BaseGTP(Algorithm):
    def __init__(self, obs_dim: int, act_dim: int, goal_dim: int = 0, horizon: int = 16, n_steps: int = 20,
                 unet_dim: int = 32, x0_embed_dim: int = 32,
                 unet_dim_mults: Tuple[int, ...] = (1, 2, 4),
                 reward_hidden_dims: Tuple[int, ...] = (256, 256),
                 dynamics_hidden_dims: Tuple[int, ...] = (256, 256), n_dynamics: int = 5,
                 topk_train: int = 4, num_samples: int = 32, beam_k: int = 8, num_beam: int = 2, lambda0: float = 1.0,
                 update_noise_schedule: str = "0.2", lr: float = 3e-4, tau: float = 0.005, gamma: float = 0.99, 
                 normalize_obs: bool = True, resample_temp: float = 1.0, resample_steps: int = 5,
                 delay_update: int = 2, horizon_curriculum: Optional[str] = None,
                 infer_dtype: jnp.dtype = 'bfloat16', train_dtype: jnp.dtype = 'float32',
                 use_ema_policy: bool = True, target_ess: Optional[float] = 0.8,
                 up_agg: str = 'mean', rol_agg: str = 'mean', chunk_size: int = 1):

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.goal_dim = goal_dim
        self.horizon = horizon
        self.unet_dim = unet_dim
        self.x0_embed_dim = x0_embed_dim
        self.unet_dim_mults = unet_dim_mults
        self.n_steps = n_steps
        assert num_samples % beam_k == 0, "num_samples must be divisible by beam_k"
        assert beam_k * num_beam * 2 >= topk_train, "beam_k * num_beam must be greater than or equal to topk_train for proper training sample selection"
        self.beam_k = beam_k
        self.beam_d = num_samples // beam_k
        self.topk_train = topk_train
        self.num_samples = num_samples
        self.num_beam = num_beam
        self.lr = lr
        self.lambda0 = lambda0
        self.normalize_obs = normalize_obs
        self.obs_normalizer = Standardizer(obs_dim)
        self.resample_temp = resample_temp
        self.resample_steps = resample_steps
        self.use_ema_policy = use_ema_policy
        self.tau = tau
        self.gamma = gamma

        self.update_noise_schedule = partial(schedule, update_noise_schedule)
        assert train_dtype == "float32", "Only float32 is supported for train_dtype currently."


        self.train_dtype = train_dtype
        self.infer_dtype = infer_dtype
        self.delay_update = delay_update
        self.horizon_curriculum = horizon_curriculum
        self.current_curriculum_horizon = horizon
        self.target_ess = target_ess
        
        self.up_agg = up_agg
        self.rol_agg = rol_agg
        self.chunk_size = chunk_size

        self.policy = DualDtypeWrapper(
            MPCPolicy,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            nx=obs_dim,
            nu=act_dim,
            horizon=horizon,
            unet_dim=unet_dim,
            x0_embed_dim=x0_embed_dim,
            unet_dim_mults=unet_dim_mults,
        )

        self.n_dynamics = n_dynamics
        self.dynamics_model = DualDtypeWrapper(
            EnsembleDynamicsModel,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            nx=obs_dim,  # remove goal for dynamics model
            nu=act_dim,
            goal_dim=goal_dim,
            hidden_dims=dynamics_hidden_dims,
            use_layer_norm=False,
            n_dynamics=n_dynamics,
            activation=jax.nn.swish,
        )

        self.reward_model = DualDtypeWrapper(
            RewardModel,
            infer_dtype=infer_dtype,
            train_dtype=train_dtype,
            nx=obs_dim,
            nu=act_dim,
            hidden_dims=reward_hidden_dims,
            use_layer_norm=False,
            activation=jax.nn.swish,
        )

        self.policy_opt = get_optimizer(lr0=lr)
        self.dynamics_opt = get_optimizer(lr0=lr)
        self.reward_opt = get_optimizer(lr0=lr)
        self.lambda_opt = get_optimizer(lr0=lr)

    def encode(self, state, obs, use_train: bool = False):
        del state, use_train
        return jnp.asarray(obs)

    def normalize_observation(self, state, obs):
        if not self.normalize_obs or state.obs_norm_stats is None:
            return obs
        return self.obs_normalizer.apply(state.obs_norm_stats, obs, method='normalize')

    def maybe_update_obs_stats(self, state, obs):
        if not self.normalize_obs or state.obs_norm_stats is None:
            return state
        new_obs_norm_stats = self.obs_normalizer.apply(
            state.obs_norm_stats, obs, method='update_stats')
        return state.replace(obs_norm_stats=new_obs_norm_stats)

    def prepare_transition_obs(self, state, transition, use_train: bool = False):
        encoded_obs = self.encode(state, transition.obs, use_train=use_train)
        model_obs = self.normalize_observation(state, encoded_obs)
        return encoded_obs, model_obs

    def get_policy_obs(self, model_obs):
        return model_obs[:, 0]

    def predict_reward_and_alive(self, reward_params, obs, act, use_train: bool = False,
                                 return_alive_logit: bool = False):
        return self.reward_model.apply(
            reward_params,
            obs,
            act,
            use_train=use_train,
            return_alive_logit=return_alive_logit,
        )

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, predict_noise=True)

    def init_train_state(self, cfg) -> TrainState:
        poli_key, dyn_key, rew_key, onorm_key = jax.random.split(
            jax.random.PRNGKey(get_seed()), 4)
        
        # Initialize network
        batch_template = (jnp.zeros((1, self.horizon, self.act_dim)),
                          jnp.zeros((1,), dtype=jnp.int32),
                          jnp.zeros((1, self.obs_dim)))
        policy_vars = self.policy.init(poli_key, *batch_template)
        variables = {'policy': policy_vars}
        ema_variables = {'policy': policy_vars}

        # Initialize optimizer
        poli_opt_state = self.policy_opt.init(policy_vars)
        opt_state = {'policy': poli_opt_state}

        lambda_ = jnp.array(jnp.log(jnp.exp(self.lambda0)-1),
                            dtype=jnp.float32)  # inverse softplus
        lambda_opt_state = self.lambda_opt.init(lambda_)
        variables['lambda'] = lambda_
        opt_state['lambda'] = lambda_opt_state

        dummy_obs = jnp.zeros((1, self.obs_dim))
        dummy_u = jnp.zeros((1, self.act_dim))

        dynamics_vars = self.dynamics_model.init(
            dyn_key, dummy_obs, dummy_u, tile=True)
        dynamics_opt_state = self.dynamics_opt.init(dynamics_vars)
        variables['dynamics'] = dynamics_vars
        opt_state['dynamics'] = dynamics_opt_state

        reward_vars = self.reward_model.init(rew_key, dummy_obs, dummy_u)
        reward_opt_state = self.reward_opt.init(reward_vars)
        variables['reward'] = reward_vars
        opt_state['reward'] = reward_opt_state

        obs_norm_stats = None
        if self.normalize_obs:
            obs_norm_stats = self.obs_normalizer.init(onorm_key, dummy_obs)

        # initialize curriculum
        if self.horizon_curriculum is not None:
            self.current_curriculum_horizon = 0

        return TrainState(
            variables=variables,
            ema_variables=ema_variables,
            opt_state=opt_state,
            obs_norm_stats=obs_norm_stats
        )

    def _get_action(self, state, obs, key, num_beam=2, beam_k=8, beam_d=4, use_ema=True, resample_n_steps=5,
                    resample_temp=0.25, temperature=1.0, trunc_horizon: Optional[int] = None,
                    init_us=None, replan_n_steps=-1, aggregate_obj='mean'):
        raise NotImplementedError(
            "This method should be implemented in a subclass to define the specific sampling strategy.")

    def get_action(self, state, obs, key, aggregate_obj='max'):
        # Cast obs to specified dtype
        obs = self.encode(state, obs, use_train=False)
        obs = self.normalize_observation(state, obs).astype(self.infer_dtype)
        key, r = jax.random.split(key)
        sampled_us, _, _, _, _ = self._get_action(state, obs, r, num_beam=self.num_beam, beam_k=self.beam_k, beam_d=self.beam_d, use_ema=self.use_ema_policy,
                                                  resample_n_steps=self.resample_steps, resample_temp=self.resample_temp,
                                                  trunc_horizon=self.current_curriculum_horizon, aggregate_obj=aggregate_obj, chunk_size=self.chunk_size)
        action = sampled_us[:, 0, 0]  # first action in the best sequence
        return jnp.clip(action, -1.0, 1.0)

    def get_rollout_action(self, state, obs, key, step=0) -> jax.Array:
        return self.get_action(state, obs, key, aggregate_obj=self.rol_agg)

    def get_eval_action(self, state, obs, key, step=0) -> jax.Array:
        return self.get_action(state, obs, key, aggregate_obj=self.up_agg)

    def compute_weights(self, lambda_, objectives):
        lambda_ = jax.nn.softplus(lambda_)
        objectives = (objectives - objectives.mean(axis=-1, keepdims=True)) / \
            (objectives.std(axis=-1, keepdims=True) + 1e-6)  # normalize objectives??
        weights = jax.nn.softmax(objectives / lambda_, axis=-1)
        return weights

    def compute_ess(self, weights):
        return 1.0 / (jnp.sum(weights**2, axis=-1) * weights.shape[-1])

    def _lambda_loss(self, lambda_, objectives, target_ess):
        """
        logp: unconstrained scalar parameter (or shape [..]) being optimized
        objectives: [B, M] per-batch objectives (higher = better)
        """
        # obtain positive lambda from unconstrained param
        weights = self.compute_weights(lambda_, objectives)
        ess = self.compute_ess(weights).mean()

        loss = jnp.mean((ess - target_ess)**2)

        return loss, {'lambda_loss': loss, 'lambda': lambda_}

    def _policy_loss(self, policy_params, obs, sampled_us, weights, key, horizons=None):
        k1, k2 = jax.random.split(key, 2)

        # for curriculum learning, fix actions after current horizon
        # sampled_us = sampled_us.at[:, self.current_curriculum_horizon:].set(
        #    sampled_us[:, self.current_curriculum_horizon-1][:, None])

        t = jax.random.randint(
            k1, (sampled_us.shape[0] // self.topk_train,), 0, self.n_steps).repeat(self.topk_train)
        noise = jax.random.normal(k2, sampled_us.shape)
        sampled_us_noisy = jax.vmap(
            self.diffusion.q_sample)(sampled_us, t, noise)

        preds = self.policy.apply(
            policy_params, sampled_us_noisy, t, jnp.repeat(obs, self.topk_train, axis=0), use_train=True)

        # only supervise the part of each sequence that was actually optimized
        err = jnp.square(preds - noise)
        if horizons is None:
            # every sequence was planned at the current curriculum horizon
            err = err[..., :self.current_curriculum_horizon, :]
            policy_loss = jnp.mean(weights * err.mean(axis=range(1, preds.ndim)))
        else:
            # cached elites may have been planned at a shorter curriculum horizon each
            mask = horizon_mask(horizons, err.shape[-2], self.current_curriculum_horizon)
            err = (err * mask).sum(axis=range(1, err.ndim)) / \
                (mask.sum(axis=range(1, mask.ndim)) * err.shape[-1])
            policy_loss = jnp.mean(weights * err)
        info = {'policy_loss': policy_loss}
        return policy_loss, info

    def _dynamics_loss(self, dyn_params, key, obs, act, next_obs, segment_len_mask):
        # remove goal for dynamics learning
        batch_indices = jax.random.randint(
            key, (self.n_dynamics, obs.shape[0],), 0, obs.shape[0])
        obs, act, next_obs, segment_len_mask = obs[batch_indices], act[batch_indices], next_obs[batch_indices], segment_len_mask[batch_indices]
        obs_wg_dim = self.obs_dim - self.goal_dim
        # normalize by number of valid state dimensions
        n_samples = segment_len_mask.sum() * obs_wg_dim
        pred_next_obs = self.dynamics_model.apply(
            dyn_params, obs, act, use_train=True, tile=False)
        loss = jnp.sum(jnp.square(pred_next_obs[..., :obs_wg_dim] - next_obs[..., :obs_wg_dim]) * segment_len_mask[..., None]) / n_samples
        return loss, {'dynamics_loss': loss}

    def _reward_loss(self, reward_params, obs, act, rewards, alives, segment_len_mask):
        pred_rewards, pred_alive_logits = self.predict_reward_and_alive(
            reward_params, obs, act, use_train=True, return_alive_logit=True)
        n_samples = segment_len_mask.sum()
        reward_loss = jnp.sum(jnp.square(pred_rewards - rewards) * segment_len_mask) / n_samples
        alive_loss = jnp.sum(bce(pred_alive_logits, alives) * segment_len_mask) / n_samples
        loss = reward_loss + alive_loss
        alive_acc = jnp.sum(((pred_alive_logits > 0) == alives) * segment_len_mask) / n_samples
        return loss, {'reward_loss': loss, 'reward_reward_loss': reward_loss, 'reward_alive_loss': alive_loss, 'alive_acc': alive_acc}

    def update_model_modules(self, state: TrainState, transition, model_obs: jnp.ndarray,
                             encoded_obs: jnp.ndarray, segment_len_mask: jnp.ndarray,
                             key: jax.Array):
        del encoded_obs
        action = transition.action
        reward = transition.reward
        alives = 1.0 - transition.done.astype(jnp.float32)

        dynamics_loss_fn = partial(
            self._dynamics_loss,
            key=key,
            obs=model_obs[:, :-1],
            act=action,
            next_obs=model_obs[:, 1:],
            segment_len_mask=segment_len_mask,
        )
        dynamics_params, dynamics_opt_state, dynamics_info = update_params(
            dynamics_loss_fn,
            state.variables['dynamics'],
            state.opt_state['dynamics'],
            self.dynamics_opt,
            has_aux=True,
        )

        reward_loss_fn = partial(
            self._reward_loss,
            obs=model_obs[:, :-1],
            act=action,
            rewards=reward,
            alives=alives,
            segment_len_mask=segment_len_mask,
        )
        reward_params, reward_opt_state, reward_info = update_params(
            reward_loss_fn,
            state.variables['reward'],
            state.opt_state['reward'],
            self.reward_opt,
            has_aux=True,
        )

        variables = {
            'dynamics': dynamics_params,
            'reward': reward_params,
        }
        opt_state = {
            'dynamics': dynamics_opt_state,
            'reward': reward_opt_state,
        }
        info = {**dynamics_info, **reward_info}
        return variables, opt_state, info

    # for reuse in subclasses
    def update(self, state: TrainState, transition, key: jax.Array, step: int, policy_obs: jnp.ndarray,
               sampled_us: jnp.ndarray, objectives: jnp.ndarray, horizons: Optional[jnp.ndarray] = None,
               prepared_obs=None) -> Tuple[TrainState, dict]:
        if prepared_obs is None:
            encoded_obs, model_obs = self.prepare_transition_obs(state, transition, use_train=True)
        else:
            encoded_obs, model_obs = prepared_obs
        segment_lengths = transition.segment_length
        seg_len_mask = jnp.arange(self.horizon)[None, :] < segment_lengths[:, None]
        key, poli_key, dyn_key = jax.random.split(key, 3)

        variables, opt_state, ema_variables, info = {}, {}, {}, {}
        model_variables, model_opt_state, model_info = self.update_model_modules(
            state, transition, model_obs, encoded_obs, seg_len_mask, dyn_key)
        variables.update(model_variables)
        opt_state.update(model_opt_state)
        info.update(model_info)

        # select topk samples for policy update
        sampled_objectives = objectives[:, :self.topk_train]  # before noising
        indices = jnp.argsort(objectives, axis=1, descending=True)[
            :, :self.topk_train]
        obj_info = {'obj_avg_sampled': jnp.mean(sampled_objectives),
                    'obj_std_sampled': sampled_objectives.std(axis=1).mean(),
                    'obj_max_sampled': sampled_objectives.max(axis=1).mean(),
                    'obj_min_sampled': sampled_objectives.min(axis=1).mean()}
        info.update(obj_info)

        objectives = jnp.take_along_axis(objectives, indices, axis=1)
        sampled_us = jnp.take_along_axis(
            sampled_us, indices[..., None, None], axis=1)

        # select topk samples
        weights = self.compute_weights(state.variables['lambda'], objectives)
        wgt_sampled = jnp.where(indices < self.topk_train, weights, 0)
        topk_info = {'ess_topk': self.compute_ess(weights).mean(),
                     'obj_avg_topk': jnp.mean(objectives),
                     'obj_std_topk': objectives.std(axis=1).mean(),
                     'obj_max_topk': objectives.max(axis=1).mean(),
                     'obj_min_topk': objectives.min(axis=1).mean(),
                     'wgt_sampled': wgt_sampled.sum(axis=1).mean()}
        info.update(topk_info)

        weights = weights.flatten()
        sampled_us = sampled_us.reshape(-1, self.horizon, self.act_dim)
        # one horizon per (batch entry, topk sample), matching the flattening above
        horizons = None if horizons is None else jnp.repeat(horizons, self.topk_train)
        policy_loss_fn = partial(
            self._policy_loss, obs=policy_obs, sampled_us=sampled_us, weights=weights, key=poli_key,
            horizons=horizons)
        policy_params, policy_opt_state, policy_info = update_params(
            policy_loss_fn, state.variables['policy'], state.opt_state['policy'], self.policy_opt, has_aux=True)
        variables['policy'] = policy_params
        opt_state['policy'] = policy_opt_state
        info.update(policy_info)

        # update lambda
        if self.target_ess is not None:
            lambda_loss_fn = partial(
                self._lambda_loss, objectives=objectives, target_ess=self.target_ess)
            lambda_, lambda_opt_state, lambda_info = update_params(
                lambda_loss_fn, state.variables['lambda'], state.opt_state['lambda'], self.lambda_opt, has_aux=True)
            # clip lambda to reasonable range to avoid instability
            lambda_ = jnp.clip(lambda_, -5, 20)
            variables['lambda'] = lambda_
            opt_state['lambda'] = lambda_opt_state
            info.update(lambda_info)

        # update ema variables
        ema_policy_params = delay_target_update(
            step, state.variables['policy'], state.ema_variables['policy'], tau=self.tau, update_every=self.delay_update)
        ema_variables['policy'] = ema_policy_params

        state.variables.update(variables)
        state.opt_state.update(opt_state)
        state.ema_variables.update(ema_variables)

        return self.maybe_update_obs_stats(state, encoded_obs[:, 0]), info

    def update_curriculum(self, step) -> bool:
        if self.horizon_curriculum is None:
            return None
        prev_horizon = self.current_curriculum_horizon
        new_horizon = int(schedule(self.horizon_curriculum, step))
        new_horizon = min(new_horizon, self.horizon)
        if new_horizon != prev_horizon:
            self.current_curriculum_horizon = new_horizon
            return {'horizon': new_horizon}
        return None


def horizon_mask(horizons, length: int, current_horizon: int, dtype=jnp.float32):
    """Mask of shape [N, length, 1] that is 1 up to each entry's own planning horizon.

    `horizons=None` falls back to masking every entry at the current curriculum horizon.
    """
    if horizons is None:
        horizons = jnp.full((1,), current_horizon, dtype=jnp.int32)
    horizons = jnp.clip(horizons, 1, min(length, current_horizon))
    return (jnp.arange(length)[None] < horizons[:, None]).astype(dtype)[..., None]


def aggregate_fn(objs, aggregate_obj='mean'):
    if aggregate_obj == 'mean':
        return objs.mean(axis=0)
    elif aggregate_obj == 'max':
        return objs.max(axis=0)
    elif aggregate_obj == 'min':
        return objs.min(axis=0)
    elif aggregate_obj == 'median':
        return jnp.median(objs, axis=0)
    elif isinstance(aggregate_obj, int):
        return jnp.percentile(objs, aggregate_obj, axis=0)
    else:
        raise ValueError(f"Invalid aggregate_obj: {aggregate_obj}")
