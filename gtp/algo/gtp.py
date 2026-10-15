from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from gtp.utils import get_optimizer, update_params, delay_target_update, get_seed
from gtp.algo.common import MultiValueCritic, DualDtypeWrapper, clamped_huber_loss

from .base_gtp import BaseGTP, TrainState, aggregate_fn


class GTP(BaseGTP):
    def __init__(self, obs_dim: int, act_dim: int, goal_dim: int=0, horizon: int = 16, n_steps: int = 20,
                 unet_dim: int = 32, x0_embed_dim: int = 32,
                 unet_dim_mults: Tuple[int, ...] = (1, 2, 4),
                 reward_hidden_dims: Tuple[int, ...] = (256, 256),
                 dynamics_hidden_dims: Tuple[int, ...] = (256, 256), n_dynamics: int = 5,
                 critics_hidden_dims: Tuple[int, ...] = (256, 256), num_critics: int = 2,
                 topk_train: int = 4, num_samples: int = 32, beam_k: int = 8, num_beam: int = 2, lambda0: float = 1.0,
                 update_noise_schedule: str = "0.2", update_noise_typ: str = "langevin", 
                 lr: float = 3e-4, tau: float = 0.005, gamma: float = 0.99, 
                 normalize_obs: bool = True, resample_temp: float = 1.0, resample_steps: int = 5,
                 delay_update: int = 2, horizon_curriculum: Optional[str] = None,
                 infer_dtype: jnp.dtype = 'float32', train_dtype: jnp.dtype = 'float32',
                 use_ema_policy: bool = False, target_ess: Optional[float] = 0.8,
                 up_agg='min', rol_agg='median', chunk_size: int = 1,
                 n_elites: Optional[int] = None, langevin_steps: int = 4,
                 elite_refresh_every: int = 50, elite_refresh_batch_size: int = 256):

        super().__init__(obs_dim=obs_dim, act_dim=act_dim, goal_dim=goal_dim, horizon=horizon, n_steps=n_steps,
                         unet_dim=unet_dim, x0_embed_dim=x0_embed_dim,
                         unet_dim_mults=unet_dim_mults,
                         reward_hidden_dims=reward_hidden_dims,
                         dynamics_hidden_dims=dynamics_hidden_dims, n_dynamics=n_dynamics,
                         topk_train=topk_train, num_samples=num_samples, beam_k=beam_k, num_beam=num_beam, lambda0=lambda0,
                         update_noise_schedule=update_noise_schedule, lr=lr, tau=tau, gamma=gamma,
                         normalize_obs=normalize_obs, resample_temp=resample_temp, resample_steps=resample_steps,
                         delay_update=delay_update, horizon_curriculum=horizon_curriculum,
                         infer_dtype=infer_dtype, train_dtype=train_dtype,
                         use_ema_policy=use_ema_policy, target_ess=target_ess, up_agg=up_agg, rol_agg=rol_agg, chunk_size=chunk_size)

        self.critic = DualDtypeWrapper(MultiValueCritic, infer_dtype=infer_dtype, train_dtype=train_dtype,
                                       hidden_dims=critics_hidden_dims, activation=jax.nn.swish, num_critics=num_critics)
        self.critic_opt = get_optimizer(lr)
        self.use_ema_policy = use_ema_policy
        self.update_noise_typ = update_noise_typ

        # Elite bootstrapping (opt-in: set `n_elites`; None is the paper's GTP). The planner the
        # environment loop runs anyway -- diffusion beam search plus langevin refinement -- hands
        # back its best `n_elites` sequences, which the replay buffer caches with the transition
        # (action sequences only). The update then rescores those cached elites online with the
        # current models instead of planning a fresh pool for every batch row, and every
        # `elite_refresh_every` env steps the elites of `elite_refresh_batch_size` buffer rows are
        # replanned so they never get arbitrarily stale.
        self.n_elites = n_elites
        self.langevin_steps = langevin_steps
        self.elite_refresh_every = elite_refresh_every
        self.elite_refresh_batch_size = elite_refresh_batch_size
        if self.bootstrap:
            refine = self.refine_elites
            assert refine or update_noise_typ is None or update_noise_typ.lower() == 'none', \
                f"update_noise_typ={update_noise_typ} is not supported with elite bootstrapping " \
                "(use langevin, ga or none)"
            assert topk_train <= n_elites <= (2 if refine else 1) * beam_k, \
                "n_elites must be in [topk_train, beam_k] (2 * beam_k with langevin refinement)"

    @property
    def bootstrap(self) -> bool:
        """Whether the update trains on cached elites (see `__init__`)."""
        return self.n_elites is not None

    @property
    def refine_elites(self) -> bool:
        return self.update_noise_typ in ('langevin', 'ga')


    def _get_action(self, state, obs, key, num_beam=2, beam_k=8, beam_d=4, use_ema=True, resample_n_steps=5,
                    resample_temp=0.25, temperature=1.0, trunc_horizon: Optional[int] = None,
                    init_us=None, replan_n_steps=-1, aggregate_obj='mean', chunk_size: Optional[int] = 1):
        # Cast obs to specified dtype
        trunc_horizon = self.horizon if trunc_horizon is None else trunc_horizon
        key, r = jax.random.split(key, 2)
        gamma = self.gamma
        assert self.horizon % chunk_size == 0, "horizon must be divisible by chunk_size"

        def propagate_actions(us, obs, init_gamma, alives_old=None, rews_old=None):
            us_rep = jnp.repeat(us[None], self.n_dynamics, axis=0)
            rollout_obs = self.dynamics_model.apply(
                state.variables['dynamics'], obs, us_rep, tile=False, method='compute_rollout')
            rollout_obs = jnp.concatenate(
                [obs[..., None, :], rollout_obs], axis=-2)
            rewards, alives = self.predict_reward_and_alive(
                state.variables['reward'], rollout_obs[..., :-1, :], us_rep)
            values = self.critic.apply(
                state.variables['critic'], rollout_obs[..., -1, :], method='compute_target')
            # the networks may predict in bfloat16 (infer_dtype); the objective that ranks the
            # candidates is accumulated in float32 so close candidates still sort correctly
            rewards, alives, values = (x.astype(jnp.float32) for x in (rewards, alives, values))
            if alives_old is None:
                alives_old = jnp.ones_like(alives[..., -1])
            alives = jnp.concatenate([alives_old[..., None], alives], axis=-1)
            alives = jnp.cumprod(alives, axis=-1)
            rews_acc = jnp.cumsum(
                rewards * alives[..., :-1] * init_gamma * gamma ** jnp.arange(rewards.shape[-1]), axis=-1)
            if rews_old is not None:
                rews_acc = rews_acc + rews_old[..., None]
            objectives = rews_acc[..., -1] + alives[..., -1] * \
                init_gamma * (gamma ** rewards.shape[-1]) * values
            objectives = aggregate_fn(objectives, aggregate_obj=aggregate_obj)
            return objectives, rollout_obs[..., 1:, :], rews_acc, alives[..., 1:]

        def select_topk(us, objectives, nexts, rews_acc, alives, topk):
            topk_indices = jnp.argsort(objectives, descending=True)[:topk]
            topk_us = us[topk_indices]
            objectives = objectives[topk_indices]
            topk_indices_expanded = jnp.expand_dims(
                topk_indices, axis=(0, *range(2, nexts.ndim)))
            nexts = jnp.take_along_axis(nexts, topk_indices_expanded, axis=1)
            rews_acc = jnp.take_along_axis(
                rews_acc, topk_indices_expanded[..., 0], axis=1)
            alives = jnp.take_along_axis(
                alives, topk_indices_expanded[..., 0], axis=1)
            return topk_us, objectives, nexts, rews_acc, alives

        def sample(obs, key, init_us=None):
            obs = jnp.atleast_2d(obs)

            def sample_fn(x, t):
                policy_vars = state.ema_variables['policy'] if use_ema else state.variables['policy']
                return self.policy.apply(policy_vars, x, t, obs)

            def p_sample_fn(x, key, replan_n_steps=-1, temperature=1.0, mask_cond=None, us_cond=None):
                def _condition_fn(us):
                    return jnp.where(mask_cond[None, :, None], us_cond, us)

                condition_fn = _condition_fn if us_cond is not None else None
                sampled_us = self.diffusion.p_sample_loop(x, sample_fn, key, temperature=temperature,
                                                        replan_n_steps=replan_n_steps, condition_fn=condition_fn)
                return jnp.clip(sampled_us, -1.0, 1.0)

            def p_sample_fn_vmap(u0s, key, replan_n_steps=-1, temperature=1.0, mask_cond=None, us_cond=None):
                keys = jax.random.split(key, u0s.shape[0])
                return jax.vmap(partial(p_sample_fn, replan_n_steps=replan_n_steps,
                                        temperature=temperature, mask_cond=mask_cond))(u0s, keys, us_cond=us_cond)[:, 0]

            def loop_fn(carry, t, lim=None):
                key, topk_us, _, obss_hist, rews_acc_hist, alives_hist = carry

                # repeat
                topk_us = jnp.repeat(topk_us, beam_d, axis=0)[:, None]
                obss_hist = jnp.repeat(obss_hist, beam_d, axis=1)
                rews_acc_hist = jnp.repeat(rews_acc_hist, beam_d, axis=1)
                alives_hist = jnp.repeat(alives_hist, beam_d, axis=1)

                # forward noise
                key, k1, k2 = jax.random.split(key, 3)
                noise = jax.random.normal(
                    k1, topk_us.shape, dtype=topk_us.dtype) * resample_temp
                q_sampled = jax.vmap(self.diffusion.q_sample, (0, None, 0))(
                    topk_us, resample_n_steps, noise)

                # re-sample
                mask_cond = jnp.arange(
                    topk_us.shape[-2], dtype=topk_us.dtype) < (t*chunk_size)
                sampled_us = p_sample_fn_vmap(q_sampled, k2, replan_n_steps=resample_n_steps,
                                            temperature=resample_temp, mask_cond=mask_cond, us_cond=topk_us)
                # compute objective
                curr_chunk = sampled_us.reshape(
                    *sampled_us.shape[:-2], -1, chunk_size, self.act_dim)[..., t, :, :]
                if lim is not None:
                    curr_chunk = curr_chunk[..., :lim, :]
                chunk_start = t * chunk_size
                curr_obs = obss_hist[..., chunk_start - 1, :]
                curr_rews_acc = rews_acc_hist[..., chunk_start - 1]
                curr_alives = alives_hist[..., chunk_start - 1]
                objectives, nexts, rews_acc, alives = propagate_actions(
                    curr_chunk,
                    curr_obs,
                    init_gamma=gamma ** chunk_start,
                    alives_old=curr_alives,
                    rews_old=curr_rews_acc,
                )

                # update histories
                obss_hist = jax.lax.dynamic_update_slice(
                    obss_hist, nexts, (0, 0, chunk_start, 0))
                rews_acc_hist = jax.lax.dynamic_update_slice(
                    rews_acc_hist, rews_acc, (0, 0, chunk_start))
                alives_hist = jax.lax.dynamic_update_slice(
                    alives_hist, alives, (0, 0, chunk_start))
                
                # select topk
                topk_us, objectives, obss_hist, rews_acc_hist, alives_hist = select_topk(
                    sampled_us, objectives, obss_hist, rews_acc_hist, alives_hist, per_beam_k)

                return (key, topk_us, objectives, obss_hist, rews_acc_hist, alives_hist), None

            key, k1, k2 = jax.random.split(key, 3)
            # initial noise
            num_samples = beam_k * beam_d
            per_beam_k = beam_k // num_beam
            u0s = jax.random.normal(
                k1, (num_samples, 1, self.horizon, self.act_dim), dtype=obs.dtype) * temperature
            if init_us is not None:
                init_us = init_us.astype(obs.dtype)[
                    None, None, ...]  # add sample dim
                init_us = jnp.repeat(init_us, u0s.shape[0], axis=0)
                _t = (replan_n_steps -
                    1) if replan_n_steps >= 0 else (self.n_steps-1)
                _t = jnp.ones(u0s.shape[0], dtype=jnp.int32) * _t
                u0s = jax.vmap(self.diffusion.q_sample)(init_us, _t, u0s)

            # initial sampling
            sampled_us = p_sample_fn_vmap(
                u0s, k2, replan_n_steps=replan_n_steps)

            obs_rep = jnp.tile(obs, (self.n_dynamics, sampled_us.shape[0], 1))
            objectives, nexts, rews, alives = propagate_actions(
                sampled_us[..., :min(chunk_size, trunc_horizon), :], obs_rep, init_gamma=1.0)
            sampled_us, objectives, nexts, rews, alives = select_topk(
                sampled_us, objectives, nexts, rews, alives, beam_k)

            # if trunc_horizon is less than or equal to chunk_size, we can skip the beam search and return the results directly
            if trunc_horizon <= chunk_size:
                return sampled_us, objectives, nexts, rews, alives

            # reshape for beam search
            sampled_us = sampled_us.reshape(
                num_beam, per_beam_k, self.horizon, self.act_dim)
            rews = rews.reshape(self.n_dynamics, num_beam,
                                per_beam_k, chunk_size).swapaxes(0, 1)
            alives = alives.reshape(self.n_dynamics, num_beam,
                                    per_beam_k, chunk_size).swapaxes(0, 1)
            nexts = nexts.reshape(self.n_dynamics, num_beam,
                                per_beam_k, chunk_size, self.obs_dim).swapaxes(0, 1)

            def beam_search(key, us, obs, rews, alives0):
                n_full_chunks = trunc_horizon // chunk_size
                objectives = jnp.zeros((us.shape[0],), dtype=jnp.float32)
                obss = jnp.zeros(
                    (obs.shape[0], obs.shape[1], trunc_horizon, self.obs_dim), dtype=obs.dtype)
                rews_acc = jnp.zeros(
                    (rews.shape[0], rews.shape[1], trunc_horizon), dtype=rews.dtype)
                alives = jnp.zeros(
                    (alives0.shape[0], alives0.shape[1], trunc_horizon), dtype=alives0.dtype)
                obss = jax.lax.dynamic_update_slice(obss, obs, (0, 0, 0, 0))
                rews_acc = jax.lax.dynamic_update_slice(rews_acc, rews, (0, 0, 0))
                alives = jax.lax.dynamic_update_slice(alives, alives0, (0, 0, 0))
                (key, topk_us, objectives, obss, rews_acc, alives), _ = jax.lax.scan(
                    loop_fn,
                    (key, us, objectives, obss, rews_acc, alives),
                    jnp.arange(1, n_full_chunks),
                )

                if trunc_horizon % chunk_size != 0:
                    lim = trunc_horizon % chunk_size
                    key, topk_us, objectives, obss, rews_acc, alives = loop_fn(
                        (key, topk_us, objectives, obss, rews_acc, alives),
                        t=n_full_chunks,
                        lim=lim,
                    )[0]

                return topk_us, objectives, obss, rews_acc, alives

            topk_us, objectives, obss, rews_acc, alives = jax.vmap(beam_search)(
                jax.random.split(key, sampled_us.shape[0]), sampled_us, nexts, rews, alives)

            # swap dynamics and beam axes
            obss = obss.swapaxes(0, 1)
            rews_acc = rews_acc.swapaxes(0, 1)
            alives = alives.swapaxes(0, 1)

            # reshape
            topk_us = topk_us.reshape(beam_k, self.horizon, self.act_dim)
            objectives = objectives.reshape(beam_k)
            obss = obss.reshape(self.n_dynamics, beam_k,
                                trunc_horizon, self.obs_dim)
            rews_acc = rews_acc.reshape(self.n_dynamics, beam_k, trunc_horizon)
            alives = alives.reshape(self.n_dynamics, beam_k, trunc_horizon)

            return select_topk(topk_us, objectives, obss, rews_acc, alives, beam_k)

        topk_us, objectives, obss, rews_acc, alives = jax.vmap(
            sample)(obs, jax.random.split(r, obs.shape[0]), init_us=init_us)
        # swap dynamics and beam axes
        obss = obss.swapaxes(0, 1)
        rews_acc = rews_acc.swapaxes(0, 1)
        alives = alives.swapaxes(0, 1)
        return topk_us, objectives, obss, rews_acc, alives

    def _plan_elites(self, state, obs, key, step=0, aggregate_obj='mean'):
        """Diffusion beam search followed by langevin refinement of its elites (bootstrapping).

        Returns the best `n_elites` sequences (objective-descending) out of the union of the
        sampled elites and their refined counterparts, with their objectives. `obs` is expected
        to be already encoded and normalized.
        """
        k1, k2 = jax.random.split(key, 2)
        us, objectives, _, _, _ = self._get_action(
            state, obs, k1, num_beam=self.num_beam, beam_k=self.beam_k, beam_d=self.beam_d,
            use_ema=self.use_ema_policy, resample_n_steps=self.resample_steps,
            resample_temp=self.resample_temp, trunc_horizon=self.current_curriculum_horizon,
            aggregate_obj=aggregate_obj, chunk_size=self.chunk_size)
        if self.refine_elites:
            eta = self.update_noise_schedule(step)**2 / 2
            refined_us, refined_objs = self.langevin(
                state, jnp.repeat(obs[:, None], us.shape[1], axis=1), us, key=k2, eta=eta,
                use_gaussian_noise=self.update_noise_typ == 'langevin',
                n_steps=self.langevin_steps, aggregate_obj=aggregate_obj)
            us = jnp.concatenate([us, refined_us], axis=1)
            objectives = jnp.concatenate([objectives, refined_objs], axis=1)
        indices = jnp.argsort(objectives, axis=1, descending=True)[:, :self.n_elites]
        us = jnp.take_along_axis(us, indices[..., None, None], axis=1)
        objectives = jnp.take_along_axis(objectives, indices, axis=1)
        return jnp.clip(us, -1.0, 1.0), objectives

    def get_elites(self, state, obs, key, step=0, aggregate_obj='mean'):
        """Elite action sequences for raw `obs` (images for a vision algo):
        [B, n_elites, horizon, act_dim]. Planning runs at `infer_dtype`."""
        obs = self.encode(state, obs, use_train=False)
        obs = self.normalize_observation(state, obs).astype(self.infer_dtype)
        return self._plan_elites(state, obs, key, step, aggregate_obj=aggregate_obj)[0]

    def get_action(self, state, obs, key, step=0, aggregate_obj='max'):
        if not self.bootstrap:
            return super().get_action(state, obs, key, aggregate_obj=aggregate_obj)
        return self.get_elites(state, obs, key, step, aggregate_obj=aggregate_obj)[:, 0, 0]

    def get_rollout_action(self, state, obs, key, step=0):
        """With bootstrapping on, also the elites the action was selected from, which the
        trainer caches with the transition for the update to train on."""
        if not self.bootstrap:
            return self.get_action(state, obs, key, step, aggregate_obj=self.rol_agg)
        elites = self.get_elites(state, obs, key, step, aggregate_obj=self.rol_agg)
        return elites[:, 0, 0], elites

    def get_eval_action(self, state, obs, key, step=0):
        return self.get_action(state, obs, key, step, aggregate_obj=self.up_agg)

    def refresh_elites(self, state, obs, key, step=0):
        """Replan the elites of buffer rows with the current models (bootstrapping)."""
        return self.get_elites(state, obs, key, step, aggregate_obj=self.rol_agg)

    def rollout_elites(self, state, obs, us, horizons, aggregate_obj='min'):
        """Rescore cached elites with the current models, at `train_dtype`.

        Only the action sequences are cached; every state and objective along them is
        recomputed here. Each row is only evaluated up to the planning horizon its elites were
        optimized for (`horizons`, a curriculum may have been shorter then): the suffix past it
        was never optimized, so the objective is bootstrapped at `horizons[b]` and the returned
        mask excludes it from the critic targets.

        Args:
            obs: encoded, normalized observations [B, obs_dim]
            us: elite action sequences [B, K, horizon, act_dim]
            horizons: per-row planning horizon [B], clipped to the current curriculum horizon
        Returns:
            objectives [B, K], rollout observations [n_dynamics, B, K, H, obs_dim], discounted
            reward-to-go [n_dynamics, B, K, H], alive masks [n_dynamics, B, K, H] and the per-row
            validity mask [B, H], where H is the current curriculum horizon.
        """
        horizon = self.current_curriculum_horizon
        obs = obs.astype(self.train_dtype)
        us = us[..., :horizon, :].astype(self.train_dtype)
        shp = (self.n_dynamics, *us.shape[:2])  # [n_dynamics, B, K]
        horizons = jnp.clip(horizons, 1, horizon)

        us_rep = jnp.repeat(us[None], self.n_dynamics, axis=0)
        obs_rep = jnp.broadcast_to(obs[None, :, None, :], (*shp, self.obs_dim))
        rollout_obs = self.dynamics_model.apply(
            state.variables['dynamics'], obs_rep, us_rep, tile=False, method='compute_rollout',
            use_train=True)
        # states visited before each action, i.e. [obs, rollout_obs[:-1]]
        curr_obs = jnp.concatenate([obs_rep[..., None, :], rollout_obs[..., :-1, :]], axis=-2)
        rewards, alives = self.predict_reward_and_alive(
            state.variables['reward'], curr_obs, us_rep, use_train=True)
        values = self.critic.apply(
            state.variables['critic'], rollout_obs, method='compute_target', use_train=True)

        alives = jnp.concatenate([jnp.ones_like(alives[..., :1]), alives], axis=-1)
        alives = jnp.cumprod(alives, axis=-1)
        rews_acc = jnp.cumsum(
            rewards * alives[..., :-1] * self.gamma ** jnp.arange(horizon, dtype=rewards.dtype), axis=-1)

        # truncate each row's objective at its own planning horizon ([1, B, 1, 1] time index)
        last = (horizons - 1)[None, :, None, None]
        objectives = jnp.take_along_axis(rews_acc, last, axis=-1)[..., 0] \
            + jnp.take_along_axis(alives[..., 1:], last, axis=-1)[..., 0] \
            * (self.gamma ** horizons.astype(rewards.dtype))[None, :, None] \
            * jnp.take_along_axis(values, last, axis=-1)[..., 0]
        objectives = aggregate_fn(objectives, aggregate_obj=aggregate_obj)
        mask = jnp.arange(horizon)[None] < horizons[:, None]  # [B, H]
        return objectives, rollout_obs, rews_acc, alives[..., 1:], mask

    def compute_objective(self, state, obs, actions, use_train: bool = False, aggregate_obj='min'):
        obs = obs.astype(self.infer_dtype)
        obss = self.dynamics_model.apply(
            state.variables['dynamics'], obs, actions, use_train=use_train, method="compute_rollout")
        obss = jnp.concatenate([obs[None, ..., None, :].repeat(
            self.n_dynamics, axis=0), obss], axis=-2)
        actions = actions[None].repeat(self.n_dynamics, axis=0)
        rewards, alives = self.predict_reward_and_alive(
            state.variables['reward'], obss[..., :-1, :], actions, use_train=use_train)
        rewards, alives = rewards.astype(jnp.float32), alives.astype(jnp.float32)
        alives = jnp.concatenate(
            [jnp.ones_like(alives[..., :1]), alives], axis=-1)
        alives = jnp.cumprod(alives, axis=-1)

        v = self.critic.apply(
            state.variables['critic'], obss[..., -1, :], method="compute_target", use_train=use_train)
        rewards = jnp.concatenate([rewards, v[..., None].astype(jnp.float32)], axis=-1)

        rewards = rewards * \
            self.gamma ** jnp.arange(rewards.shape[-1],
                                     dtype=rewards.dtype) * alives
        return aggregate_fn(rewards.sum(axis=-1), aggregate_obj=aggregate_obj).astype(jnp.float32)

    def init_train_state(self, cfg) -> TrainState:
        state = super().init_train_state(cfg)

        # Initialize critic
        key = jax.random.PRNGKey(get_seed())
        dummy_obs = jnp.zeros((1, self.obs_dim))
        critic_vars = self.critic.init(key, dummy_obs)
        state = state.replace(
            variables={**state.variables, 'critic': critic_vars},
            opt_state={**state.opt_state,
                       'critic': self.critic_opt.init(critic_vars)},
            ema_variables={**state.ema_variables, 'critic': critic_vars},
        )
        return state

    def _critic_loss(self, critic_params, state, policy_obs, rollout_obss, rews_acc, alives, mask=None,
                     delta=5.0, aggregate_obj='mean'):
        v = self.critic.apply(critic_params, policy_obs, use_train=True)

        next_v = self.critic.apply(
            state.ema_variables['critic'], rollout_obss, method="compute_target", use_train=True)
        next_v = alives * next_v * \
            self.gamma ** (jnp.arange(1,
                           next_v.shape[-1]+1, dtype=next_v.dtype))
        target_v = rews_acc + next_v
        vars = jnp.var(target_v, axis=0)
        target_v = aggregate_fn(target_v, aggregate_obj=aggregate_obj)
        if mask is None:
            mask = jnp.ones_like(target_v, dtype=jnp.bool_)
        # last valid step of each row (a cached elite was only optimized up to there)
        last = jnp.sum(mask, axis=-1) - 1
        target_v_last = jnp.take_along_axis(target_v, last[:, None], axis=-1)[:, 0]
        target_v_diff = jnp.max(target_v_last - target_v[:, 0])

        # only mix the TD targets of the steps that were actually optimized
        wgt = jax.nn.softmax(jnp.where(mask, 1 / (vars+1e-6), -jnp.inf), axis=-1)
        target_v = (target_v * wgt).sum(axis=-1)

        loss = clamped_huber_loss(
            target_v[..., None], v, delta=delta, cap_mult=2.0)
        return loss, {'critic_loss': loss, 'target_v_diff_max': target_v_diff, 'target_v_diff': target_v_diff, 'v_mean': v.mean(), 'target_v_mean': target_v.mean()}

    def update(self, state: TrainState, transition, key: jax.Array, step: int) -> Tuple[TrainState, dict]:
        if self.bootstrap:
            return self._update_bootstrapped(state, transition, key, step)
        prepared_obs = self.prepare_transition_obs(state, transition, use_train=True)
        _, obs = prepared_obs
        key, k1, k2 = jax.random.split(key, 3)
        policy_obs = self.get_policy_obs(obs)
        sampled_us, objectives, rollout_obss, rews_acc, alives = self._get_action(state, policy_obs, k1, num_beam=self.num_beam, beam_k=self.beam_k, beam_d=self.beam_d,
                                                                                  use_ema=self.use_ema_policy, resample_n_steps=self.resample_steps, resample_temp=self.resample_temp,
                                                                                  trunc_horizon=self.current_curriculum_horizon, aggregate_obj=self.up_agg, chunk_size=self.chunk_size)
        sampled_us_std = sampled_us.std(axis=1).mean(
            axis=(0, -1)).astype(jnp.float32)  # B, N, H, D
        info_u_std = {f'act_std_{i}': std for i,
                      std in enumerate(sampled_us_std)}
        info_u_std['act_std_all'] = sampled_us_std.mean()

        # select best action sequence and compute critic loss
        critic_loss = partial(self._critic_loss, state=state,  policy_obs=policy_obs, rollout_obss=rollout_obss[..., 0, :, :],
                              rews_acc=rews_acc[..., 0, :], alives=alives[..., 0, :], aggregate_obj=self.up_agg)
        critic_params, critic_opt_state, critic_info = update_params(
            critic_loss, state.variables['critic'], state.opt_state['critic'], self.critic_opt, has_aux=True)

        # add noise
        if self.update_noise_typ == "normal":
            update_noise = self.update_noise_schedule(step)
            sampled_us_rep = jnp.repeat(sampled_us, self.beam_d-1, axis=1)
            sampled_us_rep += jax.random.normal(
               k2, sampled_us_rep.shape, dtype=sampled_us_rep.dtype) * update_noise
            sampled_us_rep = jnp.clip(sampled_us_rep, -1.0, 1.0)
            sampled_us = jnp.concatenate([sampled_us, sampled_us_rep], axis=1)
            objectives = jax.vmap(partial(self.compute_objective, aggregate_obj=self.up_agg), in_axes=(None, None, 1))(
               state, policy_obs, sampled_us[..., :self.current_curriculum_horizon, :]).T
        elif self.update_noise_typ in ["langevin", "ga"]:
            n_steps = 4
            update_noise = self.update_noise_schedule(step)
            eta = update_noise**2 / 2 
            us, objs_lang = self.langevin(state, policy_obs[:, None].repeat(sampled_us.shape[1], axis=1),
                                            sampled_us, key=k2, eta=eta, use_gaussian_noise=self.update_noise_typ == 'langevin', 
                                            n_steps=n_steps, aggregate_obj=self.up_agg)
            sampled_us = jnp.concatenate([sampled_us, us], axis=1)
            objectives = jnp.concatenate([objectives, objs_lang], axis=1)
        elif self.update_noise_typ is None or self.update_noise_typ.lower() == "none":
            pass
        else:
            raise ValueError(f"Invalid update noise type: {self.update_noise_typ}")

        # Update target networks
        target_critic_params = delay_target_update(step=step, vars=critic_params,
                                                   target_vars=state.ema_variables['critic'],
                                                   tau=self.tau, update_every=self.delay_update)
        state = state.replace(
            variables={**state.variables, 'critic': critic_params},
            opt_state={**state.opt_state, 'critic': critic_opt_state},
            ema_variables={**state.ema_variables,
                           'critic': target_critic_params},
        )
        state, info = super().update(state, transition, key, step,
                                     policy_obs, sampled_us, objectives, prepared_obs=prepared_obs)
        info = {**info, **critic_info, **info_u_std}
        info = {(f'loss/{k}' if 'loss' in k else k): v for k, v in info.items()}
        return state, info

    def _update_bootstrapped(self, state: TrainState, transition, key: jax.Array, step: int) -> Tuple[TrainState, dict]:
        """`update` with elite bootstrapping: train on the elites cached with each transition
        (rescored with the current models) instead of planning a new pool per batch row."""
        prepared_obs = self.prepare_transition_obs(state, transition, use_train=True)
        _, obs = prepared_obs
        policy_obs = self.get_policy_obs(obs)

        sampled_us = transition.elites.astype(jnp.float32)  # [B, n_elites, H, D]
        horizons = jnp.clip(transition.elite_horizon, 1, self.current_curriculum_horizon)
        objectives, rollout_obss, rews_acc, alives, mask = self.rollout_elites(
            state, policy_obs, sampled_us, horizons, aggregate_obj=self.up_agg)

        sampled_us_std = sampled_us.std(axis=1).mean(axis=(0, -1)).astype(jnp.float32)
        info_u_std = {f'act_std_{i}': std for i, std in enumerate(sampled_us_std)}
        info_u_std['act_std_all'] = sampled_us_std.mean()

        # critic loss along the best elite under the current models
        best = jnp.argmax(objectives, axis=1)  # [B]
        best_obss = jnp.take_along_axis(rollout_obss, best[None, :, None, None, None], axis=2)[:, :, 0]
        best_rews_acc = jnp.take_along_axis(rews_acc, best[None, :, None, None], axis=2)[:, :, 0]
        best_alives = jnp.take_along_axis(alives, best[None, :, None, None], axis=2)[:, :, 0]
        critic_loss = partial(self._critic_loss, state=state, policy_obs=policy_obs, rollout_obss=best_obss,
                              rews_acc=best_rews_acc, alives=best_alives, mask=mask, aggregate_obj=self.up_agg)
        critic_params, critic_opt_state, critic_info = update_params(
            critic_loss, state.variables['critic'], state.opt_state['critic'], self.critic_opt, has_aux=True)

        target_critic_params = delay_target_update(step=step, vars=critic_params,
                                                   target_vars=state.ema_variables['critic'],
                                                   tau=self.tau, update_every=self.delay_update)
        state = state.replace(
            variables={**state.variables, 'critic': critic_params},
            opt_state={**state.opt_state, 'critic': critic_opt_state},
            ema_variables={**state.ema_variables, 'critic': target_critic_params},
        )
        state, info = super().update(state, transition, key, step, policy_obs, sampled_us, objectives,
                                     horizons, prepared_obs=prepared_obs)
        info = {**info, **critic_info, **info_u_std, 'elite_horizon': horizons.mean()}
        info = {(f'loss/{k}' if 'loss' in k else k): v for k, v in info.items()}
        return state, info

    def langevin(self, state, policy_obs, us, key, eta, n_steps=2, use_gaussian_noise=True, aggregate_obj='min'):
        orig_shp = us.shape
        policy_obs = policy_obs.reshape(-1, self.obs_dim)
        us = us.reshape(-1, self.horizon, self.act_dim)

        def _compute_objective(us, obs):
            return self.compute_objective(state, obs, us[..., :self.current_curriculum_horizon, :], aggregate_obj=aggregate_obj)

        grad_fn = jax.grad(_compute_objective)

        def _langevin(carry, _):
            us, key = carry
            key, subkey = jax.random.split(key)
            grads = jax.vmap(grad_fn)(us, policy_obs)
            # grad ascent; cast back so a bfloat16 carry is not promoted by the float32 schedule
            step_ = eta * grads + use_gaussian_noise * jnp.sqrt(2*eta) * jax.random.normal(subkey, us.shape)
            us = jnp.clip(us + step_.astype(us.dtype), -1, 1)
            return (us, key), None

        (us, _), _ = jax.lax.scan(_langevin, (us, key), None, length=n_steps)
        objs = _compute_objective(us, policy_obs)
        return us.reshape(orig_shp), objs.reshape(*orig_shp[:-2])
