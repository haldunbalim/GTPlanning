import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from gtp.algo.common import Algorithm, TrainState, DoubleCritic
from gtp.algo.networks import MLP, mish
from gtp.utils import get_optimizer, delay_target_update, update_params, normal_log_prob, get_seed


LOG_STD_MIN = -20
LOG_STD_MAX = 0.5

class Actor(nn.Module):
    """Gaussian policy with tanh squashing."""
    hidden_dims: list
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        x = MLP(self.hidden_dims + (2*self.action_dim,), hidden_act_class=mish)(obs)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

class SAC(Algorithm):
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 256],
                 lr=3e-4, alpha_lr=7e-3, gamma=0.99, tau=0.005, alpha=1.0,
                 target_entropy=None, auto_alpha=True, reward_scale=0.2, delay_update=2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.tau = tau
        self.alpha_value = alpha
        self.auto_alpha = auto_alpha
        self.reward_scale = reward_scale
        self.delay_update = delay_update

        # Set target entropy to -dim(A) if not specified
        if target_entropy is None:
            self.target_entropy = -act_dim
        else:
            self.target_entropy = target_entropy

        # Initialize networks
        self.actor = Actor(hidden_dims=hidden_dims, action_dim=act_dim)
        self.critic = DoubleCritic(hidden_dims=hidden_dims, use_layer_norm=False)
        
        # Initialize optimizers
        self.actor_opt = get_optimizer(lr)
        self.critic_opt = get_optimizer(lr)
        self.alpha_opt = get_optimizer(alpha_lr) if self.auto_alpha else None

    def init_train_state(self, cfg) -> TrainState:
        key = jax.random.PRNGKey(get_seed())
        key, actor_key, critic_key = jax.random.split(key, 3)

        # Initialize actor
        dummy_obs = jnp.zeros((1, self.obs_dim))
        actor_params = self.actor.init(actor_key, dummy_obs)

        # Initialize critic
        dummy_action = jnp.zeros((1, self.act_dim))
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)

        # Initialize log_alpha if using automatic temperature tuning
        if self.auto_alpha:
            log_alpha = jnp.array([jnp.log(self.alpha_value)])
        else:
            log_alpha = None

        variables = {
            'actor': actor_params,
            'critic': critic_params,
            'log_alpha': log_alpha
        }

        # Target network is a copy of the critic
        ema_variables = {
            'critic': critic_params
        }

        actor_opt_state = self.actor_opt.init(actor_params)
        critic_opt_state = self.critic_opt.init(critic_params)
        alpha_opt_state = self.alpha_opt.init(
            log_alpha) if self.auto_alpha else None

        opt_state = {
            'actor': (actor_opt_state),
            'critic': (critic_opt_state),
            'log_alpha': (alpha_opt_state) if self.auto_alpha else None
        }

        return TrainState(
            variables=variables,
            ema_variables=ema_variables,
            opt_state=opt_state
        )

    def get_action(self, state, obs, key, deterministic=False):
        """Sample action from policy."""
        mean, log_std = self.actor.apply(state.variables['actor'], obs)

        if deterministic:
            action = jnp.tanh(mean)
        else:
            std = jnp.exp(log_std)
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, shape=mean.shape)
            action_unscaled = mean + std * noise
            action = jnp.tanh(action_unscaled)

        return action
    
    def _sample_action(self, actor_params, obs, key):
        """Sample action and compute log probability."""
        mean, log_std = self.actor.apply(actor_params, obs)
        std = jnp.exp(log_std)

        # Sample action
        noise = jax.random.normal(key, shape=mean.shape)
        action_unscaled = mean + std * noise
        action = jnp.tanh(action_unscaled)

        # Compute log probability with tanh correction
        log_prob = normal_log_prob(action_unscaled, mean, std)
        log_prob -= 2 * (jnp.log(2) - action_unscaled - jax.nn.softplus(-2 * action_unscaled))
        return action, log_prob.sum(axis=-1)

    def _critic_loss(self, critic_params, target_critic_params, actor_params,
                     log_alpha, obs, action, reward, next_obs, done, key):
        """Compute critic loss."""
        # Current Q-values
        q1, q2 = self.critic.apply(critic_params, obs, action)

        # Sample next actions from current policy
        next_action, next_log_prob = self._sample_action(
            actor_params, next_obs, key)

        # Target Q-values
        target_q = self.critic.apply(target_critic_params, next_obs, next_action, 
                                     method="compute_target")


        # Add entropy term
        alpha = jnp.exp(
            log_alpha) if log_alpha is not None else self.alpha_value
        target_q = target_q - alpha * next_log_prob

        # Bellman backup
        backup = reward + self.gamma * (1 - done) * target_q
        backup = jax.lax.stop_gradient(backup)

        # TD errors
        critic_loss = ((q1 - backup) ** 2 + (q2 - backup) ** 2).mean()

        info = {
            'critic_loss': critic_loss,
            'q1_mean': q1.mean(),
            'q2_mean': q2.mean(),
            'target_q_mean': target_q.mean()
        }

        return critic_loss, info

    def _actor_loss(self, actor_params, critic_params, log_alpha, obs, key):
        """Compute actor loss."""
        # Sample actions from current policy
        action, log_prob = self._sample_action(actor_params, obs, key)

        # Q-values
        q = self.critic.apply(critic_params, obs, action, method="compute_target")

        # Actor loss is negative expected return (plus entropy)
        alpha = jnp.exp(
            log_alpha) if log_alpha is not None else self.alpha_value
        actor_loss = (alpha * log_prob - q).mean()

        info = {
            'actor_loss': actor_loss,
            'entropy': -log_prob.mean(),
            'alpha': alpha
        }

        return actor_loss, (info, log_prob)

    def _alpha_loss(self, log_alpha, log_prob):
        """Compute temperature loss."""
        # WARNING: original SAC paper uses exp(log_alpha) but sb3 does not
        alpha_loss = -(log_alpha * (log_prob + self.target_entropy)).mean()

        info = {
            'alpha_loss': alpha_loss
        }

        return alpha_loss, info

    def update(self, state, experience, key, step):
        """Update SAC networks."""
        obs = experience.obs[:, 0]
        action = experience.action[:, 0]
        reward = experience.reward[:, 0] * self.reward_scale
        next_obs = experience.obs[:, 1]
        done = experience.done[:, 0].astype(jnp.float32)

        # Split keys
        key, critic_key, actor_key = jax.random.split(key, 3)

        # ---- Update critic ----
        critic_opt_state = state.opt_state['critic']
        critic_loss_fn = partial(
            self._critic_loss,
            target_critic_params=state.ema_variables['critic'],
            actor_params=state.variables['actor'],
            log_alpha=state.variables['log_alpha'],
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            done=done,
            key=critic_key
        )
        critic_params, critic_opt_state, critic_info = update_params(
            critic_loss_fn, state.variables['critic'], critic_opt_state, self.critic_opt, has_aux=True)
        
        # Update target network with EMA
        target_critic_params = delay_target_update(
            step, critic_params, state.ema_variables['critic'],
            tau=self.tau, update_every=self.delay_update)

        # ---- Update actor ----
        actor_opt_state = state.opt_state['actor']
        actor_loss_fn = partial(
            self._actor_loss,
            critic_params=critic_params,
            log_alpha=state.variables['log_alpha'],
            obs=obs,
            key=actor_key
        )

        actor_params, actor_opt_state, (actor_info, log_prob) = update_params(
            actor_loss_fn, state.variables['actor'], actor_opt_state, self.actor_opt, has_aux=True)

        # Create new state
        new_variables = {
            'actor': actor_params,
            'critic': critic_params,
        }

        new_ema_variables = {
            'critic': target_critic_params
        }

        new_opt_state = {
            'actor': actor_opt_state,
            'critic': critic_opt_state,
        }

        # Update alpha if using automatic temperature tuning
        if self.auto_alpha:
            alpha_opt_state = state.opt_state['log_alpha']
            alpha_loss_fn = partial(
                self._alpha_loss,
                log_prob=log_prob,
            )
            new_variables['log_alpha'], new_opt_state['log_alpha'], alpha_info = update_params(
                alpha_loss_fn, state.variables['log_alpha'], alpha_opt_state, self.alpha_opt, has_aux=True)

        # Assemble new state
        state = state.replace(variables=new_variables, 
                              ema_variables=new_ema_variables,
                              opt_state=new_opt_state)
        # Collect info
        info = {**critic_info, **actor_info, **alpha_info}

        return state, info
