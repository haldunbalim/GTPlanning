import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from gtp.algo.common import Algorithm, TrainState, MultiCritic
from gtp.algo.networks import MLP, mish
from gtp.utils import get_optimizer, delay_target_update, update_params, get_seed

class Actor(nn.Module):
    """Gaussian policy with tanh squashing."""
    hidden_dims: list
    action_dim: int

    @nn.compact
    def __call__(self, obs):
        return MLP(self.hidden_dims + (self.action_dim,), hidden_act_class=mish, act_out=jnp.tanh)(obs)

class TD3(Algorithm):
    def __init__(self, obs_dim, act_dim, hidden_dims=[256, 256, 256],
                 lr=3e-4, gamma=0.99, tau=0.005, 
                 rollout_noise_std=0.1, target_noise_std=0.2, noise_clip=0.5, 
                 reward_scale=0.2, delay_update=2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.delay_update = delay_update
        self.rollout_noise_std = rollout_noise_std
        self.target_noise_std = target_noise_std
        self.noise_clip = noise_clip
        

        # Initialize networks
        self.actor = Actor(hidden_dims=hidden_dims, action_dim=act_dim)
        self.critic = MultiCritic(hidden_dims=hidden_dims, use_layer_norm=False, num_critics=2)
        
        # Initialize optimizers
        self.actor_opt = get_optimizer(lr)
        self.critic_opt = get_optimizer(lr)

    def init_train_state(self, cfg) -> TrainState:
        key = jax.random.PRNGKey(get_seed())
        key, actor_key, critic_key = jax.random.split(key, 3)

        # Initialize actor
        dummy_obs = jnp.zeros((1, self.obs_dim))
        actor_params = self.actor.init(actor_key, dummy_obs)

        # Initialize critic
        dummy_action = jnp.zeros((1, self.act_dim))
        critic_params = self.critic.init(critic_key, dummy_obs, dummy_action)

        variables = {
            'actor': actor_params,
            'critic': critic_params,
        }

        # Target network is a copy of the critic
        ema_variables = {
            'actor': actor_params, 
            'critic': critic_params
        }

        actor_opt_state = self.actor_opt.init(actor_params)
        critic_opt_state = self.critic_opt.init(critic_params)

        opt_state = {
            'actor': (actor_opt_state),
            'critic': (critic_opt_state),
        }

        return TrainState(
            variables=variables,
            ema_variables=ema_variables,
            opt_state=opt_state
        )

    def get_action(self, state, obs, key=None, deterministic=False):
        """Sample action from policy."""
        action = self.actor.apply(state.variables['actor'], obs)

        if not deterministic:
            action += jax.random.normal(key, shape=action.shape) * self.rollout_noise_std
            action = jnp.clip(action, -1.0, 1.0)

        return action

    def _critic_loss(self, critic_params, state, obs, action, reward, next_obs, done, key):
        """Compute critic loss."""
        # Current Q-values
        q = self.critic.apply(critic_params, obs, action)

        # Sample next actions from target policy
        next_action = self.actor.apply(state.ema_variables['actor'], next_obs)
        noise =  jax.random.normal(key, shape=next_action.shape) * self.target_noise_std
        next_action += jnp.clip(noise, -self.noise_clip, self.noise_clip)
        next_action = jnp.clip(next_action, -1.0, 1.0)

        # Target Q-values
        target_q = self.critic.apply(state.ema_variables['critic'], next_obs, next_action, method="compute_target")

        # Bellman backup
        backup = reward + self.gamma * (1 - done) * target_q
        backup = jax.lax.stop_gradient(backup)

        # TD errors
        critic_loss = jnp.mean((backup[..., None] - q)**2)

        info = {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'target_q_mean': target_q.mean()
        }

        return critic_loss, info

    def _actor_loss(self, actor_params, state, obs):
        """Compute actor loss."""
        # Sample actions from current policy
        action = self.actor.apply(actor_params, obs)

        # Q-values
        q = self.critic.apply(state.variables['critic'], obs, action, method="compute_target")
        actor_loss = -q.mean()

        info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, (info)


    def update(self, state, experience, key, step):
        """Update SAC networks."""
        obs = experience.obs[:, 0]
        action = experience.action[:, 0]
        reward = experience.reward[:, 0] * self.reward_scale
        next_obs = experience.obs[:, 1]
        done = experience.done[:, 0].astype(jnp.float32)

        # Split keys
        key, critic_key = jax.random.split(key, 2)

        # ---- Update critic ----
        critic_opt_state = state.opt_state['critic']
        critic_loss_fn = partial(
            self._critic_loss,
            state=state,
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
            state=state,
            obs=obs,
        )

        actor_params, actor_opt_state, actor_info = update_params(
            actor_loss_fn, state.variables['actor'], actor_opt_state, self.actor_opt, has_aux=True)
        target_actor_params = delay_target_update(
            step, actor_params, state.ema_variables['actor'],
            tau=self.tau, update_every=self.delay_update)

        # Create new state
        new_variables = {
            'actor': actor_params,
            'critic': critic_params,
        }

        new_ema_variables = {
            'critic': target_critic_params,
            'actor': target_actor_params  # Actor target network is not used in TD3
        }

        new_opt_state = {
            'actor': actor_opt_state,
            'critic': critic_opt_state,
        }

        # Assemble new state
        state = state.replace(variables=new_variables, 
                              ema_variables=new_ema_variables,
                              opt_state=new_opt_state)
        # Collect info
        info = {**critic_info, **actor_info}

        return state, info
