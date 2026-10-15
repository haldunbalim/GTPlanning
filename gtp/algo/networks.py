import jax.numpy as jnp
import flax.linen as nn
import math
import chex
import jax
from typing import Optional, Callable


def mish(x):
    return x * jnp.tanh(nn.softplus(x))
    
def sinusoidal_pos_emb(x, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb


class Downsample1d(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, kernel_size=3, strides=2, padding="SAME", dtype=self.dtype)(x)


class Upsample1d(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.ConvTranspose(self.dim, kernel_size=4, strides=2, padding="SAME", dtype=self.dtype)(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Swish
    '''
    out_channels: int
    kernel_size: int
    n_groups: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, self.kernel_size,
                    padding='SAME', dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=self.n_groups)(x)
        x = jax.nn.swish(x)
        return x

class FiLMBlock(nn.Module):
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, cond):
        '''
            x : [ batch_size x horizon x inp_channels]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x horizon x out_channels ]
        '''
        # film
        scale_bias = nn.Dense(x.shape[-1] * 2, dtype=self.dtype)(jax.nn.swish(cond))
        scale, bias = jnp.split(scale_bias, 2, axis=-1)
        return x * (scale[:, None] + 1) + bias[:, None]
    

class Conv1dFilmBlock(nn.Module):
    '''
        GroupNorm --> FiLM --> Swish --> Conv1d
    '''
    out_channels: int
    kernel_size: int
    n_groups: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t):
        if x.shape[-1] % self.n_groups != 0:
            x = nn.LayerNorm(dtype=self.dtype)(x)
        else:
            x = nn.GroupNorm(self.n_groups, dtype=self.dtype)(x)
        x = FiLMBlock(dtype=self.dtype)(x, t)
        x = jax.nn.swish(x)
        x = nn.Conv(self.out_channels, self.kernel_size,
                    padding='SAME', dtype=self.dtype)(x)
        return x

class ResidualTemporalFiLMBlock(nn.Module):

    out_channels: int
    kernel_size: int = 3
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t):
        out = Conv1dFilmBlock(
            self.out_channels, self.kernel_size, dtype=self.dtype)(x, t)
        out = Conv1dFilmBlock(
            self.out_channels, self.kernel_size, dtype=self.dtype)(out, t)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=1, padding='same', dtype=self.dtype)(x)
        return out + x

class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        hidden_dim = self.dim_head * self.heads
        qkv = nn.Conv(hidden_dim * 3, kernel_size=1, strides=1, padding="SAME", use_bias=False)(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        qkv = map(lambda t: jnp.reshape(t, (t.shape[0], -1, self.heads, self.dim_head)), qkv) # b, t, h, d
        q, k, v = map(lambda t: jnp.transpose(t, (0, 2, 3, 1)), qkv) # b, h, d, t
        q = q * (self.dim_head ** -0.5)

        k = jax.nn.softmax(k, axis=1)
        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)

        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.reshape(out.shape[0], -1, out.shape[-1]) # b, h*d, t
        out = jnp.transpose(out, (0, 2, 1))  # b, t, h*d
        out = nn.Conv(dim, kernel_size=1, strides=1, padding="SAME")(out)
        return out

class TemporalUnetFiLM(nn.Module):
    dim: int = 32
    cond_embed_dim: int = 64
    dim_mults: tuple = (1, 2, 4, 8)
    out_dim: int = 1
    attention: bool = False
    input_transform_fn: Optional[Callable] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, time, conds, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        if self.input_transform_fn is not None:
            x = self.input_transform_fn(x, *args)
        chex.assert_is_divisible(x.shape[1], 2 ** (len(self.dim_mults)-1))

        t = sinusoidal_pos_emb(time, self.dim)
        t = nn.Dense(self.dim * 4, dtype=self.dtype)(t)
        t = jax.nn.swish(t)
        t = nn.Dense(self.dim, dtype=self.dtype)(t)

        # Process the conditions
        if not isinstance(conds, list):
            conds = [conds]
        _conds = []
        for cond in conds:
            cond = nn.Dense(self.cond_embed_dim * 4, dtype=self.dtype)(cond)
            cond = jax.nn.swish(cond)
            cond = nn.Dense(self.cond_embed_dim, dtype=self.dtype)(cond)
            _conds.append(cond)

        t = jnp.concatenate([t, *_conds], axis=-1)  # concatenate time and cond

        dims = list(map(lambda m: self.dim * m, self.dim_mults))
        h = []
        for ind, d in enumerate(dims):
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            if self.attention:
                x = x + LinearAttention(dtype=self.dtype)(nn.LayerNorm()(x))
            h.append(x)
            if ind < len(dims) - 1:
                x = Downsample1d(d, dtype=self.dtype)(x)

        x = ResidualTemporalFiLMBlock(dims[-1], dtype=self.dtype)(x, t)
        x = ResidualTemporalFiLMBlock(dims[-1], dtype=self.dtype)(x, t)

        for ind, d in enumerate(reversed(dims[:-1])):
            x = jnp.concatenate((x, h.pop()), axis=-1)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            if self.attention:
                x = x + LinearAttention(dtype=self.dtype)(nn.LayerNorm()(x))
            if ind < len(dims) - 1:
                x = Upsample1d(d, dtype=self.dtype)(x)

        #x = Conv1dBlock(self.dim, kernel_size=5, dtype=self.dtype)(x)
        x = ResidualTemporalFiLMBlock(self.dim, dtype=self.dtype)(x, t)
        x = nn.Conv(self.out_dim, kernel_size=1, padding='SAME', dtype=self.dtype)(x)
        return x


class MLP(nn.Module):
    feature_sizes: list
    hidden_act_class: type = nn.relu
    act_out: Optional[callable] = None
    use_layer_norm: bool = False
    squeeze_output: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, sz in enumerate(self.feature_sizes):
            x = nn.Dense(sz, dtype=self.dtype)(x)  # Linear layer
            if i != len(self.feature_sizes) - 1:
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.hidden_act_class(x)
        if self.act_out is not None:
            x = self.act_out(x)  # Optional output activation
        if self.squeeze_output:
            x = jnp.squeeze(x, axis=-1)
        return x
