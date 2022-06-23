from typing import Type

import flax.linen as nn
import jax.numpy as jnp

from rlpd.networks import default_init


class TanhDeterministic(nn.Module):
    base_cls: Type[nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> jnp.ndarray:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(), name="OutputDenseMean"
        )(x)

        means = nn.tanh(means)

        return means
