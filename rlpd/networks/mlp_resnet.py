from typing import Any, Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any


class MLPResNetV2Block(nn.Module):
    """MLPResNet block."""

    features: int
    act: Callable

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.LayerNorm()(x)
        y = self.act(y)
        y = nn.Dense(self.features)(y)
        y = nn.LayerNorm()(y)
        y = self.act(y)
        y = nn.Dense(self.features)(y)

        if residual.shape != y.shape:
            residual = nn.Dense(self.features)(residual)

        return residual + y


class MLPResNetV2(nn.Module):
    """MLPResNetV2."""

    num_blocks: int
    features: int = 256
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.features)(x)
        for _ in range(self.num_blocks):
            x = MLPResNetV2Block(self.features, act=self.act)(x)
        x = nn.LayerNorm()(x)
        x = self.act(x)
        return x
