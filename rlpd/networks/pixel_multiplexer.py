from typing import Dict, Optional, Tuple, Type, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from rlpd.networks import default_init


class PixelMultiplexer(nn.Module):
    encoder_cls: Type[nn.Module]
    network_cls: Type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_keys: Tuple[str, ...] = ("pixels",)
    depth_keys: Tuple[str, ...] = ()

    @nn.compact
    def __call__(
        self,
        observations: Union[FrozenDict, Dict],
        actions: Optional[jnp.ndarray] = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        if len(self.depth_keys) == 0:
            depth_keys = [None] * len(self.pixel_keys)
        else:
            depth_keys = self.depth_keys

        xs = []
        for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
            x = observations[pixel_key].astype(jnp.float32) / 255.0
            if depth_key is not None:
                # The last dim is always for stacking, even if it's 1.
                x = jnp.concatenate([x, observations[depth_key]], axis=-2)

            x = jnp.reshape(x, (*x.shape[:-2], -1))

            x = self.encoder_cls(name=f"encoder_{i}")(x)

            if self.stop_gradient:
                # We do not update conv layers with policy gradients.
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        if "state" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(
                observations["state"]
            )
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)
