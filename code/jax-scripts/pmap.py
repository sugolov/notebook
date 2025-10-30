"""
Simple test case to understand JAX pmap (parallel map)
pmap maps a function across multiple devices (GPUs/CPUs) in parallel
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from equinox.nn import Linear
from typing import Callable

sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
SiLU    = lambda x: x * sigmoid(x)

class MLP(eqx.Module):
    proj: Linear
    # activation: Callable

    def __init__(self, dim_in, dim_out, key=jax.random.PRNGKey(0)):
        self.proj = Linear(dim_in, dim_out, key=key)

    def __call__(self, x):
        return jax.vmap(SiLU)(self.proj(x))

print("=== setup ===")
B = 16
D = 3
_divider = "=" * 6
print(f"batch size {B}, dim {D}")

print(f"{_divider} forward pass {_divider}")
model = MLP(3, 3)
print(model)
x = jnp.ones(3)
print(x)
y = model(x)
print(y)

print(f"{_divider} grads {_divider}")

@jax.jit
@jax.grad
def loss(model, x):
    y = jax.vmap(model)(x) 
    return jnp.mean(y ** 2)

key = jax.random.PRNGKey(0)
x   = jax.random.normal(key, (16, 3))
key, _ = jax.random.split(key, 2)
y   = jax.random.normal(key, (16, 3))
grads = loss(model, x)

print("original weights")
print(model.proj.weight)
print("grads")
print(grads.proj.weight)

# learning_rate = 0.1
# model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)

print(f"{_divider} data parallelism {_divider}")
print(f"jax devices: {jax.devices()}")
print(f"`x` device: {x.device}")
print(f"`x` sharding: {x.sharding}")

