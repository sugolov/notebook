import os

USE_CPU_ONLY = True

flags = os.environ.get("XLA_FLAGS", "")
if USE_CPU_ONLY:
    flags += " --xla_force_host_platform_device_count=8"  # Simulate 8 devices
    # Enforce CPU-only execution
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    # GPU flags
    flags += (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=false "
        "--xla_gpu_enable_async_collectives=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
os.environ["XLA_FLAGS"] = flags

import numpy as np

import jax
import jax.numpy as jnp
import equinox as eqx

from jax import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from equinox.nn import Linear
from typing import Callable

# ============================================================================ 
# Module
# ============================================================================

sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
SiLU    = lambda x: x * sigmoid(x)

class MLP(eqx.Module):
    proj: Linear
    # activation: Callable

    def __init__(self, dim_in, dim_out, key=jax.random.PRNGKey(0)):
        self.proj = Linear(dim_in, dim_out, key=key)

    def __call__(self, x):
        return jax.vmap(SiLU)(self.proj(x))
    

# ============================================================================ 
# Forward pass
# ============================================================================

_divider = "=" * 6
print(f"{_divider}  setup {_divider}")
B = 16
D = 3
print(f"batch size {B}, dim {D}")

print("\n\n")


print(f"{_divider} forward pass {_divider}")
model = MLP(3, 3)
print(model)
x = jnp.ones(3)
print(x)
y = model(x)
print(y)

print("\n\n")


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

print("\n\n")

# learning_rate = 0.1
# model = jax.tree_util.tree_map(lambda m, g: m - learning_rate * g, model, grads)



# ============================================================================ 
# Sharding
# ============================================================================
print(f"{_divider} sharding {_divider} \n")

print(f"jax devices: {jax.devices()}")
print(f"`x` device: {x.device}")
print(f"`x` sharding: {x.sharding}")

print("\n\n")

mesh = Mesh(np.array(jax.devices()), ("i", ))
print(mesh)
print(f"device ids: {mesh.device_ids}")
print(f"axis names: {mesh.axis_names}")

sharding = NamedSharding(
    mesh,
    P("i"),
)

x_sharded = jax.device_put(x, sharding)
print(f"`x` sharded device: {x_sharded.device}")
print(f"`x` sharded sharding: {x_sharded.sharding}")

print(f"{_divider} data parallel {_divider} \n")
key, _ = jax.random.split(key, 2)
x = jax.random.normal(key, (8,8,3))

sharding = NamedSharding(
    mesh,
    P("i", None, None),
)

x = jax.device_put(x, sharding)
print(f"`x` device: {x.device}")
print(f"`x` sharding: {x.sharding}")




