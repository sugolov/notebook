"""
Simple test case to understand JAX pmap (parallel map)
pmap maps a function across multiple devices (GPUs/CPUs) in parallel
"""

import jax
import jax.numpy as jnp
import equinox as eqx

print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {jax.device_count()}\n")

def square(x):
    """Simple function to square a number"""
    return x ** 2

# Regular vmap: vectorizes across batch dimension (runs on single device)
data = jnp.array([1.0, 2.0, 3.0, 4.0])
print("=== Regular vmap (single device) ===")
print(f"Input: {data}")
result_vmap = jax.vmap(square)(data)
print(f"Output: {result_vmap}\n")

# pmap: splits data across devices and runs in parallel
# Reshape data to [num_devices, batch_per_device]
num_devices = jax.device_count()
data_reshaped = data.reshape(num_devices, -1)  # Shape: [devices, items_per_device]
print(f"=== pmap (parallel across {num_devices} devices) ===")
print(f"Input shape: {data_reshaped.shape}")
print(f"Input:\n{data_reshaped}")
result_pmap = jax.pmap(square)(data_reshaped)
print(f"Output shape: {result_pmap.shape}")
print(f"Output:\n{result_pmap}\n")

# ============================================
# 2. Simple neural network with Equinox
# ============================================

class SimpleModel(eqx.Module):
    """Tiny neural network"""
    linear: eqx.nn.Linear
    
    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 1, key=key)
    
    def __call__(self, x):
        return self.linear(x)

# Create model
key = jax.random.PRNGKey(0)
model = SimpleModel(key)

# Create batch of data [batch_size, features]
batch = jax.random.normal(key, (8, 2))

print("=== Model inference ===")
print(f"Batch shape: {batch.shape}")

# Regular vmap for batch processing
batched_model = jax.vmap(model)
output_vmap = batched_model(batch)
print(f"vmap output shape: {output_vmap.shape}")

# pmap for parallel processing across devices
# Reshape: [num_devices, batch_per_device, features]
batch_parallel = batch.reshape(num_devices, -1, 2)
print(f"\nBatch reshaped for pmap: {batch_parallel.shape}")

# Need to replicate model across devices first
model_replicated = jax.device_put_replicated(model, jax.devices())
print(f"Model replicated across devices")

# Apply pmap (processes each device's data in parallel)
pmap_model = jax.pmap(jax.vmap(lambda m, x: m(x), in_axes=(None, 0)))
output_pmap = pmap_model(model_replicated, batch_parallel)
print(f"pmap output shape: {output_pmap.shape}")
print(f"pmap output:\n{output_pmap}\n")

# ============================================
# KEY TAKEAWAYS
# ============================================
print("=" * 50)
print("KEY POINTS:")
print("=" * 50)
print("1. vmap: Vectorizes function (single device)")
print("2. pmap: Parallelizes across multiple devices")
print("3. Data shape for pmap: [num_devices, batch_per_device, ...]")
print("4. Each device processes its chunk independently")
print("5. Use jax.device_put_replicated() for models/params")
print("\nTry modifying:")
print("- Change the data size")
print("- Add print statements inside functions (won't work with jit!)")
print("- Try different model architectures")