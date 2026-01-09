import jax
import jax.numpy as jnp
import jax.profiler

print(f"Devices: {jax.devices()}")

with jax.profiler.trace("./profile_test_trace"):
    x = jnp.arange(10000000)
    y = jnp.sum(x ** 2)
    y.block_until_ready()
