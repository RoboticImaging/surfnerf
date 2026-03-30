import jax.numpy as jnp
import jax

@jax.custom_jvp
def gradientScaling(origins, gaussians, vals):
  return vals



@gradientScaling.defjvp
def lgradientScaling_jvp(primals, tangents):
  origins, gaussians, vals = primals
  origins_dot, gaussians_dot, vals_dot = tangents
  ans = gradientScaling(origins, gaussians, vals)
  scaling_values = jnp.square(jnp.linalg.norm(gaussians[0] - origins[...,None,:], axis=-1)).clip(0,1)
  if scaling_values.ndim < vals.ndim:
    scaling = scaling_values[...,None]
  else:
    scaling = scaling_values
  ans_dot=vals_dot * scaling
  return ans, ans_dot 