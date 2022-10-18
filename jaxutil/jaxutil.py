import jax.numpy as jnp
import jax

def jax_index_add_dim0(target, index, source):
  """Replace torch.Tensor.index_add_ method with jax equivalent.
  """
  return target.at[index].add(source)

def jax_flatten(input, start_dim=0, end_dim=-1):
  """Replace torch.Tensor.flatten method with jax equivalent.
  """
  if end_dim == -1:
    return input.reshape(-1)
  new_shape = []
  combined = 1
  # Flatten only starting from start_dim to end_dim
  for si, s in enumerate(input.shape):
    if si < start_dim or si > end_dim:
      new_shape.append(s)
    elif si >= start_dim and si < end_dim:
      combined *= s
    elif si == end_dim:
      new_shape.append(combined*s)
    else:
      raise ValueError("jax_flatten input shape is illegal")
  return input.reshape(tuple(new_shape))

def jax_nonzero(input):
  """Replace torch.Tensor.nonzero method with jax equivalent.
  """
  return jnp.stack(input.nonzero()).T

def jax_index_select(input, index, axis):
  """Replace torch.Tensor.index_select method with jax equivalent.
  """
  return jnp.take(input, index, axis)

def jax_unique(input):
  """Replace torch.Tensor.unique_consecutive method with jax equivalent.
  """
  return jnp.unique(input, return_inverse=False, return_counts=True)

def jax_sort(input):
  """Replace torch.Tensor.sort method with jax equivalent.
  """
  return jnp.sort(input), jnp.argsort(input)

def jax_div(x, y):
  """Replace torch.Tensor.div method with jax equivalent.
  """
  return jnp.asarray(jnp.fix(jnp.divide(x, y)), dtype=jnp.int64)

def jax_repeat_interleave(input):
  """Replace torch.Tensor.repeat_interleave method with jax equivalent.
  """
  return jnp.repeat(jnp.arange(input.size), input)

def jax_unsqueeze(input, dim):
  """Replace torch.Tensor.unsqueeze method with jax equivalent.
  """
  return jnp.expand_dims(input, axis=dim)

def jax_tril_indices(row, col, offset=0):
  """Replace torch.Tensor.tril_indices method with jax equivalent.
  """
  return jnp.stack(jnp.tril_indices(row, offset, col))

def jax_expand(input, repeats, axis):
  """Replace torch.Tensor.expand method with jax equivalent.
  """
  return jnp.repeat(input, repeats=repeats, axis=axis)

def jax_norm(input, ord, dim):
  """Replace torch.Tensor.norm method with jax equivalent.
  """
  return jnp.linalg.norm(input, ord=ord, axis=dim)

def jax_cumsum_from_zero(input_: jnp.ndarray, dim=0) -> jnp.ndarray:
  """Replace torch.Tensor.cumsum method with jax equivalent.
  """
  return jnp.concatenate(
    (jnp.zeros((1,) + input_.shape[1:]), 
    jnp.cumsum(input_[:-1], axis=dim)))

def jax_clamp(input, min=None, max=None):
  """Replace torch.Tensor.clamp method with jax equivalent.
  """
  return jnp.clip(input, a_min=min, a_max=max)

def jax_unbind(input, dim=0):
  """Replace torch.Tensor.unbind method with jax equivalent.
  """
  # Adapted from jakevdp: https://github.com/google/jax/discussions/11028
  return [jax.lax.index_in_dim(input, i, axis=dim, keepdims=False) for i in range(input.shape[dim])]

def jax_masked_scatter_(input, mask, source):
  """Replace torch.Tensor.masked_scatter_ method with jax equivalent.
  """
  return input.at[mask.nonzero()[0]].set(source)