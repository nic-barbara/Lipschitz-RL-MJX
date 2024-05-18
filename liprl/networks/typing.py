import jax
import numpy as np
from typing import Union, Tuple, Any, Protocol

"""
Define custom types from jax.typing and flax.typing
Older versions of jax/flax don't have these exposed.
"""

# From jax.typing
DType = np.dtype

class SupportsDType(Protocol):
  @property
  def dtype(self) -> DType: ...
  
DTypeLike = Union[
  str,            # like 'float32', 'int32'
  type[Any],      # like np.float32, np.int32, float, int
  np.dtype,       # like np.dtype('float32'), np.dtype('int32')
  SupportsDType,  # like jnp.float32, jnp.int32
]

# Edited from flax.typing
Dtype = Union[DTypeLike, Any]

PrecisionLike = Union[
  None,
  str,
  jax.lax.Precision,
  Tuple[str, str],
  Tuple[jax.lax.Precision, jax.lax.Precision],
]
