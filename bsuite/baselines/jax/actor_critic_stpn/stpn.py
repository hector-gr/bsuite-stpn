import math
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
# from haiku._src.recurrent import RNNCore
import haiku as hk

# Following https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
class STPNState(NamedTuple):
  """A STPN core state consists of synaptic and neuronal states.
  Attributes:
    hidden: Hidden state.
    synaptic: Synaptic state.
  """
  hidden: jax.Array
  synaptic: jax.Array


class STPN(hk.RNNCore):
  r"""Short-term plasticity neuron (STPN) RNN core.
  The implementation is based on :cite:`rodriguez2022short`. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, S_{t-1})` the core
  computes
  .. math::
     \begin{array}{ll}
     h_t = \tanh((W + S_t)[x_t, h_{t-1}] + b) \\
     S_t = \lambda S_{t-1} + \gamma h_t x_t
     \end{array}
  where :math:`S_t`, is synaptic state.
  The output is equal to the new hidden, :math:`h_t`.
  Notes:
    The implementation is based on :cite:`rodriguez2022short`.
  """

  def __init__(self, hidden_size: int, name: Optional[str] = None):
    """Constructs a STPN.
    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(
      self,
      inputs: jax.Array,
      prev_state: STPNState,
  ) -> Tuple[jax.Array, STPNState]:
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("STPN input must be rank-1 or rank-2.")
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
    # neuronal update
    init_k = 1 / math.sqrt(self.hidden_size)
    b =  hk.get_parameter("b", shape=[self.hidden_size], dtype=inputs.dtype, init=hk.initializers.RandomUniform(-init_k, init_k)) # TODO: init zero ?
    w =  hk.get_parameter("w", shape=[self.hidden_size, self.hidden_size + inputs.shape[1]], dtype=inputs.dtype, init=hk.initializers.RandomUniform(-init_k, init_k)) # hk.Linear(self.hidden_size, with_bias=False, w_init=hk.initializers.RandomUniform(-init_k, init_k))(x_and_h) + 
    G = w + prev_state.synaptic
    y = jnp.einsum('bf,bhf->bh', x_and_h, G)
    norm = jnp.linalg.norm(G, axis=2, keepdims=True)
    h = jnp.tanh(y/norm.squeeze(-1) + b)
    # synaptic update
    l = hk.get_parameter("l", shape=[self.hidden_size, self.hidden_size + inputs.shape[1]], dtype=inputs.dtype, init=hk.initializers.RandomUniform(0, 1))
    g = hk.get_parameter("g", shape=[self.hidden_size, self.hidden_size + inputs.shape[1]], dtype=inputs.dtype, init=hk.initializers.RandomUniform(-0.001*init_k, 0.001*init_k)) # TODO: init normal ?
    s = l * prev_state.synaptic / norm + g * jnp.outer(h, x_and_h)
    
    return h, STPNState(h, s)

  def initial_state(self, batch_size: Optional[int], input_size: Optional[int]) -> STPNState:
    state = STPNState(hidden=jnp.zeros([self.hidden_size]),
                      cell=jnp.zeros([self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    if input_size is not None:
      state.synaptic = add_input(state.synaptic, input_size + self.hidden_size) # TODO: the tree_map is not needed since this is directly a jnp.array.
    return state

def add_batch(nest, batch_size: Optional[int]):
  """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
  return jax.tree_util.tree_map(broadcast, nest)

def add_input(nest, input_size: Optional[int]):
  """Adds an input dimension at last axis to the leaves of a nested structure."""
  broadcast = lambda x: jnp.broadcast_to(x, x.shape + (input_size,))
  return jax.tree_util.tree_map(broadcast, nest)