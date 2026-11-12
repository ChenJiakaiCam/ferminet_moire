# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""Multiplicative envelopes appropriate for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

import itertools
from typing import Mapping, Optional, Sequence, Tuple, Union

from ferminet import envelopes

import jax.numpy as jnp
import numpy as np


def migrate_multiwave_envelope_params(
    envelope_params: Sequence[Mapping[str, jnp.ndarray]],
) -> Tuple[Sequence[Mapping[str, jnp.ndarray]], bool]:
  migrated = []
  changed = False
  for block in envelope_params:
    if isinstance(block, Mapping) and 'linear_sigma' not in block and 'sigma' in block:
      new_block = dict(block)
      new_block['linear_sigma'] = new_block.pop('sigma') ** 2.0
      migrated.append(new_block)
      changed = True
    else:
      migrated.append(block)
  if isinstance(envelope_params, tuple):
    return tuple(migrated), changed
  return migrated, changed

def _independent_wave_row_indices(kpoints: jnp.ndarray) -> jnp.ndarray:
  """Returns row indices in ``[cos(k.r), sin(k.r)]`` for independent modes.

  In ``make_multiwave_envelope``, ``waves`` concatenates ``cos(k_p . r)``
  (rows ``0 .. nk - 1``) followed by ``sin(k_p . r)`` (rows ``nk .. 2*nk - 1``),
  while ``make_kpoints`` includes both ``+k`` and ``-k`` for each non-zero wave
  vector. To avoid duplicate ``cos(-k.r) == cos(k.r)`` columns and the
  identically zero ``sin(0.r) == 0`` row:
    - For ``k_p == 0``, selects row ``p`` (``cos(0) = 1``).
    - For the first occurrence ``p`` of a ``+-k`` pair, selects row ``p``
      (``cos(k_p . r)``).
    - For the partner ``q > p`` with ``k_q == -k_p``, selects row ``nk + p``
      (``sin(k_p . r)``).
  """
  kpts = np.asarray(kpoints, dtype=np.float64)
  nk = kpts.shape[0]
  norms = np.linalg.norm(kpts, axis=-1)
  row_indices = []
  for p in range(nk):
    if norms[p] <= 1e-12:
      row_indices.append(p)
      continue
    partner = None
    for q in range(p):
      if norms[q] > 1e-12 and np.linalg.norm(kpts[p] + kpts[q]) <= 1e-5 * norms[p]:
        partner = q
        break
    if partner is None:
      row_indices.append(p)
    else:
      row_indices.append(nk + partner)
  return jnp.asarray(row_indices, dtype=jnp.int32)


def make_multiwave_envelope(
    kpoints: jnp.ndarray,
    distinct_kpoints_init: bool = False,
) -> envelopes.Envelope:
  """Returns an oscillatory envelope.

  Envelope consists of a sum of truncated 3D Fourier series, one centered on
  each atom, with Fourier frequencies given by kpoints:

    sigma_{2i}*cos(kpoints_i.r_{ae}) + sigma_{2i+1}*sin(kpoints_i.r_{ae})

  By default (``distinct_kpoints_init=False``), initialization sets the
  coefficient of the first term (``kpoints[0]``) in each series to 1 and all
  other coefficients to 0. When ``distinct_kpoints_init=True``, initialization
  sets a single 1 in each column of ``linear_sigma`` at a distinct linearly
  independent Fourier mode row (cycling through the ``nk`` independent
  ``1, cos(k.r), sin(k.r)`` modes of ``kpoints``), and 0 elsewhere.

  Args:
    kpoints: Reciprocal lattice vectors of terms included in the Fourier series.
      Shape (nkpoints, ndim)
      distinct_kpoints_init: If True, initialize each column of ``linear_sigma``
      with a single 1 at a distinct independent Fourier mode row instead of
      placing all 1s in row 0. Defaults to False for backward compatibility.

  Returns:
    An instance of ferminet.envelopes.Envelope with apply_type
    envelopes.EnvelopeType.PRE_DETERMINANT
  """
  
  mode_rows = (
      _independent_wave_row_indices(kpoints) if distinct_kpoints_init else None
  )

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    """See ferminet.envelopes.EnvelopeInit."""
    del natom, ndim  # unused
    params = []
    nk = kpoints.shape[0]
    for output_dim in output_dims:
      linear_sigma = jnp.zeros((2 * nk, output_dim))
      if distinct_kpoints_init and mode_rows is not None:
        col_indices = jnp.arange(output_dim, dtype=jnp.int32)
        row_indices = mode_rows[col_indices % nk]
        linear_sigma = linear_sigma.at[row_indices, col_indices].set(1.0)
      else:
        linear_sigma = linear_sigma.at[0, :].set(1.0)
      params.append({'linear_sigma': linear_sigma})
      
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            **kwargs: jnp.ndarray) -> jnp.ndarray:
    """See ferminet.envelopes.EnvelopeApply."""
    del r_ae, r_ee  # unused
    if 'linear_sigma' not in kwargs:
      if 'sigma' in kwargs:
        raise ValueError(
            "Found legacy multiwave envelope parameter 'sigma' instead of "
            "'linear_sigma'. Migrate checkpoint parameters with "
            "ferminet.pbc.envelopes.migrate_multiwave_envelope_params or "
            "ferminet.train.migrate_checkpoint_params before applying the "
            "network."
        )
      raise KeyError("Missing envelope parameter 'linear_sigma'.")
    phase_coords = ae @ kpoints.T
    waves = jnp.concatenate((jnp.cos(phase_coords), jnp.sin(phase_coords)),
                            axis=2)
    # env = waves @ (kwargs['sigma'] ** 2.0)
    env = waves @ kwargs['linear_sigma']
    return jnp.sum(env, axis=1)

  return envelopes.Envelope(envelopes.EnvelopeType.PRE_DETERMINANT, init, apply)


def make_kpoints(
    lattice: Union[np.ndarray, jnp.ndarray],
    spins: Tuple[int, int],
    min_kpoints: Optional[int] = None,
    *,
    ndim: int = 3,
) -> jnp.ndarray:
  """Generates an array of reciprocal lattice vectors.

  Args:
    lattice: Matrix whose columns are the primitive lattice vectors of the
      system, shape (ndim, ndim).
    spins: Tuple of the number of spin-up and spin-down electrons.
    min_kpoints: If specified, the number of kpoints which must be included in
      the output. The number of kpoints returned will be the first filled shell
      which is larger than this value. Defaults to None, which results in
      min_kpoints == sum(spins).
    ndim: Number of dimensions.

  Raises:
    ValueError: Fewer kpoints requested by min_kpoints than number of
      electrons in the system.

  Returns:
    jnp.ndarray, shape (nkpoints, ndim), an array of reciprocal lattice
      vectors sorted in ascending order according to length.
  """
  rec_lattice = 2 * jnp.pi * jnp.linalg.inv(lattice)
  # Calculate required no. of k points
  if min_kpoints is None:
    min_kpoints = sum(spins)
  elif min_kpoints < sum(spins):
    raise ValueError(
        'Number of kpoints must be equal or greater than number of electrons')

  dk = 1 + 1e-5
  # Generate ordinals of the lowest min_kpoints kpoints
  max_k = int(jnp.ceil(min_kpoints * dk) ** (1 / ndim))
  ordinals = sorted(range(-max_k, max_k+1), key=abs)
  ordinals = jnp.asarray(list(itertools.product(ordinals, repeat=ndim)))

  kpoints = ordinals @ rec_lattice
  kpoints = jnp.asarray(sorted(kpoints, key=jnp.linalg.norm))
  k_norms = jnp.linalg.norm(kpoints, axis=1)

  return kpoints[k_norms <= k_norms[min_kpoints - 1] * dk]
