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

"""Feature layer for periodic boundary conditions.

See Cassella, G., Sutterud, H., Azadi, S., Drummond, N.D., Pfau, D.,
Spencer, J.S. and Foulkes, W.M.C., 2022. Discovering Quantum Phase Transitions
with Fermionic Neural Networks. arXiv preprint arXiv:2202.05183.
"""

from typing import Any, Optional, Sequence, Tuple

import chex
from ferminet import networks
import jax.numpy as jnp


def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
  """Returns the periodic norm of a set of vectors.

  Args:
    metric: metric tensor in fractional coordinate system, A.T A, where A is the
      lattice vectors.
    scaled_r: vectors in fractional coordinates of the lattice cell, with
      trailing dimension ndim, to compute the periodic norm of.
  """
  return jnp.sqrt(periodic_norm_sq(metric, scaled_r))


def periodic_norm_sq(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
  """Returns the squared periodic norm of a set of vectors without sqrt."""
  
  chex.assert_rank(metric, expected_ranks=2)
  a = (1 - jnp.cos(2 * jnp.pi * scaled_r))
  b = jnp.sin(2 * jnp.pi * scaled_r)
  cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
  sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
  return (1 / (2 * jnp.pi) ** 2) * (cos_term + sin_term)

def make_pbc_feature_layer(
    natoms: int,
    nspins: Optional[Sequence[int]] = None,
    ndim: int = 3,
    rescale_inputs: bool = False,
    lattice: Optional[jnp.ndarray] = None,
    include_r_ae: bool = True,
    interlayer_distance: float = 0.0,
    **kwargs: Any,
) -> networks.FeatureLayer:
  """Returns the init and apply functions for periodic features.

  Args:
      natoms: number of atoms.
      nspins: tuple of the number of particles in each spin/species channel,
        e.g. ``(n_up, n_down)`` or ``(n_e_up, n_e_down, n_h_up, n_h_down)``.
      ndim: dimension of the system.
      rescale_inputs: If true, rescales r_ae for stability. Note that unlike in
        the OBC case, we do not rescale r_ee as well.
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).
      include_r_ae: Flag to enable electron-atom distance features. Set to False
        to avoid cusps with ghost atoms in, e.g., homogeneous electron gas. If
        not needed, set cfg.network.make_feature_layer_kwargs['include_r_ae'] =
        False
      interlayer_distance: Optional out-of-plane bilayer separation ``d >= 0``.
        When positive and ``nspins`` has length 2 or 4, interlayer electron-hole
        pair distance features use the 3D distance
        ``sqrt(r_periodic^2 + d^2)``.
      **kwargs: Unused additional keyword arguments.
  """

  del kwargs

  if lattice is None:
    lattice = jnp.eye(ndim)

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice
  
  # z_sq is just a n_total*n_total matrix that is d**2 between all pairs of particles 
  # that are on different layers, and 0 between pairs of particles on the same layer
  z_sq = None
  if interlayer_distance > 0.0 and nspins is not None:
    if len(nspins) == 4:
      n_lower = int(nspins[0]) + int(nspins[1])
    elif len(nspins) == 2:
      n_lower = int(nspins[0])
    else:
      raise ValueError(
          'interlayer_distance > 0 in make_pbc_feature_layer requires '
          f'len(nspins) in (2, 4), got nspins={nspins}.'
      )
    n_total = sum(int(x) for x in nspins)
    z = float(interlayer_distance) * jnp.ones((n_total, n_total))
    z = z.at[:n_lower, :n_lower].set(0.0)
    z = z.at[n_lower:, n_lower:].set(0.0)
    z_sq = z**2

  def init() -> Tuple[Tuple[int, int], networks.Param]:
    if include_r_ae:
      return (natoms * (2 * ndim + 1), 2 * ndim + 1), {}
    else:
      return (natoms * (2 * ndim), 2 * ndim + 1), {}

  def apply(
      ae, r_ae, ee, r_ee, **params: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    del params
    # One e features in phase coordinates, (s_ae)_i = k_i . ae
    s_ae = jnp.einsum('il,jkl->jki', reciprocal_vecs, ae)
    # Two e features in phase coordinates
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)
    # Periodized features
    ae = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ae), jnp.cos(2 * jnp.pi * s_ae)), axis=-1)
    ee = jnp.concatenate(
        (jnp.sin(2 * jnp.pi * s_ee), jnp.cos(2 * jnp.pi * s_ee)), axis=-1)
    # Distance features defined on orthonormal projections
    r_ae = periodic_norm(lattice_metric, s_ae)
    if rescale_inputs:
      r_ae = jnp.log(1 + r_ae)
    # Don't take gradients through |0|
    n = ee.shape[0]
    # s_ee += jnp.eye(n)[..., None]
    r_ee_sq = periodic_norm_sq(lattice_metric, s_ee) #Avoided adding 1 to periodic fractional coordinate (which was used in OBC) since shifting s by 1 will make periodic coordinates back to 0
    eye = jnp.eye(n, dtype=r_ee_sq.dtype)
    
    if z_sq is not None:
      # r_ee_sq = periodic_norm_sq(lattice_metric, s_ee)
      # r_ee is a periodic scalar distance feature between particle i and j
      # So here essentially we're replacing the original periodic distance feature r_ij with 
      # sqrt(r_ij^2 + d^2) where d is the actual interlayer distance
      # Rationale is that firstly the original periodic distance feature Taylor expands to the real distance at small in-plane separation r, and here we preserved that property
      # Secondly this also removes the cusp when two particles on different layers have vanishing in-plane separation
      r_ee = jnp.sqrt(r_ee_sq + z_sq.astype(r_ee_sq.dtype) + eye) * (
          1.0 - eye
      ) 
    else:
      r_ee = jnp.sqrt(r_ee_sq + eye) * (1.0 - eye)

    if include_r_ae:
      ae_features = jnp.concatenate((r_ae[..., None], ae), axis=2)
    else:
      ae_features = ae
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    ee_features = jnp.concatenate((r_ee[..., None], ee), axis=2)
    return ae_features, ee_features

  return networks.FeatureLayer(init=init, apply=apply)
