# Copyright 2020 DeepMind Technologies Limited.
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
# limitations under the License.

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from ferminet import network_blocks
import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  BILAYER = enum.auto()


def safe_norm(ee: jnp.ndarray):
  n = ee.shape[0]
  return (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))
  )[
      ..., None
  ]  # extra dim for consistency with previous code versions


def make_periodic_r_ee(lattice: jnp.ndarray):
  """Util function for transforming r_ee into the periodic version.

  Some of this could be refactored with ferminet.pbc.feature_layer

  Args:
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).

  Returns:
      Function mapping electron-electron displacement vectors (ee) to periodic
      distances of shape (..., 1).
  """

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice

  def apply(ee: jnp.ndarray):
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)

    n = ee.shape[0]
    s_ee += jnp.eye(n)[..., None]

    a = 1 - jnp.cos(2 * jnp.pi * s_ee)
    b = jnp.sin(2 * jnp.pi * s_ee)
    cos_term = jnp.einsum('...m,mn,...n->...', a, lattice_metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, lattice_metric, b)
    periodic_r_ee = (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

    periodic_r_ee = periodic_r_ee * (1.0 - jnp.eye(n))
    return periodic_r_ee[..., None]

  return apply


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    ndim: int,
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 1 / (ndim + 1), params['ee_par'])  # pyrefly: ignore[bad-argument-type, bad-index]
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(
        jastrow_fun(r_ees[0][1], 1 / (ndim - 1), params['ee_anti'])  # pyrefly: ignore[bad-argument-type, bad-index]
    )
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(lattice: jnp.ndarray | None = None, ndim: int = 3):
  """Creates a Jastrow factor for electron-electron cusps."""

  # If working in PBC, use periodic distance for the Jastrow
  if lattice is not None:
    norm = make_periodic_r_ee(lattice)
  else:
    norm = safe_norm

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    r_ee = norm(ee)
    return _jastrow_ee(
        r_ee, params, nspins, ndim, jastrow_fun=simple_ee_cusp_fun
    )

  return init, apply


def make_bilayer_jastrow(
    lattice: jnp.ndarray | None = None,
    ndim: int = 2,
    mass_e: float = 1.0,
    mass_h: float = 1.0,
    dielectric_constant: float = 1.0,
    interlayer_distance: float = 1.0,
):
  """Creates a Jastrow factor for 2- or 4-channel electron-hole bilayers.

  For effective mass ``m_alpha`` and screened Coulomb repulsion
  ``1 / (epsilon_r * r)`` in ``ndim`` dimensions, the short-distance Kato cusp
  coefficients are:
    - identical spin/flavour in layer ``alpha``:
      ``m_alpha / (epsilon_r * (ndim + 1))`` (i.e. ``m_alpha / (3 * epsilon_r)``
      in 2D),
    - different spin/flavour in layer ``alpha``:
      ``m_alpha / (epsilon_r * (ndim - 1))`` (i.e. ``m_alpha / epsilon_r`` in
      2D),
    - interlayer electron-hole pairs: ``0`` when ``interlayer_distance > 0``
      (since ``-1 / sqrt(r^2 + d^2)`` has no cusp at ``r = 0``), and
      ``-2 * m_e * m_h / ((m_e + m_h) * epsilon_r * (ndim - 1))`` when
      ``interlayer_distance == 0``.
  """
  if mass_e <= 0.0 or mass_h <= 0.0:
    raise ValueError(
        f'Effective masses must be positive, got mass_e={mass_e}, '
        f'mass_h={mass_h}.'
    )
  if dielectric_constant <= 0.0:
    raise ValueError(
        'dielectric_constant must be positive, got '
        f'{dielectric_constant}.'
    )
  if interlayer_distance < 0.0:
    raise ValueError(
        'interlayer_distance must be >= 0, got '
        f'{interlayer_distance}.'
    )
  if lattice is not None:
    norm = make_periodic_r_ee(lattice)
  else:
    norm = safe_norm

  denom_same = float(ndim + 1.0)
  denom_diff = float(max(ndim - 1.0, 1.0))
  scale_e = float(mass_e) / float(dielectric_constant)
  scale_h = float(mass_h) / float(dielectric_constant)

  indis_e = scale_e / denom_same
  dis_e = scale_e / denom_diff
  indis_h = scale_h / denom_same
  dis_h = scale_h / denom_diff
  if interlayer_distance > 0.0:
    eh_cusp = 0.0
  else:
    two_mu_eh = 2.0 * float(mass_e) * float(mass_h) / (
        float(mass_e) + float(mass_h)
    )
    eh_cusp = -(two_mu_eh / float(dielectric_constant)) / denom_diff

  cusp_matrix_4 = jnp.array([
      [indis_e, dis_e, eh_cusp, eh_cusp],
      [dis_e, indis_e, eh_cusp, eh_cusp],
      [eh_cusp, eh_cusp, indis_h, dis_h],
      [eh_cusp, eh_cusp, dis_h, indis_h],
  ])
  cusp_matrix_2 = jnp.array([
      [indis_e, eh_cusp],
      [eh_cusp, indis_h],
  ])

  def jastrow_fun(
      r: jnp.ndarray, cusp: float | jnp.ndarray, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    return {'ee': jnp.ones((4, 4))}

  def apply(
      ee: jnp.ndarray,
      params: ParamTree,
      nspins: Sequence[int],
  ) -> jnp.ndarray:
    if len(nspins) == 4:
      cusp_matrix = cusp_matrix_4
    elif len(nspins) == 2:
      cusp_matrix = cusp_matrix_2
    else:
      raise ValueError(
          f'Bilayer Jastrow requires 2 or 4 spin channels, got {nspins}.'
      )

    if interlayer_distance > 0.0:
      # When layers are separated (d > 0), interlayer e-h pairs have zero cusp
      # and can coincide in the 2D plane. Evaluate norms only on intralayer
      # blocks so coincident e-h positions never differentiate through sqrt(0).
      if len(nspins) == 2:
        e_spins = (int(nspins[0]),)
        h_spins = (int(nspins[1]),)
        h_offset = 1
      else:
        e_spins = (int(nspins[0]), int(nspins[1]))
        h_spins = (int(nspins[2]), int(nspins[3]))
        h_offset = 2
      n_e = sum(e_spins)
      n_h = sum(h_spins)

      total = jnp.asarray(0.0, dtype=ee.dtype)
      if n_e >= 2:
        r_e = norm(ee[:n_e, :n_e])
        split_e = network_blocks.array_partitions(e_spins)
        blocks_e = [
            jnp.split(b, split_e, axis=1)
            for b in jnp.split(r_e, split_e, axis=0)
        ]
        for i, ni in enumerate(e_spins):
          for j in range(i, len(e_spins)):
            nj = e_spins[j]
            if i == j:
              if ni < 2:
                continue
              r = blocks_e[i][i][jnp.triu_indices(ni, k=1)]
            else:
              if ni == 0 or nj == 0:
                continue
              r = blocks_e[i][j].reshape(-1)
            total += jnp.sum(
                jastrow_fun(r, cusp_matrix[i, j], params['ee'][i, j])  # pyrefly: ignore[bad-index]
            )
      if n_h >= 2:
        r_h = norm(ee[n_e:, n_e:])
        split_h = network_blocks.array_partitions(h_spins)
        blocks_h = [
            jnp.split(b, split_h, axis=1)
            for b in jnp.split(r_h, split_h, axis=0)
        ]
        for i, ni in enumerate(h_spins):
          for j in range(i, len(h_spins)):
            nj = h_spins[j]
            if i == j:
              if ni < 2:
                continue
              r = blocks_h[i][i][jnp.triu_indices(ni, k=1)]
            else:
              if ni == 0 or nj == 0:
                continue
              r = blocks_h[i][j].reshape(-1)
            total += jnp.sum(
                jastrow_fun(
                    r,
                    cusp_matrix[h_offset + i, h_offset + j],
                    params['ee'][h_offset + i, h_offset + j],  # pyrefly: ignore[bad-index]
                )
            )
      return total

    r_ee = norm(ee)
    split_idx = network_blocks.array_partitions(nspins)
    row_blocks = jnp.split(r_ee, split_idx, axis=0)
    blocks = [jnp.split(b, split_idx, axis=1) for b in row_blocks]

    total = jnp.asarray(0.0, dtype=ee.dtype)
    for i, ni in enumerate(nspins):
      for j in range(i, len(nspins)):
        nj = nspins[j]
        if i == j:
          if ni < 2:
            continue
          r = blocks[i][i][jnp.triu_indices(ni, k=1)]
        else:
          if ni == 0 or nj == 0:
            continue
          r = blocks[i][j].reshape(-1)
        total += jnp.sum(
            jastrow_fun(r, cusp_matrix[i, j], params['ee'][i, j])  # pyrefly: ignore[bad-index]
        )
    return total

  return init, apply


def get_jastrow(
    jastrow: JastrowType,
    lattice: jnp.ndarray | None = None,
    ndim: int = 3,
    jastrow_kwargs: Mapping[str, Any] | None = None,
):
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(lattice, ndim)
  elif jastrow == JastrowType.BILAYER:
    kwargs = dict(jastrow_kwargs) if jastrow_kwargs is not None else {}
    jastrow_init, jastrow_apply = make_bilayer_jastrow(
        lattice, ndim, **kwargs
    )
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
