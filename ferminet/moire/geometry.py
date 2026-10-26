# Copyright 2026 DeepMind Technologies Limited.
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

"""2D periodic geometry utilities for moire bilayer systems.

All lattice matrices follow FermiNet's column-vector convention: a 2D lattice
matrix ``L`` has shape ``(2, 2)`` where ``L[:, 0]`` and ``L[:, 1]`` are the two
direct lattice vectors ``a_1`` and ``a_2``. Consequently, for a row vector
``r = (x, y)`` or array of positions with trailing dimension ``2``:

  - Fractional coordinates: ``s = r @ inv(L).T``
  - Cartesian coordinates: ``r = s @ L.T``
  - Reciprocal lattice vectors (rows): ``B = 2 * pi * inv(L)`` so that
    ``B @ L = 2 * pi * I``.
"""

from typing import Sequence, Tuple, Union

from ferminet import mcmc
import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]


def layer_particle_counts(nspins: Sequence[int]) -> Tuple[int, int]:
  """Resolves total (n_electrons, n_holes) from 4- or 2-channel ``nspins``.

  Supports either:
    - 4 fermion channels ``(n_e_up, n_e_down, n_h_up, n_h_down)`` (spinful
      electrons and spinful holes), or
    - 2 fermion channels ``(n_electrons, n_holes)`` (spin-polarized per layer).

  Args:
    nspins: Sequence of non-negative integer particle counts per channel.

  Returns:
    Tuple ``(n_electrons, n_holes)`` of total carrier counts per layer.

  Raises:
    ValueError: If ``len(nspins)`` is not 2 or 4, or if any count is negative.
  """
  counts = [int(n) for n in nspins]
  if any(n < 0 for n in counts):
    raise ValueError(f'All entries in nspins must be >= 0, got {nspins}.')
  if len(counts) == 4:
    return counts[0] + counts[1], counts[2] + counts[3]
  if len(counts) == 2:
    return counts[0], counts[1]
  raise ValueError(
      'Expected nspins to have length 4 (n_e_up, n_e_down, n_h_up, n_h_down) '
      f'or length 2 (n_e, n_h), got {nspins}.'
  )


def triangular_lattice(lattice_constant: float = 1.0) -> np.ndarray:
  """Returns a 2x2 triangular primitive lattice matrix (columns are vectors).

  The primitive vectors are ``a_1 = a_M * (sqrt(3)/2, -1/2)`` and
  ``a_2 = a_M * (0, 1)``, matching the canonical 120-degree hexagonal/triangular
  moire basis.

  Args:
    lattice_constant: Moire lattice constant ``a_M`` in Bohr (or dimensionless
      units).

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are ``a_1`` and ``a_2``.
  """
  if lattice_constant <= 0.0:
    raise ValueError(
        f'lattice_constant must be positive, got {lattice_constant}.'
    )
  basis = np.array(
      [
          [np.sqrt(3.0) / 2.0, 0.0],
          [-0.5, 1.0],
      ],
      dtype=np.float64,
  )
  return float(lattice_constant) * basis


def validate_lattice_2d(lattice: Array, name: str = 'lattice') -> np.ndarray:
  """Validates that ``lattice`` is a non-singular 2x2 matrix."""
  lat = np.asarray(lattice, dtype=np.float64)
  if lat.shape != (2, 2):
    raise ValueError(f'{name} must have shape (2, 2), got {lat.shape}.')
  if not np.all(np.isfinite(lat)):
    raise ValueError(f'{name} must have finite entries.')
  area = abs(np.linalg.det(lat))
  if area <= 1e-12:
    raise ValueError(f'{name} must be non-singular, got det={area}.')
  return lat


def supercell_lattice(
    primitive_lattice: Array,
    supercell_matrix: Array,
) -> np.ndarray:
  """Constructs a 2D simulation supercell from a primitive lattice.

  Following the column-vector convention, ``L_super = L_prim @ S``, where each
  column of the integer matrix ``S`` specifies the integer combination of
  primitive lattice vectors forming a supercell lattice vector.

  Args:
    primitive_lattice: Shape ``(2, 2)`` matrix whose columns are primitive
      lattice vectors.
    supercell_matrix: Shape ``(2, 2)`` integer matrix with non-zero determinant.

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are the supercell lattice
    vectors.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sc = np.asarray(supercell_matrix, dtype=np.float64)
  if sc.shape != (2, 2):
    raise ValueError(
        f'supercell_matrix must have shape (2, 2), got {sc.shape}.'
    )
  if not np.allclose(sc, np.round(sc), atol=1e-8):
    raise ValueError('supercell_matrix must have integer entries.')
  sc_int = np.round(sc)
  det = abs(round(float(np.linalg.det(sc_int))))
  if det <= 0:
    raise ValueError('supercell_matrix must have non-zero determinant.')
  return prim @ sc_int


def validate_commensurate(
    primitive_lattice: Array,
    simulation_lattice: Array,
    atol: float = 1e-5,
) -> np.ndarray:
  """Checks that ``simulation_lattice`` is an integer supercell of ``primitive_lattice``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).
    simulation_lattice: Shape ``(2, 2)`` simulation supercell lattice (columns).
    atol: Tolerance for integer check on ``inv(L_prim) @ L_sim``.

  Returns:
    The rounded ``(2, 2)`` integer supercell matrix ``S = inv(L_prim) @ L_sim``.

  Raises:
    ValueError: If ``simulation_lattice`` is not commensurate with
      ``primitive_lattice``.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sim = validate_lattice_2d(simulation_lattice, name='simulation_lattice')
  ratio = np.linalg.inv(prim) @ sim
  rounded = np.round(ratio)
  if not np.allclose(ratio, rounded, atol=atol):
    raise ValueError(
        'Simulation supercell is not commensurate with primitive moire '
        f'lattice: inv(L_prim) @ L_sim = {ratio.tolist()}.'
    )
  return rounded.astype(int)


def cell_area(lattice: Array) -> jnp.ndarray:
  """Returns the area ``|det(lattice)|`` of a 2D cell."""
  return jnp.abs(jnp.linalg.det(jnp.asarray(lattice)))


def reciprocal_lattice(lattice: Array) -> jnp.ndarray:
  """Returns the reciprocal lattice matrix whose columns are ``b_1, b_2``.

  Satisfies ``lattice.T @ reciprocal_lattice(lattice) = 2 * pi * I``.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix with direct vectors as columns.

  Returns:
    Array of shape ``(ndim, ndim)`` with reciprocal vectors as columns.
  """
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat).T


def reciprocal_vectors(lattice: Array) -> jnp.ndarray:
  """Returns the reciprocal lattice vectors as rows of a ``(ndim, ndim)`` array.

  Row ``i`` is ``b_i`` satisfying ``b_i . a_j = 2 * pi * delta_{ij}`` where
  ``a_j = lattice[:, j]``. This matches ``rec = 2 * pi * inv(lattice)`` in
  ``ferminet.pbc.ewald``.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix with direct vectors as columns.

  Returns:
    Array of shape ``(ndim, ndim)`` whose rows are the reciprocal vectors.
  """
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat)


def _triangular_reciprocal_basis_rows(primitive_lattice: Array) -> np.ndarray:
  """Returns validated (b_1, b_2) reciprocal basis rows for a triangular lattice."""
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  rec_rows = 2.0 * np.pi * np.linalg.inv(prim)

  def _is_triangular_pair(rows: np.ndarray) -> bool:
    norms = np.linalg.norm(rows, axis=1)
    if not np.isclose(norms[0], norms[1], rtol=1e-5, atol=1e-8):
      return False
    cos_theta = np.dot(rows[0], rows[1]) / (norms[0] * norms[1])
    return bool(np.isclose(abs(cos_theta), 0.5, rtol=1e-5, atol=1e-6))

  if not _is_triangular_pair(rec_rows):
    # Check whether an unimodular LLL reduction of the reciprocal basis yields
    # a standard triangular pair.
    rec_rows = lll_reduced_basis(rec_rows.T).T
    if not _is_triangular_pair(rec_rows):
      raise ValueError(
          'primitive_lattice must define a 2D triangular lattice with six '
          f'equal-length first-shell reciprocal vectors, got {prim.tolist()}.'
      )
  return rec_rows


def moire_first_shell_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the six first-shell moire reciprocal vectors ``g_1, ..., g_6``.

  Supports both the 120-degree direct triangular basis (where ``b_1 . b_2 > 0``)
  and the 60-degree direct triangular basis (where ``b_1 . b_2 < 0``), ordering
  the six equal-length vectors of norm ``G_M = 4*pi / (sqrt(3)*a_M)`` in
  60-degree steps so that ``first_shell[0:5:2]`` always forms a C3-symmetric
  triad summing to zero.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).

  Returns:
    Array of shape ``(6, 2)`` containing the six first-shell G-vectors as rows.

  Raises:
    ValueError: If ``primitive_lattice`` is not a 2D triangular lattice.
  """
  rec_rows = _triangular_reciprocal_basis_rows(primitive_lattice)
  b1 = jnp.asarray(rec_rows[0])
  b2 = jnp.asarray(rec_rows[1])
  if float(np.dot(rec_rows[0], rec_rows[1])) > 0.0:
    # 120-degree direct basis -> 60-degree reciprocal basis (|b_2 - b_1| = G_M).
    return jnp.stack([b1, b2, b2 - b1, -b1, -b2, b1 - b2], axis=0)
  # 60-degree direct basis -> 120-degree reciprocal basis (|b_1 + b_2| = G_M).
  return jnp.stack([b1, b1 + b2, b2, -b1, -(b1 + b2), -b2], axis=0)


def moire_harmonic_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the three C3-related first-shell moire reciprocal vectors.

  Returns three equal-length first-shell reciprocal vectors ``(g_1, g_2, g_3)``
  separated by 120 degrees satisfying ``g_1 + g_2 + g_3 = 0``, defining the
  three-harmonic continuum moire potential
  ``V(r) = 2 V_0 sum_{m=1}^3 cos(g_m . (r - r_0) + phi)``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).

  Returns:
    Array of shape ``(3, 2)`` containing the three C3 harmonic G-vectors.
  """
  first_shell = moire_first_shell_gvectors(primitive_lattice)
  return first_shell[0:5:2]


def cartesian_to_fractional(
    positions: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts Cartesian coordinates to fractional coordinates in the cell.

  Supports both reshaped coordinates ``(..., ndim)`` and flattened walker
  arrays ``(..., nelec * ndim)``.

  Args:
    positions: Cartesian coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Fractional coordinates with the same shape as ``positions``.
  """
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  inv_lat = jnp.linalg.inv(lat)
  orig_shape = pos.shape
  coords = jnp.reshape(pos, (-1, ndim))
  fractional = coords @ inv_lat.T
  return jnp.reshape(fractional, orig_shape)


def fractional_to_cartesian(
    fractional: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts fractional cell coordinates to Cartesian coordinates.

  Supports both reshaped coordinates ``(..., ndim)`` and flattened walker
  arrays ``(..., nelec * ndim)``.

  Args:
    fractional: Fractional coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Cartesian coordinates with the same shape as ``fractional``.
  """
  frac = jnp.asarray(fractional)
  lat = jnp.asarray(lattice, dtype=frac.dtype)
  orig_shape = frac.shape
  coords = jnp.reshape(frac, (-1, ndim))
  cartesian = coords @ lat.T
  return jnp.reshape(cartesian, orig_shape)


def wrap_positions(
    positions: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Wraps Cartesian positions into the primary simulation cell ``[0, 1)^ndim``.

  Delegates to ``ferminet.mcmc.map_to_simulation_cell`` so coordinate wrapping
  is identical across MCMC sampling, Ewald summation, and geometry utilities.

  Args:
    positions: Particle positions in Cartesian coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Wrapped Cartesian coordinates with the same shape as ``positions``.
  """
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  rec = 2.0 * jnp.pi * jnp.linalg.inv(lat)
  return mcmc.map_to_simulation_cell(pos, lat, rec, ndim)


def lll_reduced_basis(lattice: Array, delta: float = 0.75) -> np.ndarray:
  """LLL-reduces a column-vector lattice basis so 1-shell MIC search is exact.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix whose columns are lattice vectors.
    delta: Lovasz reduction parameter in ``(0.25, 1.0)``.

  Returns:
    NumPy array of shape ``(ndim, ndim)`` whose columns span the same lattice.
  """
  # Work with rows internally, then transpose back to columns.
  basis = np.asarray(lattice, dtype=np.float64).T.copy()
  ndim = basis.shape[0]
  if not np.all(np.isfinite(basis)):
    return np.asarray(lattice, dtype=np.float64)
  norm_scale = np.linalg.norm(basis, axis=1).max()
  if abs(np.linalg.det(basis)) <= np.finfo(np.float64).eps * norm_scale**ndim:
    return np.asarray(lattice, dtype=np.float64)

  def gram_schmidt() -> Tuple[np.ndarray, np.ndarray]:
    _, r = np.linalg.qr(basis.T)
    b_star_norm = np.diag(r)
    mu = np.zeros((ndim, ndim), dtype=np.float64)
    for i in range(ndim):
      mu[i, :i] = r[:i, i] / b_star_norm[:i]
    return b_star_norm, mu

  k = 1
  while k < ndim:
    b_star_norm, mu = gram_schmidt()
    for j in range(k - 1, -1, -1):
      if abs(mu[k, j]) > 0.5:
        basis[k] = basis[k] - np.round(mu[k, j]) * basis[j]
        b_star_norm, mu = gram_schmidt()
    lovasz = (delta - mu[k, k - 1] ** 2) * b_star_norm[k - 1] ** 2
    if b_star_norm[k] ** 2 >= lovasz:
      k += 1
    else:
      basis[[k, k - 1]] = basis[[k - 1, k]]
      k = max(k - 1, 1)
  return basis.T


def make_minimum_image_distance_fn(lattice: Array):
  """Builds a minimum-image displacement and distance function for a 2D cell.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix whose columns are lattice vectors.

  Returns:
    Callable ``dist_fn(pos_a, pos_b) -> (disp, dist)`` where ``pos_a`` has shape
    ``(..., n_a, ndim)``, ``pos_b`` has shape ``(..., n_b, ndim)``, ``disp`` has
    shape ``(..., n_a, n_b, ndim)`` giving minimum-image ``pos_a - pos_b``, and
    ``dist`` has shape ``(..., n_a, n_b)``.
  """
  reduced_np = lll_reduced_basis(lattice)
  ndim = reduced_np.shape[0]
  inv_reduced_np = np.linalg.inv(reduced_np)
  grid = np.meshgrid(*[np.array([-1, 0, 1]) for _ in range(ndim)], indexing='ij')
  point_list = np.stack([g.ravel() for g in grid], axis=-1)  # (3^ndim, ndim)
  shifts_np = point_list @ reduced_np.T  # (3^ndim, ndim)

  def distance_fn(
      pos_a: jnp.ndarray, pos_b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dtype = pos_a.dtype
    reduced = jnp.asarray(reduced_np, dtype=dtype)
    inv_reduced = jnp.asarray(inv_reduced_np, dtype=dtype)
    shifts = jnp.asarray(shifts_np, dtype=dtype)

    diff = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    frac_diff = diff @ inv_reduced.T
    wrapped = diff - jnp.round(frac_diff) @ reduced.T
    diff_all = wrapped[..., None, :] + shifts
    dists_all = jnp.linalg.norm(diff_all, axis=-1)
    min_idx = jnp.argmin(dists_all, axis=-1)
    best_disp = jnp.take_along_axis(
        diff_all, min_idx[..., None, None], axis=-2
    )[..., 0, :]
    best_dist = jnp.linalg.norm(best_disp, axis=-1)
    return best_disp, best_dist

  return distance_fn
