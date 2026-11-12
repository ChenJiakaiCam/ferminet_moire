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

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

from typing import Any, Callable, Literal, Mapping, Sequence, overload
import chex
from ferminet import constants
from ferminet import networks
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np


def map_to_simulation_cell(pos, lattice, rec, ndim):
  """If working in PBC, map electrons back to the supercell."""
  pos_ = jnp.reshape(pos, [-1, ndim])
  phase = jnp.einsum('il,kl->ki', rec / (2 * jnp.pi), pos_)
  phase_prim = phase % 1
  prim = jnp.einsum('il,kl->ki', lattice, phase_prim)
  return prim.reshape(pos.shape)


def _harmonic_mean(x, atoms):
  """Calculates the harmonic mean of each electron distance to the nuclei.

  Args:
    x: electron positions. Shape (batch, nelectrons, 1, ndim). Note the third
      dimension is already expanded, which allows for avoiding additional
      reshapes in the MH algorithm.
    atoms: atom positions. Shape (natoms, ndim)

  Returns:
    Array of shape (batch, nelectrons, 1, 1), where the (i, j, 0, 0) element is
    the harmonic mean of the distance of the j-th electron of the i-th MCMC
    configuration to all atoms.
  """
  ae = x - atoms[None, ...]
  r_ae = jnp.linalg.norm(ae, axis=-1, keepdims=True)
  return 1.0 / jnp.mean(1.0 / r_ae, axis=-2, keepdims=True)


def _log_prob_gaussian(x, mu, sigma):
  """Calculates the log probability of Gaussian with diagonal covariance.

  Args:
    x: Positions. Shape (batch, nelectron, 1, ndim) - as used in mh_update.
    mu: means of Gaussian distribution. Same shape as or broadcastable to x.
    sigma: standard deviation of the distribution. Same shape as or
      broadcastable to x.

  Returns:
    Log probability of Gaussian distribution with shape as required for
    mh_update - (batch, nelectron, 1, 1).
  """
  numer = jnp.sum(-0.5 * ((x - mu)**2) / (sigma**2), axis=[1, 2, 3])
  denom = x.shape[-1] * jnp.sum(jnp.log(sigma), axis=[1, 2, 3])
  return numer - denom


def mh_accept(x1, x2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, key, lp_new, num_accepts


def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev: float | jnp.ndarray = 0.02,
    atoms=None,
    ndim=3,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks=1,
    i: int | jnp.ndarray = 0,
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: If not None, atom positions. Shape (natoms, 3). If present, then the
      Metropolis-Hastings move proposals are drawn from a Gaussian distribution,
      N(0, (h_i stddev)^2), where h_i is the harmonic mean of distances between
      the i-th electron and the atoms, otherwise the move proposal drawn from
      N(0, stddev^2).
    ndim: dimensionality of system.
    map_to_cell: If PBC, callable which maps electrons back to the simulation
      cell. If OBC, identity function.
    blocks: Ignored.
    i: Ignored.

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  if atoms is None:  # symmetric proposal, same stddev everywhere
    x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
    x2 = map_to_cell(x2)
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    n = x1.shape[0]
    x1 = jnp.reshape(x1, [n, -1, 1, ndim])
    hmean1 = _harmonic_mean(x1, atoms)  # harmonic mean of distances to nuclei

    x2 = x1 + stddev * hmean1 * jax.random.normal(subkey, shape=x1.shape)
    x2 = map_to_cell(x2)
    x2 = x2.reshape(x1.shape)
    lp_2 = 2.0 * f(
        params, x2, data.spins, data.atoms, data.charges
    )  # log prob of proposal
    hmean2 = _harmonic_mean(x2, atoms)  # needed for probability of reverse jump

    lq_1 = _log_prob_gaussian(x1, x2, stddev * hmean1)  # forward probability
    lq_2 = _log_prob_gaussian(x2, x1, stddev * hmean2)  # reverse probability
    ratio = lp_2 + lq_2 - lp_1 - lq_1

    x1 = jnp.reshape(x1, [n, -1])
    x2 = jnp.reshape(x2, [n, -1])
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))  # pyrefly: ignore[no-matching-overload]
  return new_data, key, lp_new, num_accepts


def mh_block_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev: float | jnp.ndarray = 0.02,
    atoms=None,
    ndim=3,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks=1,
    i: int | jnp.ndarray = 0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    atoms: Not implemented. Raises an error if not None.
    ndim: dimensionality of system.
    map_to_cell: If PBC, callable which maps electrons back to the simulation
      cell. If OBC, identity function.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  if atoms is None:  # symmetric prop, same stddev everywhere
    x2 = x1.at[:, ii].add(
        stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
    x2 = jnp.reshape(x2, [batch_size, -1])
    x2 = map_to_cell(x2)
    if pad > 0:
      x2 = x2[..., :-pad*ndim]
    # log prob of proposal
    lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
    ratio = lp_2 - lp_1
  else:  # asymmetric proposal, stddev propto harmonic mean of nuclear distances
    raise NotImplementedError('Still need to work out reverse probabilities '
                              'for asymmetric moves.')

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))  # pyrefly: ignore[no-matching-overload]
  return new_data, key, lp_new, num_accepts


def _safe_pmean(x: jnp.ndarray) -> jnp.ndarray:
  """Computes pmean over PMAP_AXIS_NAME when present, else returns x."""
  try:
    return constants.pmean(x)
  except ValueError:
    return x


def _safe_psum(x: jnp.ndarray) -> jnp.ndarray:
  """Computes psum over PMAP_AXIS_NAME when present, else returns x."""
  try:
    return constants.psum(x)
  except ValueError:
    return x


def _resolve_layer_counts(
    nspins: Sequence[int] | None,
    layer_counts: tuple[int, int] | None,  # Try to resolve (n_electrons, n_holes) using layer_counts first, then nspins. If both are None, return (0, 0).
) -> tuple[int, int]:
  """Resolves (n_electrons, n_holes) from explicit layer_counts or nspins."""
  if layer_counts is not None:
    if len(layer_counts) != 2:
      raise ValueError(
          f'layer_counts must have length 2 (n_e, n_h), got {layer_counts}.'
      )
    n_e, n_h = int(layer_counts[0]), int(layer_counts[1])
    if n_e < 0 or n_h < 0:
      raise ValueError(
          f'layer_counts entries must be non-negative, got {layer_counts}.'
      )
    return n_e, n_h
  if nspins is None:
    return 0, 0
  counts = [int(n) for n in nspins]
  if any(n < 0 for n in counts):
    raise ValueError(f'All entries in nspins must be >= 0, got {nspins}.')
  if len(counts) == 4:
    return counts[0] + counts[1], counts[2] + counts[3]
  if len(counts) == 2:
    return counts[0], counts[1]
  return 0, 0


def _lll_reduced_basis(lattice: np.ndarray, delta: float = 0.75) -> np.ndarray:  # lattice is the simulation cell lattice vectors
  """LLL-reduces a column-vector lattice basis for exact 1-shell MIC search."""
  # Lenstra–Lenstra–Lovász (LLL) makes the basis vectors as short and as orthogonal as possible, which is useful for exact 1-shell MIC search.
  basis = np.asarray(lattice, dtype=np.float64).T.copy()
  ndim = basis.shape[0]
  if not np.all(np.isfinite(basis)):
    return np.asarray(lattice, dtype=np.float64)
  norm_scale = np.linalg.norm(basis, axis=1).max()
  if abs(np.linalg.det(basis)) <= np.finfo(np.float64).eps * norm_scale**ndim:
    return np.asarray(lattice, dtype=np.float64)

  def gram_schmidt():
    _, r = np.linalg.qr(basis.T)
    b_star_norm = np.diag(r)
    mu = np.zeros((ndim, ndim), dtype=np.float64)
    for idx in range(ndim):
      mu[idx, :idx] = r[:idx, idx] / b_star_norm[:idx]
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
  return basis.T  # Transposed back so that the columns are the reduced basis vectors, as in the input lattice.


def _make_minimum_image_dist_sq_fn(
    lattice: jnp.ndarray | np.ndarray | None,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Builds a pairwise squared distance function (with MIC in PBC)."""
  if lattice is None:  # OBC
    def obc_dist_sq_fn(pos_a: jnp.ndarray, pos_b: jnp.ndarray) -> jnp.ndarray:
      diff = pos_a[..., :, None, :] - pos_b[..., None, :, :]
      return jnp.sum(diff**2, axis=-1)
    return obc_dist_sq_fn

  reduced_np = _lll_reduced_basis(np.asarray(lattice, dtype=np.float64))
  ndim = reduced_np.shape[0]
  inv_reduced_np = np.linalg.inv(reduced_np)
  grid = np.meshgrid(
      *[np.array([-1, 0, 1]) for _ in range(ndim)], indexing='ij'
  )
  point_list = np.stack([g.ravel() for g in grid], axis=-1)
  shifts_np = point_list @ reduced_np.T

  def pbc_dist_sq_fn(pos_a: jnp.ndarray, pos_b: jnp.ndarray) -> jnp.ndarray:
    dtype = pos_a.dtype
    reduced = jnp.asarray(reduced_np, dtype=dtype)
    inv_reduced = jnp.asarray(inv_reduced_np, dtype=dtype)
    shifts = jnp.asarray(shifts_np, dtype=dtype)

    diff = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    frac_diff = diff @ inv_reduced.T
    wrapped = diff - jnp.round(frac_diff) @ reduced.T
    diff_all = wrapped[..., None, :] + shifts
    dists_sq_all = jnp.sum(diff_all**2, axis=-1)
    return jnp.min(dists_sq_all, axis=-1)  # Note this is actual minimum image distance, not the periodic distance feature. Shape (for ground state) is (batch, n_e, n_h).

  return pbc_dist_sq_fn


def _find_mutual_eh_pairs(
    pos_e: jnp.ndarray,
    pos_h: jnp.ndarray,
    pair_radius: float,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Identifies mutual nearest-neighbor (e, h) pairs within pair_radius.

  # i.e. Heuristic for finding excitons: find pairs such that interaction between the pair is stronger than interaction with other electrons and holes. Note 1,2 need not be true/false at the same time.
  An electron i and hole j are paired if and only if:
    1. Hole j is the nearest hole to electron i,
    2. Electron i is the nearest electron to hole j, and
    3. Their minimum-image in-plane distance satisfies r_ij < pair_radius.

  Args:
    pos_e: Electron coordinates of shape (batch, n_e, ndim).
    pos_h: Hole coordinates of shape (batch, n_h, ndim).
    pair_radius: Cutoff radius R_c (in Bohr) for pairing.
    dist_sq_fn: Callable returning squared distances of shape (batch, n_e, n_h).

  Returns:
    Tuple (partner_h_for_e, is_paired_e), where:
      partner_h_for_e: Integer array of shape (batch, n_e) with the paired hole
        index in [0, n_h) for paired electrons and -1 for unpaired electrons.
      is_paired_e: Boolean array of shape (batch, n_e) indicating whether each
        electron is part of a valid mutual pair.
  """
  n_e = pos_e.shape[1]
  dist_sq = dist_sq_fn(pos_e, pos_h)  # (batch, n_e, n_h)
  nearest_h_for_e = jnp.argmin(dist_sq, axis=-1)  # (batch, n_e)
  min_dist_sq_e = jnp.min(dist_sq, axis=-1)  # (batch, n_e)
  nearest_e_for_h = jnp.argmin(dist_sq, axis=-2)  # (batch, n_h)

  partner_e_of_nearest_h = jnp.take_along_axis(
      nearest_e_for_h, nearest_h_for_e, axis=-1
  )  # (batch, n_e)
  e_indices = jnp.arange(n_e, dtype=nearest_h_for_e.dtype)[None, :]
  is_mutual = partner_e_of_nearest_h == e_indices
  is_paired_e = is_mutual & (min_dist_sq_e < (pair_radius**2))
  partner_h_for_e = jnp.where(is_paired_e, nearest_h_for_e, -1)
  return partner_h_for_e, is_paired_e  # partner_h_for_e shaped (batch, n_e), gives the index of the paired hole for each electron, or -1 if unpaired. is_paired_e shaped (batch, n_e), gives True if electron is paired, False otherwise.


def _make_moire_hop_vectors(
    pair_lattice: jnp.ndarray | np.ndarray,
    ndim: int = 2,
) -> jnp.ndarray:
  """Constructs nearest-neighbor direct lattice translation vectors for inter-well hops.

  Given a primitive moire lattice matrix `pair_lattice` of shape `(ndim, ndim)`
  whose columns are the primitive lattice vectors:
    1. LLL-reduces the basis so the basis vectors lie in the first shell.
    2. Enumerates all non-zero integer combinations `n in {-1, 0, 1}^ndim \\ {0}`
       and retains the shortest-shell vectors (within relative tolerance 1e-4 of
       the shortest non-zero lattice vector, plus all unit basis vectors
       `+/- a_k` so anisotropic/rectangular lattices span every axis).

  For a 2D triangular moire lattice (either 120-degree or 60-degree basis), this
  returns the exact 6 nearest-neighbor moire lattice vectors of length `a_M`
  (`{+/- a_1, +/- a_2, +/-(a_1 + a_2)}` or `{+/- a_1, +/- a_2, +/-(a_2 - a_1)}`).
  Because both the L1 integer norm and Euclidean vector norm are invariant under
  `n -> -n`, opposite directions `+a` and `-a` are always included in pairs and
  sampled with equal probability.

  Args:
    pair_lattice: Shape `(ndim, ndim)` matrix with direct lattice vectors as
      columns.
    ndim: Spatial dimensionality.

  Returns:
    Float32 JAX array of shape `(n_vectors, ndim)` containing the nearest-neighbor
    lattice translation vectors as rows.
  """
  lat_np = np.asarray(pair_lattice, dtype=np.float64)
  if lat_np.shape != (ndim, ndim):
    raise ValueError(
        f'pair_lattice must have shape ({ndim}, {ndim}), got {lat_np.shape}.'
    )
  if not np.all(np.isfinite(lat_np)):
    raise ValueError('pair_lattice must have finite entries.')
  if abs(np.linalg.det(lat_np)) <= 1e-12:
    raise ValueError('pair_lattice must be non-singular.')

  reduced_np = _lll_reduced_basis(lat_np)
  grid = np.meshgrid(
      *[np.array([-1, 0, 1], dtype=np.int64) for _ in range(ndim)],
      indexing='ij',
  )
  point_list = np.stack([g.ravel() for g in grid], axis=-1)
  nonzero_mask = np.any(point_list != 0, axis=-1)
  point_list = point_list[nonzero_mask]

  candidates = point_list @ reduced_np.T  # (3^ndim - 1, ndim)
  norms = np.linalg.norm(candidates, axis=-1)
  min_norm = float(np.min(norms))
  is_basis_axis = np.sum(np.abs(point_list), axis=-1) == 1
  is_first_shell = norms <= min_norm * (1.0 + 1e-4)
  keep_mask = is_basis_axis | is_first_shell
  hop_vectors_np = candidates[keep_mask]
  return jnp.asarray(hop_vectors_np, dtype=jnp.float32)


def _validate_hop_vectors(
    hop_vectors: jnp.ndarray | np.ndarray,
    ndim: int = 2,
    atol: float = 1e-5,
) -> jnp.ndarray:
  """Validates custom hop vectors for shape, finiteness, and +/- symmetry.

  To preserve detailed balance without a Hastings proposal-ratio correction,
  every hop vector `+a` must appear with the exact same multiplicity as its
  reverse `-a` (and be non-zero and finite).

  Args:
    hop_vectors: Array of shape `(n_vectors, ndim)`.
    ndim: Spatial dimensionality.
    atol: Tolerance for matching opposite vectors `+a` and `-a`.

  Returns:
    Validated `jnp.float32` array of shape `(n_vectors, ndim)`.
  """
  if isinstance(hop_vectors, jax.core.Tracer):
    if hop_vectors.ndim != 2 or hop_vectors.shape[1] != ndim or hop_vectors.shape[0] == 0:
      raise ValueError(
          f'hop_vectors must have shape (n_vectors >= 1, {ndim}), got '
          f'{hop_vectors.shape}.'
      )
    return jnp.asarray(hop_vectors, dtype=jnp.float32)

  vecs_np = np.asarray(hop_vectors, dtype=np.float64)
  if vecs_np.ndim != 2 or vecs_np.shape[1] != ndim or vecs_np.shape[0] == 0:
    raise ValueError(
        f'hop_vectors must have shape (n_vectors >= 1, {ndim}), got '
        f'{vecs_np.shape}.'
    )
  if not np.all(np.isfinite(vecs_np)):
    raise ValueError('hop_vectors must contain only finite values.')
  norms = np.linalg.norm(vecs_np, axis=-1)
  if np.any(norms <= atol):
    raise ValueError(
        f'hop_vectors must all be non-zero (norm > {atol}), got min norm '
        f'{float(np.min(norms)):.3e}.'
    )
  n_vecs = vecs_np.shape[0]
  matched = np.zeros(n_vecs, dtype=bool)
  for idx in range(n_vecs):
    if matched[idx]:
      continue
    v = vecs_np[idx]
    tol = atol * max(1.0, float(norms[idx]))
    partner_idx = -1
    for jdx in range(idx + 1, n_vecs):
      if not matched[jdx] and np.linalg.norm(v + vecs_np[jdx]) <= tol:
        partner_idx = jdx
        break
    if partner_idx < 0:
      raise ValueError(
          'hop_vectors must be symmetric under negation (+a and -a with '
          f'equal multiplicity); no unmatched opposite vector found for '
          f'hop_vectors[{idx}] = {v.tolist()}.'
      )
    matched[idx] = True
    matched[partner_idx] = True
  return jnp.asarray(vecs_np, dtype=jnp.float32)


def _mh_pair_interwell_update_with_diagnostics(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    hop_vectors: jnp.ndarray,
    atoms: Any = None,
    ndim: int = 2,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks: int = 1,
    i: int | jnp.ndarray = 0,
    layer_counts: tuple[int, int] = (0, 0),
    pair_radius: float = 0.0,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[
    networks.FermiNetData,
    chex.PRNGKey,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
  """Internal single-pair inter-well lattice-hop MH update with diagnostics.

  For each walker in the batch:
    1. Identifies all mutual nearest-neighbor (e, h) pairs within `pair_radius`
       under minimum-image distance (`_find_mutual_eh_pairs`).
    2. Selects ONE matched pair `(i*, j*)` uniformly at random among the
       `K(X)` mutual pairs currently present in the walker (and active state if
       `num_states > 1`).
    3. Selects ONE nearest-neighbor moire lattice translation vector `a`
       uniformly at random from `hop_vectors` (where `+a` and `-a` have equal
       probability).
    4. Proposes `r_e[i*]' = r_e[i*] + a` and `r_h[j*]' = r_h[j*] + a`, leaving
       all other `N_e + N_h - 2` particles strictly fixed, and wraps positions
       into the simulation cell via `map_to_cell`.
    5. Recomputes the mutual nearest-neighbor pairing map on `X'` and requires
       exact pairing-map preservation (`partner_h_2 == partner_h_1`). Because
       `partner_h_2 == partner_h_1` implies `K(X') = K(X)` and `(i*, j*)`
       remains a mutual pair in `X'`, the reverse move `X' -> X` selects the
       same pair `(i*, j*)` with probability `1 / K(X)` and the opposite lattice
       vector `-a` with probability `1 / len(hop_vectors)`, ensuring exact
       proposal symmetry `q(X' | X) = q(X | X')` and standard Metropolis
       acceptance `min(1, |Psi(X')|^2 / |Psi(X)|^2)`.
  """
  if atoms is not None:
    raise NotImplementedError(
        'Asymmetric nuclear-distance scaling is not supported for pair moves.'
    )
  n_e, n_h = layer_counts
  if dist_sq_fn is None:
    dist_sq_fn = _make_minimum_image_dist_sq_fn(None)

  key, subkey_pair, subkey_vec = jax.random.split(key, 3)
  x1 = data.positions
  batch_size = x1.shape[0]
  nelec = n_e + n_h
  num_states = max(1, x1.shape[1] // (nelec * ndim))
  total_states = batch_size * num_states

  pos_1 = jnp.reshape(x1, [total_states, nelec, ndim])
  pos_e_1 = pos_1[:, :n_e, :]
  pos_h_1 = pos_1[:, n_e:, :]

  partner_h_1, is_paired_e_1 = _find_mutual_eh_pairs(
      pos_e_1, pos_h_1, pair_radius, dist_sq_fn
  )

  if blocks > 1 and num_states > 1:
    blocks_per_state = max(1, blocks // num_states)
    ii = i % blocks
    s_active = (ii // blocks_per_state) % num_states
    state_idx = jnp.tile(jnp.arange(num_states), batch_size)[:, None]
    active_state_mask = state_idx == s_active
    active_paired_e = is_paired_e_1 & active_state_mask
  else:
    active_paired_e = is_paired_e_1

  # Uniformly select 1 matched pair among the K_b active mutual pairs in each
  # walker/state: assign ranks 0, 1, ..., K_b - 1 to the True entries of
  # active_paired_e and draw chosen_rank ~ Uniform({0, ..., K_b - 1}).
  num_pairs_active = jnp.sum(active_paired_e.astype(jnp.int32), axis=-1)
  has_pair_state = num_pairs_active > 0
  pair_rank = jnp.cumsum(active_paired_e.astype(jnp.int32), axis=-1) - 1
  u_pair = jax.random.uniform(subkey_pair, shape=(total_states,))
  chosen_rank = jnp.floor(
      u_pair * jnp.maximum(num_pairs_active, 1).astype(u_pair.dtype)
  ).astype(jnp.int32)
  selected_e_mask = active_paired_e & (pair_rank == chosen_rank[:, None])

  chosen_h_idx = jnp.sum(
      jnp.where(selected_e_mask, partner_h_1, 0), axis=-1
  )
  selected_h_mask = has_pair_state[:, None] & (
      jnp.arange(n_h, dtype=chosen_h_idx.dtype)[None, :] == chosen_h_idx[:, None]
  )

  # Uniformly select 1 nearest-neighbor moire lattice vector per walker/state.
  hop_vecs = jnp.asarray(hop_vectors, dtype=x1.dtype)
  n_vecs = hop_vecs.shape[0]
  vec_idx = jax.random.randint(
      subkey_vec, shape=(total_states,), minval=0, maxval=n_vecs
  )
  hop_vec = hop_vecs[vec_idx]  # (total_states, ndim)

  eta_e = jnp.where(selected_e_mask[..., None], hop_vec[:, None, :], 0.0)
  eta_h = jnp.where(selected_h_mask[..., None], hop_vec[:, None, :], 0.0)

  pos_e_2 = pos_e_1 + eta_e
  pos_h_2 = pos_h_1 + eta_h
  pos_2 = jnp.concatenate([pos_e_2, pos_h_2], axis=1)
  x2_wrapped = map_to_cell(jnp.reshape(pos_2, [batch_size, -1]))
  pos_2_wrapped = jnp.reshape(x2_wrapped, [total_states, nelec, ndim])

  # Preserve unselected particles' coordinates bit-for-bit to avoid float32
  # re-wrapping roundoff on particles that were not proposed to move.
  pos_e_2_w = jnp.where(
      selected_e_mask[..., None], pos_2_wrapped[:, :n_e, :], pos_e_1
  )
  pos_h_2_w = jnp.where(
      selected_h_mask[..., None], pos_2_wrapped[:, n_e:, :], pos_h_1
  )

  # Detect non-trivial geometric displacement of the selected pair (excluding
  # zero-pair walkers and periodic identity translations where `a` is a
  # simulation supercell lattice vector that wraps back to the original point).
  pos_1_wrapped = jnp.reshape(map_to_cell(x1), [total_states, nelec, ndim])
  sel_e_1 = jnp.sum(
      jnp.where(selected_e_mask[..., None], pos_e_1, 0.0),
      axis=1,
      keepdims=True,
  )
  sel_e_1_w = jnp.sum(
      jnp.where(selected_e_mask[..., None], pos_1_wrapped[:, :n_e, :], 0.0),
      axis=1,
      keepdims=True,
  )
  sel_e_2 = jnp.sum(
      jnp.where(selected_e_mask[..., None], pos_e_2_w, 0.0),
      axis=1,
      keepdims=True,
  )
  mic_move_sq = dist_sq_fn(sel_e_2, sel_e_1)[:, 0, 0]
  wrapped_move_sq = jnp.sum((sel_e_2[:, 0, :] - sel_e_1_w[:, 0, :]) ** 2, axis=-1)
  moved_dist_sq = jnp.minimum(mic_move_sq, wrapped_move_sq)
  min_hop_sq = jnp.min(jnp.sum(hop_vecs**2, axis=-1))
  hop_tol_sq = jnp.maximum(1e-6, 1e-4 * min_hop_sq)
  moved_state = has_pair_state & (moved_dist_sq > hop_tol_sq)

  active_move_e = selected_e_mask & moved_state[:, None]
  active_move_h = selected_h_mask & moved_state[:, None]
  pos_e_2_w = jnp.where(active_move_e[..., None], pos_e_2_w, pos_e_1)
  pos_h_2_w = jnp.where(active_move_h[..., None], pos_h_2_w, pos_h_1)
  x2 = jnp.reshape(
      jnp.concatenate([pos_e_2_w, pos_h_2_w], axis=1), [batch_size, -1]
  )

  partner_h_2, _ = _find_mutual_eh_pairs(
      pos_e_2_w, pos_h_2_w, pair_radius, dist_sq_fn
  )
  same_pairing = jnp.all(
      jnp.reshape(partner_h_1 == partner_h_2, [batch_size, -1]), axis=-1
  )
  has_active_pair = jnp.any(
      jnp.reshape(selected_e_mask, [batch_size, -1]), axis=-1
  )
  moved = jnp.any(jnp.reshape(moved_state, [batch_size, -1]), axis=-1)
  valid_active_pair = has_active_pair & moved & same_pairing

  lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
  ratio = jnp.asarray(jnp.where(valid_active_pair, lp_2 - lp_1, -jnp.inf))
  key, accept_subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(accept_subkey, shape=ratio.shape))
  cond = (ratio > rnd) & valid_active_pair
  x_new = jnp.asarray(jnp.where(cond[..., None], x2, x1))
  lp_new = jnp.asarray(jnp.where(cond, lp_2, lp_1))
  num_accepts = num_accepts + jnp.sum(cond)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))  # type: ignore[arg-type]  # pyrefly: ignore[no-matching-overload]

  num_map_rejects = jnp.sum(
      (has_active_pair & moved & (~same_pairing)).astype(jnp.float32)
  )
  pairs_per_walker = (
      jnp.sum(
          jnp.reshape(is_paired_e_1, [batch_size, -1]).astype(jnp.float32),
          axis=-1,
      )
      / float(num_states)
  )
  total_pairs = jnp.sum(pairs_per_walker)
  active_pair_tries = jnp.sum(valid_active_pair.astype(jnp.float32))
  active_pair_accepts = jnp.sum(cond.astype(jnp.float32))
  return (
      new_data,
      key,
      lp_new,
      num_accepts,
      num_map_rejects,
      total_pairs,
      active_pair_tries,
      active_pair_accepts,
  )


def mh_pair_interwell_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    pair_lattice: jnp.ndarray | np.ndarray | None = None,
    hop_vectors: jnp.ndarray | np.ndarray | None = None,
    atoms: Any = None,
    ndim: int = 2,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks: int = 1,
    i: int | jnp.ndarray = 0,
    layer_counts: tuple[int, int] = (0, 0),
    pair_radius: float = 0.0,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    return_pair_diagnostics: bool = False,
):
  """Performs one Metropolis-Hastings inter-well pair hop by a moire lattice vector.

  Selects one mutual nearest-neighbor (e, h) pair uniformly at random per walker
  and translates both particles by a randomly chosen nearest-neighbor moire
  lattice vector `a` from `hop_vectors` (or derived from `pair_lattice`),
  leaving all other particles fixed. Enforces exact detailed balance via full
  mutual-nearest-neighbor pairing-map preservation (`partner_h_2 == partner_h_1`).
  """
  if hop_vectors is None:
    if pair_lattice is None:
      raise ValueError(
          'mh_pair_interwell_update requires either pair_lattice or hop_vectors.'
      )
    resolved_hop_vecs = _make_moire_hop_vectors(pair_lattice, ndim=ndim)
  else:
    resolved_hop_vecs = _validate_hop_vectors(hop_vectors, ndim=ndim)

  out = _mh_pair_interwell_update_with_diagnostics(
      params=params,
      f=f,
      data=data,
      key=key,
      lp_1=lp_1,
      num_accepts=num_accepts,
      hop_vectors=resolved_hop_vecs,
      atoms=atoms,
      ndim=ndim,
      map_to_cell=map_to_cell,
      blocks=blocks,
      i=i,
      layer_counts=layer_counts,
      pair_radius=pair_radius,
      dist_sq_fn=dist_sq_fn,
  )
  if return_pair_diagnostics:
    return out
  return out[:4]


def _mh_pair_update_with_diagnostics(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    stddev: float | jnp.ndarray = 0.02,
    atoms: Any = None,
    ndim: int = 2,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks: int = 1, # number of blocks to split electron updates into
    i: int | jnp.ndarray = 0,
    layer_counts: tuple[int, int] = (0, 0),
    pair_radius: float = 0.0,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    unpaired_stddev: float | jnp.ndarray | None = None,
    pair_lattice: jnp.ndarray | np.ndarray | None = None,
    hop_vectors: jnp.ndarray | np.ndarray | None = None,
) -> tuple[
    networks.FermiNetData,
    chex.PRNGKey,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
  """Internal pair MH update returning full 8-tuple diagnostics."""
  if atoms is not None:
    raise NotImplementedError(
        'Asymmetric nuclear-distance scaling is not supported for pair moves.'
    )
  n_e, n_h = layer_counts
  if n_e <= 0 or n_h <= 0 or pair_radius <= 0.0:
    fallback_fn = mh_block_update if blocks > 1 else mh_update
    out_data, out_key, out_lp, out_acc = fallback_fn(
        params,
        f,
        data,
        key,
        lp_1,
        num_accepts,
        stddev=stddev,
        atoms=atoms,
        ndim=ndim,
        map_to_cell=map_to_cell,
        blocks=blocks,
        i=i,
    )
    zero = jnp.zeros((), dtype=jnp.float32)
    return (out_data, out_key, out_lp, out_acc, zero, zero, zero, zero)

  if hop_vectors is not None or pair_lattice is not None:
    resolved_hop_vecs = (
        _validate_hop_vectors(hop_vectors, ndim=ndim)
        if hop_vectors is not None
        else _make_moire_hop_vectors(pair_lattice, ndim=ndim)  # type: ignore[arg-type]
    )
    return _mh_pair_interwell_update_with_diagnostics(
        params=params,
        f=f,
        data=data,
        key=key,
        lp_1=lp_1,
        num_accepts=num_accepts,
        hop_vectors=resolved_hop_vecs,
        atoms=atoms,
        ndim=ndim,
        map_to_cell=map_to_cell,
        blocks=blocks,
        i=i,
        layer_counts=layer_counts,
        pair_radius=pair_radius,
        dist_sq_fn=dist_sq_fn,
    )

  if dist_sq_fn is None:
    dist_sq_fn = _make_minimum_image_dist_sq_fn(None)

  key, subkey_h, subkey_e = jax.random.split(key, 3)
  x1 = data.positions
  batch_size = x1.shape[0]
  nelec = n_e + n_h
  num_states = max(1, x1.shape[1] // (nelec * ndim))

  pos_1 = jnp.reshape(x1, [batch_size * num_states, nelec, ndim])
  pos_e_1 = pos_1[:, :n_e, :]
  pos_h_1 = pos_1[:, n_e:, :]

  partner_h_1, is_paired_e_1 = _find_mutual_eh_pairs(
      pos_e_1, pos_h_1, pair_radius, dist_sq_fn
  )

  unpaired_std = stddev if unpaired_stddev is None else unpaired_stddev
  h_indices = jnp.arange(n_h)[None, None, :]
  is_paired_h_1 = jnp.any(partner_h_1[..., :, None] == h_indices, axis=-2)
  sigma_h = jnp.where(is_paired_h_1[..., None], stddev, unpaired_std)

  # Contains stdev at paired hole indices and unpaired_std at unpaired hole indices
  eta_h = sigma_h * jax.random.normal(subkey_h, shape=pos_h_1.shape)
  eta_e_indep = unpaired_std * jax.random.normal(subkey_e, shape=pos_e_1.shape)

  if blocks > 1:
    blocks_per_state = max(1, blocks // num_states)
    h_blocks_per_state = min(blocks_per_state, n_h)
    e_blocks_per_state = min(blocks_per_state, n_e)
    ii = i % blocks
    s_active = (ii // blocks_per_state) % num_states
    b_active = ii % blocks_per_state
    h_b_active = b_active % h_blocks_per_state
    e_b_active = b_active % e_blocks_per_state
    state_idx = jnp.tile(jnp.arange(num_states), batch_size)[:, None, None]
    active_state = state_idx == s_active
    active_h_block = (
        (jnp.arange(n_h) % h_blocks_per_state) == h_b_active
    )[None, :, None]
    active_e_block = (
        (jnp.arange(n_e) % e_blocks_per_state) == e_b_active
    )[None, :, None]
    eta_h = jnp.where(active_state & active_h_block, eta_h, 0.0)
    eta_e_indep = jnp.where(active_state & active_e_block, eta_e_indep, 0.0)
    active_paired_h = is_paired_h_1 & (
        active_state[..., 0] & active_h_block[..., 0]
    )
  else:
    active_paired_h = is_paired_h_1

  safe_partner_h_1 = jnp.maximum(partner_h_1, 0)
  paired_eta_e = jnp.take_along_axis(
      eta_h, safe_partner_h_1[..., None], axis=1
  )
  # Contains stdev at paired hole indices and unpaired_std at unpaired hole indices
  eta_e = jnp.where(is_paired_e_1[..., None], paired_eta_e, eta_e_indep)

  #Updates positions of electrons and holes, including both paired and unpaired particles
  pos_e_2 = pos_e_1 + eta_e
  pos_h_2 = pos_h_1 + eta_h
  pos_2 = jnp.concatenate([pos_e_2, pos_h_2], axis=1)
  x2_wrapped = map_to_cell(jnp.reshape(pos_2, [batch_size, -1]))

  pos_2_wrapped = jnp.reshape(x2_wrapped, [batch_size * num_states, nelec, ndim])
  moved_e = jnp.any(eta_e != 0.0, axis=-1, keepdims=True)
  moved_h = jnp.any(eta_h != 0.0, axis=-1, keepdims=True)
  pos_e_2_w = jnp.where(moved_e, pos_2_wrapped[:, :n_e, :], pos_e_1)
  pos_h_2_w = jnp.where(moved_h, pos_2_wrapped[:, n_e:, :], pos_h_1)
  x2 = jnp.reshape(
      jnp.concatenate([pos_e_2_w, pos_h_2_w], axis=1), [batch_size, -1]
  )
  partner_h_2, _ = _find_mutual_eh_pairs(
      pos_e_2_w, pos_h_2_w, pair_radius, dist_sq_fn
  )
  same_pairing = jnp.all(
      jnp.reshape(partner_h_1 == partner_h_2, [batch_size, -1]), axis=-1
  )
  moved = jnp.any(x2 != x1, axis=-1)

  lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
  ratio = jnp.asarray(jnp.where(same_pairing & moved, lp_2 - lp_1, -jnp.inf))
  key, accept_subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(accept_subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.asarray(jnp.where(cond[..., None], x2, x1))
  lp_new = jnp.asarray(jnp.where(cond, lp_2, lp_1))
  num_accepts = num_accepts + jnp.sum(cond)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))  # type: ignore[arg-type]  # pyrefly: ignore[no-matching-overload]

  num_map_rejects = jnp.sum((moved & (~same_pairing)).astype(jnp.float32))
  pairs_per_walker = (
      jnp.sum(
          jnp.reshape(is_paired_e_1, [batch_size, -1]).astype(jnp.float32),
          axis=-1,
      )
      / float(num_states)
  )
  total_pairs = jnp.sum(pairs_per_walker)
  has_active_pair = jnp.any(
      jnp.reshape(active_paired_h, [batch_size, -1]), axis=-1
  )
  valid_active_pair = has_active_pair & same_pairing
  active_pair_tries = jnp.sum(valid_active_pair.astype(jnp.float32))
  active_pair_accepts = jnp.sum(
      (cond & valid_active_pair).astype(jnp.float32)
  )
  return (
      new_data,
      key,
      lp_new,
      num_accepts,
      num_map_rejects,
      total_pairs,
      active_pair_tries,
      active_pair_accepts,
  )

# This and the next overload just provide type hints for the return value of mh_pair_update depending on the value of return_pair_diagnostics.
@overload
def mh_pair_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    stddev: float | jnp.ndarray = ...,
    atoms: Any = ...,
    ndim: int = ...,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = ...,
    blocks: int = ...,
    i: int | jnp.ndarray = ...,
    layer_counts: tuple[int, int] = ...,
    pair_radius: float = ...,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = ...,
    unpaired_stddev: float | jnp.ndarray | None = ...,
    pair_lattice: jnp.ndarray | np.ndarray | None = ...,
    hop_vectors: jnp.ndarray | np.ndarray | None = ...,
    return_pair_diagnostics: Literal[False] = ...,
) -> tuple[networks.FermiNetData, chex.PRNGKey, jnp.ndarray, jnp.ndarray]:
  ...


@overload
def mh_pair_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    stddev: float | jnp.ndarray = ...,
    atoms: Any = ...,
    ndim: int = ...,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = ...,
    blocks: int = ...,
    i: int | jnp.ndarray = ...,
    layer_counts: tuple[int, int] = ...,
    pair_radius: float = ...,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = ...,
    unpaired_stddev: float | jnp.ndarray | None = ...,
    pair_lattice: jnp.ndarray | np.ndarray | None = ...,
    hop_vectors: jnp.ndarray | np.ndarray | None = ...,
    *,
    return_pair_diagnostics: Literal[True],
) -> tuple[
    networks.FermiNetData,
    chex.PRNGKey,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
]:
  ...


def mh_pair_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike | Callable[..., jnp.ndarray],
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1: Any,
    num_accepts: Any,
    stddev: float | jnp.ndarray = 0.02,
    atoms: Any = None,
    ndim: int = 2,
    map_to_cell: Callable[[jnp.ndarray], jnp.ndarray] = lambda x: x,
    blocks: int = 1,
    i: int | jnp.ndarray = 0,
    layer_counts: tuple[int, int] = (0, 0),
    pair_radius: float = 0.0,
    dist_sq_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] | None = None,
    unpaired_stddev: float | jnp.ndarray | None = None,
    pair_lattice: jnp.ndarray | np.ndarray | None = None,
    hop_vectors: jnp.ndarray | np.ndarray | None = None,
    return_pair_diagnostics: bool = False,
):
  """Performs one Metropolis-Hastings step shifting bound (e, h) pairs together.

  When `pair_lattice` or `hop_vectors` is provided (the default in moire
  bilayer systems), performs an explicit inter-well pair hop: selects one
  matched mutual nearest-neighbor (e, h) pair uniformly at random per walker
  and translates both particles by a randomly chosen nearest-neighbor moire
  lattice vector while keeping all other particles fixed. When neither
  `pair_lattice` nor `hop_vectors` is provided, falls back to shifting mutual
  nearest-neighbor pairs by Gaussian displacements of width `stddev`. Exact
  detailed balance is enforced in both modes by requiring the mutual
  nearest-neighbor pairing map within `pair_radius` to be identical in the
  initial and proposed configurations.

  Args:
    params: Wavefunction parameters.
    f: Callable with LogFermiNetLike signature returning log|psi|.
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: Log probability 2*log|psi(x1)|.
    num_accepts: Running total of accepted MH moves.
    stddev: Width of Gaussian move proposal for matched pairs (used when
      `pair_lattice` and `hop_vectors` are None).
    atoms: Must be None (asymmetric nuclear scaling is not supported).
    ndim: Dimensionality of system.
    map_to_cell: Callable mapping positions into the simulation cell in PBC.
    blocks: Number of blocks to split updates into.
    i: Index of active block when blocks > 1.
    layer_counts: Tuple (n_electrons, n_holes) of particle counts per layer.
    pair_radius: Cutoff radius R_c (in Bohr) for pairing an electron and hole.
    dist_sq_fn: Pairwise squared distance function (minimum-image in PBC).
    unpaired_stddev: Optional width of Gaussian move proposal for unpaired
      particles when `pair_lattice` and `hop_vectors` are None.
    pair_lattice: Optional `(ndim, ndim)` primitive moire lattice matrix whose
      nearest-neighbor lattice vectors define discrete inter-well pair hops.
    hop_vectors: Optional `(n_vectors, ndim)` array of explicit inter-well
      lattice translation vectors.
    return_pair_diagnostics: If True, also returns `(num_map_rejects,
      total_pairs, active_pair_tries, active_pair_accepts)` for the step.

  Returns:
    Tuple `(new_data, key, lp_new, num_accepts)` by default, or
    `(new_data, key, lp_new, num_accepts, num_map_rejects, total_pairs,
    active_pair_tries, active_pair_accepts)` when
    `return_pair_diagnostics=True`.
  """
  out = _mh_pair_update_with_diagnostics(
      params=params,
      f=f,
      data=data,
      key=key,
      lp_1=lp_1,
      num_accepts=num_accepts,
      stddev=stddev,
      atoms=atoms,
      ndim=ndim,
      map_to_cell=map_to_cell,
      blocks=blocks,
      i=i,
      layer_counts=layer_counts,
      pair_radius=pair_radius,
      dist_sq_fn=dist_sq_fn,
      unpaired_stddev=unpaired_stddev,
      pair_lattice=pair_lattice,
      hop_vectors=hop_vectors,
  )
  if return_pair_diagnostics:
    return out
  return out[:4]


def make_mcmc_step(
    batch_network,
    batch_per_device,
    steps=10,
    atoms=None,
    ndim=3,
    blocks=1,
    lattice=None,
    nspins: Sequence[int] | None = None,
    layer_counts: tuple[int, int] | None = None,
    pair_prob: float = 0.0,
    pair_radius: float = 0.0,
    pair_width_scale: float = 1.0,
    pair_min_fraction: float = 2.0 / 3.0,
    pair_lattice: jnp.ndarray | np.ndarray | None = None,
    hop_vectors: jnp.ndarray | np.ndarray | None = None,
    return_stats: bool = False,
):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    ndim: Dimensionality of the system (usually 3).
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.
    lattice: If None, assume OBC. Otherwise matrix with supercell lattice
      vectors
    nspins: Optional sequence of particle counts per spin/layer channel used to
      infer (n_electrons, n_holes) when `layer_counts` is not explicitly given.
    layer_counts: Optional explicit tuple (n_electrons, n_holes).
    pair_prob: Probability in [0, 1) of performing a joint electron-hole pair
      move on each MCMC sweep. Defaults to 0.0 (disabled). Must be strictly less
      than 1.0 so ordinary single-particle moves remain present to equilibrate
      internal pair separations.
    pair_radius: Cutoff distance (in Bohr) within which mutual nearest-neighbor
      electron-hole pairs are shifted together during pair moves.
    pair_width_scale: Initial or fallback multiplicative scale factor applied to
      `width` during fallback Gaussian pair moves when `pair_lattice` and
      `hop_vectors` are None. Defaults to 1.0.
    pair_min_fraction: Minimum fraction in [0, 1] of `min(n_electrons, n_holes)`
      that must be bound into mutual nearest-neighbor pairs on average across
      walkers before joint pair sweeps are enabled. Defaults to 2/3.
    pair_lattice: Optional `(ndim, ndim)` primitive moire lattice matrix whose
      nearest-neighbor lattice vectors define discrete inter-well pair hops
      (translating one randomly selected matched pair per walker by a nearest-
      neighbor moire lattice vector while leaving all other particles fixed).
    hop_vectors: Optional `(n_vectors, ndim)` array of explicit inter-well
      lattice translation vectors.
    return_stats: If True, `mcmc_step` returns `(new_data, stats_dict)` with
      separate ordinary/pair attempt and acceptance counts, acceptance rates,
      pair-map rejection rate, Metropolis rejection rate, and mean number of
      identified pairs per walker. Defaults to False for full backward
      compatibility.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  pair_prob = float(pair_prob)
  if not 0.0 <= pair_prob < 1.0:
    raise ValueError(
        'pair_prob must be in [0, 1) so ordinary single-particle moves are '
        'present to equilibrate internal pair separations and partner '
        f'changes, got {pair_prob}.'
    )
  pair_radius = float(pair_radius)
  pair_width_scale = float(pair_width_scale) # Ratio of pair move width to ordinary move width.
  if pair_width_scale <= 0.0:
    raise ValueError(
        f'pair_width_scale must be positive, got {pair_width_scale}.'
    )
  pair_min_fraction = float(pair_min_fraction)
  if not 0.0 <= pair_min_fraction <= 1.0:
    raise ValueError(
        f'pair_min_fraction must be in [0, 1], got {pair_min_fraction}.'
    )

  resolved_layer_counts = _resolve_layer_counts(nspins, layer_counts)
  n_e, n_h = resolved_layer_counts
  use_pair_moves = (pair_prob > 0.0) and (n_e > 0) and (n_h > 0)
  min_pairs_required = pair_min_fraction * float(min(n_e, n_h))
  if use_pair_moves and pair_radius <= 0.0:
    raise ValueError(
        'pair_radius must be > 0 when pair_prob > 0 and both carrier layers '
        f'are occupied, got pair_radius={pair_radius}.'
    )

  if hop_vectors is not None:
    validated_hop_vectors = _validate_hop_vectors(hop_vectors, ndim=ndim)
  else:
    validated_hop_vectors = None

  if use_pair_moves and (
      validated_hop_vectors is not None or pair_lattice is not None
  ):
    resolved_hop_vectors = (
        validated_hop_vectors
        if validated_hop_vectors is not None
        else _make_moire_hop_vectors(pair_lattice, ndim=ndim)  # type: ignore[arg-type]
    )
  else:
    resolved_hop_vectors = None

  inner_fun = mh_block_update if blocks > 1 else mh_update

  # If PBC, map electrons back to the supercell
  if lattice is not None:
    rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
    map_to_cell = lambda p: map_to_simulation_cell(p, lattice, rec, ndim)
  else:
    rec = None
    map_to_cell = lambda p: p

  dist_sq_fn = (
      _make_minimum_image_dist_sq_fn(lattice) if use_pair_moves else None
  )
  latch_state = {'pair_moves_enabled': pair_min_fraction <= 0.0}

  def mcmc_step(params, data, key, width) -> tuple[networks.FermiNetData, Any]:
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal. May be a scalar
        (in which case ordinary moves use `width` and pair moves use
        `width * pair_width_scale`), a length-2 array `[width_normal,
        width_pair]`, or a length-3 array `[width_normal, width_pair,
        pair_moves_enabled]` where the third entry is a latched warm-up flag
        (0.0 before activation, 1.0 once activated).

    Returns:
      `(data, pmove)` when `return_stats=False`, or `(data, stats_dict)` when
      `return_stats=True`.
    """
    pos = data.positions
    width_arr = jnp.asarray(width)
    has_explicit_enable_flag = (
        width_arr.ndim > 0 and width_arr.shape[-1] == 3
    )
    if has_explicit_enable_flag:
      normal_width = width_arr[..., 0]
      pair_width = width_arr[..., 1]
      explicit_pair_enabled = jnp.any(width_arr[..., 2] > 0.5)
    elif width_arr.ndim > 0 and width_arr.shape[-1] == 2:
      normal_width = width_arr[..., 0]
      pair_width = width_arr[..., 1]
      explicit_pair_enabled = None
    else:
      normal_width = width
      pair_width = width * pair_width_scale
      explicit_pair_enabled = None

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    
    #Just normal MH update, no pair moves
    if not use_pair_moves and not return_stats:
      def legacy_single_step_fn(i, x):
        return inner_fun( 
            params,
            batch_network,
            *x,
            stddev=normal_width,
            atoms=atoms,
            ndim=ndim,
            map_to_cell=map_to_cell,
            blocks=blocks,
            i=i,
        )

      new_data, key, _, num_accepts = lax.fori_loop(
          0, nsteps, legacy_single_step_fn, (data, key, logprob, 0.0)
      )
      pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
      pmove = _safe_pmean(pmove)
      return new_data, pmove

    if use_pair_moves:
      nelec = n_e + n_h
      num_states = max(1, pos.shape[-1] // (nelec * ndim))
      pos_init_3d = jnp.reshape(
          pos, [batch_per_device * num_states, nelec, ndim]
      )
      _, is_paired_init = _find_mutual_eh_pairs(
          pos_init_3d[:, :n_e, :],
          pos_init_3d[:, n_e:, :],
          pair_radius,
          dist_sq_fn,
      )
      init_pairs_per_walker = (
          jnp.sum(
              jnp.reshape(is_paired_init, [batch_per_device, -1]).astype(
                  jnp.float32
              ),
              axis=-1,
          )
          / float(num_states)
      )
      init_mean_pairs = _safe_pmean(jnp.mean(init_pairs_per_walker))
      if has_explicit_enable_flag:
        assert explicit_pair_enabled is not None
        pair_gate_open = explicit_pair_enabled
      elif min_pairs_required <= 0.0:
        pair_gate_open = jnp.asarray(True, dtype=bool)
      elif not isinstance(init_mean_pairs, jax.core.Tracer):
        if (
            not latch_state['pair_moves_enabled']
            and float(init_mean_pairs) >= min_pairs_required
        ):
          latch_state['pair_moves_enabled'] = True
        pair_gate_open = jnp.asarray(
            latch_state['pair_moves_enabled'], dtype=bool
        )
      else:
        raise ValueError(
            'JIT/pmap-compiled pair MCMC with pair_min_fraction > 0 requires '
            'a 3-element width state [width_normal, width_pair, '
            'pair_moves_enabled] (see mcmc.init_pair_mcmc_width), or set '
            'pair_min_fraction=0.0 for a fixed schedule.'
        )

      key, mode_key = jax.random.split(key)
      do_pair_sweeps = jax.random.bernoulli(
          mode_key, p=pair_prob, shape=(max(1, steps),)
      )
      if steps >= 2:
        # Guarantee at least one ordinary single-particle sweep per mcmc_step
        # call so internal relative coordinates always equilibrate.
        do_pair_sweeps = jnp.where(
            jnp.all(do_pair_sweeps),
            do_pair_sweeps.at[0].set(False),
            do_pair_sweeps,
        )
      do_pair_sweeps = jnp.where(
          pair_gate_open,
          do_pair_sweeps,
          jnp.zeros_like(do_pair_sweeps),
      )
    else:
      init_mean_pairs = jnp.asarray(jnp.nan, dtype=jnp.float32)
      pair_gate_open = jnp.asarray(False, dtype=bool)
      do_pair_sweeps = jnp.zeros((max(1, steps),), dtype=bool)

    batch_f = jnp.asarray(float(batch_per_device), dtype=jnp.float32)

    # carry=(cur_data, cur_key, cur_lp, acc_n, try_n, acc_p, try_p, acc_p_active, try_p_active, map_rej_p, sum_pairs_p) and holds the current MCMC state and running diagnostics for the ordinary and pair moves. 
    def tracked_normal_step(i, carry):
      (
          cur_data,
          cur_key,
          cur_lp,
          acc_n,
          try_n,
          acc_p,
          try_p,
          acc_p_active,
          try_p_active,
          map_rej_p,
          sum_pairs_p,
      ) = carry
      next_data, next_key, next_lp, step_acc = inner_fun(
          params,
          batch_network,
          cur_data,
          cur_key,
          cur_lp,
          0.0,
          stddev=normal_width,
          atoms=atoms,
          ndim=ndim,
          map_to_cell=map_to_cell,
          blocks=blocks,
          i=i,
      )
      acc_n = acc_n + jnp.sum(step_acc).astype(jnp.float32) #step_acc is number of walkers accepted in this step on this GPU 
      try_n = try_n + batch_f #batch_f is number of walkers on this GPU
      return (
          next_data,
          next_key,
          next_lp,
          acc_n,
          try_n,
          acc_p,
          try_p,
          acc_p_active,
          try_p_active,
          map_rej_p,
          sum_pairs_p,
      )

    def tracked_pair_step(i, carry):
      (
          cur_data,
          cur_key,
          cur_lp,
          acc_n,
          try_n,
          acc_p,
          try_p,
          acc_p_active,
          try_p_active,
          map_rej_p,
          sum_pairs_p,
      ) = carry
      (
          next_data,
          next_key,
          next_lp,
          step_acc,
          step_map_rej,
          step_pairs,
          step_try_active,
          step_acc_active,
      ) = _mh_pair_update_with_diagnostics(
          params,
          batch_network,
          cur_data,
          cur_key,
          cur_lp,
          0.0,
          stddev=pair_width,
          atoms=atoms,
          ndim=ndim,
          map_to_cell=map_to_cell,
          blocks=blocks,
          i=i,
          layer_counts=resolved_layer_counts,
          pair_radius=pair_radius,
          dist_sq_fn=dist_sq_fn,
          unpaired_stddev=normal_width,
          hop_vectors=resolved_hop_vectors,
      )
      acc_p = acc_p + jnp.sum(step_acc).astype(jnp.float32) #step_acc is number of walkers accepted in this step on this GPU
      try_p = try_p + batch_f #batch_f is number of walkers on this GPU
      acc_p_active = acc_p_active + jnp.sum(step_acc_active).astype(jnp.float32) #Walker being active just means that is has at least one pair
      try_p_active = try_p_active + jnp.sum(step_try_active).astype(jnp.float32)
      map_rej_p = map_rej_p + jnp.sum(step_map_rej).astype(jnp.float32) #Number of walker proposals rejected due to pairing map changed after proposal
      sum_pairs_p = sum_pairs_p + jnp.sum(step_pairs).astype(jnp.float32) #step_pairs is sum of number of pairs over all walkers on this GPU, sum_pairs_p accumulates over all paired-update steps. This is to calculate mean_num_pairs, which is the mean number of pairs per walker over all paired-update steps.
      return (
          next_data,
          next_key,
          next_lp,
          acc_n,
          try_n,
          acc_p,
          try_p,
          acc_p_active,
          try_p_active,
          map_rej_p,
          sum_pairs_p,
      )

    # To get the normal or pair step depending on do_pair_sweeps
    def step_fn(i, carry):
      if not use_pair_moves:
        return tracked_normal_step(i, carry)
      # do_pair_sweeps is boolean array of length steps, indicating whether to do a pair move or not for each step.
      do_pair = do_pair_sweeps[i // blocks] # i//blocks determine which block we're in
      return lax.cond(
          do_pair,
          lambda args: tracked_pair_step(*args),
          lambda args: tracked_normal_step(*args),
          (i, carry),
      )

    zero_f = jnp.zeros((), dtype=jnp.float32)
    init_carry = (
        data,
        key,
        logprob,
        zero_f,
        zero_f,
        zero_f,
        zero_f,
        zero_f,
        zero_f,
        zero_f,
        zero_f,
    )
    
    (
        new_data,
        key,
        _,
        n_acc_normal,
        n_try_normal,
        n_acc_pair,
        n_try_pair,
        n_acc_pair_active,
        n_try_pair_active,
        n_map_rej_pair,
        sum_pairs,
    ) = lax.fori_loop(0, nsteps, step_fn, init_carry) #So here we run the MCMC steps, and get the final data, key, and the accumulated statistics for normal and pair moves.

    n_acc_normal = _safe_psum(n_acc_normal) #constants.psum sums a value across the devices participating in the parallel calculation
    n_try_normal = _safe_psum(n_try_normal)
    n_acc_pair = _safe_psum(n_acc_pair)
    n_try_pair = _safe_psum(n_try_pair)
    n_acc_pair_active = _safe_psum(n_acc_pair_active)
    n_try_pair_active = _safe_psum(n_try_pair_active)
    n_map_rej_pair = _safe_psum(n_map_rej_pair)
    sum_pairs = _safe_psum(sum_pairs)

    total_acc = n_acc_normal + n_acc_pair
    total_try = n_try_normal + n_try_pair
    pmove = jnp.where(
        total_try > 0.0, total_acc / jnp.maximum(total_try, 1.0), 0.0
    )
    if not return_stats:
      return new_data, pmove

    pmove_normal = jnp.where(
        n_try_normal > 0.0,
        n_acc_normal / jnp.maximum(n_try_normal, 1.0),
        jnp.nan,
    )
    pmove_pair = jnp.where(
        n_try_pair > 0.0,
        n_acc_pair / jnp.maximum(n_try_pair, 1.0),
        jnp.nan,
    )
    pmove_pair_active = jnp.where(
        n_try_pair_active > 0.0,
        n_acc_pair_active / jnp.maximum(n_try_pair_active, 1.0),
        jnp.nan,
    )
    pair_map_reject_rate = jnp.where(
        n_try_pair > 0.0,
        n_map_rej_pair / jnp.maximum(n_try_pair, 1.0),
        jnp.nan,
    )
    n_mh_rej_pair = jnp.maximum(n_try_pair_active - n_acc_pair_active, 0.0)
    pair_mh_reject_rate = jnp.where(
        n_try_pair > 0.0,
        n_mh_rej_pair / jnp.maximum(n_try_pair, 1.0),
        jnp.nan,
    )
    mean_num_pairs = jnp.where(
        n_try_pair > 0.0,
        sum_pairs / jnp.maximum(n_try_pair, 1.0),
        init_mean_pairs,
    )
    stats = {
        'pmove': pmove,
        'pmove_normal': pmove_normal,
        'pmove_pair': pmove_pair,
        'pmove_pair_active': pmove_pair_active,
        'n_try_normal': n_try_normal,
        'n_acc_normal': n_acc_normal,
        'n_try_pair': n_try_pair,
        'n_acc_pair': n_acc_pair,
        'n_try_pair_active': n_try_pair_active,
        'n_acc_pair_active': n_acc_pair_active,
        'n_map_rej_pair': n_map_rej_pair,
        'n_mh_rej_pair': n_mh_rej_pair,
        'pair_map_reject_rate': pair_map_reject_rate,
        'pair_mh_reject_rate': pair_mh_reject_rate,
        'mean_num_pairs': mean_num_pairs,
        'min_pairs_required': jnp.asarray(min_pairs_required, dtype=jnp.float32),
        'pair_moves_enabled': jnp.asarray(pair_gate_open, dtype=bool),
    }
    return new_data, stats

  mcmc_step.pair_state = latch_state  # type: ignore[attr-defined]
  return mcmc_step


def init_pair_mcmc_width(
    move_width: float,
    pair_width_scale: float = 1.0,
    pair_min_fraction: float = 2.0 / 3.0,
    pair_moves_enabled: bool | None = None,
    pair_hop_distance: float | None = None,
) -> jnp.ndarray:
  """Initializes `[width_normal, width_pair, pair_moves_enabled]` state."""
  if pair_moves_enabled is None:
    pair_moves_enabled = float(pair_min_fraction) <= 0.0
  w0 = float(move_width)
  wp = (
      float(pair_hop_distance)
      if pair_hop_distance is not None
      else w0 * float(pair_width_scale)
  )
  return jnp.asarray(
      [
          w0,
          wp,
          1.0 if pair_moves_enabled else 0.0,
      ],
      dtype=jnp.float32,
  )


def init_pair_mcmc_adaptation_state(
    adapt_frequency: int,
    pair_moves_enabled: bool = True,
    adapt_pair_width: bool = True,
) -> dict[str, Any]:
  """Initializes window buffers and latch state for pair MCMC adaptation."""
  n = max(1, int(adapt_frequency))
  return {
      'try_normal': np.zeros(n, dtype=np.float64),
      'acc_normal': np.zeros(n, dtype=np.float64),
      'try_pair': np.zeros(n, dtype=np.float64),
      'acc_pair': np.zeros(n, dtype=np.float64),
      'pair_moves_enabled': bool(pair_moves_enabled),
      'pair_enabled_steps': n if pair_moves_enabled else 0,
      'adapt_pair_width': bool(adapt_pair_width),
  }


def update_pair_mcmc_widths(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    mcmc_stats: Mapping[str, jnp.ndarray | np.ndarray | float],
    adapt_state: dict[str, Any],
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
    min_pairs_required: float | None = None,
    freeze_pair_schedule: bool = False,
    adapt_pair_width: bool | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
  """Adapts `normal_width` (and optionally `pair_width`) and latches pair moves.

  Accumulates attempt and acceptance counts for ordinary and pair moves across
  the `adapt_frequency` window before dividing:
    - `A_n = sum(N_{acc,n}) / sum(N_{try,n})` updates `width[..., 0]` (sigma_n)
    - When `adapt_pair_width` is True, `A_p = sum(N_{acc,p,active}) /
      sum(N_{try,p,active})` updates `width[..., 1]` (sigma_p). For discrete
      inter-well lattice hops (`adapt_pair_width=False`), `width[..., 1]` is
      kept fixed at the moire lattice spacing and never adapted toward 50-55%.
  If `width` has a 3rd entry (`pair_moves_enabled`) or `adapt_state` starts
  with `pair_moves_enabled=False`, once `mean_num_pairs >= min_pairs_required`
  is reached, `pair_moves_enabled` latches permanently to `True` (`width[..., 2]
  = 1.0`) and never switches back off.
  """
  if adapt_pair_width is None:
    do_adapt_pair_width = bool(adapt_state.get('adapt_pair_width', True))
  else:
    do_adapt_pair_width = bool(adapt_pair_width)

  has_enable_slot = width.ndim > 0 and width.shape[-1] == 3
  if has_enable_slot:
    currently_enabled = bool(np.asarray(width[..., 2]).reshape(-1)[0] > 0.5)
  else:
    currently_enabled = bool(adapt_state.get('pair_moves_enabled', True))
  currently_enabled = currently_enabled or bool(
      adapt_state.get('pair_moves_enabled', False)
  )
  was_enabled_on_step = currently_enabled

  if not currently_enabled and not freeze_pair_schedule:
    if min_pairs_required is None:
      if 'min_pairs_required' in mcmc_stats:
        threshold = float(
            np.asarray(mcmc_stats['min_pairs_required']).reshape(-1)[0]
        )
      else:
        threshold = 0.0
    else:
      threshold = float(min_pairs_required)
    mean_pairs = float(
        np.asarray(mcmc_stats.get('mean_num_pairs', np.nan)).reshape(-1)[0]
    )
    if not np.isnan(mean_pairs) and mean_pairs >= threshold:
      currently_enabled = True
      adapt_state['try_pair'][:] = 0.0
      adapt_state['acc_pair'][:] = 0.0
      adapt_state['pair_enabled_steps'] = 0

  adapt_state['pair_moves_enabled'] = currently_enabled
  if has_enable_slot and currently_enabled:
    width = width.at[..., 2].set(1.0)

  if adapt_frequency <= 0 or freeze_pair_schedule:
    return width, adapt_state
  t_since_update = t % adapt_frequency
  try_p_key = (
      'n_try_pair_active'
      if 'n_try_pair_active' in mcmc_stats
      else 'n_try_pair'
  )
  acc_p_key = (
      'n_acc_pair_active'
      if 'n_acc_pair_active' in mcmc_stats
      else 'n_acc_pair'
  )
  adapt_state['try_normal'][t_since_update] = float(
      np.asarray(mcmc_stats['n_try_normal']).reshape(-1)[0]
  )
  adapt_state['acc_normal'][t_since_update] = float(
      np.asarray(mcmc_stats['n_acc_normal']).reshape(-1)[0]
  )
  adapt_state['try_pair'][t_since_update] = float(
      np.asarray(mcmc_stats[try_p_key]).reshape(-1)[0]
  )
  adapt_state['acc_pair'][t_since_update] = float(
      np.asarray(mcmc_stats[acc_p_key]).reshape(-1)[0]
  )
  if was_enabled_on_step:
    adapt_state['pair_enabled_steps'] = (
        int(adapt_state.get('pair_enabled_steps', adapt_frequency)) + 1
    )

  if t > 0 and t_since_update == 0:
    total_try_n = float(np.sum(adapt_state['try_normal']))
    if total_try_n > 0.0:
      a_n = float(np.sum(adapt_state['acc_normal'])) / total_try_n
      if a_n > pmove_max:
        width = width.at[..., 0].multiply(1.1)
      elif a_n < pmove_min:
        width = width.at[..., 0].divide(1.1)

    pair_ready = (
        do_adapt_pair_width
        and currently_enabled
        and int(adapt_state.get('pair_enabled_steps', adapt_frequency))
        >= adapt_frequency
    )
    total_try_p = float(np.sum(adapt_state['try_pair']))
    if pair_ready and total_try_p > 0.0:
      a_p = float(np.sum(adapt_state['acc_pair'])) / total_try_p
      if a_p > pmove_max:
        width = width.at[..., 1].multiply(1.1)
      elif a_p < pmove_min:
        width = width.at[..., 1].divide(1.1)

  return width, adapt_state


def update_mcmc_width(
    t: int,
    width: jnp.ndarray,
    adapt_frequency: int,
    pmove: jnp.ndarray,
    pmoves: np.ndarray,
    pmove_max: float = 0.55,
    pmove_min: float = 0.5,
) -> tuple[jnp.ndarray, np.ndarray]:
  """Updates the width in MCMC steps.

  Args:
    t: Current step.
    width: Current MCMC width.
    adapt_frequency: The number of iterations after which the update is applied.
    pmove: Acceptance ratio in the last step.
    pmoves: Acceptance ratio over the last N steps, where N is the number of
      steps between MCMC width updates.
    pmove_max: The upper threshold for the range of allowed pmove values
    pmove_min: The lower threshold for the range of allowed pmove values

  Returns:
    width: Updated MCMC width.
    pmoves: Updated `pmoves`.
  """

  t_since_mcmc_update = t % adapt_frequency
  # update `pmoves`; `pmove` should be the same across devices
  pmoves[t_since_mcmc_update] = pmove.reshape(-1)[0].item()
  if t > 0 and t_since_mcmc_update == 0:
    if np.mean(pmoves) > pmove_max:
      width *= 1.1
    elif np.mean(pmoves) < pmove_min:
      width /= 1.1
  return width, pmoves
