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

"""2D-periodic Ewald summation for electron-hole bilayers with finite separation."""

from typing import Callable, Optional, Sequence, Tuple

from ferminet.moire import geometry
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special as scipy_special

# Reciprocal vectors whose coplanar Ewald weight falls below this threshold
# are dropped during setup.
_GWEIGHT_TOL = 1e-12


def _halfspace_gpoint_indices_2d(ewald_gmax: int) -> np.ndarray:
  """Enumerates non-zero 2D integer indices covering half of reciprocal space.

  Returns all integer pairs ``(m_1, m_2)`` in ``[-ewald_gmax, ewald_gmax]^2``
  whose first non-zero entry is positive, avoiding double-counting of ``G`` and
  ``-G``.

  Args:
    ewald_gmax: Maximum integer index in each reciprocal lattice direction.

  Returns:
    NumPy integer array of shape ``(npoints, 2)``.
  """
  if ewald_gmax < 1:
    raise ValueError(f'ewald_gmax must be >= 1, got {ewald_gmax}.')
  full = np.arange(-ewald_gmax, ewald_gmax + 1, dtype=np.int32)
  positive = np.arange(1, ewald_gmax + 1, dtype=np.int32)
  zero = np.array([0], dtype=np.int32)

  # Block 0: m_1 > 0, m_2 in [-gmax, gmax]
  g0_1, g0_2 = np.meshgrid(positive, full, indexing='ij')
  block0 = np.stack([g0_1.ravel(), g0_2.ravel()], axis=-1)

  # Block 1: m_1 == 0, m_2 > 0
  g1_1, g1_2 = np.meshgrid(zero, positive, indexing='ij')
  block1 = np.stack([g1_1.ravel(), g1_2.ravel()], axis=-1)

  return np.concatenate([block0, block1], axis=0)


class BilayerEwaldSum:
  r"""2D-periodic Ewald summation for an electron-hole bilayer.

  Evaluates the 2D-periodic sums of the 3D Coulomb kernels screened by a
  constant relative dielectric permittivity ``epsilon_r``:

  .. math::
      v_{ee}(r) = v_{hh}(r) = \frac{1}{\epsilon_r r}, \qquad
      v_{eh}(r) = -\frac{1}{\epsilon_r \sqrt{r^2 + d^2}},

  where ``r`` is the in-plane distance and ``d >= 0`` is the fixed interlayer
  separation between the electron plane (``z_e = 0``) and the hole plane
  (``z_h = d``).

  Uses Parry's 2D Ewald summation formula so that the real-space,
  reciprocal-space (``G != 0``), self-interaction, and zero-wavevector
  (``G = 0``) contributions are mutually consistent and strictly independent of
  the Ewald splitting parameter ``alpha``.
  """

  def __init__(
      self,
      lattice: geometry.Array,
      nspins: Sequence[int],
      interlayer_distance: float = 0.0,
      dielectric_constant: float = 1.0,
      ewald_gmax: int = 100,
      nlatvec: int = 1,
      alpha: Optional[float] = None,
      include_capacitor: bool = True,
      allow_charged_capacitor: bool = False,
  ):
    """Initializes precomputed lattice and reciprocal-space Ewald tables.

    Args:
      lattice: Shape ``(2, 2)`` simulation supercell matrix (columns are direct
        lattice vectors).
      nspins: Sequence ``(n_e_up, n_e_down, n_h_up, n_h_down)`` (4 channels) or
        ``(n_electrons, n_holes)`` (2 channels) of particle counts.
      interlayer_distance: Out-of-plane distance ``d >= 0`` between electron and
        hole layers in Bohr.
      dielectric_constant: Relative dielectric constant ``epsilon_r > 0``.
      ewald_gmax: Reciprocal-space cutoff index in each direction.
      nlatvec: Real-space image shell cutoff around the minimum-image vector.
      alpha: Optional override for Ewald splitting parameter ``alpha``. Defaults
        to ``5.0 / h_min`` where ``h_min`` is the smallest perpendicular height
        of the simulation cell.
      include_capacitor: If True (default), include the physical ``G = 0``
        parallel-plate capacitor contribution ``+(2 * pi * d / (epsilon_r * A))
        * N_e * N_h`` for separated layers. If False, subtract this constant
        capacitor shift (corresponding to per-layer neutralizing backgrounds).
      allow_charged_capacitor: If False (default), raises ``ValueError`` when
        ``n_electrons != n_holes``, ``interlayer_distance > 0``, and
        ``include_capacitor=True``, since a net-charged periodic bilayer has an
        ambiguous ``G = 0`` electrostatic energy unless the neutralizing
        background convention is explicitly chosen.
    """
    self.n_electrons, self.n_holes = geometry.layer_particle_counts(nspins)
    if interlayer_distance < 0.0:
      raise ValueError(
          f'interlayer_distance must be >= 0, got {interlayer_distance}.'
      )
    if dielectric_constant <= 0.0:
      raise ValueError(
          f'dielectric_constant must be > 0, got {dielectric_constant}.'
      )
    if nlatvec < 0:
      raise ValueError(f'nlatvec must be >= 0, got {nlatvec}.')

    self.latvec_np = geometry.validate_lattice_2d(lattice, name='lattice')
    self.interlayer_distance = float(interlayer_distance)
    self.dielectric_constant = float(dielectric_constant)
    self.include_capacitor = bool(include_capacitor)
    self.allow_charged_capacitor = bool(allow_charged_capacitor)

    if (
        self.n_electrons != self.n_holes
        and self.interlayer_distance > 0.0
        and self.include_capacitor
        and not self.allow_charged_capacitor
    ):
      raise ValueError(
          f'Charge-imbalanced bilayer (n_electrons={self.n_electrons}, '
          f'n_holes={self.n_holes}) with interlayer_distance='
          f'{self.interlayer_distance} > 0 has an ambiguous G=0 capacitor '
          'energy unless the neutralizing-background convention is specified. '
          'Set include_capacitor=False for per-layer neutralizing backgrounds, '
          'or set allow_charged_capacitor=True to explicitly evaluate the '
          'uncompensated interlayer G=0 term.'
      )

    self.area = float(abs(np.linalg.det(self.latvec_np)))
    self.dist_fn = geometry.make_minimum_image_distance_fn(self.latvec_np)

    # Real-space lattice displacements L = n_1 a_1 + n_2 a_2.
    grid = np.meshgrid(
        np.arange(-nlatvec, nlatvec + 1, dtype=np.int32),
        np.arange(-nlatvec, nlatvec + 1, dtype=np.int32),
        indexing='ij',
    )
    indices = np.stack([grid[0].ravel(), grid[1].ravel()], axis=-1)
    self.lattice_displacements = jnp.asarray(
        indices @ self.latvec_np.T, dtype=jnp.float32
    )
    self.center_image_idx = int(
        np.argmin(np.linalg.norm(indices @ self.latvec_np.T, axis=-1))
    )

    # Reciprocal-space basis (rows are b_1 / (2*pi), b_2 / (2*pi)).
    inv_lat = np.linalg.inv(self.latvec_np)
    smallest_height = float(np.amin(1.0 / np.linalg.norm(inv_lat, axis=1)))
    self.alpha = float(5.0 / smallest_height if alpha is None else alpha)
    if self.alpha <= 0.0:
      raise ValueError(f'Ewald alpha must be > 0, got {self.alpha}.')

    g_indices = _halfspace_gpoint_indices_2d(ewald_gmax)
    rec_rows = 2.0 * np.pi * inv_lat  # rows are b_1, b_2
    gpoints_np = g_indices @ rec_rows
    gnorm_np = np.linalg.norm(gpoints_np, axis=-1)

    w0_np = (
        2.0
        * np.pi
        * scipy_special.erfc(gnorm_np / (2.0 * self.alpha))
        / (self.area * gnorm_np)
    )
    keep = w0_np > _GWEIGHT_TOL
    gpoints_kept = gpoints_np[keep]
    gnorm_kept = gnorm_np[keep]
    w0_kept = w0_np[keep]

    self.gpoints = jnp.asarray(gpoints_kept)
    self.gweight_coplanar = jnp.asarray(w0_kept)

    # Interlayer reciprocal weight for half-space G-vectors:
    # W_d(G) = (pi / (A G)) * [exp(-G^2/(4 alpha^2) - alpha^2 d^2) *
    #          erfcx(G/(2 alpha) + alpha d) + exp(-G d) * erfc(G/(2 alpha) - alpha d)]
    d = self.interlayer_distance
    if d == 0.0:
      self.gweight_interlayer = self.gweight_coplanar
    else:
      u_plus = gnorm_kept / (2.0 * self.alpha) + self.alpha * d
      u_minus = gnorm_kept / (2.0 * self.alpha) - self.alpha * d
      term_plus = np.exp(
          -(gnorm_kept**2) / (4.0 * self.alpha**2) - (self.alpha * d) ** 2
      ) * scipy_special.erfcx(u_plus)
      term_minus = np.exp(-gnorm_kept * d) * scipy_special.erfc(u_minus)
      wd_kept = np.pi * (term_plus + term_minus) / (self.area * gnorm_kept)
      self.gweight_interlayer = jnp.asarray(wd_kept)

    # Self-interaction and G=0 constants.
    self.self_const_factor = -self.alpha / np.sqrt(np.pi)
    self.g0_coplanar_factor = -2.0 * np.sqrt(np.pi) / (self.area * self.alpha)
    if include_capacitor:
      self.g0_interlayer_factor = float(
          (2.0 * np.sqrt(np.pi) / self.area)
          * (
              np.exp(-((self.alpha * d) ** 2)) / self.alpha
              + np.sqrt(np.pi) * d * scipy_special.erf(self.alpha * d)
          )
      )
    else:
      self.g0_interlayer_factor = float(
          (2.0 * np.sqrt(np.pi) / self.area)
          * (
              np.exp(-((self.alpha * d) ** 2)) / self.alpha
              - np.sqrt(np.pi) * d * scipy_special.erfc(self.alpha * d)
          )
      )

  def _coplanar_energy(self, coords: jnp.ndarray) -> jnp.ndarray:
    """Computes unscreened periodic energy for one coplanar species."""
    n_parts = coords.shape[0]
    if n_parts == 0:
      return jnp.asarray(0.0, dtype=coords.dtype)

    dtype = coords.dtype
    lat_disp = self.lattice_displacements.astype(dtype)
    gpoints = self.gpoints.astype(dtype)
    gweight = self.gweight_coplanar.astype(dtype)

    displacements, _ = self.dist_fn(coords, coords)
    rvec = displacements[None, ...] + lat_disp[:, None, None, :]
    r = jnp.linalg.norm(rvec, axis=-1)

    n_imgs = lat_disp.shape[0]
    mask = jnp.ones((n_imgs, n_parts, n_parts), dtype=dtype)
    mask = mask.at[self.center_image_idx].set(1.0 - jnp.eye(n_parts, dtype=dtype))

    r_safe = jnp.where(mask < 0.5, 1.0, jnp.maximum(r, 1e-12))
    pot_term = jax.lax.erfc(self.alpha * r_safe) / r_safe
    v_real = 0.5 * jnp.sum(pot_term * mask)

    phases = gpoints @ coords.T
    sf_real = jnp.sum(jnp.cos(phases), axis=-1)
    sf_imag = jnp.sum(jnp.sin(phases), axis=-1)
    v_recip = jnp.dot(gweight, sf_real**2 + sf_imag**2)

    v_self = jnp.asarray(self.self_const_factor, dtype=dtype) * n_parts
    v_g0 = 0.5 * jnp.asarray(self.g0_coplanar_factor, dtype=dtype) * (
        n_parts**2
    )
    return v_real + v_recip + v_self + v_g0

  def _interlayer_energy(
      self, pos_e: jnp.ndarray, pos_h: jnp.ndarray
  ) -> jnp.ndarray:
    """Computes unscreened periodic interaction energy between e and h layers."""
    n_e = pos_e.shape[0]
    n_h = pos_h.shape[0]
    if n_e == 0 or n_h == 0:
      return jnp.asarray(0.0, dtype=pos_e.dtype)

    dtype = pos_e.dtype
    lat_disp = self.lattice_displacements.astype(dtype)
    gpoints = self.gpoints.astype(dtype)
    gweight_d = self.gweight_interlayer.astype(dtype)
    d_sq = jnp.asarray(self.interlayer_distance**2, dtype=dtype)

    displacements, _ = self.dist_fn(pos_e, pos_h)  # (n_e, n_h, 2)
    rvec = displacements[None, ...] + lat_disp[:, None, None, :]
    r_3d = jnp.sqrt(jnp.sum(rvec**2, axis=-1) + d_sq)
    r_safe = jnp.maximum(r_3d, 1e-12)
    v_real = -jnp.sum(jax.lax.erfc(self.alpha * r_safe) / r_safe)

    phases_e = gpoints @ pos_e.T
    phases_h = gpoints @ pos_h.T
    se_real = jnp.sum(jnp.cos(phases_e), axis=-1)
    se_imag = jnp.sum(jnp.sin(phases_e), axis=-1)
    sh_real = jnp.sum(jnp.cos(phases_h), axis=-1)
    sh_imag = jnp.sum(jnp.sin(phases_h), axis=-1)
    re_se_sh_conj = se_real * sh_real + se_imag * sh_imag
    v_recip = -2.0 * jnp.dot(gweight_d, re_se_sh_conj)

    v_g0 = jnp.asarray(self.g0_interlayer_factor, dtype=dtype) * (n_e * n_h)
    return v_real + v_recip + v_g0

  def components(
      self, pos_e: jnp.ndarray, pos_h: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns screened Coulomb components ``(U_ee, U_hh, U_eh)`` in Hartree.

    Args:
      pos_e: Electron in-plane positions, shape ``(n_electrons, 2)``.
      pos_h: Hole in-plane positions, shape ``(n_holes, 2)``.

    Returns:
      Tuple ``(u_ee, u_hh, u_eh)`` of scalar JAX arrays.
    """
    inv_eps = 1.0 / self.dielectric_constant
    u_ee = inv_eps * self._coplanar_energy(pos_e)
    u_hh = inv_eps * self._coplanar_energy(pos_h)
    u_eh = inv_eps * self._interlayer_energy(pos_e, pos_h)
    return u_ee, u_hh, u_eh

  def energy(self, positions: jnp.ndarray) -> jnp.ndarray:
    """Evaluates total screened Coulomb energy ``U_ee + U_hh + U_eh``."""
    coords = jnp.reshape(positions, (self.n_electrons + self.n_holes, 2))
    pos_e = coords[: self.n_electrons]
    pos_h = coords[self.n_electrons :]
    u_ee, u_hh, u_eh = self.components(pos_e, pos_h)
    return u_ee + u_hh + u_eh


def make_bilayer_ewald_potential(
    lattice: geometry.Array,
    nspins: Sequence[int],
    interlayer_distance: float = 0.0,
    dielectric_constant: float = 1.0,
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    alpha: Optional[float] = None,
    include_capacitor: bool = True,
    allow_charged_capacitor: bool = False,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Creates a callable evaluating the 2D-periodic bilayer Coulomb energy.

  Args:
    lattice: Shape ``(2, 2)`` simulation supercell matrix (columns are vectors).
    nspins: Sequence ``(n_electrons, n_holes)``.
    interlayer_distance: Out-of-plane separation ``d >= 0`` in Bohr.
    dielectric_constant: Relative dielectric constant ``epsilon_r > 0``.
    ewald_gmax: Reciprocal-space integer cutoff per direction.
    nlatvec: Real-space image cutoff around the minimum-image displacement.
    alpha: Optional override for Ewald splitting parameter.
    include_capacitor: Whether to include the ``G = 0`` capacitor term.
    allow_charged_capacitor: Whether to permit ``include_capacitor=True`` when
      ``n_electrons != n_holes`` and ``interlayer_distance > 0``.

  Returns:
    Callable ``potential_fn(positions) -> scalar`` returning ``U_ee + U_hh + U_eh``.
  """
  ewald = BilayerEwaldSum(
      lattice=lattice,
      nspins=nspins,
      interlayer_distance=interlayer_distance,
      dielectric_constant=dielectric_constant,
      ewald_gmax=ewald_gmax,
      nlatvec=nlatvec,
      alpha=alpha,
      include_capacitor=include_capacitor,
      allow_charged_capacitor=allow_charged_capacitor,
  )
  return ewald.energy
