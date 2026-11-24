"""Strained 0-deg P-MoSe2 conduction-band Q-valley continuum model.

Constructs the DFT/STM-calibrated adiabatic 1-band continuum potential
``V_eff(r)``, multi-harmonic 2D Fourier cosine decomposition, rotated 2x2
anisotropic effective mass tensor ``M_Q``, Gamma-point Bloch miniband levels,
and top/bottom layer-spinor weights for open-surface STM parallel-stacked
twisted bilayer MoSe2 (``|theta| ~ 1.01 deg``, ``epsilon_het ~ 0.42%``).

Primary literature sources for microscopic constants:
  1. Continuum elasticity, adhesion, out-of-plane corrugation, piezoelectricity:
     - V. V. Enaldiev et al., Phys. Rev. Lett. 124, 206101 (2020)
       [arXiv:1911.12804], Table S1 & Eqs. (S8)-(S12).
     - F. Ferreira et al., Appl. Phys. Lett. 118, 241602 (2021)
       [arXiv:2103.06320v3], Table S1 (p. 6), Eqs. (S1)-(S9) (pp. 6-8),
       Table S4 (p. 10: ``d_0' = 6.70 A``).
  2. Local-stacking 2x2 P-MoSe2 Q-valley Hamiltonian:
     - F. Ferreira et al., Appl. Phys. Lett. 118, 241602 (2021)
       [arXiv:2103.06320v3], Eqs. (S20)-(S21) (pp. 11-12) & Table S6 (p. 13).
  3. Monolayer MoSe2 Q-valley in-plane strain deformation potential tensor:
     - S. Fang, S. Carr, M. A. Cazalilla & E. Kaxiras, Phys. Rev. B 98, 075106
       (2018) [arXiv:1709.07510v2] + Phys. Rev. B 92, 205108 (2015) (22-band
       Wannier + atomic SOC): ``(D_par^Q, D_perp^Q) = (+0.0642, -1.3525) eV``
       (spinless 6-band: ``(+0.0939, -1.2042) eV``).
  4. Anisotropic Q^(2) conduction-band effective mass tensor:
     - A. Kormanyos et al., 2D Mater. 2, 022001 (2015) [arXiv:1410.6666v3],
       Sec. 6 & Table 8 (p. 22): ``(m_par, m_perp) = (0.48, 1.08) m_0`` [HSE06]
       or ``(0.51, 1.15) m_0`` [PBE].
  5. Open-surface STM hBN/vacuum screening & electrostatics:
     - Z. Ge et al., Nature (2026) [arXiv:2510.12009v3]; D. Groll et al.
       [arXiv:2610.07316v1]; A. Laturia et al., npj 2D Mater. Appl. 2, 6 (2018).
"""

import dataclasses
import functools
import math
from typing import Dict, Optional, Sequence, Tuple

from ferminet.moire import fang_tmd_wannier
from ferminet.utils import units
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import least_squares, minimize

# Exact unit conversions consistent with ferminet.utils.units:
HARTREE_TO_MEV: float = 27211.386245988
NM_TO_BOHR: float = 10.0 * units.BOHR_ANGSTROM
NM_INV_TO_BOHR_INV: float = 0.1 * units.ANGSTROM_BOHR
HBAR2_OVER_2M0_MEV_NM2: float = (
    0.5 * HARTREE_TO_MEV * (0.1 * units.ANGSTROM_BOHR) ** 2
)
EV_NM2_PER_N_M: float = 6.24150907446

# Experimental STM moire primitive vectors fitted to 11 XX' vertices in
# 2025-09-18_0074.sxm (|theta| = 1.009 deg, heterostrain = 0.419%):
STM_A1_NM: Tuple[float, float] = (21.032, 2.665)  # along DW_3 (nm)
STM_A2_NM: Tuple[float, float] = (5.968, 15.308)  # along DW_2 (nm)

# Literature mechanical, dielectric, and piezoelectric constants for P-MoSe2:
A0_MOSE2_NM: float = 0.3288
DEFAULT_TWIST_ANGLE_DEG: float = 1.009
DEFAULT_CRYSTAL_ROTATION_DEG: float = 0.0
MU_LAME_EV_NM2: float = 49.6 * EV_NM2_PER_N_M
LAMBDA_LAME_EV_NM2: float = 42.3 * EV_NM2_PER_N_M
POISSON_MOSE2: float = 0.23
W1_ADHESION_EV_NM2: float = 0.1725
W2_ADHESION_EV_NM2: float = 0.0250
KAPPA_OUT_OF_PLANE_EV_NM4: float = 189.0
QZ_CORRUGATION_NM_INV: float = 29.614
D0_PRIME_ANGSTROM: float = 6.70
E11_PIEZO_C_M: float = 2.14e-10
EPS_PAR_MOSE2: float = 16.8
EPS_PERP_BL_MOSE2: float = 7.29
EPS_ENV_VAC_HBN: float = 3.05
# Phenomenological effective permittivity fitted by Ge et al. (Nature 2026)
# for open-surface STM bilayer MoSe2 CB (sweep ~2.6-4.5 for sensitivity):
DEFAULT_DIELECTRIC_CONSTANT: float = 3.73
DEFAULT_HBN_GATE_DISTANCE_NM: float = 65.0

# Ferreira et al. (2021) Table S6 local-stacking P-MoSe2 Q-valley parameters:
EPS_Q_P_MEV: float = -3700.0
FERREIRA_Q_TABLE_S6: Dict[str, Tuple[float, float]] = {
    'tQ0': (+0.176, +0.672),
    'tQ1': (+0.011, +2.358),
    'tQ2': (+0.011, +2.358),
    'tQ3': (+0.002, +1.861),
    'vQ0': (+0.004, -3.054),
    'vQ1s': (-0.004, +2.821),
    'vQ2s': (-0.004, +2.821),
    'vQ3s': (-0.002, +2.302),
    'Del_a': (+0.018, +2.623),
}

# Monolayer MoSe2 Q^(2) CBM deformation potentials computed directly from the
# 22-band (with atomic SOC) and 11-band (spinless) strained Wannier Hamiltonians
# of Fang et al., PRB 98, 075106 (2018) & PRB 92, 205108 (2015):
_Q_DEF_SOC = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
    include_soc=True
)
_Q_DEF_SPINLESS = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
    include_soc=False
)
DEFORMATION_POTENTIALS_MEV: Dict[str, Tuple[float, float]] = {
    'soc': (_Q_DEF_SOC.d_par_mev, _Q_DEF_SOC.d_perp_mev),
    'spinless': (_Q_DEF_SPINLESS.d_par_mev, _Q_DEF_SPINLESS.d_perp_mev),
    'none': (0.0, 0.0),
}

# Monolayer MoSe2 Q^(2) CBM principal effective masses (Kormanyos et al. 2015):
PRINCIPAL_MASSES_M0: Dict[str, Tuple[float, float]] = {
    'hse': (0.48, 1.08),
    'pbe': (0.51, 1.15),
    'isotropic': (0.78, 0.78),
}


@dataclasses.dataclass(frozen=True)
class ValleyContinuumData:
  """Resolved continuum potential, mass tensor, minibands, and Fourier modes."""

  ferreira_index: int  # 1, 2, or 3 in Ferreira Fig. S1 crystal frame
  crystal_angle_rad: float
  light_mass_angle_deg: float
  mass_tensor_m0: Tuple[Tuple[float, float], Tuple[float, float]]
  v_eff_mev: np.ndarray
  v_eff_zero_mean_mev: np.ndarray
  delta_tot_mev: np.ndarray
  t_tunneling_mev: np.ndarray
  min_energy_mev: float
  mean_energy_mev: float
  mean_energy_hartree: float
  gamma_energies_mev: Tuple[float, ...]
  fourier_gvectors_nm_inv: np.ndarray
  fourier_gvectors_bohr_inv: np.ndarray
  fourier_amps_mev: np.ndarray
  fourier_amps_hartree: np.ndarray
  fourier_phases_rad: np.ndarray
  fourier_rms_error_mev: float
  fourier_max_error_mev: float


@dataclasses.dataclass(frozen=True)
class TMoSe2ContinuumModel:
  """Complete strained 0-deg P-MoSe2 continuum model across all 3 Q-valley pairs."""

  primitive_lattice_nm: np.ndarray
  primitive_lattice_bohr: np.ndarray
  moire_gvectors_nm_inv: np.ndarray
  moire_gvectors_bohr_inv: np.ndarray
  ez_phys_mv_per_angstrom: float
  eta_pinning: float
  eps_tb_hom_lab: np.ndarray
  z_corrugation_angstrom: np.ndarray
  interlayer_spacing_angstrom: np.ndarray
  ephi_plus_piezo_mev: np.ndarray
  ephi_minus_piezo_mev: np.ndarray
  ephi_top_piezo_mev: np.ndarray
  valleys: Tuple[ValleyContinuumData, ValleyContinuumData, ValleyContinuumData]


def unstrained_moire_vectors_nm(
    twist_angle_deg: float = DEFAULT_TWIST_ANGLE_DEG,
    crystal_rotation_deg: float = DEFAULT_CRYSTAL_ROTATION_DEG,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
  """Returns canonical 60-deg unstrained moire primitive vectors (nm)."""
  th_rad = abs(math.radians(float(twist_angle_deg)))
  if not math.isfinite(th_rad) or th_rad <= 0.0:
    raise ValueError(
        f'twist_angle_deg must be finite and non-zero, got {twist_angle_deg!r}.'
    )
  l_m_nm = A0_MOSE2_NM / (2.0 * math.sin(0.5 * th_rad))
  rot_rad = math.radians(float(crystal_rotation_deg))
  c, s = math.cos(rot_rad), math.sin(rot_rad)
  a1_raw = (l_m_nm, 0.0)
  a2_raw = (0.5 * l_m_nm, 0.5 * math.sqrt(3.0) * l_m_nm)
  a1 = (c * a1_raw[0] - s * a1_raw[1], s * a1_raw[0] + c * a1_raw[1])
  a2 = (c * a2_raw[0] - s * a2_raw[1], s * a2_raw[0] + c * a2_raw[1])
  return a1, a2


UNSTRAINED_A1_NM, UNSTRAINED_A2_NM = unstrained_moire_vectors_nm()


def moire_primitive_lattice_bohr(
    a1_nm: Sequence[float] = STM_A1_NM,
    a2_nm: Sequence[float] = STM_A2_NM,
) -> np.ndarray:
  """Returns the ``(2, 2)`` primitive moire lattice matrix in Bohr (columns)."""
  a1 = np.asarray(a1_nm, dtype=np.float64)
  a2 = np.asarray(a2_nm, dtype=np.float64)
  if a1.shape != (2,) or a2.shape != (2,):
    raise ValueError(
        f'a1_nm and a2_nm must have shape (2,), got {a1.shape} and {a2.shape}.'
    )
  l_m = np.column_stack([a1, a2])
  if not np.all(np.isfinite(l_m)) or abs(float(np.linalg.det(l_m))) <= 1e-8:
    raise ValueError(
        f'Moire basis vectors must be finite and linearly independent: {l_m.tolist()}.'
    )
  return l_m * NM_TO_BOHR


def default_ez_phys_mv_per_angstrom(
    nu_filling: float,
    top_layer_fraction: float = 0.5,
    nu_threshold_offset: float = 0.0,
    full_slab_slope_mv_per_angstrom: float = 0.811,
    omega_m_nm2: Optional[float] = None,
) -> float:
  """Returns upward interlayer electric field ``E_z^phys`` (mV/A) from Gauss's law.

  For two parallel 2D sheets with zero electric field in vacuum above the top
  sheet (``D(z > z_t) = 0``), Gauss's law across the top sheet gives an
  interlayer displacement field ``D_interlayer = e * n_top = e * f_top *
  (nu + nu_offset) / Omega_M``. Using ``Omega_M = 306.053 nm^2`` and bilayer
  out-of-plane permittivity ``eps_perp,BL = 7.29`` yields ``0.811 mV/A`` per
  full sheet, or ``0.4055 * nu mV/A`` for approximately equal layer populations
  (``top_layer_fraction = 0.5``, ``nu_threshold_offset = 0.0``). When
  ``omega_m_nm2`` is supplied explicitly, computes the exact Gauss-law slope
  ``(e / eps_0) / (eps_perp,BL * omega_m_nm2)`` for that moire cell area.
  """
  f_top = float(top_layer_fraction)
  if not np.isfinite(f_top) or f_top < 0.0 or f_top > 1.0:
    raise ValueError(
        f'top_layer_fraction must be in [0, 1], got {top_layer_fraction!r}.'
    )
  if omega_m_nm2 is not None:
    if not np.isfinite(omega_m_nm2) or omega_m_nm2 <= 0.0:
      raise ValueError(f'omega_m_nm2 must be positive, got {omega_m_nm2!r}.')
    slope = (40.0 * math.pi * 14.399645356) / (
        EPS_PERP_BL_MOSE2 * float(omega_m_nm2)
    )
  else:
    slope = float(full_slab_slope_mv_per_angstrom)
  return float(
      f_top * slope * (float(nu_filling) + float(nu_threshold_offset))
  )


def rotated_q_valley_mass_tensor(
    light_mass_angle_deg: float,
    mass_model: str = 'hse',
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
  """Returns the 2x2 SPD effective mass tensor ``M_Q`` (in ``m_0``) in lab axes."""
  key = str(mass_model).strip().lower()
  if key not in PRINCIPAL_MASSES_M0:
    raise ValueError(
        f'Unsupported mass_model={mass_model!r}; expected one of '
        f'{sorted(PRINCIPAL_MASSES_M0)}.'
    )
  m_par, m_perp = PRINCIPAL_MASSES_M0[key]
  th = float(np.deg2rad(float(light_mass_angle_deg)))
  c, s = float(np.cos(th)), float(np.sin(th))
  rot = np.array([[c, -s], [s, c]], dtype=np.float64)
  mat = rot @ np.diag([m_par, m_perp]) @ rot.T
  mat = 0.5 * (mat + mat.T)
  mat[np.abs(mat) < 1e-14] = 0.0
  return (
      (float(mat[0, 0]), float(mat[0, 1])),
      (float(mat[1, 0]), float(mat[1, 1])),
  )


def _solve_gamma_miniband_energies_mev(
    v_eff_mev: np.ndarray,
    b_m_nm_inv: np.ndarray,
    mass_tensor_m0: Tuple[Tuple[float, float], Tuple[float, float]],
    pw_cutoff: int = 8,
    n_states: int = 4,
) -> Tuple[float, ...]:
  """Diagonalizes single-particle ``T_aniso + V_eff(r)`` at Gamma in a plane-wave basis.

  Retains the full absolute potential offset ``mean(v_eff_mev)`` so miniband
  eigenvalues can be compared directly across inequivalent Q valleys.
  """
  grid_n = int(v_eff_mev.shape[0])
  eff_cut = min(int(pw_cutoff), grid_n // 2 - 1)
  m_mat = np.asarray(mass_tensor_m0, dtype=np.float64)
  m_inv = np.linalg.inv(m_mat)

  idx_1d = np.arange(-eff_cut, eff_cut + 1, dtype=np.int64)
  m1_g, m2_g = np.meshgrid(idx_1d, idx_1d, indexing='ij')
  m1 = m1_g.ravel()
  m2 = m2_g.ravel()
  g_vecs = m1[:, None] * b_m_nm_inv[0] + m2[:, None] * b_m_nm_inv[1]
  ke_diag = HBAR2_OVER_2M0_MEV_NM2 * np.einsum('ni,ij,nj->n', g_vecs, m_inv, g_vecs)

  v_fft = np.fft.fft2(v_eff_mev) / float(grid_n * grid_n)
  dm1 = m1[:, None] - m1[None, :]
  dm2 = m2[:, None] - m2[None, :]
  valid = (np.abs(dm1) < grid_n // 2) & (np.abs(dm2) < grid_n // 2)
  h_pw = np.where(valid, v_fft[dm1 % grid_n, dm2 % grid_n], 0.0j)
  h_pw[np.arange(m1.size), np.arange(m1.size)] += ke_diag
  h_pw = 0.5 * (h_pw + h_pw.conj().T)

  n_keep = min(int(n_states), int(m1.size))
  evals = eigh(h_pw, subset_by_index=[0, n_keep - 1], eigvals_only=True)
  return tuple(float(x) for x in evals)


@functools.lru_cache(maxsize=32)
def _solve_elastic_and_piezo_fields(
    a1_nm: Tuple[float, float] = STM_A1_NM,
    a2_nm: Tuple[float, float] = STM_A2_NM,
    grid_n: int = 64,
    eta_pinning: float = 0.0,
) -> Tuple[np.ndarray, ...]:
  """Solves consistent ``eta``-split elasticity, corrugation, piezo, and strain fields."""
  if grid_n < 16 or grid_n % 2 != 0:
    raise ValueError(f'grid_n must be an even integer >= 16, got {grid_n}.')
  eta_val = float(eta_pinning)
  if not np.isfinite(eta_val) or abs(eta_val) > 1.0:
    raise ValueError(
        f'eta_pinning must be finite and in [-1, 1], got {eta_pinning!r}.'
    )

  a1 = np.asarray(a1_nm, dtype=np.float64)
  a2 = np.asarray(a2_nm, dtype=np.float64)
  l_m = np.column_stack([a1, a2])
  if abs(float(np.linalg.det(l_m))) <= 1e-8:
    raise ValueError(f'Moire basis vectors must be linearly independent: {l_m.tolist()}.')

  b_m = 2.0 * np.pi * np.linalg.inv(l_m)  # rows G_1, G_2 in nm^-1
  if float(np.dot(b_m[0], b_m[1])) > 0.0:
    g_moire_triad = np.vstack([b_m[0], -b_m[1], b_m[1] - b_m[0]])
  else:
    g_moire_triad = np.vstack([b_m[0], b_m[1], -(b_m[0] + b_m[1])])
  g_mono_mag = 4.0 * np.pi / (np.sqrt(3.0) * A0_MOSE2_NM)

  # Exact closed-form alignment for unstrained equilateral 60-deg moire cells
  # (avoids 6-fold degenerate least_squares rounding noise when ep == 0):
  triad_norms = np.linalg.norm(g_moire_triad, axis=1)
  is_unstrained_equilateral = (
      np.allclose(triad_norms, triad_norms[0], rtol=1e-11, atol=1e-12)
      and float(np.linalg.det(l_m)) > 0.0
  )
  if is_unstrained_equilateral:
    th_signed = -float(triad_norms[0]) / float(g_mono_mag)
    j_mat = np.array([[0.0, -th_signed], [th_signed, 0.0]], dtype=np.float64)
    g_mono_aligned = (np.linalg.inv(j_mat) @ g_moire_triad.T).T
    eps_tb_hom_lab = np.zeros((2, 2), dtype=np.float64)
  else:
    def make_j_and_gmono(p: np.ndarray):
      th, ep, ps, ar = p
      angs = ar + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
      gm = g_mono_mag * np.column_stack([np.cos(angs), np.sin(angs)])
      c, s = np.cos(ps), np.sin(ps)
      rm = np.array([[c, -s], [s, c]])
      e_het = rm @ np.diag([ep, -POISSON_MOSE2 * ep]) @ rm.T
      j_m = np.array([[0.0, -th], [th, 0.0]]) - e_het
      return j_m, gm, (j_m @ gm.T).T, e_het

    best = None
    perms = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2))
    for perm in perms:
      gt = g_moire_triad[list(perm)]
      for sgn in (1.0, -1.0):
        for ie in (0.004, -0.004):
          res = least_squares(
              lambda p, s_val=sgn, gt_val=gt: (
                  make_j_and_gmono(p)[2] - s_val * gt_val
              ).ravel(),
              [-0.0176, ie, 0.7, 0.3],
          )
          if best is None or res.cost < best[0]:
            best = (res.cost, perm, sgn, res.x)
    assert best is not None
    _, perm, sgn, p_opt = best
    _, gm3, _, e_het_lab = make_j_and_gmono(p_opt)
    g_mono_aligned = sgn * gm3[np.argsort(perm)]
    # From r_0(r) = J_phys @ r + u_tb(r) with G_m^moire = J_phys.T @ G_m^mono = j_mat @ G_m^mono,
    # the homogeneous relative strain tensor sym(J_phys) = sym(j_mat.T) = -e_het_lab:
    eps_tb_hom_lab = -e_het_lab

  s = np.arange(grid_n, dtype=np.float64) / float(grid_n)
  s1, s2 = np.meshgrid(s, s, indexing='ij')
  r_grid = s1[..., None] * a1 + s2[..., None] * a2
  k_int = np.fft.fftfreq(grid_n) * float(grid_n)
  k1, k2 = np.meshgrid(k_int, k_int, indexing='ij')
  qx = k1 * b_m[0, 0] + k2 * b_m[1, 0]
  qy = k1 * b_m[0, 1] + k2 * b_m[1, 1]

  # Under u^t = ((1+eta)/2)*u_tb and u^b = -((1-eta)/2)*u_tb, total monolayer
  # elastic energy scales by ((1+eta)/2)^2 + ((1-eta)/2)^2 = 0.5 * (1 + eta^2):
  stiff_scale = 1.0 + eta_val**2
  d_xx = 0.5 * stiff_scale * (
      (LAMBDA_LAME_EV_NM2 + 2.0 * MU_LAME_EV_NM2) * qx**2
      + MU_LAME_EV_NM2 * qy**2
  )
  d_yy = 0.5 * stiff_scale * (
      (LAMBDA_LAME_EV_NM2 + 2.0 * MU_LAME_EV_NM2) * qy**2
      + MU_LAME_EV_NM2 * qx**2
  )
  d_xy = 0.5 * stiff_scale * (LAMBDA_LAME_EV_NM2 + MU_LAME_EV_NM2) * qx * qy
  phi0 = np.einsum('ijk,mk->ijm', r_grid, g_moire_triad)
  c_z_nm = (
      W1_ADHESION_EV_NM2 * QZ_CORRUGATION_NM_INV
      + W2_ADHESION_EV_NM2 * g_mono_mag
  ) / (2.0 * KAPPA_OUT_OF_PLANE_EV_NM4)
  n2_cells = float(grid_n * grid_n)

  def elastic_obj(u_flat: np.ndarray) -> Tuple[float, np.ndarray]:
    u_tb = u_flat.reshape(grid_n, grid_n, 2)
    uk_x = np.fft.fft2(u_tb[..., 0]) / n2_cells
    uk_y = np.fft.fft2(u_tb[..., 1]) / n2_cells
    e_el = 0.5 * np.sum(
        d_xx * np.abs(uk_x) ** 2
        + d_yy * np.abs(uk_y) ** 2
        + 2.0 * d_xy * np.real(np.conj(uk_x) * uk_y)
    )
    grad_el_x = np.real(np.fft.ifft2((d_xx * uk_x + d_xy * uk_y) * n2_cells))
    grad_el_y = np.real(np.fft.ifft2((d_xy * uk_x + d_yy * uk_y) * n2_cells))
    phi = phi0 + np.einsum('ijk,mk->ijm', u_tb, g_mono_aligned)
    cos_sum = np.sum(np.cos(phi), axis=-1)
    z_nm = c_z_nm * cos_sum
    w_adh = (
        -KAPPA_OUT_OF_PLANE_EV_NM4 * z_nm**2
        + (W1_ADHESION_EV_NM2 + W2_ADHESION_EV_NM2) * cos_sum
    )
    dw_dcos = (
        -2.0 * KAPPA_OUT_OF_PLANE_EV_NM4 * z_nm * c_z_nm
        + (W1_ADHESION_EV_NM2 + W2_ADHESION_EV_NM2)
    )
    grad_adh = (
        -dw_dcos[..., None]
        * np.einsum('ijm,mk->ijk', np.sin(phi), g_mono_aligned)
        / n2_cells
    )
    grad = (
        np.stack([grad_el_x / n2_cells, grad_el_y / n2_cells], axis=-1)
        + grad_adh
    )
    grad -= np.mean(grad, axis=(0, 1), keepdims=True)
    return float(e_el + np.mean(w_adh)), grad.ravel()

  res_el = minimize(
      elastic_obj,
      np.zeros(grid_n * grid_n * 2, dtype=np.float64),
      jac=True,
      method='L-BFGS-B',
      options={'gtol': 1e-10, 'maxiter': 600},
  )
  u_tb = res_el.x.reshape(grid_n, grid_n, 2)
  phi = phi0 + np.einsum('ijk,mk->ijm', u_tb, g_mono_aligned)
  z_angstrom = 10.0 * c_z_nm * np.sum(np.cos(phi), axis=-1)
  d_angstrom = D0_PRIME_ANGSTROM + z_angstrom

  # Ferreira Fig. S1 crystal frame: G1^F = G_mono[2], G2^F = G_mono[0], G3^F = G_mono[1]
  g1_c, g2_c, g3_c = g_mono_aligned[2], g_mono_aligned[0], g_mono_aligned[1]
  ex_c = (g1_c - g2_c) / np.linalg.norm(g1_c - g2_c)
  ey_c = -g3_c / np.linalg.norm(g3_c)
  r_c = np.column_stack([ex_c, ey_c])

  # Two-sheet piezopotentials e * phi_+(r) and e * phi_-(r) in meV (Enaldiev Eq. S11):
  # With rho_t = (1+eta)*rho_0 and rho_b = -(1-eta)*rho_0,
  #   rho_- = 0.5*(rho_t - rho_b) = rho_0       -> e*phi_- = 0.5*e*(phi_t - phi_b),
  #   rho_+ = 0.5*(rho_t + rho_b) = eta * rho_0 -> e*phi_+ = 0.5*e*(phi_t + phi_b).
  q_c = np.stack([qx, qy], axis=-1) @ r_c
  uk_c = (
      np.stack(
          [
              np.fft.fft2(0.5 * u_tb[..., 0]) / n2_cells,
              np.fft.fft2(0.5 * u_tb[..., 1]) / n2_cells,
          ],
          axis=-1,
      )
      @ r_c
  )
  qx_c, qy_c = q_c[..., 0], q_c[..., 1]
  q_norm = np.sqrt(qx_c**2 + qy_c**2)
  e11_e_nm = E11_PIEZO_C_M / 1.602176634e-19 * 1e-9
  rho_0 = e11_e_nm * (
      2.0 * qx_c * qy_c * uk_c[..., 0] + (qx_c**2 - qy_c**2) * uk_c[..., 1]
  )
  d0_nm = 0.1 * D0_PRIME_ANGSTROM
  alpha_2d_nm = d0_nm * (EPS_PAR_MOSE2 - 1.0) / (4.0 * np.pi)
  coulomb_pref = 2.0 * np.pi * 1439.9645 / (
      EPS_ENV_VAC_HBN * np.maximum(q_norm, 1e-12)
  )

  fac_minus = 1.0 - np.exp(-q_norm * d0_nm)
  num_minus = coulomb_pref * fac_minus
  den_minus = 1.0 + (
      2.0 * np.pi * alpha_2d_nm * q_norm / EPS_ENV_VAC_HBN
  ) * fac_minus
  ephi_minus_mev = np.real(
      np.fft.ifft2(
          np.where(q_norm > 1e-12, (num_minus / den_minus) * rho_0, 0.0)
          * n2_cells
      )
  )

  fac_plus = 1.0 + np.exp(-q_norm * d0_nm)
  num_plus = coulomb_pref * fac_plus
  den_plus = 1.0 + (
      2.0 * np.pi * alpha_2d_nm * q_norm / EPS_ENV_VAC_HBN
  ) * fac_plus
  ephi_plus_mev = np.real(
      np.fft.ifft2(
          np.where(
              q_norm > 1e-12, (num_plus / den_plus) * (eta_val * rho_0), 0.0
          )
          * n2_cells
      )
  )

  # Periodic relative strain sym(grad u_tb) and homogeneous heterostrain in crystal frame:
  eps_xx = np.real(np.fft.ifft2(1j * qx * np.fft.fft2(u_tb[..., 0])))
  eps_yy = np.real(np.fft.ifft2(1j * qy * np.fft.fft2(u_tb[..., 1])))
  eps_xy = 0.5 * np.real(
      np.fft.ifft2(
          1j * qy * np.fft.fft2(u_tb[..., 0])
          + 1j * qx * np.fft.fft2(u_tb[..., 1])
      )
  )
  eps_per_lab = np.zeros((grid_n, grid_n, 2, 2), dtype=np.float64)
  eps_per_lab[..., 0, 0] = eps_xx
  eps_per_lab[..., 1, 1] = eps_yy
  eps_per_lab[..., 0, 1] = eps_xy
  eps_per_lab[..., 1, 0] = eps_xy
  eps_tb_per_c = np.einsum('ab,ijbc,cd->ijad', r_c.T, eps_per_lab, r_c)
  eps_tb_hom_c = r_c.T @ eps_tb_hom_lab @ r_c

  return (
      l_m,
      b_m,
      g_moire_triad,
      phi,
      z_angstrom,
      d_angstrom,
      ex_c,
      ephi_plus_mev,
      ephi_minus_mev,
      eps_tb_per_c,
      eps_tb_hom_c,
      eps_tb_hom_lab,
      k1,
      k2,
  )


def build_tmose2_continuum_model(
    nu_filling: float = 3.0,
    eta_pinning: float = 0.0,
    ez_phys_mv_per_angstrom: Optional[float] = None,
    ez_top_layer_fraction: float = 0.5,
    ez_nu_offset: float = 0.0,
    fourier_cutoff: int = 12,
    mass_model: str = 'hse',
    deformation_model: str = 'soc',
    include_heterostrain_deformation: bool = True,
    eps_homo_uniaxial: float = 0.0,
    phi_homo_deg: float = 167.256,
    a1_nm: Sequence[float] = STM_A1_NM,
    a2_nm: Sequence[float] = STM_A2_NM,
    grid_n: int = 64,
    delta_t_mev: float = 0.0,
    include_piezo: bool = True,
    include_ferroelectric: bool = True,
    ez_use_cell_area: bool = False,
    deterministic_valley_order: bool = False,
) -> TMoSe2ContinuumModel:
  """Builds the 0-deg P-MoSe2 continuum model and Fourier expansion.

  When ``delta_t_mev == 0`` and ``deterministic_valley_order`` is False, valleys
  in ``model.valleys`` are sorted by ascending lowest Gamma-point Bloch miniband
  energy ``gamma_energies_mev[0]`` (with absolute offsets retained), so
  ``model.valleys[0]`` is Ferreira index 3 on the STM lattice. When
  ``delta_t_mev != 0`` or ``deterministic_valley_order`` is True, valleys are
  ordered deterministically as ``(Ferreira 3, Ferreira 2, Ferreira 1 = +-Q_1)``
  so ``valley_index = 2`` always selects ``+-Q_1``.
  """
  dt_val = float(delta_t_mev)
  if not np.isfinite(dt_val):
    raise ValueError(f'delta_t_mev must be finite, got {delta_t_mev!r}.')
  fourier_cutoff = int(fourier_cutoff)
  if fourier_cutoff < 1 or fourier_cutoff >= grid_n // 2:
    raise ValueError(
        f'fourier_cutoff must be in [1, {grid_n // 2 - 1}], got {fourier_cutoff}.'
    )
  def_key = str(deformation_model).strip().lower()
  if def_key not in DEFORMATION_POTENTIALS_MEV:
    raise ValueError(
        f'Unsupported deformation_model={deformation_model!r}; expected one of '
        f'{sorted(DEFORMATION_POTENTIALS_MEV)}.'
    )
  d_par_mev, d_perp_mev = DEFORMATION_POTENTIALS_MEV[def_key]

  a1_tup = (float(a1_nm[0]), float(a1_nm[1]))
  a2_tup = (float(a2_nm[0]), float(a2_nm[1]))
  eta_val = float(eta_pinning)
  (
      l_m_nm,
      b_m_nm_inv,
      g_moire_nm_inv,
      phi,
      z_ang,
      d_ang,
      ex_c,
      ephi_plus_raw,
      ephi_minus_raw,
      eps_tb_per_c,
      eps_tb_hom_c,
      eps_tb_hom_lab,
      k1,
      k2,
  ) = _solve_elastic_and_piezo_fields(a1_tup, a2_tup, int(grid_n), eta_val)

  ephi_plus_mev = (
      ephi_plus_raw if bool(include_piezo) else np.zeros_like(ephi_plus_raw)
  )
  ephi_minus_mev = (
      ephi_minus_raw if bool(include_piezo) else np.zeros_like(ephi_minus_raw)
  )

  if bool(include_heterostrain_deformation):
    eps_tb_total_c = eps_tb_per_c + eps_tb_hom_c[None, None, :, :]
  else:
    eps_tb_total_c = eps_tb_per_c

  omega_m_nm2 = (
      abs(float(np.linalg.det(l_m_nm))) if bool(ez_use_cell_area) else None
  )
  ez_val = (
      default_ez_phys_mv_per_angstrom(
          nu_filling,
          top_layer_fraction=ez_top_layer_fraction,
          nu_threshold_offset=ez_nu_offset,
          omega_m_nm2=omega_m_nm2,
      )
      if ez_phys_mv_per_angstrom is None
      else float(ez_phys_mv_per_angstrom)
  )

  def f_exp(table_key: str) -> np.ndarray:
    a_ev, q_a = FERREIRA_Q_TABLE_S6[table_key]
    return a_ev * np.exp(-q_a * z_ang) * 1.0e3

  s_p0 = (
      f_exp('Del_a') * np.sum(np.sin(phi), axis=-1)
      if bool(include_ferroelectric)
      else np.zeros_like(z_ang)
  )

  # Optional external homogeneous uniaxial homostrain (common to both layers):
  if float(eps_homo_uniaxial) != 0.0:
    th_h = float(np.deg2rad(float(phi_homo_deg)))
    ch, sh = np.cos(th_h), np.sin(th_h)
    r_h = np.array([[ch, -sh], [sh, ch]], dtype=np.float64)
    eps_homo_lab = (
        r_h
        @ np.diag(
            [float(eps_homo_uniaxial), -POISSON_MOSE2 * float(eps_homo_uniaxial)]
        )
        @ r_h.T
    )
    ey_c = np.array([-ex_c[1], ex_c[0]], dtype=np.float64)
    r_c_mat = np.column_stack([ex_c, ey_c])
    eps_homo_c = r_c_mat.T @ eps_homo_lab @ r_c_mat
  else:
    eps_homo_c = np.zeros((2, 2), dtype=np.float64)

  phi_f = (phi[..., 2], phi[..., 0], phi[..., 1])
  valley_specs = (
      (3, -2.0 * np.pi / 3.0, (phi_f[2], phi_f[0], phi_f[1])),
      (1, 0.0, (phi_f[0], phi_f[1], phi_f[2])),
      (2, 2.0 * np.pi / 3.0, (phi_f[1], phi_f[2], phi_f[0])),
  )

  n2_cells = float(grid_n * grid_n)
  mask_cut = (np.abs(k1) <= fourier_cutoff) & (np.abs(k2) <= fourier_cutoff)
  valley_list = []

  for f_idx, ang_v, (pa, pb, pc) in valley_specs:
    if d_par_mev == 0.0 and d_perp_mev == 0.0:
      v_def_minus = np.zeros_like(z_ang)
      v_def_plus = np.zeros_like(z_ang)
    else:
      eq_par = np.array([np.cos(ang_v), np.sin(ang_v)], dtype=np.float64)
      eq_perp = np.array([-np.sin(ang_v), np.cos(ang_v)], dtype=np.float64)
      e_par_tb = np.einsum('a,ijab,b->ij', eq_par, eps_tb_total_c, eq_par)
      e_perp_tb = np.einsum('a,ijab,b->ij', eq_perp, eps_tb_total_c, eq_perp)
      v_def_minus = d_par_mev * e_par_tb + d_perp_mev * e_perp_tb
      e_homo_shift = float(
          d_par_mev * (eq_par @ eps_homo_c @ eq_par)
          + d_perp_mev * (eq_perp @ eps_homo_c @ eq_perp)
      )
      v_def_plus = 0.5 * eta_val * v_def_minus + e_homo_shift

    eps_q = (
        EPS_Q_P_MEV
        + f_exp('vQ0')
        + f_exp('vQ1s') * np.cos(pa)
        + f_exp('vQ2s') * np.cos(pb)
        + f_exp('vQ3s') * np.cos(pc)
    )
    if f_idx == 1 and abs(dt_val) > 1e-12:
      t1_scale = (11.0 + dt_val) / 11.0
      t2_scale = (11.0 - dt_val) / 11.0
    else:
      t1_scale = 1.0
      t2_scale = 1.0
    t_q = (
        f_exp('tQ0')
        + t1_scale * f_exp('tQ1') * np.exp(-1j * pa)
        + t2_scale * f_exp('tQ2') * np.exp(+1j * pb)
        + f_exp('tQ3') * (np.exp(1j * pc) + np.exp(-1j * pc))
    )
    # Electron potential energy diag(-e*phi_t, -e*phi_b) contributes:
    #   common shift  = -0.5 * e * (phi_t + phi_b) = -ephi_plus_mev
    #   H_bb - H_tt   = (-e*phi_b) - (-e*phi_t)    = +2 * ephi_minus_mev
    delta_tot = s_p0 - ez_val * d_ang + 2.0 * ephi_minus_mev - v_def_minus
    v_eff = (
        eps_q
        + v_def_plus
        - ephi_plus_mev
        - np.sqrt(np.abs(t_q) ** 2 + 0.25 * delta_tot**2)
    )
    mean_e_mev = float(np.mean(v_eff))
    v_zero_mean = v_eff - mean_e_mev

    v_fft = np.fft.fft2(v_zero_mean) / n2_cells
    v_rec = np.real(np.fft.ifft2(np.where(mask_cut, v_fft, 0.0) * n2_cells))
    err_grid = np.abs(v_zero_mean - v_rec)
    rms_err = float(np.sqrt(np.mean(err_grid**2)))
    max_err = float(np.max(err_grid))

    modes = []
    for m1 in range(-fourier_cutoff, fourier_cutoff + 1):
      for m2 in range(-fourier_cutoff, fourier_cutoff + 1):
        if (m1 > 0) or (m1 == 0 and m2 > 0):
          c_val = v_fft[m1 % grid_n, m2 % grid_n]
          amp_mev = float(2.0 * np.abs(c_val))
          phi_rad = float(np.angle(c_val))
          g_nm_inv = m1 * b_m_nm_inv[0] + m2 * b_m_nm_inv[1]
          g_bohr_inv = g_nm_inv * NM_INV_TO_BOHR_INV
          modes.append(
              (amp_mev, g_nm_inv, g_bohr_inv, amp_mev / HARTREE_TO_MEV, phi_rad)
          )
    modes.sort(reverse=True, key=lambda item: item[0])

    g_nm_arr = np.stack([m[1] for m in modes], axis=0)
    g_bohr_arr = np.stack([m[2] for m in modes], axis=0)
    amps_mev_arr = np.array([m[0] for m in modes], dtype=np.float64)
    amps_ha_arr = np.array([m[3] for m in modes], dtype=np.float64)
    phases_arr = np.array([m[4] for m in modes], dtype=np.float64)

    th_light_deg = float(np.degrees(np.arctan2(ex_c[1], ex_c[0]) + ang_v) % 180.0)
    m_mat = rotated_q_valley_mass_tensor(th_light_deg, mass_model=mass_model)
    gamma_evals = _solve_gamma_miniband_energies_mev(
        v_eff_mev=v_eff,
        b_m_nm_inv=b_m_nm_inv,
        mass_tensor_m0=m_mat,
    )

    valley_list.append(
        ValleyContinuumData(
            ferreira_index=f_idx,
            crystal_angle_rad=float(ang_v),
            light_mass_angle_deg=th_light_deg,
            mass_tensor_m0=m_mat,
            v_eff_mev=v_eff,
            v_eff_zero_mean_mev=v_zero_mean,
            delta_tot_mev=delta_tot,
            t_tunneling_mev=t_q,
            min_energy_mev=float(np.min(v_eff)),
            mean_energy_mev=mean_e_mev,
            mean_energy_hartree=mean_e_mev / HARTREE_TO_MEV,
            gamma_energies_mev=gamma_evals,
            fourier_gvectors_nm_inv=g_nm_arr,
            fourier_gvectors_bohr_inv=g_bohr_arr,
            fourier_amps_mev=amps_mev_arr,
            fourier_amps_hartree=amps_ha_arr,
            fourier_phases_rad=phases_arr,
            fourier_rms_error_mev=rms_err,
            fourier_max_error_mev=max_err,
        )
    )

  if abs(dt_val) <= 1e-12 and not bool(deterministic_valley_order):
    valley_list.sort(key=lambda v: (v.gamma_energies_mev[0], v.min_energy_mev))
  else:
    _order_map = {3: 0, 2: 1, 1: 2}
    valley_list.sort(key=lambda v: _order_map[v.ferreira_index])

  ephi_top_mev = ephi_plus_mev + ephi_minus_mev
  return TMoSe2ContinuumModel(
      primitive_lattice_nm=l_m_nm.copy(),
      primitive_lattice_bohr=(l_m_nm * NM_TO_BOHR).copy(),
      moire_gvectors_nm_inv=g_moire_nm_inv.copy(),
      moire_gvectors_bohr_inv=(g_moire_nm_inv * NM_INV_TO_BOHR_INV).copy(),
      ez_phys_mv_per_angstrom=ez_val,
      eta_pinning=eta_val,
      eps_tb_hom_lab=eps_tb_hom_lab.copy(),
      z_corrugation_angstrom=z_ang.copy(),
      interlayer_spacing_angstrom=d_ang.copy(),
      ephi_plus_piezo_mev=ephi_plus_mev.copy(),
      ephi_minus_piezo_mev=ephi_minus_mev.copy(),
      ephi_top_piezo_mev=ephi_top_mev.copy(),
      valleys=(valley_list[0], valley_list[1], valley_list[2]),
  )


def evaluate_layer_weights(
    model: TMoSe2ContinuumModel,
    coords_bohr: np.ndarray,
    valley_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates adiabatic ``(w_top(r), w_bottom(r))`` spinor weights for STM post-processing.

  Uses bilinear periodic interpolation of the lower-band Bloch spinor
  ``w_top(r) = 0.5 * (1 + Delta_tot(r) / (2 * sqrt(|T_Q(r)|^2 + 0.25 * Delta_tot(r)^2)))``
  (where ``Delta_tot(r) = H_bb(r) - H_tt(r)`` in Ferreira Eq. S20's ``(top, bottom)``
  basis) at arbitrary 2D walker coordinates ``coords_bohr`` (shape ``(..., 2)``).
  """
  valley = model.valleys[int(valley_index)]
  denom = 2.0 * np.sqrt(
      np.abs(valley.t_tunneling_mev) ** 2 + 0.25 * valley.delta_tot_mev**2
  )
  w_top_grid = 0.5 * (1.0 + valley.delta_tot_mev / np.maximum(denom, 1e-12))

  pts = np.asarray(coords_bohr, dtype=np.float64)
  orig_shape = pts.shape[:-1]
  pts_2d = pts.reshape(-1, 2)
  inv_lat = np.linalg.inv(model.primitive_lattice_bohr)
  frac = (pts_2d @ inv_lat.T) % 1.0

  grid_n = w_top_grid.shape[0]
  u = frac[:, 0] * grid_n
  v = frac[:, 1] * grid_n
  i0 = np.floor(u).astype(int) % grid_n
  j0 = np.floor(v).astype(int) % grid_n
  i1 = (i0 + 1) % grid_n
  j1 = (j0 + 1) % grid_n
  du = u - np.floor(u)
  dv = v - np.floor(v)

  w_top = (
      (1.0 - du) * (1.0 - dv) * w_top_grid[i0, j0]
      + du * (1.0 - dv) * w_top_grid[i1, j0]
      + (1.0 - du) * dv * w_top_grid[i0, j1]
      + du * dv * w_top_grid[i1, j1]
  )
  w_top = w_top.reshape(orig_shape)
  return w_top, 1.0 - w_top
