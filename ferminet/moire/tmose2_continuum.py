"""Strained 0-deg P-MoSe2 conduction-band Q-valley continuum model.

Constructs the zero-free-parameter adiabatic 1-band continuum potential
``V_eff(r)``, multi-harmonic 2D Fourier cosine decomposition, rotated 2x2
anisotropic effective mass tensor ``M_Q``, and top/bottom layer-spinor weights
for open-surface STM parallel-stacked twisted bilayer MoSe2 (``theta ~ 1.01 deg``,
``epsilon_het ~ 0.44%``).

Primary literature sources for all microscopic constants:
  1. Continuum elasticity, adhesion, out-of-plane corrugation, piezoelectricity:
     - V. V. Enaldiev et al., Phys. Rev. Lett. 124, 206101 (2020)
       [arXiv:1911.12804], Table S1 & Eq. (S10).
     - F. Ferreira et al., Appl. Phys. Lett. 118, 241602 (2021)
       [arXiv:2103.06320v3], Table S1 (p. 6), Eqs. (S5)-(S9) (pp. 7-8),
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
     - A. Kormanyos et al., 2D Mater. 2, 022001 (2015) [arXiv:1410.6666],
       Table 7 (p. 23): ``(m_par, m_perp) = (0.48, 1.08) m_0`` [HSE06] or
       ``(0.51, 1.15) m_0`` [PBE].
  5. Open-surface STM hBN/vacuum screening & electrostatics:
     - Z. Ge et al., Nature (2026) [arXiv:2510.12009v3]; D. Groll et al.
       [arXiv:2610.07316v1]; A. Laturia et al., npj 2D Mater. Appl. 2, 6 (2018).
"""

import dataclasses
import functools
from typing import Dict, Sequence, Tuple

from ferminet.utils import units
import numpy as np
from scipy.optimize import least_squares, minimize

# Exact unit conversions consistent with ferminet.utils.units:
HARTREE_TO_MEV: float = 27211.386245988
NM_TO_BOHR: float = 10.0 * units.BOHR_ANGSTROM
NM_INV_TO_BOHR_INV: float = 0.1 * units.ANGSTROM_BOHR
EV_NM2_PER_N_M: float = 6.24150907446

# Experimental STM moire primitive vectors fitted to 11 XX' vertices in
# 2025-09-18_0074.sxm (theta ~ 1.011 deg, heterostrain ~ 0.440%):
STM_A1_NM: Tuple[float, float] = (21.032, 2.665)  # along DW_3 (nm)
STM_A2_NM: Tuple[float, float] = (5.968, 15.308)  # along DW_2 (nm)

# Literature mechanical, dielectric, and piezoelectric constants for P-MoSe2:
A0_MOSE2_NM: float = 0.3288
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

# Monolayer MoSe2 Q^(2) CBM deformation potentials (Fang et al. PRB 98, 075106):
DEFORMATION_POTENTIALS_MEV: Dict[str, Tuple[float, float]] = {
    'soc': (+64.2, -1352.5),       # 22-band Wannier + atomic SOC (default)
    'spinless': (+93.9, -1204.2),  # 6-band spinless Wannier
    'none': (0.0, 0.0),            # unstrained / zero deformation potential
}

# Monolayer MoSe2 Q^(2) CBM principal effective masses (Kormanyos et al. 2015):
PRINCIPAL_MASSES_M0: Dict[str, Tuple[float, float]] = {
    'hse': (0.48, 1.08),
    'pbe': (0.51, 1.15),
    'isotropic': (0.78, 0.78),
}


@dataclasses.dataclass(frozen=True)
class ValleyContinuumData:
  """Resolved continuum potential, mass tensor, and Fourier modes for one Q valley."""

  ferreira_index: int  # 1, 2, or 3 in Ferreira Fig. S1 crystal frame
  crystal_angle_rad: float
  light_mass_angle_deg: float
  mass_tensor_m0: Tuple[Tuple[float, float], Tuple[float, float]]
  v_eff_mev: np.ndarray
  v_eff_zero_mean_mev: np.ndarray
  delta_tot_mev: np.ndarray
  t_tunneling_mev: np.ndarray
  min_energy_mev: float
  fourier_gvectors_nm_inv: np.ndarray
  fourier_gvectors_bohr_inv: np.ndarray
  fourier_amps_mev: np.ndarray
  fourier_amps_hartree: np.ndarray
  fourier_phases_rad: np.ndarray
  fourier_rms_error_mev: float


@dataclasses.dataclass(frozen=True)
class TMoSe2ContinuumModel:
  """Complete strained 0-deg P-MoSe2 continuum model across all 3 Q-valley pairs."""

  primitive_lattice_nm: np.ndarray
  primitive_lattice_bohr: np.ndarray
  moire_gvectors_nm_inv: np.ndarray
  moire_gvectors_bohr_inv: np.ndarray
  ez_phys_mv_per_angstrom: float
  eta_pinning: float
  z_corrugation_angstrom: np.ndarray
  interlayer_spacing_angstrom: np.ndarray
  ephi_top_piezo_mev: np.ndarray
  valleys: Tuple[ValleyContinuumData, ValleyContinuumData, ValleyContinuumData]


def default_ez_phys_mv_per_angstrom(
    nu_filling: float,
    nu_threshold_offset: float = 0.89,
    slope_mv_per_angstrom: float = 0.811,
) -> float:
  """Returns upward internal vertical electric field ``E_z^phys`` (mV/A) from Gauss's law.

  Calibrated to Slide 3 gate-voltage linear regression ``V_G(nu) = 1.0768 * nu +
  0.960 V`` (``d_hBN ~ 65 nm``, cell area ``306.053 nm^2``, ``eps_perp,BL = 7.29``):
  ``D_bot / (eps_0 * eps_perp,BL) = 0.811 mV/A * (nu + 0.89)``.
  """
  return float(slope_mv_per_angstrom * (float(nu_filling) + float(nu_threshold_offset)))


def rotated_q_valley_mass_tensor(
    light_mass_angle_deg: float,
    mass_model: str = 'hse',
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
  """Returns the 2x2 SPD effective mass tensor ``M_Q`` (in ``m_0``) in STM lab axes."""
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
  return (
      (float(mat[0, 0]), float(mat[0, 1])),
      (float(mat[1, 0]), float(mat[1, 1])),
  )


@functools.lru_cache(maxsize=32)
def _solve_elastic_and_piezo_fields(
    a1_nm: Tuple[float, float] = STM_A1_NM,
    a2_nm: Tuple[float, float] = STM_A2_NM,
    grid_n: int = 64,
) -> Tuple[np.ndarray, ...]:
  """Solves 2D continuum elasticity relaxation, corrugation ``Z(r)``, and piezo potential."""
  if grid_n < 16 or grid_n % 2 != 0:
    raise ValueError(f'grid_n must be an even integer >= 16, got {grid_n}.')

  a1 = np.asarray(a1_nm, dtype=np.float64)
  a2 = np.asarray(a2_nm, dtype=np.float64)
  l_m = np.column_stack([a1, a2])
  if abs(float(np.linalg.det(l_m))) <= 1e-8:
    raise ValueError(f'Moire basis vectors must be linearly independent: {l_m.tolist()}.')

  b_m = 2.0 * np.pi * np.linalg.inv(l_m)  # rows G_1, G_2 in nm^-1
  g_moire_triad = np.vstack([b_m[0], b_m[1], -(b_m[0] + b_m[1])])
  g_mono_mag = 4.0 * np.pi / (np.sqrt(3.0) * A0_MOSE2_NM)

  def make_j_and_gmono(p: np.ndarray):
    th, ep, ps, ar = p
    angs = ar + np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
    gm = g_mono_mag * np.column_stack([np.cos(angs), np.sin(angs)])
    c, s = np.cos(ps), np.sin(ps)
    rm = np.array([[c, -s], [s, c]])
    e_het = rm @ np.diag([ep, -POISSON_MOSE2 * ep]) @ rm.T
    j_mat = np.array([[0.0, -th], [th, 0.0]]) - e_het
    return j_mat, gm, (j_mat @ gm.T).T

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
  _, gm3, _ = make_j_and_gmono(p_opt)
  g_mono_aligned = sgn * gm3[np.argsort(perm)]

  s = np.arange(grid_n, dtype=np.float64) / float(grid_n)
  s1, s2 = np.meshgrid(s, s, indexing='ij')
  r_grid = s1[..., None] * a1 + s2[..., None] * a2
  k_int = np.fft.fftfreq(grid_n) * float(grid_n)
  k1, k2 = np.meshgrid(k_int, k_int, indexing='ij')
  qx = k1 * b_m[0, 0] + k2 * b_m[1, 0]
  qy = k1 * b_m[0, 1] + k2 * b_m[1, 1]

  d_xx = 0.5 * (
      (LAMBDA_LAME_EV_NM2 + 2.0 * MU_LAME_EV_NM2) * qx**2
      + MU_LAME_EV_NM2 * qy**2
  )
  d_yy = 0.5 * (
      (LAMBDA_LAME_EV_NM2 + 2.0 * MU_LAME_EV_NM2) * qy**2
      + MU_LAME_EV_NM2 * qx**2
  )
  d_xy = 0.5 * (LAMBDA_LAME_EV_NM2 + MU_LAME_EV_NM2) * qx * qy
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

  # Piezopotential e * phi^t(r) in meV (Enaldiev PRL 2020 Eq. S10):
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
  rho_k = e11_e_nm * (
      2.0 * qx_c * qy_c * uk_c[..., 0] + (qx_c**2 - qy_c**2) * uk_c[..., 1]
  )
  alpha_2d = 0.67 * (EPS_PAR_MOSE2 - 1.0) / (4.0 * np.pi)
  num_p = (
      (2.0 * np.pi * 1439.9645 / (EPS_ENV_VAC_HBN * np.maximum(q_norm, 1e-12)))
      * (1.0 - np.exp(-q_norm * 0.67))
      * 0.5
  )
  den_p = 1.0 + (2.0 * np.pi * alpha_2d * q_norm / EPS_ENV_VAC_HBN) * (
      1.0 - np.exp(-q_norm * 0.67)
  )
  ephi_t_mev = np.real(
      np.fft.ifft2(np.where(q_norm > 1e-12, (num_p / den_p) * rho_k, 0.0) * n2_cells)
  )

  # Relative strain tensor eps_tb = eps^t - eps^b in crystal frame:
  eps_xx = np.real(np.fft.ifft2(1j * qx * np.fft.fft2(u_tb[..., 0])))
  eps_yy = np.real(np.fft.ifft2(1j * qy * np.fft.fft2(u_tb[..., 1])))
  eps_xy = 0.5 * np.real(
      np.fft.ifft2(
          1j * qy * np.fft.fft2(u_tb[..., 0])
          + 1j * qx * np.fft.fft2(u_tb[..., 1])
      )
  )
  eps_lab = np.zeros((grid_n, grid_n, 2, 2), dtype=np.float64)
  eps_lab[..., 0, 0] = eps_xx
  eps_lab[..., 1, 1] = eps_yy
  eps_lab[..., 0, 1] = eps_xy
  eps_lab[..., 1, 0] = eps_xy
  eps_tb_c = np.einsum('ab,ijbc,cd->ijad', r_c.T, eps_lab, r_c)

  return (
      l_m,
      b_m,
      g_moire_triad,
      phi,
      z_angstrom,
      d_angstrom,
      ex_c,
      ephi_t_mev,
      eps_tb_c,
      k1,
      k2,
  )


def build_tmose2_continuum_model(
    nu_filling: float = 3.0,
    eta_pinning: float = 1.0,
    ez_phys_mv_per_angstrom: float | None = None,
    fourier_cutoff: int = 5,
    mass_model: str = 'hse',
    deformation_model: str = 'soc',
    eps_homo_uniaxial: float = 0.0,
    phi_homo_deg: float = 167.256,
    a1_nm: Sequence[float] = STM_A1_NM,
    a2_nm: Sequence[float] = STM_A2_NM,
    grid_n: int = 64,
) -> TMoSe2ContinuumModel:
  """Builds the strained 0-deg P-MoSe2 continuum model and Fourier expansion.

  Valleys in ``model.valleys`` are sorted by ascending global minimum energy
  ``min_r V_eff^(a)(r)`` so ``model.valleys[0]`` is the heterostrain-selected
  lowest-energy Kramers pair ``+-Q_*`` (aligned with ``DW_1 = A_2 - A_1``,
  light-mass angle ``167.256 deg``) and ``model.valleys[1]`` is the runner-up
  valley pair relevant for two-valley simulations at ``nu >= 5``.
  """
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
  (
      l_m_nm,
      b_m_nm_inv,
      g_moire_nm_inv,
      phi,
      z_ang,
      d_ang,
      ex_c,
      ephi_t_mev,
      eps_tb_c,
      k1,
      k2,
  ) = _solve_elastic_and_piezo_fields(a1_tup, a2_tup, int(grid_n))

  ez_val = (
      default_ez_phys_mv_per_angstrom(nu_filling)
      if ez_phys_mv_per_angstrom is None
      else float(ez_phys_mv_per_angstrom)
  )

  def f_exp(table_key: str) -> np.ndarray:
    a_ev, q_a = FERREIRA_Q_TABLE_S6[table_key]
    return a_ev * np.exp(-q_a * z_ang) * 1.0e3

  s_p0 = f_exp('Del_a') * np.sum(np.sin(phi), axis=-1)

  # Optional homogeneous uniaxial strain tensor in crystal frame:
  if float(eps_homo_uniaxial) != 0.0:
    th_h = float(np.deg2rad(float(phi_homo_deg)))
    ch, sh = np.cos(th_h), np.sin(th_h)
    r_h = np.array([[ch, -sh], [sh, ch]], dtype=np.float64)
    eps_homo_lab = (
        r_h
        @ np.diag([float(eps_homo_uniaxial), -POISSON_MOSE2 * float(eps_homo_uniaxial)])
        @ r_h.T
    )
    ey_c = np.array([-ex_c[1], ex_c[0]], dtype=np.float64)
    r_c_mat = np.column_stack([ex_c, ey_c])
    eps_homo_c = r_c_mat.T @ eps_homo_lab @ r_c_mat
  else:
    eps_homo_c = np.zeros((2, 2), dtype=np.float64)

  # Three C3-related Q valleys in Ferreira Fig. S1 crystal frame, ordered
  # canonically starting with the STM-observed active DW_1 valley (f_idx=3,
  # DW_1 = A_2 - A_1, light-mass angle 167.256 deg), followed by DW_2 (f_idx=1)
  # and DW_3 (f_idx=2). If explicit uniaxial homostrain is applied, re-sort by
  # min_r V_eff^(a)(r).
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
    eq_par = np.array([np.cos(ang_v), np.sin(ang_v)], dtype=np.float64)
    eq_perp = np.array([-np.sin(ang_v), np.cos(ang_v)], dtype=np.float64)
    e_par_tb = np.einsum('a,ijab,b->ij', eq_par, eps_tb_c, eq_par)
    e_perp_tb = np.einsum('a,ijab,b->ij', eq_perp, eps_tb_c, eq_perp)
    v_def_minus = d_par_mev * e_par_tb + d_perp_mev * e_perp_tb
    e_homo_shift = float(
        d_par_mev * (eq_par @ eps_homo_c @ eq_par)
        + d_perp_mev * (eq_perp @ eps_homo_c @ eq_perp)
    )
    v_def_plus = 0.5 * float(eta_pinning) * v_def_minus + e_homo_shift

    eps_q = (
        EPS_Q_P_MEV
        + f_exp('vQ0')
        + f_exp('vQ1s') * np.cos(pa)
        + f_exp('vQ2s') * np.cos(pb)
        + f_exp('vQ3s') * np.cos(pc)
    )
    t_q = (
        f_exp('tQ0')
        + f_exp('tQ1') * np.exp(-1j * pa)
        + f_exp('tQ2') * np.exp(+1j * pb)
        + f_exp('tQ3') * (np.exp(1j * pc) + np.exp(-1j * pc))
    )
    delta_tot = s_p0 - ez_val * d_ang + 2.0 * ephi_t_mev - v_def_minus
    v_eff = eps_q + v_def_plus - np.sqrt(np.abs(t_q) ** 2 + 0.25 * delta_tot**2)
    v_zero_mean = v_eff - float(np.mean(v_eff))

    v_fft = np.fft.fft2(v_zero_mean) / n2_cells
    v_rec = np.real(np.fft.ifft2(np.where(mask_cut, v_fft, 0.0) * n2_cells))
    rms_err = float(np.sqrt(np.mean((v_zero_mean - v_rec) ** 2)))

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
            fourier_gvectors_nm_inv=g_nm_arr,
            fourier_gvectors_bohr_inv=g_bohr_arr,
            fourier_amps_mev=amps_mev_arr,
            fourier_amps_hartree=amps_ha_arr,
            fourier_phases_rad=phases_arr,
            fourier_rms_error_mev=rms_err,
        )
    )

  if float(eps_homo_uniaxial) != 0.0:
    valley_list.sort(key=lambda v: v.min_energy_mev)

  return TMoSe2ContinuumModel(
      primitive_lattice_nm=l_m_nm.copy(),
      primitive_lattice_bohr=(l_m_nm * NM_TO_BOHR).copy(),
      moire_gvectors_nm_inv=g_moire_nm_inv.copy(),
      moire_gvectors_bohr_inv=(g_moire_nm_inv * NM_INV_TO_BOHR_INV).copy(),
      ez_phys_mv_per_angstrom=ez_val,
      eta_pinning=float(eta_pinning),
      z_corrugation_angstrom=z_ang.copy(),
      interlayer_spacing_angstrom=d_ang.copy(),
      ephi_top_piezo_mev=ephi_t_mev.copy(),
      valleys=(valley_list[0], valley_list[1], valley_list[2]),
  )


def evaluate_layer_weights(
    model: TMoSe2ContinuumModel,
    coords_bohr: np.ndarray,
    valley_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates adiabatic ``(w_top(r), w_bottom(r))`` spinor weights for STM post-processing.

  Uses bilinear periodic interpolation of the lower-band Bloch spinor
  ``w_top(r) = 0.5 * (1 - Delta_tot(r) / (2 * sqrt(|T_Q(r)|^2 + 0.25 * Delta_tot(r)^2)))``
  at arbitrary 2D walker coordinates ``coords_bohr`` (shape ``(..., 2)``).
  """
  valley = model.valleys[int(valley_index)]
  denom = 2.0 * np.sqrt(
      np.abs(valley.t_tunneling_mev) ** 2 + 0.25 * valley.delta_tot_mev**2
  )
  w_top_grid = 0.5 * (1.0 - valley.delta_tot_mev / np.maximum(denom, 1e-12))

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
  ).reshape(orig_shape)
  return w_top, 1.0 - w_top
