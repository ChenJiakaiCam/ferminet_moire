"""Monolayer MoSe2 11-band & 22-band strained Wannier tight-binding Hamiltonian.

Implements the ab initio DFT-Wannier strained tight-binding Hamiltonian and
atomic spin-orbit coupling (SOC) for monolayer 2H-MoSe2 from:
  1. S. Fang, S. Carr, M. A. Cazalilla & E. Kaxiras, Phys. Rev. B 98, 075106
     (2018) [arXiv:1709.07510v2], Sec. II.C, Eqs. (7)-(13) (pp. 6-7), Table II
     (p. 6), Table IV (p. 10), and Tables V-VIII (pp. 12-14).
  2. S. Fang, R. Kuate Defo, S. N. Shirodkar, S. Lieu, G. A. Tritsaris &
     E. Kaxiras, Phys. Rev. B 92, 205108 (2015) [arXiv:1506.08860v3], Sec. IV,
     Table II (p. 4), Eqs. (11)-(12) (p. 7), and Table VIII (p. 15).

All Wannier on-site energies in Fang et al. (2018) are referenced to the DFT
slab vacuum level ``E_vac = 0`` (20 A vacuum slab), providing vacuum-referenced
conduction-band Q-valley deformation potentials ``(D_par^Q, D_perp^Q)`` via
analytic Hellmann-Feynman derivatives as well as fixed-k and 2D k-reminimized
central finite differences.
"""

import dataclasses
import functools
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, minimize_scalar

# Unstrained PBE lattice constant of monolayer MoSe2 (Fang 2018 Table II, p. 6):
A0_FANG_MOSE2_ANGSTROM: float = 3.317

# Atomic spin-orbit coupling strengths in eV/hbar^2 (Fang 2015 Table VIII, p. 15):
LAMBDA_SO_MO_EV: float = 0.0836
LAMBDA_SO_SE_EV: float = 0.2470

# 11-orbital Wannier basis in Fang (2018) Sec. II.C (p. 6):
#   Odd  M_1 = -1 (5 orbitals): Psi_A = (d_xz, d_yz), Psi_B = (p_x^o, p_y^o, p_z^o)
#   Even M_1 = +1 (6 orbitals): Psi_C = (d_xy, d_{x^2-y^2}, d_{z^2}),
#                               Psi_D = (p_x^e, p_y^e, p_z^e)
#
# Exact MoSe2 parameters from Fang (2018) Tables V-VIII (pp. 12-14), in eV:
#   H_0 tuples: (e0, e1, alpha0, alpha1, beta0, beta1)
_H0_AA: Tuple[float, ...] = (0.0, -4.547, 0.0, -2.341, -0.810, 0.0)
_H0_BB: Tuple[float, ...] = (-5.986, -6.502, 1.396, -1.440, -0.121, 0.270)
_H0_CC: Tuple[float, ...] = (-5.559, -5.314, -1.090, -2.023, -0.296, 0.004)
_H0_DD: Tuple[float, ...] = (-8.231, -7.110, -0.742, -3.316, -1.146, 0.829)

# 1st- and 3rd-neighbor M-X hopping blocks (Table VII, p. 13):
_T_1BA: Tuple[float, ...] = (-0.695, 1.941, 0.0, -1.326, 0.0)
_AL_1BA: Tuple[float, ...] = (0.408, -0.417, 0.0, 1.718, 0.0)
_BE_1BA: Tuple[float, ...] = (
    -0.897, 0.264, 0.0, -1.874, 0.0, 0.770, 0.0, -0.469, -0.717
)

_T_1DC: Tuple[float, ...] = (1.268, 0.554, -0.874, -0.858, -0.772)
_AL_1DC: Tuple[float, ...] = (-0.407, 0.825, 1.928, 0.272, -0.298)
_BE_1DC: Tuple[float, ...] = (
    1.530, -0.367, -1.995, -0.510, -0.727, 0.761, -0.475, -1.841, 1.005
)

_T_3DC: Tuple[float, ...] = (0.017, -0.215, -0.155, -0.223, -0.069)
_AL_3DC: Tuple[float, ...] = (0.175, 0.185, 0.554, 0.760, 0.062)
_BE_3DC: Tuple[float, ...] = (
    -0.164, -0.995, -0.093, -0.292, -0.290, -0.664, -0.391, -0.299, 0.007
)

# 2nd-neighbor M-M and X-X hopping blocks (Tables V & VIII, pp. 12 & 14):
_T_2AA: Tuple[float, ...] = (-0.146, 0.017, 0.0, -0.191, 0.0, 0.0)
_AL_2AA: Tuple[float, ...] = (-0.309, -0.125, 0.0, 0.514, 0.0, 0.0)
_BE_2AA: Tuple[float, ...] = (
    -0.588, -0.118, 0.0, 0.416, 0.0, 0.0, -0.063, 0.0, 0.0
)

_T_2BB: Tuple[float, ...] = (0.964, -0.172, -0.211, -0.068, 0.076, -0.074)
_AL_2BB: Tuple[float, ...] = (-1.979, -0.103, 0.536, -0.059, -0.123, 0.142)
_BE_2BB: Tuple[float, ...] = (
    -2.378, 0.827, 0.445, -0.016, -0.146, 0.112, 0.567, -0.128, -0.092
)

_T_2CC: Tuple[float, ...] = (0.251, -0.473, -0.264, -0.201, 0.096, 0.352)
_AL_2CC: Tuple[float, ...] = (-0.951, 1.333, 0.885, 0.195, 0.236, -0.596)
_BE_2CC: Tuple[float, ...] = (
    -0.793, 1.108, 0.333, 0.008, 0.126, -0.667, -0.565, 0.255, -0.110
)

_T_2DD: Tuple[float, ...] = (0.991, -0.004, -0.217, -0.039, -0.121, 0.005)
_AL_2DD: Tuple[float, ...] = (-1.586, -0.072, 0.668, -0.162, -0.202, 0.050)
_BE_2DD: Tuple[float, ...] = (
    -2.180, 0.884, 0.576, 0.155, -0.026, 0.073, 0.777, 0.066, -0.127
)

# 3-fold rotation matrix U_R in Eq. (13) (p. 7):
_U_R: np.ndarray = np.array(
    [
        [-0.5, 0.5 * np.sqrt(3.0), 0.0],
        [-0.5 * np.sqrt(3.0), -0.5, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclasses.dataclass(frozen=True)
class QValleyDeformationResult:
  """Resolved Q-valley minimum and vacuum-referenced deformation potentials."""

  k_q_over_k: float
  k_q_ang_inv: float
  energy_ev: float
  spin_splitting_2delta_q_mev: float
  d_par_mev: float
  d_perp_mev: float
  d_shear_mev: float


def _rot_strain_120(
    uxx: float, uyy: float, two_uxy: float
) -> Tuple[float, float, float]:
  """Rotates strain components for a bond rotated +2*pi/3 CCW (Eq. 12, p. 7)."""
  uxy = 0.5 * two_uxy
  return (
      0.25 * uxx + 0.75 * uyy - 0.5 * np.sqrt(3.0) * uxy,
      0.75 * uxx + 0.25 * uyy + 0.5 * np.sqrt(3.0) * uxy,
      0.5 * np.sqrt(3.0) * uxx - 0.5 * np.sqrt(3.0) * uyy - uxy,
  )


def _rot_vec_120(vec: np.ndarray) -> np.ndarray:
  """Rotates a 2D vector by +2*pi/3 CCW."""
  c, s = -0.5, 0.5 * np.sqrt(3.0)
  return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _make_h0_block(
    params: Tuple[float, ...], uxx: float, uyy: float, two_uxy: float
) -> np.ndarray:
  """Builds 3x3 strained on-site block H^(0) (Eq. 9, p. 6)."""
  e0, e1, a0_p, a1_p, b0, b1 = params
  tr = uxx + uyy
  sh = uxx - uyy
  return (
      np.diag([e1, e1, e0])
      + tr * np.diag([a1_p, a1_p, a0_p])
      + sh * np.array([[b0, 0.0, 0.0], [0.0, -b0, b1], [0.0, b1, 0.0]])
      + two_uxy * np.array([[0.0, b0, b1], [b0, 0.0, 0.0], [b1, 0.0, 0.0]])
  )


def _make_h13_block(
    t: Tuple[float, ...],
    al: Tuple[float, ...],
    be: Tuple[float, ...],
    uxx: float,
    uyy: float,
    two_uxy: float,
) -> np.ndarray:
  """Builds 3x3 M-X 1st/3rd-neighbor block H^(n) along yz mirror (Eq. 10, p. 7)."""
  tr = uxx + uyy
  sh = uxx - uyy
  return (
      np.array([[t[0], 0.0, 0.0], [0.0, t[1], t[2]], [0.0, t[3], t[4]]])
      + tr * np.array([[al[0], 0.0, 0.0], [0.0, al[1], al[2]], [0.0, al[3], al[4]]])
      + sh * np.array([[be[0], 0.0, 0.0], [0.0, be[1], be[2]], [0.0, be[3], be[4]]])
      + two_uxy * np.array([[0.0, be[5], be[6]], [be[7], 0.0, 0.0], [be[8], 0.0, 0.0]])
  )


def _make_h2_block(
    t: Tuple[float, ...],
    al: Tuple[float, ...],
    be: Tuple[float, ...],
    uxx: float,
    uyy: float,
    two_uxy: float,
) -> np.ndarray:
  """Builds 3x3 M-M/X-X 2nd-neighbor block H^(2) along +a_1 (Eq. 11, p. 7)."""
  tr = uxx + uyy
  sh = uxx - uyy
  return (
      np.array([[t[0], t[3], t[4]], [-t[3], t[1], t[5]], [-t[4], t[5], t[2]]])
      + tr
      * np.array([
          [al[0], al[3], al[4]],
          [-al[3], al[1], al[5]],
          [-al[4], al[5], al[2]],
      ])
      + sh
      * np.array([
          [be[0], be[3], be[4]],
          [-be[3], be[1], be[5]],
          [-be[4], be[5], be[2]],
      ])
      + two_uxy
      * np.array([[0.0, be[6], be[7]], [be[6], 0.0, be[8]], [be[7], -be[8], 0.0]])
  )


def build_mose2_11band_hamiltonian(
    k_ang_inv: np.ndarray,
    uxx: float = 0.0,
    uyy: float = 0.0,
    two_uxy: float = 0.0,
    a_angstrom: float = A0_FANG_MOSE2_ANGSTROM,
    lab_frame_k: bool = False,
) -> np.ndarray:
  """Returns the 11x11 spinless strained Wannier Hamiltonian H_11(k; u) in eV.

  Basis order:
    [0..4]  M_1-odd:  (d_xz, d_yz, p_x^o, p_y^o, p_z^o)
    [5..10] M_1-even: (d_xy, d_{x^2-y^2}, d_{z^2}, p_x^e, p_y^e, p_z^e)

  Args:
    k_ang_inv: 2D wavevector ``(k_x, k_y)`` in ``Angstrom^-1`` (``Gamma -> K_+``
      is along ``+x``).
    uxx: Normal strain ``epsilon_{xx}`` along ``x`` (parallel to ``Gamma-K_+``).
    uyy: Normal strain ``epsilon_{yy}`` along ``y`` (perp to ``Gamma-K_+``).
    two_uxy: Engineering shear strain ``2 * epsilon_{xy}``.
    a_angstrom: Unstrained in-plane lattice constant in Angstrom.
    lab_frame_k: If True, evaluates Bloch phases at strained bond vectors
      ``(I + u) r_0``; if False (default), uses crystal bond vectors ``r_0``.
  """
  k_vec = np.asarray(k_ang_inv, dtype=np.float64)
  if lab_frame_k:
    u_mat = np.array(
        [[uxx, 0.5 * two_uxy], [0.5 * two_uxy, uyy]], dtype=np.float64
    )
    k_eff = (np.eye(2) + u_mat).T @ k_vec
  else:
    k_eff = k_vec

  a1 = a_angstrom * np.array([1.0, 0.0])
  a2 = a_angstrom * np.array([-0.5, 0.5 * np.sqrt(3.0)])
  r1_0 = -(a1 + 2.0 * a2) / 3.0
  r3_0 = +2.0 * (a1 + 2.0 * a2) / 3.0
  r2_0 = a1

  h_aa = _make_h0_block(_H0_AA, uxx, uyy, two_uxy)[:2, :2].astype(np.complex128)
  h_bb = _make_h0_block(_H0_BB, uxx, uyy, two_uxy).astype(np.complex128)
  h_ba = np.zeros((3, 2), dtype=np.complex128)
  h_cc = _make_h0_block(_H0_CC, uxx, uyy, two_uxy).astype(np.complex128)
  h_dd = _make_h0_block(_H0_DD, uxx, uyy, two_uxy).astype(np.complex128)
  h_dc = np.zeros((3, 3), dtype=np.complex128)

  u0 = (float(uxx), float(uyy), float(two_uxy))
  u1 = _rot_strain_120(*u0)
  u2 = _rot_strain_120(*u1)
  u_rot0 = np.eye(3, dtype=np.float64)
  u_rot1 = _U_R
  u_rot2 = _U_R @ _U_R

  for r1, r3, r2, um, u_rot in (
      (r1_0, r3_0, r2_0, u0, u_rot0),
      (_rot_vec_120(r1_0), _rot_vec_120(r3_0), _rot_vec_120(r2_0), u1, u_rot1),
      (
          _rot_vec_120(_rot_vec_120(r1_0)),
          _rot_vec_120(_rot_vec_120(r3_0)),
          _rot_vec_120(_rot_vec_120(r2_0)),
          u2,
          u_rot2,
      ),
  ):
    # Standard tight-binding Bloch phase H_{ij}(k) = sum_r H_{ij}(r) exp(-i k.r)
    # matching Fang (2015) Eqs. (6)-(8) and Fang (2018) Table IV at K_+:
    e1 = np.exp(-1.0j * float(np.dot(k_eff, r1)))
    e3 = np.exp(-1.0j * float(np.dot(k_eff, r3)))
    e2 = np.exp(-1.0j * float(np.dot(k_eff, r2)))

    h1_ba = (u_rot.T @ _make_h13_block(_T_1BA, _AL_1BA, _BE_1BA, *um) @ u_rot)[
        :, :2
    ]
    h_ba += h1_ba * e1

    h1_dc = u_rot.T @ _make_h13_block(_T_1DC, _AL_1DC, _BE_1DC, *um) @ u_rot
    h3_dc = u_rot.T @ _make_h13_block(_T_3DC, _AL_3DC, _BE_3DC, *um) @ u_rot
    h_dc += h1_dc * e1 + h3_dc * e3

    h2_aa = (u_rot.T @ _make_h2_block(_T_2AA, _AL_2AA, _BE_2AA, *um) @ u_rot)[
        :2, :2
    ]
    h2_bb = u_rot.T @ _make_h2_block(_T_2BB, _AL_2BB, _BE_2BB, *um) @ u_rot
    h2_cc = u_rot.T @ _make_h2_block(_T_2CC, _AL_2CC, _BE_2CC, *um) @ u_rot
    h2_dd = u_rot.T @ _make_h2_block(_T_2DD, _AL_2DD, _BE_2DD, *um) @ u_rot

    h_aa += h2_aa * e2 + h2_aa.T * np.conj(e2)
    h_bb += h2_bb * e2 + h2_bb.T * np.conj(e2)
    h_cc += h2_cc * e2 + h2_cc.T * np.conj(e2)
    h_dd += h2_dd * e2 + h2_dd.T * np.conj(e2)

  h_odd = np.block([[h_aa, h_ba.conj().T], [h_ba, h_bb]])
  h_even = np.block([[h_cc, h_dc.conj().T], [h_dc, h_dd]])
  return np.block(
      [[h_odd, np.zeros((5, 6), dtype=np.complex128)], [np.zeros((6, 5), dtype=np.complex128), h_even]]
  )


@functools.lru_cache(maxsize=4)
def build_mose2_22band_soc_matrix(
    lambda_mo_ev: float = LAMBDA_SO_MO_EV,
    lambda_se_ev: float = LAMBDA_SO_SE_EV,
) -> np.ndarray:
  """Returns the 22x22 on-site atomic L.S matrix in eV (Fang 2015 Eq. 12).

  Spin-orbital ordering: ``[spin_up (11 orbitals), spin_down (11 orbitals)]``.
  Uses Condon-Shortley spherical-harmonic conventions matching Fang (2015)
  Sec. IV.B (p. 6) and Table II (p. 4).
  """
  lp_x = np.array(
      [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0j], [0.0, 1.0j, 0.0]],
      dtype=np.complex128,
  )
  lp_y = np.array(
      [[0.0, 0.0, 1.0j], [0.0, 0.0, 0.0], [-1.0j, 0.0, 0.0]],
      dtype=np.complex128,
  )
  lp_z = np.array(
      [[0.0, -1.0j, 0.0], [1.0j, 0.0, 0.0], [0.0, 0.0, 0.0]],
      dtype=np.complex128,
  )

  # Unitary map from |l=2, m=+2,+1,0,-1,-2> to [d_xz, d_yz, d_xy, d_{x^2-y^2}, d_{z^2}]:
  v_d = np.zeros((5, 5), dtype=np.complex128)
  v_d[1, 0] = -1.0 / np.sqrt(2.0)
  v_d[3, 0] = +1.0 / np.sqrt(2.0)
  v_d[1, 1] = +1.0j / np.sqrt(2.0)
  v_d[3, 1] = +1.0j / np.sqrt(2.0)
  v_d[0, 2] = -1.0j / np.sqrt(2.0)
  v_d[4, 2] = +1.0j / np.sqrt(2.0)
  v_d[0, 3] = +1.0 / np.sqrt(2.0)
  v_d[4, 3] = +1.0 / np.sqrt(2.0)
  v_d[2, 4] = 1.0

  m_vals = np.array([2.0, 1.0, 0.0, -1.0, -2.0], dtype=np.float64)
  j_plus = np.zeros((5, 5), dtype=np.float64)
  for idx in range(4):
    m = m_vals[idx + 1]
    j_plus[idx, idx + 1] = np.sqrt(6.0 - m * (m + 1.0))
  j_minus = j_plus.T

  ld_x = v_d.conj().T @ (0.5 * (j_plus + j_minus)) @ v_d
  ld_y = v_d.conj().T @ (-0.5j * (j_plus - j_minus)) @ v_d
  ld_z = v_d.conj().T @ np.diag(m_vals) @ v_d

  d_idx = [0, 1, 5, 6, 7]
  lx = np.zeros((11, 11), dtype=np.complex128)
  ly = np.zeros((11, 11), dtype=np.complex128)
  lz = np.zeros((11, 11), dtype=np.complex128)

  lx[np.ix_(d_idx, d_idx)] = float(lambda_mo_ev) * ld_x
  ly[np.ix_(d_idx, d_idx)] = float(lambda_mo_ev) * ld_y
  lz[np.ix_(d_idx, d_idx)] = float(lambda_mo_ev) * ld_z

  # In the M_1 parity basis p_{x,y}^{(e,o)} = (p_{x,y}^A +/- p_{x,y}^B)/sqrt(2),
  # p_z^{(e,o)} = (p_z^A -/+ p_z^B)/sqrt(2), L_z acts within each parity block
  # while L_x and L_y couple odd <-> even:
  lz[2:5, 2:5] = float(lambda_se_ev) * lp_z
  lz[8:11, 8:11] = float(lambda_se_ev) * lp_z
  lx[2:5, 8:11] = float(lambda_se_ev) * lp_x
  lx[8:11, 2:5] = float(lambda_se_ev) * lp_x
  ly[2:5, 8:11] = float(lambda_se_ev) * lp_y
  ly[8:11, 2:5] = float(lambda_se_ev) * lp_y

  sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
  sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
  sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
  return np.kron(sx, lx) + np.kron(sy, ly) + np.kron(sz, lz)


def build_mose2_22band_hamiltonian(
    k_ang_inv: np.ndarray,
    uxx: float = 0.0,
    uyy: float = 0.0,
    two_uxy: float = 0.0,
    a_angstrom: float = A0_FANG_MOSE2_ANGSTROM,
    lab_frame_k: bool = False,
) -> np.ndarray:
  """Returns the 22x22 spinful strained Wannier + atomic SOC Hamiltonian (eV)."""
  h11 = build_mose2_11band_hamiltonian(
      k_ang_inv,
      uxx=uxx,
      uyy=uyy,
      two_uxy=two_uxy,
      a_angstrom=a_angstrom,
      lab_frame_k=lab_frame_k,
  )
  return np.kron(np.eye(2, dtype=np.complex128), h11) + build_mose2_22band_soc_matrix()


def compute_mose2_k_valley_kp_parameters(
    fd_eps: float = 1.0e-5,
) -> Dict[str, float]:
  """Computes spinless K_+ massive-Dirac k.p parameters (Fang 2018 Table IV).

  Fang et al. (2018) Table IV (p. 10) lists for MoSe2 (in eV):
    ``f0 = -4.59, f1 = 1.55, f2 = 0.88, f3 = -5.01, f4 = -2.28, f5 = 1.84``.
  """
  k_plus = np.array([4.0 * np.pi / (3.0 * A0_FANG_MOSE2_ANGSTROM), 0.0])
  h0 = build_mose2_11band_hamiltonian(k_plus)[5:, 5:]
  evals0, evecs0 = eigh(h0)
  ev, ec = float(evals0[3]), float(evals0[4])
  psi_v, psi_c = evecs0[:, 3], evecs0[:, 4]

  dk = 1.0e-5
  dh_dkx = (
      build_mose2_11band_hamiltonian(k_plus + np.array([dk, 0.0]))[5:, 5:]
      - build_mose2_11band_hamiltonian(k_plus - np.array([dk, 0.0]))[5:, 5:]
  ) / (2.0 * dk)
  f2 = abs(complex(psi_c.conj() @ dh_dkx @ psi_v)) / A0_FANG_MOSE2_ANGSTROM

  evals_iso_p = np.linalg.eigvalsh(
      build_mose2_11band_hamiltonian(k_plus, uxx=fd_eps, uyy=fd_eps)[5:, 5:]
  )
  evals_iso_m = np.linalg.eigvalsh(
      build_mose2_11band_hamiltonian(k_plus, uxx=-fd_eps, uyy=-fd_eps)[5:, 5:]
  )
  dec_dtr = float(evals_iso_p[4] - evals_iso_m[4]) / (4.0 * fd_eps)
  dev_dtr = float(evals_iso_p[3] - evals_iso_m[3]) / (4.0 * fd_eps)

  dh_dsh = 0.5 * (
      build_mose2_11band_hamiltonian(k_plus, uxx=1.0, uyy=-1.0)[5:, 5:] - h0
  )
  f5 = abs(complex(psi_c.conj() @ dh_dsh @ psi_v))

  return {
      'f0': 0.5 * (ec + ev),
      'f1': ec - ev,
      'f2': float(f2),
      'f3': 0.5 * (dec_dtr + dev_dtr),
      'f4': 0.5 * (dec_dtr - dev_dtr),
      'f5': float(f5),
  }


@functools.lru_cache(maxsize=32)
def compute_mose2_q_deformation_potentials_mev(
    include_soc: bool = True,
    band: str = 'lower',
    method: str = 'hellmann_feynman',
    fd_eps: float = 1.0e-5,
    lab_frame_k: bool = False,
) -> QValleyDeformationResult:
  """Computes monolayer MoSe2 Q-valley CBM deformation potentials (in meV).

  Evaluates the conduction-band Q-valley minimum along ``Gamma -> K_+``
  (``hat{e}_par = hat{x}``, ``hat{e}_perp = hat{y}``) and returns
  ``(D_par^Q, D_perp^Q, D_shear^Q)`` in meV:
    - ``include_soc=True, band='lower'``: 22-band Wannier + atomic SOC lower
      spin-split ``Q^(2)`` CBM (index 14; ``(+64.209, -1352.455) meV``).
    - ``include_soc=True, band='upper'``: 22-band Wannier + atomic SOC upper
      spin-split ``Q^(1)`` CBM (index 15; ``(+117.611, -1086.035) meV``).
    - ``include_soc=False``: 11-band (6-band M_1-even) spinless ``Q`` CBM
      (index 7 in 11-band / index 4 in 6-band; ``(+93.917, -1204.205) meV``).

  Args:
    include_soc: Whether to include the 22-band atomic SOC Hamiltonian.
    band: ``'lower'`` (``Q^(2)``) or ``'upper'`` (``Q^(1)``) when
      ``include_soc=True``.
    method: One of ``'hellmann_feynman'``, ``'fixed_k_fd'``, or
      ``'reminimized_fd'``.
    fd_eps: Strain step ``epsilon`` for central finite-difference methods.
    lab_frame_k: Whether Bloch phases use ``(I + u) r_0`` in FD checks.
  """
  k_mag = 4.0 * np.pi / (3.0 * A0_FANG_MOSE2_ANGSTROM)
  if include_soc:
    band_key = str(band).strip().lower()
    if band_key == 'lower':
      cb_idx = 14
    elif band_key == 'upper':
      cb_idx = 15
    else:
      raise ValueError(f"band must be 'lower' or 'upper', got {band!r}.")
    ham_fn = lambda kv, uxx=0.0, uyy=0.0, two_uxy=0.0: build_mose2_22band_hamiltonian(
        kv, uxx=uxx, uyy=uyy, two_uxy=two_uxy, lab_frame_k=lab_frame_k
    )
  else:
    cb_idx = 7
    ham_fn = lambda kv, uxx=0.0, uyy=0.0, two_uxy=0.0: build_mose2_11band_hamiltonian(
        kv, uxx=uxx, uyy=uyy, two_uxy=two_uxy, lab_frame_k=lab_frame_k
    )

  res_k = minimize_scalar(
      lambda s: float(np.linalg.eigvalsh(ham_fn(np.array([s * k_mag, 0.0])))[cb_idx]),
      bounds=(0.42, 0.72),
      method='bounded',
      options={'xatol': 1.0e-13},
  )
  s_q = float(res_k.x)
  k_q = np.array([s_q * k_mag, 0.0], dtype=np.float64)
  evals0, evecs0 = eigh(ham_fn(k_q))
  e_q_ev = float(evals0[cb_idx])
  split_mev = (
      float(evals0[15] - evals0[14]) * 1.0e3 if include_soc else 0.0
  )

  method_key = str(method).strip().lower()
  if method_key == 'hellmann_feynman':
    if lab_frame_k:
      raise ValueError("method='hellmann_feynman' requires lab_frame_k=False.")
    psi_q = evecs0[:, cb_idx]
    h0 = ham_fn(k_q, 0.0, 0.0, 0.0)
    dh_duxx = ham_fn(k_q, 1.0, 0.0, 0.0) - h0
    dh_duyy = ham_fn(k_q, 0.0, 1.0, 0.0) - h0
    dh_d2uxy = ham_fn(k_q, 0.0, 0.0, 1.0) - h0
    d_par_mev = float(np.real(psi_q.conj() @ dh_duxx @ psi_q)) * 1.0e3
    d_perp_mev = float(np.real(psi_q.conj() @ dh_duyy @ psi_q)) * 1.0e3
    d_shear_mev = float(np.real(psi_q.conj() @ dh_d2uxy @ psi_q)) * 1.0e3
  elif method_key == 'fixed_k_fd':
    eps = float(fd_eps)
    ep_p = float(np.linalg.eigvalsh(ham_fn(k_q, +eps, 0.0, 0.0))[cb_idx])
    ep_m = float(np.linalg.eigvalsh(ham_fn(k_q, -eps, 0.0, 0.0))[cb_idx])
    eq_p = float(np.linalg.eigvalsh(ham_fn(k_q, 0.0, +eps, 0.0))[cb_idx])
    eq_m = float(np.linalg.eigvalsh(ham_fn(k_q, 0.0, -eps, 0.0))[cb_idx])
    es_p = float(np.linalg.eigvalsh(ham_fn(k_q, 0.0, 0.0, +2.0 * eps))[cb_idx])
    es_m = float(np.linalg.eigvalsh(ham_fn(k_q, 0.0, 0.0, -2.0 * eps))[cb_idx])
    d_par_mev = (ep_p - ep_m) / (2.0 * eps) * 1.0e3
    d_perp_mev = (eq_p - eq_m) / (2.0 * eps) * 1.0e3
    d_shear_mev = (es_p - es_m) / (4.0 * eps) * 1.0e3
  elif method_key == 'reminimized_fd':
    eps = float(fd_eps)

    def _min_energy(uxx_v: float, uyy_v: float, two_uxy_v: float) -> float:
      opt = minimize(
          lambda kv: float(
              np.linalg.eigvalsh(ham_fn(kv, uxx_v, uyy_v, two_uxy_v))[cb_idx]
          ),
          k_q,
          method='BFGS',
          options={'gtol': 1.0e-11},
      )
      return float(opt.fun)

    d_par_mev = (_min_energy(+eps, 0.0, 0.0) - _min_energy(-eps, 0.0, 0.0)) / (
        2.0 * eps
    ) * 1.0e3
    d_perp_mev = (_min_energy(0.0, +eps, 0.0) - _min_energy(0.0, -eps, 0.0)) / (
        2.0 * eps
    ) * 1.0e3
    d_shear_mev = (
        _min_energy(0.0, 0.0, +2.0 * eps) - _min_energy(0.0, 0.0, -2.0 * eps)
    ) / (4.0 * eps) * 1.0e3
  else:
    raise ValueError(
        f'Unsupported method={method!r}; expected hellmann_feynman, '
        'fixed_k_fd, or reminimized_fd.'
    )

  return QValleyDeformationResult(
      k_q_over_k=s_q,
      k_q_ang_inv=float(k_q[0]),
      energy_ev=e_q_ev,
      spin_splitting_2delta_q_mev=split_mev,
      d_par_mev=d_par_mev,
      d_perp_mev=d_perp_mev,
      d_shear_mev=d_shear_mev,
  )
