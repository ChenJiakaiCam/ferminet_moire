"""Hamiltonian for 2D electron-hole moire bilayer systems."""

from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import chex
from ferminet import hamiltonian
from ferminet import networks
from ferminet.moire import ewald as moire_ewald
from ferminet.moire import geometry
from ferminet.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

def _coordinate_inverse_masses(
    nspins: Sequence[int], #Number of each species, vector of length 2 or 4.
    masses: Sequence[float], #Masses of each species, vector of length 2 or 4. 
    ndim: int = 2,
) -> np.ndarray:
    """Builds per-coordinate inverse-mass weights ``1 / m_alpha^*``.
    Note nspins below is essentially the number of different particle species, with different spins counting as different species.

    Supports both 4-channel ``nspins = (n_e_up, n_e_down, n_h_up, n_h_down)`` and
    2-channel ``nspins = (n_e, n_h)``. When ``len(nspins) == 4`` and
    ``len(masses) == 2``, ``masses = (mass_e, mass_h)`` is automatically expanded
    to ``(mass_e, mass_e, mass_h, mass_h)``.

    e.g. For four channels, the outputs is shaped 2*(N_e_up + N_e_down + N_h_up + N_h_down) like (1/m_e, 1/m_e ,....., 1/m_h, 1/m_h....) with factor two because it's outputing one for x and y coordinates
    """
    #Just some input type checks
    geometry.layer_particle_counts(nspins)
    ndim_int = geometry.validate_nonnegative_int(ndim, 'ndim')
    if ndim_int <= 0:
      raise ValueError(f'ndim must be positive, got {ndim}.')
      
    
    if len(nspins) == 4 and len(masses) == 2:
        masses = (masses[0], masses[0], masses[1], masses[1])
    if len(nspins) != len(masses):
        raise ValueError(
            f'Length of nspins ({nspins}) must match masses ({masses}) or be 4 '
            'channels with 2 layer masses (mass_e, mass_h).'
        )
        
    for m in masses:
        if m <= 0.0:
            raise ValueError(f'Effective masses must be positive, got {masses}.')
    counts = [
        geometry.validate_nonnegative_int(n, 'nspins entry') * ndim_int
        for n in nspins
    ] # Number of each particle species
    inv_m = [1.0 / float(m) for m in masses]
    if sum(counts) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.repeat(np.asarray(inv_m, dtype=np.float64), counts)

def local_kinetic_energy(
    f: networks.FermiNetLike,
    nspins: Sequence[int],
    masses: Sequence[float] = (1.0, 1.0),
    ndim: int = 2,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> hamiltonian.KineticEnergy:
  r"""Creates a function for the mass-weighted local kinetic energy.

  Evaluates: T_L(R) = Tphi(R)/phi(R) = -1/2 sum_i (1/m_i)(del^2 phi(R))/phi(R)

  Args:
    f: Callable evaluating the wavefunction as a ``(phase_or_sign, logabs)``
      tuple.
    nspins: Tuple ``(n_electrons, n_holes)`` of particle counts per channel.
    masses: Tuple ``(mass_e, mass_h)`` of effective masses in units of ``m_0``.
    ndim: Spatial dimensionality of each particle (default 2).
    use_scan: Whether to use ``lax.scan`` for the default Laplacian loop.
    complex_output: Whether ``f`` outputs a complex phase in its first slot.
    laplacian_method: Either ``'default'`` or ``'folx'``. folx for forward laplacian

  Returns:
    Callable ``ke(params, data) -> scalar`` evaluating the local kinetic energy.
  """
  
  inv_masses_np = _coordinate_inverse_masses(nspins, masses, ndim=ndim)
  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)
  
  if laplacian_method == 'default':
      
    def _lapl_over_f_default(params, data): #params stored the NN's trainable parameters, data contains positions, spins, atoms, charges
      pos = data.positions #flattened 1D array of [x1, y1, ...,xN, yN]
      n = pos.shape[0] #Number of coordinate compoments = 2*N_particles
      inv_masses = jnp.asarray(inv_masses_np, dtype=pos.dtype)
      eye = jnp.eye(n, dtype=pos.dtype)
      grad_f = jax.grad(logabs_f, argnums=1) #logabs_f takes 5 inputs: params, positions, spins, atoms, charges. So argnums = 1 differentiates w.r.t electron/hole positions

      def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges)

      # jax.linearize takes a function and a position, and outputs the function at that position, and a new function that multiplies the Jacobian (here it's d^2(f)/dx_idx_j) at that position with a vector, useful for linear approximations. 
      primal, dgrad_f = jax.linearize(grad_f_closure, pos)
    
      phase_primal = None
    
      # Here we compute hessian_diagonal which contains 1/m_i * d^2(log(phi))/dR_i^2, for calculation of Laplacian of log(phi) later
      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)

        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins, data.atoms, data.charges)

        phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, pos)
        hessian_diagonal = lambda i: inv_masses[i] * (
            dgrad_f(eye[i])[i] + 1.0j * dgrad_phase(eye[i])[i]
        ) #ln(phi) = u + i*theta, so here just sum the hessian of the two components
      else:
        hessian_diagonal = lambda i: inv_masses[i] * dgrad_f(eye[i])[i]
        
      
      # Sum over elements of hessian_diagonal, either using lax.scan or lax.fori_loop
      if use_scan: #Note hessian_diagonal is function of coordinate indices
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n
        )
        result = -0.5 * jnp.sum(diagonal)
      else:
        init_val = jnp.asarray(0.0 + 0.0j if complex_output else 0.0, dtype=(
            jnp.complex128
            if (complex_output and pos.dtype == jnp.float64)
            else (jnp.complex64 if complex_output else pos.dtype)
        ))
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), init_val
        )
        
        
      #For real phi, d_i^2(phi)/phi = d_i^2(log(|phi|)) + (d_i(log(|phi|)))^2, below we add the second term
      result -= 0.5 * jnp.sum(inv_masses * (primal**2)) #primal = d_i(log(|phi|))
      
      #Let phi = exp(u+i*theta), u = log(|phi|), primal = d_i(u), phase_primal = d_i(theta)
      # Then d_i^2(phi)/phi includes a few more terms in terms of derivatives of theta or phase_primal
      if complex_output:
        assert phase_primal is not None
        result += 0.5 * jnp.sum(inv_masses * (phase_primal**2))
        result -= 1.0j * jnp.sum(inv_masses * primal * phase_primal)
      return result # -1/2 * sum_i[(1/m_i) * d_i^2(phi)/phi] where i runs over each coordinates of each particle
    
    return _lapl_over_f_default
  

  elif laplacian_method == 'folx':
    sqrt_inv_masses_np = np.sqrt(inv_masses_np) #Using sqrt(mass) is for coordinate scaling, so that later when folx computes the second derivatives the 1/m_i automatically shows up
    
    def _lapl_over_f_folx(params, data):
      pos = data.positions
      scale = jnp.asarray(sqrt_inv_masses_np, dtype=pos.dtype)
      scaled_pos = pos / scale # List of r_i*sqrt(m_i)
      
      
      f_closure = lambda y: f( #Inputs y*scale so that evaluating f_closure at scaled_pos gives the original wavefunction value
          params, y * scale, data.spins, data.atoms, data.charges
      )
      #Note d^2/dy^2 = 1/m d^2/dR^2
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6) #sparsity_threshold settings follow ferminet/hamiltonian.py
      output = f_wrapped(scaled_pos) 
      # forward_laplacian augments each output of the original function with its derivatives
      # so since f returns (sign_or_phase, logabs), output has output[0] for sign or phase derivatives and output[1] for log|phi| derivatives
      # each output[i] contains x (the actual value), jacobian (the first derivatives) and laplacian 
      
      result = -(
          output[1].laplacian + jnp.sum(output[1].jacobian.dense_array ** 2) #note inverse mass is already included in the jacobian**2 and laplacian
      ) / 2.0
      if complex_output:
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.0j * jnp.sum(
            output[0].jacobian.dense_array * output[1].jacobian.dense_array #dense_array in the sense it doesn't throw away 0's, so outputs all du/dr_i
        )
      return result # -1/2 * sum_i[(1/m_i) * d_i^2(phi)/phi]
      
    return _lapl_over_f_folx


  else:
    raise NotImplementedError(
        f'Laplacian method {laplacian_method} not implemented.'
    )

  


HARTREE_TO_MEV: float = 27211.386245988
ScalarOrSequence = Union[float, Sequence[float], geometry.Array]


# Built-in moire potential parameter presets in the hole convention
# U_h(r) = -V_v(r) = 2 * sum_{s=1}^{N_shells} V_{h,s} * sum_{j=1}^3 cos(G_j^{(s)} . (r - r_0) + phi_{h,s}),
# where amplitudes V_{h,s} are in Hartree (1 Ha = 27211.386245988 meV) and
# phases phi_{h,s} are in radians, ordered according to
# geometry.moire_harmonic_shells(primitive_lattice, num_shells).
MOIRE_POTENTIAL_PRESETS: Mapping[
    str, Tuple[Tuple[float, ...], Tuple[float, ...]]
] = {
    # 1. Zhang, Yuan, and Fu (2020) [arXiv:1910.14061] 1-shell continuum model:
    # V_h = -15.0 meV (-5.512e-4 Ha), phi_h = +45 deg (+pi/4 rad).
    'zhang_2020_1shell': (
        (-5.512e-4,),
        (float(np.pi / 4.0),),
    ),
    # 2. Luskin, Geier, Fu, and Zhu (2026) [arXiv:2608.08917] Table S1
    # 3-shell rigid DFT fits (hole convention U_h = -V_v):
    #   SCAN-rVV10: V_{h,1..3} = (-3.659, +0.279, -0.603) meV,
    #               phi_{1..3} = (+50.89, +10.60, -110.2) deg.
    'luskin_2026_3shell_scan': (
        (-1.344658e-4, +1.025306e-5, -2.215984e-5),
        (0.888198, 0.185005, -1.923353),
    ),
    #   DFT-D3: V_{h,1..3} = (-4.902, +0.303, -0.716) meV,
    #           phi_{1..3} = (+53.97, -0.155, -98.43) deg.
    'luskin_2026_3shell_dftd3': (
        (-1.801452e-4, +1.113504e-5, -2.631251e-5),
        (0.941954, -0.002705, -1.717928),
    ),
    # 3. Luskin, Geier, Fu, and Zhu (2026) [arXiv:2608.08917] full untwisted
    # (theta = 0 deg, delta = 0.04) relaxed scalar hole potential
    # U_h(r) = U_{m,relax}(r) + U_Phi(r) (reconstructed moire potential +
    # pseudoelectric strain deformation potential Phi(r) = C * eps_0(r)),
    # retained up to 5 shells (6 C3-symmetric triads: Q = 1, 3, 4, 7a, 7b, 9)
    # or 3 leading shells (Q = 1, 3, 4):
    'luskin_2026_relaxed_scan': (
        (
            -2.808078e-4,
            -2.625710e-5,
            -1.665482e-5,
            -3.403920e-6,
            -3.511100e-6,
            -2.563096e-6,
        ),
        (-0.455925, -3.070619, 1.947429, 2.755679, -2.674509, 1.952711),
    ),
    'luskin_2026_relaxed_3shell_scan': (
        (-2.808078e-4, -2.625710e-5, -1.665482e-5),
        (-0.455925, -3.070619, 1.947429),
    ),
    'luskin_2026_relaxed_dftd3': (
        (
            -3.201871e-4,
            -2.646910e-5,
            -1.827854e-5,
            -3.606763e-6,
            -3.604609e-6,
            -2.913338e-6,
        ),
        (-0.554206, 3.140469, 1.646516, 2.616314, -2.617504, 1.737927),
    ),
    'luskin_2026_relaxed_3shell_dftd3': (
        (-3.201871e-4, -2.646910e-5, -1.827854e-5),
        (-0.554206, 3.140469, 1.646516),
    ),
    'luskin_2026_vm_relax_only_scan': (
        (
            -1.405746e-4,
            -4.876900e-6,
            -1.466914e-5,
            -1.807245e-6,
            -1.997600e-6,
            -2.294530e-6,
        ),
        (-0.936193, -2.749852, 1.797346, 2.455897, -2.333573, 1.818268),
    ),
    'luskin_2026_vm_relax_only_dftd3': (
        (
            -1.889297e-4,
            -4.785667e-6,
            -1.709443e-5,
            -2.158672e-6,
            -2.155135e-6,
            -2.720477e-6,
        ),
        (-0.991179, 3.135378, 1.485998, 2.255780, -2.257292, 1.604413),
    ),
    # 4. Wang, Gao, and Tong (2026) [arXiv:2602.09526] 3-shell and 1-shell fits
    # to the 19-point relaxed untwisted (theta = 0 deg) WS2/WSe2 diagonal DFT
    # profiles (Fig. 6d for R-type 0 deg, Fig. 7d for H-type 180 deg):
    'wang_2026_rtype_3shell': (
        (-3.116975e-4, -8.268598e-7, -4.673433e-5),
        (1.820520, 0.0, 1.800620),
    ),
    'wang_2026_htype_3shell': (
        (-1.845635e-4, -2.097774e-6, -3.614323e-5),
        (-2.056730, 0.0, -0.841550),
    ),
    'wang_2026_rtype_1shell': (
        (-2.671676e-4,),
        (-0.865683,),
    ),
    'wang_2026_htype_1shell': (
        (-1.782342e-4,),
        (1.103048,),
    ),
}


def resolve_moire_potential_params(
    amplitude: ScalarOrSequence = 0.0,
    phase: ScalarOrSequence = 0.0,
    model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes scalar, multi-shell sequence, or preset moire potential parameters.

  Args:
    amplitude: Scalar float or 1D sequence of shell amplitudes ``V_{alpha,s}``
      in Hartree. Ignored if ``model`` is provided and not ``'none'``.
    phase: Scalar float or 1D sequence of shell phases ``phi_{alpha,s}`` in
      radians. Ignored if ``model`` is provided and not ``'none'``.
    model: Optional preset name in ``MOIRE_POTENTIAL_PRESETS`` (or ``'none'``).

  Returns:
    Tuple ``(v_shells, phi_shells)`` of 1D ``np.float64`` arrays of matching
    length ``num_shells >= 1``.
  """
  if model is not None and str(model).strip().lower() not in ('', 'none'):
    key = str(model).strip().lower()
    if key not in MOIRE_POTENTIAL_PRESETS:
      valid = ', '.join(sorted(MOIRE_POTENTIAL_PRESETS.keys()))
      raise ValueError(
          f'Unknown moire potential preset {model!r}. Available presets: '
          f'{valid}.'
      )
    v_preset, phi_preset = MOIRE_POTENTIAL_PRESETS[key]
    return (
        np.asarray(v_preset, dtype=np.float64),
        np.asarray(phi_preset, dtype=np.float64),
    )

  v_arr = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
  phi_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))
  if v_arr.ndim != 1 or v_arr.size < 1:
    raise ValueError(
        f'Moire amplitude must be a scalar or non-empty 1D sequence, got shape '
        f'{v_arr.shape}.'
    )
  if phi_arr.ndim != 1 or phi_arr.size < 1:
    raise ValueError(
        f'Moire phase must be a scalar or non-empty 1D sequence, got shape '
        f'{phi_arr.shape}.'
    )
  if not (np.all(np.isfinite(v_arr)) and np.all(np.isfinite(phi_arr))):
    raise ValueError('Moire amplitude and phase entries must be finite.')

  if phi_arr.size == 1 and v_arr.size > 1:
    phi_arr = np.broadcast_to(phi_arr, v_arr.shape).copy()
  elif v_arr.size != phi_arr.size:
    raise ValueError(
        f'Moire amplitude and phase must have the same number of shells, got '
        f'len(amplitude)={v_arr.size} and len(phase)={phi_arr.size}.'
    )
  return v_arr, phi_arr


def solve_luskin_continuum_relaxed_potential(
    theta_deg: float = 0.0,
    delta: float = 0.04,
    functional: str = 'scan',
    include_deformation_potential: bool = True,
    num_shells: int = 6,
    grid_size: int = 36,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes multi-shell hole potential coefficients from Luskin et al. (2026).

  Solves the continuum configuration-space elasticity + Generalized Stacking
  Fault Energy (GSFE) minimization of Luskin, Geier, Fu, and Zhu
  (arXiv:2608.08917, Secs. S1.3-S2.2) on an ``N x N`` stacking grid and returns
  the hole-convention Fourier shell parameters ``(v_shells_ha, phi_shells_rad)``
  for ``U_h(r) = -(V_{m,relax}(r) + Phi(r))``.

  Args:
    theta_deg: Interlayer twist angle in degrees (default ``0.0``).
    delta: Lattice mismatch ``(a_WSe2 - a_WS2) / a_WS2`` (default ``0.04``).
    functional: DFT functional for Table S1 rigid coefficients, either
      ``'scan'`` (SCAN-rVV10) or ``'dftd3'`` (DFT-D3).
    include_deformation_potential: Whether to include the scalar strain
      deformation potential ``Phi(r) = (f_3 - f_4) eps_0(r)`` with
      ``f_3 - f_4 = -2.25 eV``.
    num_shells: Number of C3-symmetric harmonic shells/triads to extract.
    grid_size: Configuration-space discretization ``N`` (default ``36``).

  Returns:
    Tuple ``(v_shells_ha, phi_shells_rad)`` of 1D ``np.float64`` arrays of
    length ``num_shells`` in Hartree and radians, respectively.
  """
  import scipy.optimize

  n_shells = geometry.validate_nonnegative_int(num_shells, 'num_shells')
  if n_shells < 1:
    raise ValueError(f'num_shells must be >= 1, got {num_shells}.')
  func_key = functional.strip().lower()
  if func_key in ('scan', 'scan-rvv10', 'scan_rvv10'):
    v_rigid_mev = (3.659, -0.279, 0.603)
    phi_rigid_rad = tuple(np.radians([50.89, 10.60, -110.2]))
  elif func_key in ('dftd3', 'dft-d3', 'dft_d3'):
    v_rigid_mev = (4.902, -0.303, 0.716)
    phi_rigid_rad = tuple(np.radians([53.97, -0.155, -98.43]))
  else:
    raise ValueError(
        f"functional must be 'scan' or 'dftd3', got {functional!r}."
    )

  a_mono = 3.316
  a_mat = a_mono * np.array(
      [[np.sqrt(3.0) / 2.0, np.sqrt(3.0) / 2.0], [-0.5, 0.5]],
      dtype=np.float64,
  )
  k1, g1 = 53.37 * 1000.0, 33.23 * 1000.0
  k2, g2 = 46.87 * 1000.0, 31.21 * 1000.0
  k_bar = 0.5 * (k1 + k2)
  g_bar = 0.5 * (g1 + g2)
  c1, c2, c3, c4, c5 = 12.8351, -2.0714, -0.5827, -0.7188, -0.1370

  theta = np.radians(float(theta_deg))
  c_th, s_th = np.cos(theta), np.sin(theta)
  r_theta = np.array([[c_th, -s_th], [s_th, c_th]], dtype=np.float64)
  m_mat = r_theta / (1.0 + float(delta))
  a_sc = np.linalg.inv(m_mat - np.eye(2)) @ a_mat
  b_sc = 2.0 * np.pi * np.linalg.inv(a_sc)

  n_grid = int(grid_size)
  grid_1d = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
  s0, t0 = np.meshgrid(grid_1d, grid_1d, indexing='ij')
  freqs = np.fft.fftfreq(n_grid, d=1.0 / n_grid)
  mm, nn = np.meshgrid(freqs, freqs, indexing='ij')
  qx = mm * b_sc[0, 0] + nn * b_sc[1, 0]
  qy = mm * b_sc[0, 1] + nn * b_sc[1, 1]
  dxx = (k_bar + g_bar) * qx**2 + g_bar * qy**2
  dyy = (k_bar + g_bar) * qy**2 + g_bar * qx**2
  dxy = k_bar * qx * qy
  inv_a_2pi = 2.0 * np.pi * np.linalg.inv(a_mat)

  def _energy_and_grad(x_flat: np.ndarray) -> Tuple[float, np.ndarray]:
    u = x_flat.reshape(2, n_grid, n_grid)
    ux_k = np.fft.fft2(u[0]) / (n_grid * n_grid)
    uy_k = np.fft.fft2(u[1]) / (n_grid * n_grid)
    e_el = 0.5 * np.sum(
        dxx * np.abs(ux_k) ** 2
        + dyy * np.abs(uy_k) ** 2
        + 2.0 * dxy * np.real(ux_k * np.conj(uy_k))
    )
    grad_el_ux = np.real(
        np.fft.ifft2((dxx * ux_k + dxy * uy_k) * (n_grid * n_grid))
    ) / (n_grid * n_grid)
    grad_el_uy = np.real(
        np.fft.ifft2((dxy * ux_k + dyy * uy_k) * (n_grid * n_grid))
    ) / (n_grid * n_grid)
    ds = inv_a_2pi[0, 0] * u[0] + inv_a_2pi[0, 1] * u[1]
    dt = inv_a_2pi[1, 0] * u[0] + inv_a_2pi[1, 1] * u[1]
    s, t = s0 + ds, t0 + dt
    v_gsfe = (
        c1 * (np.cos(s) + np.cos(t) + np.cos(s + t))
        + c2 * (np.cos(s + 2.0 * t) + np.cos(s - t) + np.cos(2.0 * s + t))
        + c3 * (np.cos(2.0 * s) + np.cos(2.0 * t) + np.cos(2.0 * (s + t)))
        + c4 * (np.sin(s) + np.sin(t) - np.sin(s + t))
        + c5 * (np.sin(2.0 * s + 2.0 * t) - np.sin(2.0 * s) - np.sin(2.0 * t))
    )
    dv_ds = (
        -c1 * (np.sin(s) + np.sin(s + t))
        - c2 * (
            np.sin(s + 2.0 * t)
            + np.sin(s - t)
            + 2.0 * np.sin(2.0 * s + t)
        )
        - 2.0 * c3 * (np.sin(2.0 * s) + np.sin(2.0 * (s + t)))
        + c4 * (np.cos(s) - np.cos(s + t))
        + 2.0 * c5 * (np.cos(2.0 * s + 2.0 * t) - np.cos(2.0 * s))
    )
    dv_dt = (
        -c1 * (np.sin(t) + np.sin(s + t))
        - c2 * (
            2.0 * np.sin(s + 2.0 * t)
            - np.sin(s - t)
            + np.sin(2.0 * s + t)
        )
        - 2.0 * c3 * (np.sin(2.0 * t) + np.sin(2.0 * (s + t)))
        + c4 * (np.cos(t) - np.cos(s + t))
        + 2.0 * c5 * (np.cos(2.0 * s + 2.0 * t) - np.cos(2.0 * t))
    )
    grad_ux = grad_el_ux + (
        dv_ds * inv_a_2pi[0, 0] + dv_dt * inv_a_2pi[1, 0]
    ) / (n_grid * n_grid)
    grad_uy = grad_el_uy + (
        dv_ds * inv_a_2pi[0, 1] + dv_dt * inv_a_2pi[1, 1]
    ) / (n_grid * n_grid)
    return float(e_el + np.mean(v_gsfe)), np.concatenate(
        [grad_ux.ravel(), grad_uy.ravel()]
    )

  res = scipy.optimize.minimize(
      _energy_and_grad,
      np.zeros(2 * n_grid * n_grid),
      jac=True,
      method='L-BFGS-B',
      options={'gtol': 1e-12},
  )
  u_opt = res.x.reshape(2, n_grid, n_grid)
  ds = inv_a_2pi[0, 0] * u_opt[0] + inv_a_2pi[0, 1] * u_opt[1]
  dt = inv_a_2pi[1, 0] * u_opt[0] + inv_a_2pi[1, 1] * u_opt[1]
  s, t = s0 + ds, t0 + dt

  v1, v2, v3 = v_rigid_mev
  p1, p2, p3 = phi_rigid_rad
  vm_relax = (
      2.0 * v1 * (np.cos(s + p1) + np.cos(t + p1) + np.cos(-(s + t) + p1))
      + 2.0 * v2 * (
          np.cos(s + 2.0 * t + p2)
          + np.cos(t - s + p2)
          + np.cos(-(2.0 * s + t) + p2)
      )
      + 2.0 * v3 * (
          np.cos(2.0 * s + p3)
          + np.cos(2.0 * t + p3)
          + np.cos(-2.0 * (s + t) + p3)
      )
  )
  if include_deformation_potential:
    ux_k = np.fft.fft2(u_opt[0]) / (n_grid * n_grid)
    uy_k = np.fft.fft2(u_opt[1]) / (n_grid * n_grid)
    eps0 = np.real(
        np.fft.ifft2(1j * (qx * ux_k + qy * uy_k) * (n_grid * n_grid))
    )
    # Strain tensor eps_ij(r) is computed from relative displacement Delta_u
    # via spectral differentiation (Sec. S2.2); C = f3 - f4 = -2.25 eV.
    phi_def = -2250.0 * eps0
    v_total_mev = vm_relax + phi_def
  else:
    v_total_mev = vm_relax

  u_hole_mev = -(v_total_mev - np.mean(v_total_mev))
  u_k = np.conj(np.fft.fft2(u_hole_mev)) / (n_grid * n_grid)

  # Map (p, q) in geometry.moire_harmonic_shells ordering to (m, n) on the
  # (s0, t0) grid where g_1 -> (1, 0), g_2 -> (0, 1), g_3 -> (-1, -1), so
  # G_1^{(p,q)} = p g_1 - q g_3 -> (p + q, q), sorted with (q_metric, -q, p)
  # so Shell 4a is (p=1, q=2) -> (3, 2) or (1, 3) and Shell 4b is (p=2, q=1) -> (3, 1).
  max_idx = int(np.ceil(np.sqrt(float(n_shells)))) + 2
  candidates = []
  for p in range(1, max_idx + 1):
    for q in range(0, max_idx + 1):
      candidates.append((p * p + p * q + q * q, q, p))
  candidates.sort()

  v_shells_ha = []
  phi_shells_rad = []
  for _, q, p in candidates[:n_shells]:
    # In (s0, t0), G_1^{(p,q)} = p g_1 - q g_3 has coordinates (p + q, q).
    m_idx = (p + q) % n_grid
    n_idx = q % n_grid
    c_gm = u_k[m_idx, n_idx]
    amp_mev = float(np.abs(c_gm))
    # Retain negative V_h convention and absorb the minus sign into phi_h.
    v_shells_ha.append(-amp_mev / HARTREE_TO_MEV)
    phi_shells_rad.append(float(np.angle(-c_gm)))

  return (
      np.asarray(v_shells_ha, dtype=np.float64),
      np.asarray(phi_shells_rad, dtype=np.float64),
  )


def single_species_moire_potential(
    coords: jnp.ndarray,
    harmonic_gvectors: geometry.Array,
    amplitude: ScalarOrSequence,
    phase: ScalarOrSequence = 0.0,
    offset: Sequence[float] | geometry.Array = (0.0, 0.0),
) -> jnp.ndarray:
  r"""Evaluates the single- or multi-shell moire potential for one carrier layer.

  Evaluates the general multi-shell C3-symmetric harmonic expansion:

  .. math::
    V_\alpha(\mathbf{r}) = 2 \sum_{s=1}^{N_{\mathrm{shells}}} V_{\alpha,s}
      \sum_{m=1}^{3} \cos\!\left(
        \mathbf{G}_m^{(s)} \cdot (\mathbf{r} - \mathbf{r}_{0,\alpha})
        + \phi_{\alpha,s}
      \right)

  where ``G_1^{(s)}, G_2^{(s)}, G_3^{(s)}`` are the three C3-related reciprocal
  lattice vectors in shell ``s`` satisfying ``G_1^{(s)} + G_2^{(s)} + G_3^{(s)} = 0``.
  Supports both single-shell scalar parameters (``amplitude: float``,
  ``harmonic_gvectors`` of shape ``(3, 2)``) and multi-shell sequences
  (``amplitude`` of length ``N_shells``, ``harmonic_gvectors`` of shape
  ``(N_shells, 3, 2)`` or ``(3, 2)``).

  Args:
    coords: Particle coordinates in the layer, shape ``(n_particles, 2)``.
    harmonic_gvectors: Shape ``(3, 2)`` (single shell) or
      ``(num_shells, 3, 2)`` (multi-shell) array of C3 harmonic reciprocal
      vectors. If ``amplitude`` has multiple shells and ``harmonic_gvectors``
      has shape ``(3, 2)``, higher shells are constructed automatically from
      the first-shell triad via ``geometry.moire_harmonic_shells_from_first_shell``.
    amplitude: Scalar float or 1D sequence/array ``(V_{alpha,1}, ..., V_{alpha,S})``
      of shell amplitudes in Hartree (signed).
    phase: Scalar float or 1D sequence/array ``(phi_{alpha,1}, ..., phi_{alpha,S})``
      of shell phases in radians.
    offset: Spatial shift ``r_{0,alpha}`` in Cartesian Bohr coordinates.

  Returns:
    Scalar JAX array giving ``sum_i V_alpha(r_i)`` in Hartree.
  """
  if coords.shape[0] == 0:
    return jnp.asarray(0.0, dtype=coords.dtype)

  dtype = coords.dtype
  v_arr = jnp.atleast_1d(jnp.asarray(amplitude, dtype=dtype))
  phi_arr = jnp.atleast_1d(jnp.asarray(phase, dtype=dtype))
  if v_arr.ndim != 1 or phi_arr.ndim != 1:
    raise ValueError(
        'amplitude and phase in single_species_moire_potential must be '
        f'scalars or 1D sequences, got shapes {v_arr.shape} and {phi_arr.shape}.'
    )
  num_shells = int(v_arr.shape[0])
  if phi_arr.shape[0] == 1 and num_shells > 1:
    phi_arr = jnp.broadcast_to(phi_arr, (num_shells,))
  elif phi_arr.shape[0] != num_shells:
    raise ValueError(
        f'amplitude and phase must have the same number of shells, got '
        f'{num_shells} and {phi_arr.shape[0]}.'
    )

  if not isinstance(amplitude, jax.core.Tracer):
    if np.all(np.asarray(amplitude, dtype=np.float64) == 0.0):
      return jnp.asarray(0.0, dtype=dtype)

  gvecs = jnp.asarray(harmonic_gvectors, dtype=dtype)
  if gvecs.ndim == 2:
    if gvecs.shape != (3, 2):
      raise ValueError(
          f'2D harmonic_gvectors must have shape (3, 2), got {gvecs.shape}.'
      )
    if num_shells == 1:
      gvecs = gvecs[None, :, :]  # (1, 3, 2)
    else:
      gvecs = jnp.asarray(
          geometry.moire_harmonic_shells_from_first_shell(
              gvecs, num_shells=num_shells
          ),
          dtype=dtype,
      )
  elif gvecs.ndim == 3:
    if gvecs.shape[1:] != (3, 2):
      raise ValueError(
          '3D harmonic_gvectors must have shape (num_shells, 3, 2), got '
          f'{gvecs.shape}.'
      )
    if gvecs.shape[0] < num_shells:
      raise ValueError(
          f'harmonic_gvectors has {gvecs.shape[0]} shells, fewer than '
          f'len(amplitude)={num_shells}.'
      )
    gvecs = gvecs[:num_shells]
  else:
    raise ValueError(
        'harmonic_gvectors must have shape (3, 2) or (num_shells, 3, 2), got '
        f'{gvecs.shape}.'
    )

  r0 = jnp.asarray(offset, dtype=dtype)
  shifted = coords - r0[None, :]  # (n_particles, 2)
  # projections[i, s, m] = (r_i - r_0) . G_m^{(s)}, shape (n_particles, num_shells, 3)
  projections = jnp.einsum('id,smd->ism', shifted, gvecs)
  shell_cos_sum = jnp.sum(
      jnp.cos(projections + phi_arr[None, :, None]), axis=(0, 2)
  )  # (num_shells,)
  return 2.0 * jnp.sum(v_arr * shell_cos_sum)


def make_moire_potential(
    nspins: Sequence[int],
    simulation_lattice: geometry.Array,
    moire_lattice: Optional[geometry.Array] = None,
    moire_lattice_e: Optional[geometry.Array] = None,
    moire_lattice_h: Optional[geometry.Array] = None,
    v_e: ScalarOrSequence = 0.0,
    v_h: ScalarOrSequence = 0.0,
    phi_e: ScalarOrSequence = 0.0,
    phi_h: ScalarOrSequence = 0.0,
    offset_e: Sequence[float] = (0.0, 0.0),
    offset_h: Sequence[float] = (0.0, 0.0),
    potential_model_e: Optional[str] = None,
    potential_model_h: Optional[str] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds a callable evaluating the electron + hole moire potential energy.

  Validates that both the electron and hole primitive moire lattices are
  commensurate with ``simulation_lattice``. Supports 1-shell scalar parameters,
  multi-shell sequences ``(v_1, ..., v_S)``, and named literature presets in
  ``MOIRE_POTENTIAL_PRESETS`` via ``potential_model_e`` / ``potential_model_h``.

  Args:
    nspins: Tuple ``(n_electrons, n_holes)`` or 4-channel tuple.
    simulation_lattice: Shape ``(2, 2)`` simulation supercell matrix (columns).
    moire_lattice: Default ``(2, 2)`` primitive moire lattice (columns) shared
      by both layers if ``moire_lattice_e`` / ``moire_lattice_h`` are not given.
    moire_lattice_e: Optional ``(2, 2)`` primitive moire lattice for electrons.
    moire_lattice_h: Optional ``(2, 2)`` primitive moire lattice for holes.
    v_e: Electron moire potential amplitude(s) in Hartree (scalar or sequence).
    v_h: Hole moire potential amplitude(s) in Hartree (scalar or sequence).
    phi_e: Electron moire phase(s) in radians (scalar or sequence).
    phi_h: Hole moire phase(s) in radians (scalar or sequence).
    offset_e: Electron moire spatial offset ``(x_0, y_0)`` in Bohr.
    offset_h: Hole moire spatial offset ``(x_0, y_0)`` in Bohr.
    potential_model_e: Optional preset name from ``MOIRE_POTENTIAL_PRESETS``
      for the electron layer.
    potential_model_h: Optional preset name from ``MOIRE_POTENTIAL_PRESETS``
      for the hole layer.

  Returns:
    Callable ``moire_fn(positions) -> scalar`` returning
    ``sum_i V_e(r_i^e) + sum_j V_h(r_j^h)`` in Hartree.
  """
  n_e, n_h = geometry.layer_particle_counts(nspins)
  sim_lat = geometry.validate_lattice_2d(
      simulation_lattice, name='simulation_lattice'
  )

  # By default one supercell is just one moire unit cell
  default_moire = sim_lat if moire_lattice is None else moire_lattice
  lat_e = default_moire if moire_lattice_e is None else moire_lattice_e
  lat_h = default_moire if moire_lattice_h is None else moire_lattice_h

  v_e_arr, phi_e_arr = resolve_moire_potential_params(
      v_e, phi_e, model=potential_model_e
  )
  v_h_arr, phi_h_arr = resolve_moire_potential_params(
      v_h, phi_h, model=potential_model_h
  )

  if np.any(v_e_arr != 0.0):
    geometry.validate_commensurate(lat_e, sim_lat)
    gvecs_e = np.asarray(
        geometry.moire_harmonic_shells(lat_e, num_shells=len(v_e_arr)),
        dtype=np.float64,
    )
  else:
    gvecs_e = np.zeros((len(v_e_arr), 3, 2), dtype=np.float64)

  if np.any(v_h_arr != 0.0):
    geometry.validate_commensurate(lat_h, sim_lat)
    gvecs_h = np.asarray(
        geometry.moire_harmonic_shells(lat_h, num_shells=len(v_h_arr)),
        dtype=np.float64,
    )
  else:
    gvecs_h = np.zeros((len(v_h_arr), 3, 2), dtype=np.float64)

  offset_e_arr = np.asarray(offset_e, dtype=np.float64)
  offset_h_arr = np.asarray(offset_h, dtype=np.float64)
  if offset_e_arr.shape != (2,) or offset_h_arr.shape != (2,):
    raise ValueError(
        f'offset_e and offset_h must have shape (2,), got {offset_e_arr.shape} '
        f'and {offset_h_arr.shape}.'
    )

  def moire_potential_fn(positions: jnp.ndarray) -> jnp.ndarray:
    coords = jnp.reshape(positions, (n_e + n_h, 2))
    pos_e = coords[:n_e]
    pos_h = coords[n_e:]
    pot_e = single_species_moire_potential(
        pos_e,
        gvecs_e,
        amplitude=v_e_arr,
        phase=phi_e_arr,
        offset=offset_e_arr,
    )
    pot_h = single_species_moire_potential(
        pos_h,
        gvecs_h,
        amplitude=v_h_arr,
        phase=phi_h_arr,
        offset=offset_h_arr,
    )
    return pot_e + pot_h

  return moire_potential_fn


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    ndim: int = 2,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Sequence[str] | None = None,
    lattice: Optional[geometry.Array] = None,
    moire_lattice: Optional[geometry.Array] = None,
    moire_lattice_e: Optional[geometry.Array] = None,
    moire_lattice_h: Optional[geometry.Array] = None,
    masses: Optional[Sequence[float]] = None,
    mass_e: float = 1.0,
    mass_h: float = 1.0,
    interlayer_distance: float = 0.0,
    dielectric_constant: float = 1.0,
    v_e: ScalarOrSequence = 0.0,
    v_h: ScalarOrSequence = 0.0,
    phi_e: ScalarOrSequence = 0.0,
    phi_h: ScalarOrSequence = 0.0,
    offset_e: Sequence[float] = (0.0, 0.0),
    offset_h: Sequence[float] = (0.0, 0.0),
    potential_model_e: Optional[str] = None,
    potential_model_h: Optional[str] = None,
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    ewald_alpha: Optional[float] = None,
    include_capacitor: Optional[bool] = None,
    allow_charged_capacitor: bool = False,
    layer_charges: Optional[Union[Sequence[geometry.CarrierSpec], str]] = None,
    charge_e: Optional[geometry.CarrierSpec] = None,
    charge_h: Optional[geometry.CarrierSpec] = None,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function for a 2D carrier moire bilayer.

  Matches the ``MakeLocalEnergy`` calling convention used by ``ferminet.train``.

  Args:
    f: Callable returning ``(sign_or_phase, logabs)`` of the wavefunction.
    charges: Unused nuclear charges array (ghost atom has charge 0).
    nspins: Tuple ``(n_electrons, n_holes)`` or 4-channel tuple.
    use_scan: Whether to use ``lax.scan`` for computing the Laplacian.
    ndim: Spatial dimensionality of the carriers (must be 2).
    complex_output: If True, the output of ``f`` is complex-valued.
    laplacian_method: Laplacian calculation method (``'default'`` or ``'folx'``).
    states: Number of excited states (must be 0; excited states not supported).
    state_specific: Unused excited-state flag (must be False).
    pp_type: Unused pseudopotential type.
    pp_symbols: Unused pseudopotential symbols (must be None/empty).
    lattice: Shape ``(2, 2)`` simulation supercell matrix (columns are vectors).
    moire_lattice: Shape ``(2, 2)`` primitive moire lattice (columns). Defaults
      to ``lattice``.
    moire_lattice_e: Optional ``(2, 2)`` electron primitive moire lattice.
    moire_lattice_h: Optional ``(2, 2)`` hole primitive moire lattice.
    masses: Optional ``(mass_e, mass_h)`` tuple overriding ``mass_e, mass_h``.
    mass_e: Electron effective mass ``m_e^* / m_0``.
    mass_h: Hole effective mass ``m_h^* / m_0``.
    interlayer_distance: Out-of-plane bilayer separation ``d`` in Bohr.
    dielectric_constant: Relative dielectric permittivity ``epsilon_r``.
    v_e: Electron moire potential amplitude(s) in Hartree (scalar or sequence).
    v_h: Hole moire potential amplitude(s) in Hartree (scalar or sequence).
    phi_e: Electron moire phase(s) in radians (scalar or sequence).
    phi_h: Hole moire phase(s) in radians (scalar or sequence).
    offset_e: Electron moire spatial offset in Bohr.
    offset_h: Hole moire spatial offset in Bohr.
    potential_model_e: Optional preset name from ``MOIRE_POTENTIAL_PRESETS``
      for the electron layer.
    potential_model_h: Optional preset name from ``MOIRE_POTENTIAL_PRESETS``
      for the hole layer.
    ewald_gmax: Reciprocal-space cutoff index for 2D Ewald summation.
    nlatvec: Real-space shell cutoff for 2D Ewald summation.
    ewald_alpha: Optional explicit Ewald splitting parameter.
    include_capacitor: Whether to include the ``G = 0`` capacitor term.
    allow_charged_capacitor: Whether to permit ``include_capacitor=True`` when
      total mobile charge is non-zero and ``interlayer_distance > 0``.
    layer_charges: Optional 2-tuple ``(q_1, q_2)`` (or 4-tuple or string
      shorthand ``'eh'``, ``'ee'``, ``'he'``, ``'hh'``) of per-layer carrier
      charges (``-1`` for electrons, ``+1`` for holes). Defaults to ``(-1, +1)``.
    charge_e: Optional carrier charge/type for layer 1 (defaults to ``-1``).
    charge_h: Optional carrier charge/type for layer 2 (defaults to ``+1``).

  Returns:
    Callable ``_e_l(params, key, data) -> (total_energy, None)``.
  """
  if ndim != 2:
    raise ValueError(
        f'Moire bilayer Hamiltonian requires ndim=2, got ndim={ndim}.'
    )
  if states > 0 or state_specific:
    raise NotImplementedError(
        'Excited states are not implemented for moire bilayer Hamiltonian.'
    )
  if pp_symbols:
    raise NotImplementedError(
        'Pseudopotentials are not supported for moire bilayer Hamiltonian.'
    )
  del charges, pp_type

  if lattice is None:
    lattice = np.eye(2)  # supercell lattice

  species_masses = tuple(masses) if masses is not None else (mass_e, mass_h)

  ke_fn = local_kinetic_energy(
      f=f,
      nspins=nspins,
      masses=species_masses,
      ndim=ndim,
      use_scan=use_scan,
      complex_output=complex_output,
      laplacian_method=laplacian_method,
  )

  moire_pot_fn = make_moire_potential(
      nspins=nspins,
      simulation_lattice=lattice,
      moire_lattice=moire_lattice,
      moire_lattice_e=moire_lattice_e,
      moire_lattice_h=moire_lattice_h,
      v_e=v_e,
      v_h=v_h,
      phi_e=phi_e,
      phi_h=phi_h,
      offset_e=offset_e,
      offset_h=offset_h,
      potential_model_e=potential_model_e,
      potential_model_h=potential_model_h,
  )

  coulomb_pot_fn = moire_ewald.make_bilayer_ewald_potential(
      lattice=lattice,
      nspins=nspins,
      interlayer_distance=interlayer_distance,
      dielectric_constant=dielectric_constant,
      ewald_gmax=ewald_gmax,
      nlatvec=nlatvec,
      alpha=ewald_alpha,
      include_capacitor=include_capacitor,
      allow_charged_capacitor=allow_charged_capacitor,
      charges=layer_charges,
      charge_e=charge_e,
      charge_h=charge_h,
  )

  def _e_l(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    del key  # unused
    kinetic = ke_fn(params, data)
    v_moire = moire_pot_fn(data.positions)
    v_coulomb = coulomb_pot_fn(data.positions)
    return kinetic + v_moire + v_coulomb, None

  return _e_l
  
  
  
  

  
    
    
    

  
  