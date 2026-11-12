"""Hamiltonian for 2D electron-hole moire bilayer systems."""

from typing import Callable, Optional, Sequence, Tuple, Union

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

ScalarOrSequence = Union[float, Sequence[float], geometry.Array]

def resolve_moire_potential_params(
    amplitude: ScalarOrSequence = 0.0,
    phase: ScalarOrSequence = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes scalar or multi-shell sequence moire potential parameters.

  Args:
    amplitude: Scalar float or 1D sequence of shell amplitudes ``V_{alpha,s}``
      in Hartree.
    phase: Scalar float or 1D sequence of shell phases ``phi_{alpha,s}`` in
      radians.

  Returns:
    Tuple ``(v_shells, phi_shells)`` of 1D ``np.float64`` arrays of matching
    length ``num_shells >= 1``.
  """
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


def resolve_anisotropic_moire_potential_params(
    amplitude: ScalarOrSequence = 0.0,
    phase: ScalarOrSequence = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes 1st-shell anisotropic moire params to shape ``(3,)``.

  Supports either a common scalar ``v`` / ``phi`` across all three first-shell
  reciprocal vectors ``(G_1, G_2, G_3)`` or per-harmonic 3-tuples
  ``(v_1, v_2, v_3)`` and ``(phi_1, phi_2, phi_3)``.

  Args:
    amplitude: Scalar float or length-3 sequence of harmonic amplitudes in
      Hartree.
    phase: Scalar float or length-3 sequence of harmonic phases in radians.

  Returns:
    Tuple ``(v_harmonics, phi_harmonics)`` of shape ``(3,)`` float64 arrays.
  """
  v_arr = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
  phi_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))
  if v_arr.ndim != 1 or v_arr.size not in (1, 3):
    raise ValueError(
        'Anisotropic moire potential currently supports only the first shell: '
        'amplitude must be a scalar or a length-3 sequence (v_1, v_2, v_3) '
        f'for (G_1, G_2, G_3), got shape {np.asarray(amplitude).shape}.'
    )
  if phi_arr.ndim != 1 or phi_arr.size not in (1, 3):
    raise ValueError(
        'Anisotropic moire potential currently supports only the first shell: '
        'phase must be a scalar or a length-3 sequence (phi_1, phi_2, phi_3) '
        f'for (G_1, G_2, G_3), got shape {np.asarray(phase).shape}.'
    )
  if not (np.all(np.isfinite(v_arr)) and np.all(np.isfinite(phi_arr))):
    raise ValueError('Moire amplitude and phase entries must be finite.')
  if v_arr.size == 1:
    v_arr = np.broadcast_to(v_arr, (3,)).copy()
  if phi_arr.size == 1:
    phi_arr = np.broadcast_to(phi_arr, (3,)).copy()
  return v_arr, phi_arr


def _inverse_and_inv_sqrt_mass_2d(
    mass_tensor: Optional[geometry.Array],
    scalar_mass: float,
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
  """Returns ``(M^{-1}, M^{-1/2})`` as symmetric ``(2, 2)`` float64 matrices."""
  if mass_tensor is not None:
    mat = geometry.validate_mass_tensor_2d(mass_tensor, name=name)
    eigvals, eigvecs = np.linalg.eigh(mat)
    inv_mat = (eigvecs / eigvals) @ eigvecs.T
    inv_sqrt_mat = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
    inv_mat = 0.5 * (inv_mat + inv_mat.T)
    inv_sqrt_mat = 0.5 * (inv_sqrt_mat + inv_sqrt_mat.T)
    return inv_mat, inv_sqrt_mat

  m_val = float(scalar_mass)
  if not np.isfinite(m_val) or m_val <= 0.0:
    raise ValueError(
        f'{name} must be finite and positive, got {scalar_mass!r}.'
    )
  eye2 = np.eye(2, dtype=np.float64)
  return (1.0 / m_val) * eye2, (1.0 / np.sqrt(m_val)) * eye2


def _particle_inverse_mass_blocks(
    nspins: Sequence[int],
    masses: Sequence[float],
    mass_tensor_e: Optional[geometry.Array],
    mass_tensor_h: Optional[geometry.Array],
    ndim: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Builds per-particle ``M^{-1}``, ``M^{-1/2}``, and ``(2N, 2N)`` basis."""
  if ndim != 2:
    raise ValueError(
        f'Anisotropic mass tensor requires ndim=2, got ndim={ndim}.'
    )
  if mass_tensor_e is not None and mass_tensor_h is not None:
    raise ValueError(
        'Anisotropic mass tensor is currently supported on at most one layer '
        '(either mass_tensor_e or mass_tensor_h, not both).'
    )
  n_e, n_h = geometry.layer_particle_counts(nspins)
  masses_seq = list(masses)
  if len(nspins) == 4 and len(masses_seq) == 2:
    m_e_scalar, m_h_scalar = float(masses_seq[0]), float(masses_seq[1])
  elif len(nspins) == len(masses_seq) and len(masses_seq) == 2:
    m_e_scalar, m_h_scalar = float(masses_seq[0]), float(masses_seq[1])
  elif len(nspins) == len(masses_seq) and len(masses_seq) == 4:
    if mass_tensor_e is None and not np.isclose(
        masses_seq[0], masses_seq[1], rtol=1e-12, atol=0.0
    ):
      raise ValueError(
          '4-channel masses must have equal electron-layer masses '
          f'masses[0]==masses[1], got {masses_seq}.'
      )
    if mass_tensor_h is None and not np.isclose(
        masses_seq[2], masses_seq[3], rtol=1e-12, atol=0.0
    ):
      raise ValueError(
          '4-channel masses must have equal hole-layer masses '
          f'masses[2]==masses[3], got {masses_seq}.'
      )
    m_e_scalar, m_h_scalar = float(masses_seq[0]), float(masses_seq[2])
  else:
    raise ValueError(
        f'Length of nspins ({nspins}) must match masses ({masses}) or be 4 '
        'channels with 2 layer masses (mass_e, mass_h).'
    )

  w_e, s_e = _inverse_and_inv_sqrt_mass_2d(
      mass_tensor_e, m_e_scalar, 'mass_tensor_e'
  )
  w_h, s_h = _inverse_and_inv_sqrt_mass_2d(
      mass_tensor_h, m_h_scalar, 'mass_tensor_h'
  )
  n_tot = n_e + n_h
  if n_tot == 0:
    return (
        np.zeros((0, 2, 2), dtype=np.float64),
        np.zeros((0, 2, 2), dtype=np.float64),
        np.zeros((0, 0), dtype=np.float64),
    )

  w_blocks = np.empty((n_tot, 2, 2), dtype=np.float64)
  s_blocks = np.empty((n_tot, 2, 2), dtype=np.float64)
  if n_e > 0:
    w_blocks[:n_e] = w_e
    s_blocks[:n_e] = s_e
  if n_h > 0:
    w_blocks[n_e:] = w_h
    s_blocks[n_e:] = s_h

  s_basis = np.zeros((2 * n_tot, 2 * n_tot), dtype=np.float64)
  for p_idx in range(n_tot):
    sl = slice(2 * p_idx, 2 * p_idx + 2)
    s_basis[sl, sl] = s_blocks[p_idx]
  return w_blocks, s_blocks, s_basis


def _make_anisotropic_kinetic_energy(
    f: networks.FermiNetLike,
    nspins: Sequence[int],
    masses: Sequence[float],
    ndim: int,
    use_scan: bool,
    complex_output: bool,
    laplacian_method: str,
    mass_tensor_e: Optional[geometry.Array],
    mass_tensor_h: Optional[geometry.Array],
) -> hamiltonian.KineticEnergy:
  """Builds local kinetic energy ``-1/2 sum_p d_i (M_p^{-1})_{ij} d_j psi / psi``."""
  w_blocks_np, s_blocks_np, s_basis_np = _particle_inverse_mass_blocks(
      nspins=nspins,
      masses=masses,
      mass_tensor_e=mass_tensor_e,
      mass_tensor_h=mass_tensor_h,
      ndim=ndim,
  )
  n_particles = int(w_blocks_np.shape[0])
  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  if laplacian_method == 'default':

    def _lapl_over_f_aniso_default(params, data):
      pos = data.positions
      if n_particles == 0:
        return jnp.asarray(
            0.0 + 0.0j if complex_output else 0.0,
            dtype=(
                jnp.complex128
                if (complex_output and pos.dtype == jnp.float64)
                else (jnp.complex64 if complex_output else pos.dtype)
            ),
        )
      n = pos.shape[0]
      w_blocks = jnp.asarray(w_blocks_np, dtype=pos.dtype)
      basis = jnp.asarray(s_basis_np, dtype=pos.dtype)
      grad_f = jax.grad(logabs_f, argnums=1)

      def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges)

      primal, dgrad_f = jax.linearize(grad_f_closure, pos)
      phase_primal = None
      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)

        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins, data.atoms, data.charges)

        phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, pos)

        def hessian_diagonal(i):
          return jnp.dot(basis[i], dgrad_f(basis[i])) + 1.0j * jnp.dot(
              basis[i], dgrad_phase(basis[i])
          )

      else:

        def hessian_diagonal(i):
          return jnp.dot(basis[i], dgrad_f(basis[i]))

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n
        )
        result = -0.5 * jnp.sum(diagonal)
      else:
        init_val = jnp.asarray(
            0.0 + 0.0j if complex_output else 0.0,
            dtype=(
                jnp.complex128
                if (complex_output and pos.dtype == jnp.float64)
                else (jnp.complex64 if complex_output else pos.dtype)
            ),
        )
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), init_val
        )

      primal_2d = jnp.reshape(primal, (n_particles, 2))
      w_primal_2d = jnp.einsum('pij,pj->pi', w_blocks, primal_2d)
      result -= 0.5 * jnp.sum(primal_2d * w_primal_2d)
      if complex_output:
        assert phase_primal is not None
        phase_2d = jnp.reshape(phase_primal, (n_particles, 2))
        w_phase_2d = jnp.einsum('pij,pj->pi', w_blocks, phase_2d)
        result += 0.5 * jnp.sum(phase_2d * w_phase_2d)
        result -= 1.0j * jnp.sum(primal_2d * w_phase_2d)
      return result

    return _lapl_over_f_aniso_default

  if laplacian_method == 'folx':

    def _lapl_over_f_aniso_folx(params, data):
      pos = data.positions
      if n_particles == 0:
        return jnp.asarray(
            0.0 + 0.0j if complex_output else 0.0,
            dtype=(
                jnp.complex128
                if (complex_output and pos.dtype == jnp.float64)
                else (jnp.complex64 if complex_output else pos.dtype)
            ),
        )
      s_blocks = jnp.asarray(s_blocks_np, dtype=pos.dtype)
      zeros = jnp.zeros_like(pos)

      def f_closure(z):
        z_2d = jnp.reshape(z, (n_particles, 2))
        delta_pos = jnp.einsum('pij,pj->pi', s_blocks, z_2d).reshape(pos.shape)
        return f(
            params, pos + delta_pos, data.spins, data.atoms, data.charges
        )

      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      output = f_wrapped(zeros)
      result = -(
          output[1].laplacian + jnp.sum(output[1].jacobian.dense_array ** 2)
      ) / 2.0
      if complex_output:
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.0j * jnp.sum(
            output[0].jacobian.dense_array * output[1].jacobian.dense_array
        )
      return result

    return _lapl_over_f_aniso_folx

  raise NotImplementedError(
      f'Laplacian method {laplacian_method} not implemented.'
  )

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
    laplacian_method: str = 'folx',
    mass_tensor_e: Optional[geometry.Array] = None,
    mass_tensor_h: Optional[geometry.Array] = None,
) -> hamiltonian.KineticEnergy:
  r"""Creates a function for the mass-weighted local kinetic energy.

  Evaluates: T_L(R) = Tphi(R)/phi(R) = -1/2 sum_i (1/m_i)(del^2 phi(R))/phi(R),
  or ``-1/2 sum_p del_i (M_p^{-1})_{ij} del_j phi(R) / phi(R)`` when a 2x2
  symmetric positive-definite effective mass tensor ``M_p`` is provided on one
  of the layers.

  Args:
    f: Callable evaluating the wavefunction as a ``(phase_or_sign, logabs)``
      tuple.
    nspins: Tuple ``(n_electrons, n_holes)`` (or 4-channel spinful counts) of
      particle counts per channel.
    masses: Tuple ``(mass_e, mass_h)`` of effective masses in units of ``m_0``.
    ndim: Spatial dimensionality of each particle (default 2).
    use_scan: Whether to use ``lax.scan`` for the default Laplacian loop.
    complex_output: Whether ``f`` outputs a complex phase in its first slot.
    laplacian_method: Either ``'default'`` or ``'folx'``. folx for forward laplacian
    mass_tensor_e: Optional ``(2, 2)`` SPD effective mass tensor for the
      electron layer in units of ``m_0``.
    mass_tensor_h: Optional ``(2, 2)`` SPD effective mass tensor for the hole
      layer in units of ``m_0``.

  Returns:
    Callable ``ke(params, data) -> scalar`` evaluating the local kinetic energy.
  """
  if mass_tensor_e is not None or mass_tensor_h is not None:
    return _make_anisotropic_kinetic_energy(
        f=f,
        nspins=nspins,
        masses=masses,
        ndim=ndim,
        use_scan=use_scan,
        complex_output=complex_output,
        laplacian_method=laplacian_method,
        mass_tensor_e=mass_tensor_e,
        mass_tensor_h=mass_tensor_h,
    )

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

  


def single_species_moire_potential(
    coords: jnp.ndarray,
    harmonic_gvectors: geometry.Array,
    amplitude: ScalarOrSequence,
    phase: ScalarOrSequence = 0.0,
    offset: Sequence[float] | geometry.Array = (0.0, 0.0),
) -> jnp.ndarray:
  r"""Evaluates the single- or multi-shell moire potential for one carrier layer.

  V(r) = 2V_a sum(m from 1 to 3) cos(g_m . (r-r_0_a) + phi_a)

  where ``g_1, g_2, g_3`` are the three C3-related first-shell moire reciprocal
  lattice vectors satisfying ``g_1 + g_2 + g_3 = 0``.
  a = e or h
  r_0_a (offset) shifts the whole potential pattern in real space
  phi_a is a common phase 
  
  Can also be generalized to multi-shell potentials.

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
    Scalar JAX array (i.e. just a single number) giving ``sum_i V_alpha(r_i)``.
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

  if not isinstance(amplitude, jax.Array):
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


def anisotropic_single_species_moire_potential(
    coords: jnp.ndarray,
    first_shell_gvectors: geometry.Array,
    amplitude: ScalarOrSequence,
    phase: ScalarOrSequence = 0.0,
    offset: Sequence[float] | geometry.Array = (0.0, 0.0),
) -> jnp.ndarray:
  r"""Evaluates the 1st-shell anisotropic moire potential for one carrier layer.

  Evaluates

  .. math::
      V_\alpha(r) = 2 \sum_{m=1}^3 v_{\alpha,m}
      \cos(G_m \cdot (r - r_{0,\alpha}) + \phi_{\alpha,m}),

  where ``(G_1, G_2, G_3)`` are the three (generally non-equilateral)
  first-shell moire reciprocal lattice vectors satisfying
  ``G_1 + G_2 + G_3 = 0``.

  Args:
    coords: Particle coordinates in the layer, shape ``(n_particles, 2)``.
    first_shell_gvectors: Shape ``(3, 2)`` array containing ``(G_1, G_2, G_3)``
      in ``Bohr^-1``.
    amplitude: Scalar float ``v`` or length-3 sequence ``(v_1, v_2, v_3)`` in
      Hartree.
    phase: Scalar float ``phi`` or length-3 sequence ``(phi_1, phi_2, phi_3)``
      in radians.
    offset: Spatial shift ``r_{0,\alpha}`` in Cartesian Bohr coordinates.

  Returns:
    Scalar JAX array giving ``sum_i V_alpha(r_i)`` in Hartree.
  """
  if coords.shape[0] == 0:
    return jnp.asarray(0.0, dtype=coords.dtype)
  dtype = coords.dtype
  v_raw = jnp.atleast_1d(jnp.asarray(amplitude, dtype=dtype))
  phi_raw = jnp.atleast_1d(jnp.asarray(phase, dtype=dtype))
  if v_raw.ndim != 1 or v_raw.shape[0] not in (1, 3):
    raise ValueError(
        'amplitude in anisotropic_single_species_moire_potential must be a '
        f'scalar or length-3 sequence, got shape {v_raw.shape}.'
    )
  if phi_raw.ndim != 1 or phi_raw.shape[0] not in (1, 3):
    raise ValueError(
        'phase in anisotropic_single_species_moire_potential must be a '
        f'scalar or length-3 sequence, got shape {phi_raw.shape}.'
    )
  v_vec = jnp.broadcast_to(v_raw, (3,))
  phi_vec = jnp.broadcast_to(phi_raw, (3,))
  if not isinstance(amplitude, jax.Array):
    if np.all(np.asarray(amplitude, dtype=np.float64) == 0.0):
      return jnp.asarray(0.0, dtype=dtype)

  gvecs = jnp.asarray(first_shell_gvectors, dtype=dtype)
  if gvecs.shape != (3, 2):
    raise ValueError(
        'first_shell_gvectors must have shape (3, 2), got '
        f'{gvecs.shape}.'
    )
  r0 = jnp.asarray(offset, dtype=dtype)
  shifted = coords - r0[None, :]  # (n_particles, 2)
  projections = jnp.einsum('id,md->im', shifted, gvecs)  # (n_particles, 3)
  harmonic_cos_sum = jnp.sum(
      jnp.cos(projections + phi_vec[None, :]), axis=0
  )  # (3,)
  return 2.0 * jnp.sum(v_vec * harmonic_cos_sum)


def _make_anisotropic_moire_potential(
    n_e: int,
    n_h: int,
    sim_lat: np.ndarray,
    lat_e: geometry.Array,
    lat_h: geometry.Array,
    v_e: ScalarOrSequence,
    v_h: ScalarOrSequence,
    phi_e: ScalarOrSequence,
    phi_h: ScalarOrSequence,
    offset_e_arr: np.ndarray,
    offset_h_arr: np.ndarray,
    moire_gvectors: Optional[geometry.Array],
    moire_gvectors_e: Optional[geometry.Array],
    moire_gvectors_h: Optional[geometry.Array],
    gvector_units: str,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds single-layer first-shell anisotropic moire potential callable."""
  v_e_vec, phi_e_vec = resolve_anisotropic_moire_potential_params(v_e, phi_e)
  v_h_vec, phi_h_vec = resolve_anisotropic_moire_potential_params(v_h, phi_h)
  has_v_e = bool(np.any(v_e_vec != 0.0))
  has_v_h = bool(np.any(v_h_vec != 0.0))
  if has_v_e and has_v_h:
    raise ValueError(
        'When anisotropic=True, moire potential may only be active on one '
        'layer (either v_e != 0 or v_h != 0, not both).'
    )

  g_gen = (
      geometry.validate_anisotropic_gvectors(
          moire_gvectors, gvector_units=gvector_units
      )
      if moire_gvectors is not None
      else None
  )
  g_e = (
      geometry.validate_anisotropic_gvectors(
          moire_gvectors_e, gvector_units=gvector_units
      )
      if moire_gvectors_e is not None
      else None
  )
  g_h = (
      geometry.validate_anisotropic_gvectors(
          moire_gvectors_h, gvector_units=gvector_units
      )
      if moire_gvectors_h is not None
      else None
  )
  if g_e is not None and g_h is not None:
    raise ValueError(
        'When anisotropic=True, specify at most one of moire_gvectors_e or '
        'moire_gvectors_h.'
    )
  if has_v_e and g_h is not None:
    raise ValueError(
        'Cannot set moire_gvectors_h when electron moire potential is active '
        '(v_e != 0).'
    )
  if has_v_h and g_e is not None:
    raise ValueError(
        'Cannot set moire_gvectors_e when hole moire potential is active '
        '(v_h != 0).'
    )
  if g_gen is not None:
    if has_v_e or (not has_v_h and g_e is not None):
      if g_e is not None and not np.allclose(g_gen, g_e):
        raise ValueError('Conflicting moire_gvectors and moire_gvectors_e.')
      g_e = g_gen
    else:
      if g_h is not None and not np.allclose(g_gen, g_h):
        raise ValueError('Conflicting moire_gvectors and moire_gvectors_h.')
      g_h = g_gen

  if has_v_e:
    if g_e is None:
      geometry.validate_commensurate(lat_e, sim_lat)
      g_e = np.asarray(
          geometry.anisotropic_first_shell_gvectors(lat_e), dtype=np.float64
      )
    geometry.validate_gvectors_commensurate(g_e, sim_lat)
  else:
    g_e = np.zeros((3, 2), dtype=np.float64)

  if has_v_h:
    if g_h is None:
      geometry.validate_commensurate(lat_h, sim_lat)
      g_h = np.asarray(
          geometry.anisotropic_first_shell_gvectors(lat_h), dtype=np.float64
      )
    geometry.validate_gvectors_commensurate(g_h, sim_lat)
  else:
    g_h = np.zeros((3, 2), dtype=np.float64)

  def moire_potential_fn(positions: jnp.ndarray) -> jnp.ndarray:
    coords = jnp.reshape(positions, (n_e + n_h, 2))
    pos_e = coords[:n_e]
    pos_h = coords[n_e:]
    pot_e = anisotropic_single_species_moire_potential(
        pos_e,
        g_e,
        amplitude=v_e_vec,
        phase=phi_e_vec,
        offset=offset_e_arr,
    )
    pot_h = anisotropic_single_species_moire_potential(
        pos_h,
        g_h,
        amplitude=v_h_vec,
        phase=phi_h_vec,
        offset=offset_h_arr,
    )
    return pot_e + pot_h

  return moire_potential_fn


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
    offset_e: Sequence[float] | geometry.Array = (0.0, 0.0),
    offset_h: Sequence[float] | geometry.Array = (0.0, 0.0),
    anisotropic: bool = False,
    moire_gvectors: Optional[geometry.Array] = None,
    moire_gvectors_e: Optional[geometry.Array] = None,
    moire_gvectors_h: Optional[geometry.Array] = None,
    gvector_units: str = 'bohr^-1',
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds a callable evaluating the electron + hole moire potential energy.

  Validates that both the electron and hole primitive moire lattices are
  commensurate with ``simulation_lattice``.

  Args:
    nspins: Tuple ``(n_electrons, n_holes)``.
    simulation_lattice: Shape ``(2, 2)`` simulation supercell matrix (columns).
    moire_lattice: Default ``(2, 2)`` primitive moire lattice (columns) shared
      by both layers if ``moire_lattice_e`` / ``moire_lattice_h`` are not given.
    moire_lattice_e: Optional ``(2, 2)`` primitive moire lattice for electrons.
    moire_lattice_h: Optional ``(2, 2)`` primitive moire lattice for holes.
    v_e: Electron moire potential amplitude in Hartree.
    v_h: Hole moire potential amplitude in Hartree.
    phi_e: Electron moire phase in radians.
    phi_h: Hole moire phase in radians.
    offset_e: Electron moire spatial offset ``(x_0, y_0)`` in Bohr.
    offset_h: Hole moire spatial offset ``(x_0, y_0)`` in Bohr.
    anisotropic: If ``True``, enables single-layer first-shell anisotropic moire
      potential specified by ``(G_1, G_2, G_3)``.
    moire_gvectors: Optional ``(3, 2)`` first-shell reciprocal triad for the
      active moire layer (requires ``anisotropic=True``).
    moire_gvectors_e: Optional ``(3, 2)`` electron first-shell reciprocal triad.
    moire_gvectors_h: Optional ``(3, 2)`` hole first-shell reciprocal triad.
    gvector_units: Units of ``moire_gvectors*`` (``'bohr^-1'``, ``'nm^-1'``, or
      ``'angstrom^-1'``).

  Returns:
    Callable ``moire_fn(positions) -> scalar`` returning
    ``sum_i V_e(r_i^e) + sum_j V_h(r_j^h)`` in Hartree.
  """
  n_e, n_h = geometry.layer_particle_counts(nspins)
  sim_lat = geometry.validate_lattice_2d(
      simulation_lattice, name='simulation_lattice'
  )

  # So by default one supercell is just one moire unit cell
  default_moire = sim_lat if moire_lattice is None else moire_lattice
  lat_e = default_moire if moire_lattice_e is None else moire_lattice_e
  lat_h = default_moire if moire_lattice_h is None else moire_lattice_h

  offset_e_arr = np.asarray(offset_e, dtype=np.float64)
  offset_h_arr = np.asarray(offset_h, dtype=np.float64)
  if offset_e_arr.shape != (2,) or offset_h_arr.shape != (2,):
    raise ValueError(
        f'offset_e and offset_h must have shape (2,), got {offset_e_arr.shape} '
        f'and {offset_h_arr.shape}.'
    )

  if not anisotropic:
    if (
        moire_gvectors is not None
        or moire_gvectors_e is not None
        or moire_gvectors_h is not None
        or gvector_units.strip().lower().replace(' ', '')
        not in ('bohr^-1', 'bohr**-1', '1/bohr', 'bohr_inv', 'au', 'a.u.')
    ):
      raise ValueError(
          'moire_gvectors and non-default gvector_units require '
          'anisotropic=True.'
      )
    v_e_arr, phi_e_arr = resolve_moire_potential_params(v_e, phi_e)
    v_h_arr, phi_h_arr = resolve_moire_potential_params(v_h, phi_h)

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

  return _make_anisotropic_moire_potential(
      n_e=n_e,
      n_h=n_h,
      sim_lat=sim_lat,
      lat_e=lat_e,
      lat_h=lat_h,
      v_e=v_e,
      v_h=v_h,
      phi_e=phi_e,
      phi_h=phi_h,
      offset_e_arr=offset_e_arr,
      offset_h_arr=offset_h_arr,
      moire_gvectors=moire_gvectors,
      moire_gvectors_e=moire_gvectors_e,
      moire_gvectors_h=moire_gvectors_h,
      gvector_units=gvector_units,
  )


def _resolve_local_energy_mass_tensors(
    anisotropic: bool,
    v_e: ScalarOrSequence,
    v_h: ScalarOrSequence,
    moire_gvectors_e: Optional[geometry.Array],
    moire_gvectors_h: Optional[geometry.Array],
    mass_tensor: Optional[geometry.Array],
    mass_tensor_e: Optional[geometry.Array],
    mass_tensor_h: Optional[geometry.Array],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """Validates and routes single-layer mass tensors for ``local_energy``."""
  if not anisotropic:
    if (
        mass_tensor is not None
        or mass_tensor_e is not None
        or mass_tensor_h is not None
    ):
      raise ValueError(
          'mass_tensor, mass_tensor_e, and mass_tensor_h require '
          'anisotropic=True.'
      )
    return None, None

  has_v_e = bool(np.any(np.asarray(v_e, dtype=np.float64) != 0.0))
  has_v_h = bool(np.any(np.asarray(v_h, dtype=np.float64) != 0.0))
  if has_v_e and has_v_h:
    raise ValueError(
        'When anisotropic=True, moire potential may only be active on one '
        'layer (either v_e != 0 or v_h != 0, not both).'
    )

  mt_gen = (
      geometry.validate_mass_tensor_2d(mass_tensor, name='mass_tensor')
      if mass_tensor is not None
      else None
  )
  mt_e = (
      geometry.validate_mass_tensor_2d(mass_tensor_e, name='mass_tensor_e')
      if mass_tensor_e is not None
      else None
  )
  mt_h = (
      geometry.validate_mass_tensor_2d(mass_tensor_h, name='mass_tensor_h')
      if mass_tensor_h is not None
      else None
  )
  if mt_e is not None and mt_h is not None:
    raise ValueError(
        'Anisotropic mass tensor is currently supported on at most one layer '
        '(either mass_tensor_e or mass_tensor_h, not both).'
    )
  if has_v_e and mt_h is not None:
    raise ValueError(
        'Cannot set mass_tensor_h when moire potential is active on the '
        'electron layer (v_e != 0).'
    )
  if has_v_h and mt_e is not None:
    raise ValueError(
        'Cannot set mass_tensor_e when moire potential is active on the '
        'hole layer (v_h != 0).'
    )
  if mt_gen is not None:
    if has_v_e or (not has_v_h and moire_gvectors_e is not None):
      if mt_e is not None and not np.allclose(mt_gen, mt_e):
        raise ValueError('Conflicting mass_tensor and mass_tensor_e.')
      mt_e = mt_gen
    elif has_v_h or moire_gvectors_h is not None:
      if mt_h is not None and not np.allclose(mt_gen, mt_h):
        raise ValueError('Conflicting mass_tensor and mass_tensor_h.')
      mt_h = mt_gen
    elif mt_e is not None:
      if not np.allclose(mt_gen, mt_e):
        raise ValueError('Conflicting mass_tensor and mass_tensor_e.')
    elif mt_h is not None:
      if not np.allclose(mt_gen, mt_h):
        raise ValueError('Conflicting mass_tensor and mass_tensor_h.')
    else:
      raise ValueError(
          'Ambiguous generic mass_tensor when both v_e == 0 and v_h == 0; '
          'specify mass_tensor_e or mass_tensor_h (or non-zero v_e / v_h).'
      )
  return mt_e, mt_h


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    ndim: int = 2,
    complex_output: bool = False,
    laplacian_method: str = 'folx',
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
    offset_e: Sequence[float] | geometry.Array = (0.0, 0.0),
    offset_h: Sequence[float] | geometry.Array = (0.0, 0.0),
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    ewald_alpha: Optional[float] = None,
    include_capacitor: bool = True,
    allow_charged_capacitor: bool = False,
    anisotropic: bool = False,
    moire_gvectors: Optional[geometry.Array] = None,
    moire_gvectors_e: Optional[geometry.Array] = None,
    moire_gvectors_h: Optional[geometry.Array] = None,
    gvector_units: str = 'bohr^-1',
    mass_tensor: Optional[geometry.Array] = None,
    mass_tensor_e: Optional[geometry.Array] = None,
    mass_tensor_h: Optional[geometry.Array] = None,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function for a 2D electron-hole moire bilayer.

  Matches the ``MakeLocalEnergy`` calling convention used by ``ferminet.train``.

  Args:
    f: Callable returning ``(sign_or_phase, logabs)`` of the wavefunction.
    charges: Unused nuclear charges array (ghost atom has charge 0).
    nspins: Tuple ``(n_electrons, n_holes)``.
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
    v_e: Electron moire potential amplitude in Hartree.
    v_h: Hole moire potential amplitude in Hartree.
    phi_e: Electron moire phase in radians.
    phi_h: Hole moire phase in radians.
    offset_e: Electron moire spatial offset in Bohr.
    offset_h: Hole moire spatial offset in Bohr.
    ewald_gmax: Reciprocal-space cutoff index for 2D Ewald summation.
    nlatvec: Real-space shell cutoff for 2D Ewald summation.
    ewald_alpha: Optional explicit Ewald splitting parameter.
    include_capacitor: Whether to include the ``G = 0`` capacitor term.
    allow_charged_capacitor: Whether to permit ``include_capacitor=True`` when
      ``n_electrons != n_holes`` and ``interlayer_distance > 0``.
    anisotropic: If ``True``, enables single-layer first-shell anisotropic moire
      potential and 2x2 SPD mass tensor on the active moire layer.
    moire_gvectors: Optional ``(3, 2)`` first-shell reciprocal triad for the
      active moire layer (requires ``anisotropic=True``).
    moire_gvectors_e: Optional ``(3, 2)`` electron first-shell reciprocal triad.
    moire_gvectors_h: Optional ``(3, 2)`` hole first-shell reciprocal triad.
    gvector_units: Units of ``moire_gvectors*`` (``'bohr^-1'``, ``'nm^-1'``, or
      ``'angstrom^-1'``).
    mass_tensor: Optional ``(2, 2)`` SPD effective mass tensor on the active
      moire layer (requires ``anisotropic=True``).
    mass_tensor_e: Optional ``(2, 2)`` SPD electron effective mass tensor.
    mass_tensor_h: Optional ``(2, 2)`` SPD hole effective mass tensor.

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

  mt_e, mt_h = _resolve_local_energy_mass_tensors(
      anisotropic=bool(anisotropic),
      v_e=v_e,
      v_h=v_h,
      moire_gvectors_e=moire_gvectors_e,
      moire_gvectors_h=moire_gvectors_h,
      mass_tensor=mass_tensor,
      mass_tensor_e=mass_tensor_e,
      mass_tensor_h=mass_tensor_h,
  )

  if lattice is None:
    if (
        anisotropic
        and moire_lattice is None
        and (
            moire_gvectors is not None
            or moire_gvectors_e is not None
            or moire_gvectors_h is not None
        )
    ):
      g_src = (
          moire_gvectors
          if moire_gvectors is not None
          else (
              moire_gvectors_e
              if moire_gvectors_e is not None
              else moire_gvectors_h
          )
      )
      lattice = geometry.primitive_lattice_from_gvectors(
          g_src, gvector_units=gvector_units
      )
    else:
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
      mass_tensor_e=mt_e,
      mass_tensor_h=mt_h,
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
      anisotropic=bool(anisotropic),
      moire_gvectors=moire_gvectors,
      moire_gvectors_e=moire_gvectors_e,
      moire_gvectors_h=moire_gvectors_h,
      gvector_units=gvector_units,
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
