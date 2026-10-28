"""Hamiltonian for 2D electron-hole moire bilayer systems."""

from typing import Callable, Optional, Sequence, Tuple

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
    counts = [int(n) * int(ndim) for n in nspins] # Number of each particle species
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

  


def single_species_moire_potential(
    coords: jnp.ndarray,
    harmonic_gvectors: geometry.Array,
    amplitude: float,
    phase: float = 0.0,
    offset: Sequence[float] | geometry.Array = (0.0, 0.0),
) -> jnp.ndarray:
  r"""Evaluates the three-harmonic moire potential for one carrier layer.

  V(r) = 2V_a sum(m from 1 to 3) cos(g_m . (r-r_0_a) + phi_a)

  where ``g_1, g_2, g_3`` are the three C3-related first-shell moire reciprocal
  lattice vectors satisfying ``g_1 + g_2 + g_3 = 0``.
  a = e or h
  r_0_a (offset) shifts the whole potential pattern in real space
  phi_a is a common phase 

  Args:
    coords: Particle coordinates in the layer, shape ``(n_particles, 2)``.
    harmonic_gvectors: Shape ``(3, 2)`` array of C3 harmonic reciprocal vectors.
    amplitude: Potential amplitude ``V_alpha`` in Hartree (signed).
    phase: Phase parameter ``phi_alpha`` in radians.
    offset: Spatial shift ``r_{0,alpha}`` in Cartesian Bohr coordinates.

  Returns:
    Scalar JAX array (i.e. just a single number) giving ``sum_i V_alpha(r_i)``.
  """
  
  if coords.shape[0] == 0 or amplitude == 0.0:
    return jnp.asarray(0.0, dtype=coords.dtype)
  dtype = coords.dtype
  gvecs = jnp.asarray(harmonic_gvectors, dtype=dtype)
  r0 = jnp.asarray(offset, dtype=dtype)
  shifted = coords - r0[None, :]
  projections = shifted @ gvecs.T  # (n_particles, 3)
  return (
      2.0
      * jnp.asarray(amplitude, dtype=dtype)
      * jnp.sum(jnp.cos(projections + jnp.asarray(phase, dtype=dtype)))
  )
  
  
def make_moire_potential(
    nspins: Sequence[int],
    simulation_lattice: geometry.Array,
    moire_lattice: Optional[geometry.Array] = None,
    moire_lattice_e: Optional[geometry.Array] = None,
    moire_lattice_h: Optional[geometry.Array] = None,
    v_e: float = 0.0,
    v_h: float = 0.0,
    phi_e: float = 0.0,
    phi_h: float = 0.0,
    offset_e: Sequence[float] = (0.0, 0.0),
    offset_h: Sequence[float] = (0.0, 0.0),
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

  Returns:
    Callable ``moire_fn(positions) -> scalar`` returning
    ``sum_i V_e(r_i^e) + sum_j V_h(r_j^h)`` in Hartree.
  """
  n_e, n_h = geometry.layer_particle_counts(nspins)  
  sim_lat = geometry.validate_lattice_2d( # The supercell lattice
      simulation_lattice, name='simulation_lattice'
  )
  
  #So by default one supercell is just one moire unit cell
  default_moire = sim_lat if moire_lattice is None else moire_lattice
  lat_e = default_moire if moire_lattice_e is None else moire_lattice_e
  lat_h = default_moire if moire_lattice_h is None else moire_lattice_h
  
  # i.e. to check that the simulation cell is an integer repetition of both top and bottom moire unit cells
  geometry.validate_commensurate(lat_e, sim_lat)
  geometry.validate_commensurate(lat_h, sim_lat)


  #Return the three 120 degrees separated moire potential lattice vectors
  gvecs_e = np.asarray(geometry.moire_harmonic_gvectors(lat_e))
  gvecs_h = np.asarray(geometry.moire_harmonic_gvectors(lat_h))
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
        pos_e, gvecs_e, amplitude=v_e, phase=phi_e, offset=offset_e_arr
    )
    pot_h = single_species_moire_potential(
        pos_h, gvecs_h, amplitude=v_h, phase=phi_h, offset=offset_h_arr
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
    v_e: float = 0.0,
    v_h: float = 0.0,
    phi_e: float = 0.0,
    phi_h: float = 0.0,
    offset_e: Sequence[float] = (0.0, 0.0),
    offset_h: Sequence[float] = (0.0, 0.0),
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    ewald_alpha: Optional[float] = None,
    include_capacitor: bool = True,
    allow_charged_capacitor: bool = False,
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
    lattice = np.eye(2) #supercell lattice
    
  species_masses = tuple(masses) if masses is not None else (mass_e, mass_h)
  
  ke_fn = local_kinetic_energy( #Function accepting params of trained NN wavefunction and data for electron/hole positions
      f=f,
      nspins=nspins,
      masses=species_masses,
      ndim=ndim,
      use_scan=use_scan,
      complex_output=complex_output,
      laplacian_method=laplacian_method,
  )
  
  moire_pot_fn = make_moire_potential( # Function accepting positions of electrons and holes
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
  
  
  
  

  
    
    
    

  
  