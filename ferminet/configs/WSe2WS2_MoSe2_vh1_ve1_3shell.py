"""2D electron-hole moire bilayer configuration for FermiNet.

Represents two parallel 2D carrier layers at fixed out-of-plane heights
``z_e = 0`` and ``z_h = d`` with no interlayer tunneling. Spinful electrons and
holes are mapped to four FermiNet fermion channels
``nspins = (N_e_up, N_e_down, N_h_up, N_h_down)`` (with 2-channel
``(N_e, N_h)`` also supported) and ``full_det = False`` so each determinant
is a product of four block determinants ``det(D_{e,up}) det(D_{e,down})
det(D_{h,up}) det(D_{h,down})`` while the equivariant network streams capture
all intra- and inter-layer correlations.
"""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from typing import Sequence, Tuple, Union
from ferminet import base_config
from ferminet.moire import geometry
from ferminet.utils import system
import ml_collections
import numpy as np


def _format_shell_param(arr: np.ndarray) -> Union[float, Tuple[float, ...]]:
  """Formats a 1D shell parameter array as a float (1 shell) or tuple."""
  if arr.size == 1:
    return float(arr[0])
  return tuple(float(x) for x in arr)


def _resolve_shell_params(
    amplitude: Union[float, Sequence[float], np.ndarray],
    phase: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes scalar or multi-shell sequence moire potential parameters."""
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


def _resolve_moire_nspins(
    mcfg: ml_collections.ConfigDict, 
) -> Tuple[int, ...]:
  """Resolves the 4-channel ``(n_e_up, n_e_down, n_h_up, n_h_down)`` tuple."""
  # mcfg = cfg.system.moire, containing the moire bilayer settings like particle counts, effective masses, lattices, etc
  
  if mcfg.get('nspins') is not None:
    nspins = tuple(
        geometry.validate_nonnegative_int(x, 'moire.nspins entry')
        for x in mcfg.nspins
    )
    geometry.layer_particle_counts(nspins) #Just a check that nspins is length 2 or 4 and have nonnegative elements
    return nspins

  # If this is true we set all electrons and holes to be spin up
  # If false and mcfg.spins not specified, assume both electrons and holes half spin up half spin down
  spin_polarized = bool(mcfg.get('spin_polarized', False))

  if mcfg.get('n_electrons') is not None:
    n_e_total = geometry.validate_nonnegative_int(
        mcfg.n_electrons, 'moire.n_electrons'
    )
    if spin_polarized:
      n_e_up, n_e_down = n_e_total, 0
    else:
      n_e_up = n_e_total - (n_e_total // 2)
      n_e_down = n_e_total // 2
  else:
    n_e_up = geometry.validate_nonnegative_int(
        mcfg.get('n_electrons_up', 2), 'moire.n_electrons_up'
    )
    n_e_down = geometry.validate_nonnegative_int(
        mcfg.get('n_electrons_down', 2), 'moire.n_electrons_down'
    )
    if spin_polarized:
      n_e_up, n_e_down = n_e_up + n_e_down, 0

  if mcfg.get('n_holes') is not None:
    n_h_total = geometry.validate_nonnegative_int(
        mcfg.n_holes, 'moire.n_holes'
    )
    if spin_polarized:
      n_h_up, n_h_down = n_h_total, 0
    else:
      n_h_up = n_h_total - (n_h_total // 2)
      n_h_down = n_h_total // 2
  else:
    n_h_up = geometry.validate_nonnegative_int(
        mcfg.get('n_holes_up', 2), 'moire.n_holes_up'
    )
    n_h_down = geometry.validate_nonnegative_int(
        mcfg.get('n_holes_down', 2), 'moire.n_holes_down'
    )
    if spin_polarized:
      n_h_up, n_h_down = n_h_up + n_h_down, 0

  return (n_e_up, n_e_down, n_h_up, n_h_down)


def _configure_moire_system(
    cfg: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Resolves moire geometry, carrier counts, and Hamiltonian/network kwargs."""
  mcfg = cfg.system.moire

  nspins = _resolve_moire_nspins(mcfg)
  n_e, n_h = geometry.layer_particle_counts(nspins) #Total n_e and n_h from the spin counts (assuming e/h on separate layers)
  if n_e < 0 or n_h < 0 or (n_e + n_h) <= 0:
    raise ValueError(
        f'Carrier counts must be non-negative with N_e + N_h > 0, got '
        f'nspins={nspins} (n_electrons={n_e}, n_holes={n_h}).'
    )
    
  with cfg.ignore_type():
    cfg.system.electrons = nspins
  cfg.system.ndim = 2
  # A single zero-charge 2D ghost atom defines the origin for periodic features.
  cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]
  
  # All the "primitive lattice" below just refer to a single moire unit cell lattice vectors.
  if mcfg.primitive_lattice is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
    )
  else: #Constructs triangular lattice given the lattice constant
    prim_lattice = geometry.triangular_lattice(float(mcfg.lattice_constant))
    
  if mcfg.supercell_lattice is not None:
    sim_lattice = geometry.validate_lattice_2d(
        mcfg.supercell_lattice, name='moire.supercell_lattice'
    )
  else:
    sim_lattice = geometry.supercell_lattice(
        prim_lattice, np.asarray(mcfg.supercell_matrix)
    )
    
  lat_e = (
      prim_lattice
      if mcfg.moire_lattice_e is None
      else geometry.validate_lattice_2d(
          mcfg.moire_lattice_e, name='moire.moire_lattice_e'
      )
  )
  lat_h = (
      prim_lattice
      if mcfg.moire_lattice_h is None
      else geometry.validate_lattice_2d(
          mcfg.moire_lattice_h, name='moire.moire_lattice_h'
      )
  )
  
  v_e_arr, phi_e_arr = _resolve_shell_params(mcfg.v_e, mcfg.phi_e)
  v_h_arr, phi_h_arr = _resolve_shell_params(mcfg.v_h, mcfg.phi_h)

  # Both moire potentials must be commensurate with the simulation supercell.
  if np.any(v_e_arr != 0.0):
    geometry.validate_commensurate(lat_e, sim_lattice)
  if np.any(v_h_arr != 0.0):
    geometry.validate_commensurate(lat_h, sim_lattice)
  
  with cfg.ignore_type(): #I think ignore_type() may not actually be necessasry?
    cfg.system.lattice = sim_lattice #Supercell lattice 
    if not cfg.system.make_local_energy_fn:
      cfg.system.make_local_energy_fn = 'ferminet.moire.hamiltonian.local_energy'
    local_energy_defaults = {
        'lattice': sim_lattice,
        'moire_lattice': prim_lattice,
        'moire_lattice_e': lat_e,
        'moire_lattice_h': lat_h,
        'masses': (float(mcfg.mass_e), float(mcfg.mass_h)),
        'mass_e': float(mcfg.mass_e),
        'mass_h': float(mcfg.mass_h),
        'interlayer_distance': float(mcfg.interlayer_distance),
        'dielectric_constant': float(mcfg.dielectric_constant),
        'v_e': _format_shell_param(v_e_arr),
        'v_h': _format_shell_param(v_h_arr),
        'phi_e': _format_shell_param(phi_e_arr),
        'phi_h': _format_shell_param(phi_h_arr),
        'offset_e': tuple(float(x) for x in mcfg.offset_e),
        'offset_h': tuple(float(x) for x in mcfg.offset_h),
        'ewald_gmax': int(mcfg.ewald_gmax),
        'nlatvec': int(mcfg.nlatvec),
        'ewald_alpha': mcfg.ewald_alpha,
        'include_capacitor': bool(mcfg.include_capacitor),
        'allow_charged_capacitor': bool(
            mcfg.get('allow_charged_capacitor', False)
        ),
    }
    local_energy_kwargs = dict(cfg.system.make_local_energy_kwargs)
    local_energy_kwargs.update(local_energy_defaults)
    cfg.system.make_local_energy_kwargs = local_energy_kwargs
    
    if not cfg.network.make_feature_layer_fn:
      cfg.network.make_feature_layer_fn = (
          'ferminet.pbc.feature_layer.make_pbc_feature_layer'
      )
    feature_layer_kwargs = dict(cfg.network.make_feature_layer_kwargs)
    feature_layer_kwargs.setdefault('include_r_ae', False) #r_ae (atom-electron) gives the network a scalar atom-electron distance i.e. a cusp, which we don't want here
    feature_layer_kwargs.update({
        'lattice': sim_lattice,
        # 'include_r_ae': False, #r_ae (atom-electron) gives the network a scalar atom-electron distance i.e. a cusp, which we don't want here
        'interlayer_distance': float(mcfg.interlayer_distance),
    })
    # feature_layer_defaults.update(dict(cfg.network.make_feature_layer_kwargs))
    cfg.network.make_feature_layer_kwargs = feature_layer_kwargs
    
    jastrow_choice = str(cfg.network.get('jastrow', 'default')).lower()
    if jastrow_choice == 'default':
      cfg.network.jastrow = 'bilayer'
    elif jastrow_choice == 'simple_ee':
      raise ValueError(
          "cfg.network.jastrow='simple_ee' is incompatible with electron-hole "
          "moire bilayers because it applies repulsive electron-electron cusp "
          "conditions across electron-hole pairs and ignores effective masses "
          "and dielectric screening. Use cfg.network.jastrow='bilayer' (or "
          "'none')."
      )

    if (
        cfg.network.get('network_type', 'ferminet') == 'psiformer'
        and not cfg.network.full_det
    ):
      raise ValueError(
          "Psiformer only supports full_det=True; set cfg.network.full_det=True "
          "or use cfg.network.network_type='ferminet'."
      )
    #May need to check this choice of envelope?
    jastrow_kwargs = (
        dict(cfg.network.jastrow_kwargs)
        if cfg.network.get('jastrow_kwargs') is not None
        else {}
    )
    jastrow_kwargs.update({
        'mass_e': float(mcfg.mass_e),
        'mass_h': float(mcfg.mass_h),
        'dielectric_constant': float(mcfg.dielectric_constant),
        'interlayer_distance': float(mcfg.interlayer_distance),
    })
    cfg.network.jastrow_kwargs = jastrow_kwargs
  
  return cfg

def finalize(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Finalizes moire bilayer configuration during base_config.resolve(cfg)."""
  return _configure_moire_system(cfg)


def get_config() -> ml_collections.ConfigDict:
  """Returns default ConfigDict for a 2D electron-hole moire bilayer."""
  cfg = base_config.default()

  # Strictly 2D carrier coordinates; disable molecular HF pretraining and
  # physical-spin observables since the two spin channels represent (e, h).
  cfg.system.ndim = 2
  cfg.pretrain.method = None
  cfg.pretrain.iterations = 0
  cfg.mcmc.init_width = 500.0
  cfg.mcmc.burn_in = 1000
  cfg.mcmc.steps = 30
  cfg.mcmc.scale_by_nuclear_distance = False
  cfg.mcmc.move_width = 2.0
  cfg.network.full_det = True 
  # cfg.network.jastrow = 'bilayer'
  cfg.network.jastrow = 'none'
  cfg.network.make_envelope_fn = ''
  cfg.network.make_envelope_kwargs = {}
  cfg.optim.laplacian = 'folx'
  cfg.optim.spin_energy = 0.0 # Controls spin-contamination penalty weight
  cfg.observables.s2 = False
  cfg.observables.dipole = False
  cfg.observables.density = False

  # Moire bilayer physical and numerical parameters (all lengths in Bohr and
  # energies in Hartree).
  cfg.system.moire = ml_collections.ConfigDict({
      # Four fermion channels: (N_e_up, N_e_down, N_h_up, N_h_down).
      # You can set n_electrons_up / n_electrons_down / n_holes_up / n_holes_down
      # directly, pass an explicit 4- or 2-tuple in 'nspins', or set total layer
      # counts 'n_electrons' / 'n_holes' (with 'spin_polarized' = False/True).
      
      ############################################################################################################
      'n_electrons_up': 6, #Particle number per simulation supercell, not per moire unit cell!
      'n_electrons_down': 6,
      'n_holes_up': 6,
      'n_holes_down': 6,
      'supercell_matrix': ((4, 0), (0, 3)),
      ############################################################################################################
      
      'nspins': None,
      'n_electrons': None,
      'n_holes': None,
      'spin_polarized': False,
      # Effective masses m_e^* / m_0 and m_h^* / m_0.
      'mass_e': 0.70,
      'mass_h': 0.45,
      # Out-of-plane separation d between electron (z=0) and hole (z=d) layers.
      'interlayer_distance': 37.79,
      # Primitive moire lattice constant a_M (Bohr) and optional explicit (2, 2)
      # column-vector lattice matrices.
      'lattice_constant': 151.18,
      'primitive_lattice': None, # So triangular lattice with lattice_constant automatically set up
      'supercell_lattice': None, # So 2x2 moire unit cells in a supercell
      'moire_lattice_e': None, # So both potentials use the moire primitive triangular lattice, not the larger supercell lattice!
      'moire_lattice_h': None,
      
      
      # Separate electron and hole moire potential parameters:
      # V_alpha(r) = 2 * sum_{s=1}^{N_shells} v_{alpha,s}
      #              * sum_{m=1}^3 cos(G_m^{(s)} . (r - offset_alpha) + phi_{alpha,s})
      #
      # 3-shell hole parameters (amplitudes in Hartree, phases in radians):
      #   - Luskin et al. (2026) [arXiv:2608.08917] Table S1 (SCAN-rVV10, rigid):
      #       'v_h': (-1.344658e-4, +1.025306e-5, -2.215984e-5),
      #       'phi_h': (0.888198, 0.185005, -1.923353),
      #   - Luskin et al. (2026) [arXiv:2608.08917] Table S1 (DFT-D3, rigid):
      #       'v_h': (-1.801451e-4, +1.113504e-5, -2.631251e-5),
      #       'phi_h': (0.941954, -0.002705, -1.717928),
      #   - Luskin et al. (2026) theta=0 deg, delta=0.04 relaxed scalar U_{m,relax}+U_Phi (3 shells, SCAN-rVV10):
      #       'v_h': (-2.808078e-4, -2.625710e-5, -1.665482e-5),
      #       'phi_h': (-0.455925, -3.070619, 1.947429),
      #   - Wang, Gao, and Tong (2026) [arXiv:2602.09526] R-type (0 deg, Fig. 6d) 3-shell fit:
      #       'v_h': (-2.690179e-4, -4.602059e-5, -2.299545e-5),
      #       'phi_h': (-0.875966, 3.141593, -1.367956),
      #   - Wang, Gao, and Tong (2026) [arXiv:2602.09526] H-type (180 deg, Fig. 7d) 3-shell fit:
      #       'v_h': (-1.751011e-4, +4.918113e-6, -9.988565e-6),
      #       'phi_h': (1.135562, 0.0, 0.251845),
      'v_e': 0.0,
      'v_h': (-2.808078e-4, -2.625710e-5, -1.665482e-5),
      'phi_e': 0.0,
      'phi_h': (-0.455925, -3.070619, 1.947429),
      'offset_e': (0.0, 0.0),
      'offset_h': (0.0, 0.0),
      
      
      # Screened Coulomb interaction and 2D Ewald convergence settings.
      'dielectric_constant': 5.7,
      'ewald_gmax': 100, #Search range for the k-space Ewald sum. Apparently with our choice of alpha this is more than enough?
      'nlatvec': 1, #Sets the range of real-space Ewald sum. Here we sum the 3x3 nearest periodic images of the supercell.
      'ewald_alpha': None, # Defaults to 5.0/h_min, which was also used in pyQMC and JaQMC
      'include_capacitor': True,
      'allow_charged_capacitor': False,
  }, type_safe=False)

  cfg = _configure_moire_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.moire_bilayer' #Records the config *.py script used
  return cfg
