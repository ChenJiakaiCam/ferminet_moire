"""Anisotropic 2D electron-hole moire bilayer configuration for FermiNet.

Configures a 2D electron-hole bilayer where the moire potential is active on
one layer (holes by default) with three generally non-equilateral first-shell
reciprocal lattice vectors ``(G_1, G_2, G_3)`` satisfying
``G_1 + G_2 + G_3 = 0`` and an optional 2x2 symmetric positive-definite
effective mass tensor ``M_{ij}`` (in units of ``m_0``) on that moire-active
layer so the kinetic operator is ``-1/2 del_i (M^{-1})_{ij} del_j``.

Both the periodic input distance features (``distance_scale``) and the VMC
optimization objective (``scale_by_moire_energy = True``) use the average length
of the two primitive moire lattice vectors ``a_M = 0.5 * (||a_1|| + ||a_2||)``
and ``m_eff = 0.5 * Tr(M)`` when ``anisotropic = True``. Setting
``anisotropic = False`` (with ``moire_gvectors = None`` and
``mass_tensor* = None``) falls back identically to the standard isotropic
triangular moire bilayer.
"""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from typing import Sequence, Tuple, Union
from ferminet import base_config
from ferminet.moire import geometry
from ferminet.utils import system
import ml_collections
import numpy as np


def _format_shell_param(arr: np.ndarray) -> Union[float, Tuple[float, ...]]:
  """Formats a 1D shell/harmonic parameter array as a float or tuple."""
  if arr.size == 1:
    return float(arr[0])
  return tuple(float(x) for x in arr)


def _resolve_shell_params(
    amplitude: Union[float, Sequence[float], np.ndarray],
    phase: Union[float, Sequence[float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes scalar or 1D sequence moire potential parameters."""
  v_arr = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
  phi_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))
  if v_arr.ndim != 1 or v_arr.size < 1:
    raise ValueError(
        'Moire amplitude must be a scalar or non-empty 1D sequence, got shape '
        f'{v_arr.shape}.'
    )
  if phi_arr.ndim != 1 or phi_arr.size < 1:
    raise ValueError(
        'Moire phase must be a scalar or non-empty 1D sequence, got shape '
        f'{phi_arr.shape}.'
    )
  if not (np.all(np.isfinite(v_arr)) and np.all(np.isfinite(phi_arr))):
    raise ValueError('Moire amplitude and phase entries must be finite.')
  if phi_arr.size == 1 and v_arr.size > 1:
    phi_arr = np.broadcast_to(phi_arr, v_arr.shape).copy()
  elif v_arr.size == 1 and phi_arr.size > 1:
    v_arr = np.broadcast_to(v_arr, phi_arr.shape).copy()
  elif v_arr.size != phi_arr.size:
    raise ValueError(
        'Moire amplitude and phase must have matching lengths, got '
        f'len(amplitude)={v_arr.size} and len(phase)={phi_arr.size}.'
    )
  return v_arr, phi_arr


def _resolve_moire_nspins(
    mcfg: ml_collections.ConfigDict,
) -> Tuple[int, ...]:
  """Resolves the 4-channel ``(n_e_up, n_e_down, n_h_up, n_h_down)`` tuple."""
  if mcfg.get('nspins') is not None:
    nspins = tuple(
        geometry.validate_nonnegative_int(x, 'moire.nspins entry')
        for x in mcfg.nspins
    )
    geometry.layer_particle_counts(nspins)
    return nspins

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
  is_aniso = bool(
      cfg.system.get('anisotropic', False) or mcfg.get('anisotropic', False)
  )

  nspins = _resolve_moire_nspins(mcfg)
  n_e, n_h = geometry.layer_particle_counts(nspins)
  if n_e < 0 or n_h < 0 or (n_e + n_h) <= 0:
    raise ValueError(
        'Carrier counts must be non-negative with N_e + N_h > 0, got '
        f'nspins={nspins} (n_electrons={n_e}, n_holes={n_h}).'
    )

  with cfg.ignore_type():
    cfg.system.electrons = nspins
  cfg.system.ndim = 2
  cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]

  g_src = (
      mcfg.get('moire_gvectors')
      if mcfg.get('moire_gvectors') is not None
      else (
          mcfg.get('moire_gvectors_e')
          if mcfg.get('moire_gvectors_e') is not None
          else mcfg.get('moire_gvectors_h')
      )
  )
  if mcfg.primitive_lattice is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
    )
  elif is_aniso and g_src is not None:
    prim_lattice = geometry.primitive_lattice_from_gvectors(
        g_src, gvector_units=mcfg.get('gvector_units', 'bohr^-1')
    )
  else:
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

  if np.any(v_e_arr != 0.0):
    geometry.validate_commensurate(lat_e, sim_lattice)
  if np.any(v_h_arr != 0.0):
    geometry.validate_commensurate(lat_h, sim_lattice)

  eff_me = (
      geometry.effective_scalar_mass(mcfg.mass_e, 'moire.mass_e')
      if is_aniso
      else float(mcfg.mass_e)
  )
  eff_mh = (
      geometry.effective_scalar_mass(mcfg.mass_h, 'moire.mass_h')
      if is_aniso
      else float(mcfg.mass_h)
  )

  with cfg.ignore_type():
    cfg.system.lattice = sim_lattice
    prev_moire_lat = cfg.system.get('make_local_energy_kwargs', {}).get(
        'moire_lattice'
    )
    cur_pair_lat = cfg.mcmc.get('pair_lattice')
    stashed_auto = bool(
        (mcfg.get('_aniso_stashed') or {}).get('pair_lattice_was_auto', False)
    )
    if (
        cur_pair_lat is None
        or stashed_auto
        or (
            prev_moire_lat is not None
            and np.asarray(cur_pair_lat).shape == (2, 2)
            and np.asarray(prev_moire_lat).shape == (2, 2)
            and np.allclose(cur_pair_lat, prev_moire_lat)
        )
    ):
      cfg.mcmc.pair_lattice = prim_lattice
    if not cfg.system.make_local_energy_fn:
      cfg.system.make_local_energy_fn = (
          'ferminet.moire.hamiltonian.local_energy'
      )
    local_energy_defaults = {
        'lattice': sim_lattice,
        'moire_lattice': prim_lattice,
        'moire_lattice_e': lat_e,
        'moire_lattice_h': lat_h,
        'masses': (eff_me, eff_mh),
        'mass_e': eff_me,
        'mass_h': eff_mh,
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
    for stale_key in (
        'anisotropic',
        'moire_gvectors',
        'moire_gvectors_e',
        'moire_gvectors_h',
        'gvector_units',
        'mass_tensor',
        'mass_tensor_e',
        'mass_tensor_h',
    ):
      local_energy_kwargs.pop(stale_key, None)
    local_energy_kwargs.update(local_energy_defaults)
    cfg.system.make_local_energy_kwargs = local_energy_kwargs

    if not cfg.network.make_feature_layer_fn:
      cfg.network.make_feature_layer_fn = (
          'ferminet.pbc.feature_layer.make_pbc_feature_layer'
      )
    if is_aniso:
      distance_scale = geometry.average_lattice_period(prim_lattice)
    elif mcfg.primitive_lattice is None:
      distance_scale = float(mcfg.lattice_constant)
    else:
      distance_scale = float(np.linalg.norm(prim_lattice[:, 0]))

    feature_layer_kwargs = dict(cfg.network.make_feature_layer_kwargs)
    feature_layer_kwargs.setdefault('include_r_ae', False)
    feature_layer_kwargs.update({
        'lattice': sim_lattice,
        'interlayer_distance': float(mcfg.interlayer_distance),
        'distance_scale': distance_scale,
    })
    cfg.network.make_feature_layer_kwargs = feature_layer_kwargs

    jastrow_choice = str(cfg.network.get('jastrow', 'default')).lower()
    if jastrow_choice == 'default':
      cfg.network.jastrow = 'bilayer'
    elif jastrow_choice == 'simple_ee':
      raise ValueError(
          "cfg.network.jastrow='simple_ee' is incompatible with electron-hole "
          'moire bilayers. Use bilayer or none.'
      )

    if (
        cfg.network.get('network_type', 'ferminet') == 'psiformer'
        and not cfg.network.full_det
    ):
      raise ValueError(
          'Psiformer only supports full_det=True; set '
          "cfg.network.full_det=True or cfg.network.network_type='ferminet'."
      )
    jastrow_kwargs = (
        dict(cfg.network.jastrow_kwargs)
        if cfg.network.get('jastrow_kwargs') is not None
        else {}
    )
    jastrow_kwargs.update({
        'mass_e': eff_me,
        'mass_h': eff_mh,
        'dielectric_constant': float(mcfg.dielectric_constant),
        'interlayer_distance': float(mcfg.interlayer_distance),
    })
    cfg.network.jastrow_kwargs = jastrow_kwargs

  cfg = geometry.apply_moire_anisotropic_config(cfg)
  return cfg


def finalize(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Finalizes moire bilayer configuration during base_config.resolve(cfg)."""
  return _configure_moire_system(cfg)


def get_config() -> ml_collections.ConfigDict:
  """Returns ConfigDict for a single-layer anisotropic 2D moire bilayer."""
  cfg = base_config.default()

  cfg.system.ndim = 2
  cfg.pretrain.method = None
  cfg.pretrain.iterations = 0
  cfg.mcmc.init_width = 500.0
  cfg.mcmc.burn_in = 1000
  cfg.mcmc.steps = 30
  cfg.mcmc.scale_by_nuclear_distance = False
  cfg.mcmc.move_width = 2.0
  cfg.mcmc.pair_prob = 0.5
  cfg.mcmc.pair_radius = 1.5 * 37.79
  cfg.network.full_det = True
  # Default to jastrow='none': radial 2D cusps cannot cancel directional
  # 1/r terms from an anisotropic effective mass tensor simultaneously.
  cfg.network.jastrow = 'none'
  cfg.network.make_envelope_fn = ''
  cfg.network.make_envelope_kwargs = {}
  cfg.optim.laplacian = 'folx'
  cfg.optim.spin_energy = 0.0
  cfg.optim.scale_by_moire_energy = True
  cfg.observables.s2 = False
  cfg.observables.dipole = False
  cfg.observables.density = False

  cfg.system.moire = ml_collections.ConfigDict(
      {
          'anisotropic': True,
          'n_electrons_up': 3,
          'n_electrons_down': 3,
          'n_holes_up': 3,
          'n_holes_down': 3,
          'supercell_matrix': ((6, 0), (0, 3)),
          'nspins': None,
          'n_electrons': None,
          'n_holes': None,
          'spin_polarized': False,
          # Isotropic scalar masses (used on the layer without mass_tensor_*).
          'mass_e': 0.70,
          'mass_h': 0.45,
          # Optional 2x2 symmetric positive-definite effective mass tensor M_ij
          # (in units of m_0) on the moire-active layer so kinetic energy is
          # -1/2 del_i (M^{-1})_{ij} del_j. Off-diagonal entries are supported.
          'mass_tensor': None,
          'mass_tensor_e': None,
          'mass_tensor_h': ((0.40, 0.0), (0.0, 0.55)),
          'interlayer_distance': 37.79,
          # Fallback lattice_constant (used only when anisotropic=False and
          # primitive_lattice is None).
          'lattice_constant': 151.18,
          'primitive_lattice': None,
          'supercell_lattice': None,
          'moire_lattice_e': None,
          'moire_lattice_h': None,
          # First-shell anisotropic reciprocal lattice triad (G_1, G_2, G_3)
          # satisfying G_1 + G_2 + G_3 = 0 (3-decimal rounded inputs are closed
          # via G_3 <- -(G_1 + G_2)). Units: 'bohr^-1', 'nm^-1', 'angstrom^-1'.
          'moire_gvectors': (
              (-0.487, 0.204),
              (0.344, 0.269),
              (0.143, -0.474),
          ),
          'moire_gvectors_e': None,
          'moire_gvectors_h': None,
          'gvector_units': 'nm^-1',
          # Single-layer first-shell moire potential:
          #   V_alpha(r) = 2 * sum_{m=1}^3 v_{alpha,m}
          #                * cos(G_m . (r - offset_alpha) + phi_{alpha,m})
          # Note: V(r) = -(v_q / 4) sum_{m=1}^3 cos(G_m . r) corresponds to
          # scalar v = -v_q / 8 and phi = 0.0. Both scalar v_h / phi_h and
          # per-harmonic 3-tuples (v_1, v_2, v_3) / (phi_1, phi_2, phi_3) are
          # supported when anisotropic=True.
          'v_e': 0.0,
          'v_h': -5.512e-4,
          'phi_e': 0.0,
          'phi_h': 0.0,
          'offset_e': (0.0, 0.0),
          'offset_h': (0.0, 0.0),
          'dielectric_constant': 5.7,
          'ewald_gmax': 100,
          'nlatvec': 1,
          'ewald_alpha': None,
          'include_capacitor': True,
          'allow_charged_capacitor': False,
      },
      type_safe=False,
  )

  cfg = _configure_moire_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.anisotropic_moire_bilayer'
  return cfg
