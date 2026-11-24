"""Strained 0-deg P-MoSe2 conduction-band Q-valley Wigner/Kagome configuration.

Configures neural quantum state (FermiNet) VMC simulations for electron-doped
parallel-stacked 0-deg twisted bilayer MoSe2 (``|theta| ~ 1.01 deg``,
``epsilon_het ~ 0.42%``) calibrated to open-surface STM.

Default settings model the ``nu = 3`` state on a ``2 x 2`` simulation supercell
(``12`` electrons on a single active layer, ``nspins = (6, 6, 0, 0)``,
``jastrow = 'none'``) within the lowest-Gamma-miniband Kramers valley pair
``+-Q_*`` (Ferreira index 3 on the STM lattice, aligned with ``DW_1 = A_2 - A_1``):
  * Relaxed continuum potential ``V_eff(r)`` expanded in ``312`` exact 2D
    Fourier cosine modes (``|m_1|, |m_2| <= 12``, RMS error ``0.03 meV``, max
    error ``< 0.5 meV``), incorporating self-consistent ``eta``-split continuum
    elasticity and two-sheet piezoelectricity (Enaldiev PRL 2020 Eq. S11),
    signed out-of-plane corrugation ``Z(r) = d(r) - 6.70 A``, P-MoSe2 Q-valley
    tunneling/stacking harmonics (Ferreira APL 2021 Table S6), total
    (homogeneous + periodic) 22-band SOC Q-valley deformation potentials
    ``(+64.2, -1352.5) meV`` (Fang PRB 2018), and two-sheet Gauss-law
    interlayer electric field ``E_z^phys(nu)``.
  * Rotated 2x2 HSE06 Q-valley effective mass tensor ``M_Q = ((0.5092, 0.1291),
    (0.1291, 1.0508)) m_0`` (light-mass axis ``167.256 deg`` in STM coordinates;
    Kormanyos 2D Mater. 2015 Table 8).
  * Phenomenological open-surface hBN/vacuum Coulomb permittivity
    ``dielectric_constant = 3.73`` (Ge et al., Nature 2026; sweep ``~2.6-4.5``
    for screening sensitivity).
  * Relative valley potential offsets ``N_e * (bar_V_a - bar_V_0)`` preserved
    automatically via ``fourier_const_e`` when comparing ``valley_index = 0, 1, 2``
    sectors.

Example CLI overrides:
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=1``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=2``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=3``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=4``
"""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from typing import Tuple

from ferminet import base_config
from ferminet.moire import geometry
from ferminet.moire import tmose2_continuum
from ferminet.utils import system
import ml_collections
import numpy as np


def _split_spin_counts(n_total: int, spin_polarized: bool) -> Tuple[int, int]:
  """Splits ``n_total`` electrons into ``(n_up, n_down)``."""
  n_val = geometry.validate_nonnegative_int(n_total, 'n_total')
  if spin_polarized:
    return n_val, 0
  return n_val - (n_val // 2), n_val // 2


def _same_2x2_tensor(a: object, b: object) -> bool:
  """Returns whether ``a`` and ``b`` represent the same 2x2 numeric matrix."""
  if a is None or b is None or isinstance(a, str) or isinstance(b, str):
    return False
  arr_a = np.asarray(a, dtype=np.float64)
  arr_b = np.asarray(b, dtype=np.float64)
  return (
      arr_a.shape == (2, 2)
      and arr_b.shape == (2, 2)
      and np.allclose(arr_a, arr_b)
  )


def _resolve_auto_or_user_mass_tensor_e(
    mcfg: ml_collections.ConfigDict,
    default_tensor: object,
) -> object:
  """Resolves electron 2x2 mass tensor while honoring user/CLI overrides."""
  cur_val = mcfg.get('mass_tensor_e') if 'mass_tensor_e' in mcfg else None
  auto_marker = mcfg.get('_auto_mass_tensor_e', '__UNINIT__')

  if auto_marker in ('__UNINIT__', '__AUTO_NONE__'):
    if cur_val is not None:
      return geometry._as_nested_float_tuple(
          geometry.validate_mass_tensor_2d(cur_val, name='moire.mass_tensor_e')
      )
    if mcfg.get('mass_tensor') is not None:
      return geometry._as_nested_float_tuple(
          geometry.validate_mass_tensor_2d(
              mcfg.get('mass_tensor'), name='moire.mass_tensor'
          )
      )
    return geometry._as_nested_float_tuple(default_tensor)

  if auto_marker == '__USER__':
    if cur_val is None:
      return None
    return geometry._as_nested_float_tuple(
        geometry.validate_mass_tensor_2d(cur_val, name='moire.mass_tensor_e')
    )

  if cur_val is None:
    return None
  if _same_2x2_tensor(cur_val, auto_marker):
    if mcfg.get('mass_tensor') is not None:
      return geometry._as_nested_float_tuple(
          geometry.validate_mass_tensor_2d(
              mcfg.get('mass_tensor'), name='moire.mass_tensor'
          )
      )
    return geometry._as_nested_float_tuple(default_tensor)
  return geometry._as_nested_float_tuple(
      geometry.validate_mass_tensor_2d(cur_val, name='moire.mass_tensor_e')
  )


def _resolve_supercell_count(
    prim_lattice: np.ndarray,
    sim_lattice: np.ndarray,
) -> int:
  """Returns integer number of 1x1 primitive moire cells inside ``sim_lattice``."""
  ratio = geometry.validate_commensurate(prim_lattice, sim_lattice)
  n_cells = abs(round(float(np.linalg.det(ratio))))
  if n_cells <= 0:
    raise ValueError(f'Invalid supercell cell count {n_cells}.')
  return int(n_cells)


def _has_explicit_particle_counts(mcfg: ml_collections.ConfigDict) -> bool:
  """Returns True if explicit electron/spin counts were provided on ``mcfg``."""
  return (
      mcfg.get('nspins') is not None
      or mcfg.get('n_electrons') is not None
      or mcfg.get('n_electrons_up') is not None
      or mcfg.get('n_electrons_down') is not None
  )


def _resolve_tmose2_nspins(
    mcfg: ml_collections.ConfigDict,
    n_cells: int,
) -> Tuple[int, int, int, int]:
  """Resolves single-layer 4-channel ``nspins = (n_up, n_down, 0, 0)``."""
  spin_polarized = bool(mcfg.get('spin_polarized', False))

  has_explicit_counts = (
      mcfg.get('n_electrons') is not None
      or mcfg.get('n_electrons_up') is not None
      or mcfg.get('n_electrons_down') is not None
  )
  if mcfg.get('nspins') is not None and not has_explicit_counts:
    nspins = tuple(
        geometry.validate_nonnegative_int(x, 'moire.nspins entry')
        for x in mcfg.nspins
    )
    if len(nspins) != 4:
      raise ValueError(
          f'tmose2_0deg_cb_wigner expects 4-channel nspins, got {nspins}.'
      )
    if nspins[2] != 0 or nspins[3] != 0:
      raise ValueError(
          f'Single-layer electron config requires n_holes == 0, got {nspins}.'
      )
    return nspins  # type: ignore[return-value]

  if has_explicit_counts:
    if mcfg.get('n_electrons') is not None:
      n_up, n_dn = _split_spin_counts(mcfg.n_electrons, spin_polarized)
    else:
      n_up = geometry.validate_nonnegative_int(
          mcfg.get('n_electrons_up') or 0, 'moire.n_electrons_up'
      )
      n_dn = geometry.validate_nonnegative_int(
          mcfg.get('n_electrons_down') or 0, 'moire.n_electrons_down'
      )
      if spin_polarized:
        n_up, n_dn = n_up + n_dn, 0
    return (n_up, n_dn, 0, 0)

  nu = float(mcfg.get('nu', 3.0))
  if not np.isfinite(nu) or nu <= 0.0:
    raise ValueError(f'moire.nu must be finite and positive, got {nu!r}.')
  raw_total = nu * float(n_cells)
  n_total = int(round(raw_total))
  if abs(raw_total - float(n_total)) > 1e-6 or n_total <= 0:
    raise ValueError(
        f'nu * n_cells must be a positive integer; got nu={nu} on '
        f'n_cells={n_cells} (nu * n_cells = {raw_total}).'
    )
  n_up, n_dn = _split_spin_counts(n_total, spin_polarized)
  return (n_up, n_dn, 0, 0)


def _configure_tmose2_system(
    cfg: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Resolves 0-deg tMoSe2 continuum Hamiltonian and network kwargs."""
  mcfg = cfg.system.moire
  mass_model = str(mcfg.get('mass_model', 'hse')).strip().lower()
  unstrained = bool(mcfg.get('unstrained', False))
  raw_a1 = mcfg.get('a1_nm')
  raw_a2 = mcfg.get('a2_nm')
  auto_lat_vecs = bool(mcfg.get('_auto_lattice_vecs', False))

  if unstrained:
    u_a1, u_a2 = tmose2_continuum.unstrained_moire_vectors_nm(
        twist_angle_deg=float(
            mcfg.get('twist_angle_deg', tmose2_continuum.DEFAULT_TWIST_ANGLE_DEG)
        ),
        crystal_rotation_deg=float(
            mcfg.get(
                'crystal_rotation_deg',
                tmose2_continuum.DEFAULT_CRYSTAL_ROTATION_DEG,
            )
        ),
    )
    a1_nm = (
        u_a1
        if (
            raw_a1 is None
            or (
                auto_lat_vecs
                and np.allclose(raw_a1, tmose2_continuum.STM_A1_NM)
            )
        )
        else tuple(float(x) for x in raw_a1)
    )
    a2_nm = (
        u_a2
        if (
            raw_a2 is None
            or (
                auto_lat_vecs
                and np.allclose(raw_a2, tmose2_continuum.STM_A2_NM)
            )
        )
        else tuple(float(x) for x in raw_a2)
    )
  else:
    a1_nm = (
        tmose2_continuum.STM_A1_NM
        if raw_a1 is None
        else tuple(float(x) for x in raw_a1)
    )
    a2_nm = (
        tmose2_continuum.STM_A2_NM
        if raw_a2 is None
        else tuple(float(x) for x in raw_a2)
    )

  # 1. Resolve primitive and simulation lattices FIRST so particle counts and
  #    actual physical filling nu_eff = N_e / N_cells determine electrostatics:
  base_prim_bohr = tmose2_continuum.moire_primitive_lattice_bohr(
      a1_nm=a1_nm, a2_nm=a2_nm
  )
  if mcfg.get('primitive_lattice') is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
    )
  elif bool(mcfg.get('use_lll_primitive_cell', False)):
    prim_lattice = geometry.lll_reduced_basis(base_prim_bohr)
    if float(np.linalg.det(prim_lattice)) < 0.0:
      prim_lattice = prim_lattice.copy()
      prim_lattice[:, 1] = -prim_lattice[:, 1]
  else:
    prim_lattice = base_prim_bohr.copy()

  uc_mat = cfg.system.get('unit_cell_matrix')
  if uc_mat is None:
    uc_mat = mcfg.get('unit_cell_matrix')
  uc_lat = cfg.system.get('unit_cell_lattice')
  if uc_lat is None:
    uc_lat = mcfg.get('unit_cell_lattice')
  sc_in_uc = bool(
      mcfg.get('supercell_in_unit_cells')
      if mcfg.get('supercell_in_unit_cells') is not None
      else cfg.system.get('supercell_in_unit_cells', True)
  )
  _, _, sim_lattice = geometry.resolve_moire_lattices(
      lattice_constant=geometry.average_lattice_period(prim_lattice),
      primitive_lattice=prim_lattice,
      unit_cell_matrix=uc_mat,
      unit_cell_lattice=uc_lat,
      supercell_matrix=np.asarray(mcfg.supercell_matrix),
      supercell_lattice_override=mcfg.supercell_lattice,
      supercell_in_unit_cells=sc_in_uc,
  )

  n_cells = _resolve_supercell_count(prim_lattice, sim_lattice)
  nspins = _resolve_tmose2_nspins(mcfg, n_cells=n_cells)
  n_e, _ = geometry.layer_particle_counts(nspins)
  if n_e <= 0:
    raise ValueError(f'Electron count must be > 0, got nspins={nspins}.')
  effective_nu = float(n_e) / float(n_cells)

  if _has_explicit_particle_counts(mcfg):
    cur_nu = mcfg.get('nu')
    auto_nu = mcfg.get('_auto_nu', 3.0)
    if (
        cur_nu is not None
        and auto_nu is not None
        and not np.isclose(float(cur_nu), float(auto_nu))
        and not np.isclose(float(cur_nu), effective_nu)
    ):
      raise ValueError(
          f'Explicit particle count gives filling nu={effective_nu:g} '
          f'(n_e={n_e} across n_cells={n_cells}), conflicting with explicit '
          f'moire.nu={cur_nu!r}.'
      )

  if effective_nu > 4.0 + 1e-9 and not bool(
      mcfg.get('allow_high_filling_single_valley', False)
  ):
    raise ValueError(
        f'Effective filling nu={effective_nu:g} > 4 exceeds single-valley '
        'Kramers capacity in 0-deg tMoSe2 (where STM fillings nu >= 5 populate '
        'additional valleys/minibands). Set '
        'moire.allow_high_filling_single_valley=True only if intentionally '
        'forcing all electrons into a single two-spin valley pair.'
    )

  # 2. Construct continuum model using the resolved physical filling effective_nu:
  ez_use_cell_area = (
      unstrained
      if mcfg.get('ez_use_cell_area') is None
      else bool(mcfg.get('ez_use_cell_area'))
  )
  det_valley_order = (
      unstrained
      if mcfg.get('deterministic_valley_order') is None
      else bool(mcfg.get('deterministic_valley_order'))
  )
  model = tmose2_continuum.build_tmose2_continuum_model(
      nu_filling=effective_nu,
      eta_pinning=float(mcfg.get('eta_pinning', 0.0)),
      ez_phys_mv_per_angstrom=mcfg.get('ez_phys_mv_per_angstrom', None),
      ez_top_layer_fraction=float(mcfg.get('ez_top_layer_fraction', 0.5)),
      ez_nu_offset=float(mcfg.get('ez_nu_offset', 0.0)),
      fourier_cutoff=int(mcfg.get('fourier_cutoff', 12)),
      mass_model=mass_model,
      deformation_model=str(mcfg.get('deformation_model', 'soc')),
      include_heterostrain_deformation=bool(
          mcfg.get('include_heterostrain_deformation', True)
      ),
      eps_homo_uniaxial=float(mcfg.get('eps_homo_uniaxial', 0.0)),
      phi_homo_deg=float(mcfg.get('phi_homo_deg', 167.256)),
      a1_nm=a1_nm,
      a2_nm=a2_nm,
      grid_n=int(mcfg.get('grid_n', 64)),
      delta_t_mev=float(mcfg.get('delta_t_mev', 0.0)),
      include_piezo=bool(mcfg.get('include_piezo', True)),
      include_ferroelectric=bool(mcfg.get('include_ferroelectric', True)),
      ez_use_cell_area=ez_use_cell_area,
      deterministic_valley_order=det_valley_order,
  )

  valley_idx = int(mcfg.get('valley_index', 0))
  if valley_idx < 0 or valley_idx >= len(model.valleys):
    raise ValueError(
        f'moire.valley_index must be in [0, {len(model.valleys) - 1}], '
        f'got {valley_idx}.'
    )
  v0 = model.valleys[valley_idx]
  v_ref = model.valleys[0]
  rel_valley_offset_ha = float(
      v0.mean_energy_hartree - v_ref.mean_energy_hartree
  )
  fc_e = (
      rel_valley_offset_ha
      if bool(mcfg.get('include_relative_valley_offset', True))
      else 0.0
  )

  use_isotropic_mass = mass_model == 'isotropic'
  default_mt_e = (
      None
      if use_isotropic_mass
      else geometry._as_nested_float_tuple(v0.mass_tensor_m0)
  )
  mt_e = _resolve_auto_or_user_mass_tensor_e(mcfg, default_mt_e)
  eff_me = (
      geometry.effective_scalar_mass(mt_e, 'mass_tensor_e')
      if mt_e is not None
      else float(mcfg.get('mass_e', 0.78))
  )
  eff_mh = float(mcfg.get('mass_h', 0.78))

  fg_e = tuple(
      tuple(float(x) for x in row) for row in v0.fourier_gvectors_bohr_inv
  )
  fa_e = tuple(float(x) for x in v0.fourier_amps_hartree)
  fp_e = tuple(float(x) for x in v0.fourier_phases_rad)

  interlayer_d = float(mcfg.get('interlayer_distance', 0.0))
  distance_scale = geometry.average_lattice_period(prim_lattice)

  with geometry._mutable_cfg(cfg):
    cfg.system.anisotropic = True
    cfg.system.electrons = nspins
    cfg.system.ndim = 2
    cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]
    cfg.system.lattice = sim_lattice
    mcfg.resolved_a1_nm = a1_nm
    mcfg.resolved_a2_nm = a2_nm
    if (
        not unstrained
        and raw_a1 is not None
        and raw_a2 is not None
        and (
            not np.allclose(raw_a1, tmose2_continuum.STM_A1_NM)
            or not np.allclose(raw_a2, tmose2_continuum.STM_A2_NM)
        )
    ):
      mcfg._auto_lattice_vecs = False
    mcfg.lattice_constant = float(distance_scale)
    mcfg.mass_e = float(eff_me)
    mcfg.mass_h = float(eff_mh)
    mcfg.nu = float(effective_nu)
    mcfg._auto_nu = float(effective_nu)
    mcfg.effective_nu = float(effective_nu)
    mcfg.ez_resolved_mv_per_angstrom = float(model.ez_phys_mv_per_angstrom)
    mcfg.valley_mean_energy_mev = float(v0.mean_energy_mev)
    mcfg.valley_mean_energy_hartree = float(v0.mean_energy_hartree)
    mcfg.valley_relative_offset_hartree = rel_valley_offset_ha
    mcfg.total_valley_mean_offset_hartree = float(n_e) * float(
        v0.mean_energy_hartree
    )

    if mt_e is None:
      mcfg._auto_mass_tensor_e = (
          '__AUTO_NONE__' if default_mt_e is None else '__USER__'
      )
    elif (
        default_mt_e is not None
        and _same_2x2_tensor(mt_e, default_mt_e)
        and mcfg.get('mass_tensor') is None
    ):
      mcfg._auto_mass_tensor_e = mt_e
    else:
      mcfg._auto_mass_tensor_e = '__USER__'
    mcfg.mass_tensor_e = mt_e
    mcfg.mass_tensor_h = None
    mcfg.mass_tensor = None

    if not cfg.system.make_local_energy_fn:
      cfg.system.make_local_energy_fn = 'ferminet.moire.hamiltonian.local_energy'

    le_kwargs = dict(cfg.system.make_local_energy_kwargs)
    for stale_key in (
        'moire_gvectors',
        'moire_gvectors_e',
        'moire_gvectors_h',
        'mass_tensor',
        'fourier_gvectors_e',
        'fourier_gvectors_h',
        'fourier_amps_e',
        'fourier_amps_h',
        'fourier_phases_e',
        'fourier_phases_h',
        'fourier_const_e',
        'fourier_const_h',
    ):
      le_kwargs.pop(stale_key, None)

    le_kwargs.update({
        'lattice': sim_lattice,
        'moire_lattice': prim_lattice,
        'moire_lattice_e': prim_lattice,
        'moire_lattice_h': prim_lattice,
        'masses': (eff_me, eff_mh),
        'mass_e': eff_me,
        'mass_h': eff_mh,
        'interlayer_distance': interlayer_d,
        'dielectric_constant': float(mcfg.dielectric_constant),
        'v_e': 0.0,
        'v_h': 0.0,
        'phi_e': 0.0,
        'phi_h': 0.0,
        'offset_e': tuple(float(x) for x in mcfg.offset_e),
        'offset_h': tuple(float(x) for x in mcfg.offset_h),
        'ewald_gmax': int(mcfg.ewald_gmax),
        'nlatvec': int(mcfg.nlatvec),
        'ewald_alpha': mcfg.ewald_alpha,
        'include_capacitor': bool(mcfg.include_capacitor),
        'allow_charged_capacitor': bool(mcfg.allow_charged_capacitor),
        'anisotropic': True,
        'gvector_units': 'bohr^-1',
        'mass_tensor_e': mt_e,
        'mass_tensor_h': None,
        'fourier_gvectors_e': fg_e,
        'fourier_amps_e': fa_e,
        'fourier_phases_e': fp_e,
        'fourier_const_e': fc_e,
    })
    cfg.system.make_local_energy_kwargs = le_kwargs

    if not cfg.network.make_feature_layer_fn:
      cfg.network.make_feature_layer_fn = (
          'ferminet.pbc.feature_layer.make_pbc_feature_layer'
      )
    fl_kwargs = dict(cfg.network.make_feature_layer_kwargs)
    fl_kwargs.setdefault('include_r_ae', False)
    fl_kwargs.update({
        'lattice': sim_lattice,
        'primitive_lattice': prim_lattice,
        'interlayer_distance': interlayer_d,
        'distance_scale': distance_scale,
    })
    cfg.network.make_feature_layer_kwargs = fl_kwargs

    cur_pair_lat = cfg.mcmc.get('pair_lattice')
    auto_pair_lat = mcfg.get('_auto_pair_lattice', '__UNINIT__')
    if cur_pair_lat is None and auto_pair_lat == '__UNINIT__':
      cfg.mcmc.pair_lattice = prim_lattice
      mcfg._auto_pair_lattice = prim_lattice
    elif (
        cur_pair_lat is not None
        and not isinstance(auto_pair_lat, str)
        and _same_2x2_tensor(cur_pair_lat, auto_pair_lat)
    ):
      cfg.mcmc.pair_lattice = prim_lattice
      mcfg._auto_pair_lattice = prim_lattice
    else:
      mcfg._auto_pair_lattice = '__USER__'

    jastrow_choice = str(cfg.network.get('jastrow', 'none')).lower()
    if jastrow_choice == 'simple_ee':
      raise ValueError(
          "cfg.network.jastrow='simple_ee' is incompatible with 2D moire "
          'configs. Use none.'
      )

  cfg = geometry.apply_moire_unit_cell_config(cfg)
  cfg = geometry.apply_moire_energy_scale_config(cfg)
  return cfg


def finalize(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Finalizes 0-deg tMoSe2 configuration during ``base_config.resolve(cfg)``."""
  return _configure_tmose2_system(cfg)


def get_config() -> ml_collections.ConfigDict:
  """Returns default ConfigDict for 0-deg tMoSe2 at ``nu = 3`` (2x2 supercell)."""
  cfg = base_config.default()

  cfg.system.ndim = 2
  cfg.system.anisotropic = True
  cfg.pretrain.method = None
  cfg.pretrain.iterations = 0
  cfg.mcmc.init_width = 500.0
  cfg.mcmc.burn_in = 1000
  cfg.mcmc.steps = 30
  cfg.mcmc.scale_by_nuclear_distance = False
  cfg.mcmc.move_width = 20.0
  cfg.mcmc.pair_prob = 0.0
  cfg.network.full_det = True
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
          # Moire filling factor (electrons per 1x1 moire cell):
          'nu': 3.0,
          '_auto_nu': 3.0,
          'effective_nu': 3.0,
          'allow_high_filling_single_valley': False,
          'valley_index': 0,
          'include_relative_valley_offset': True,
          'supercell_matrix': ((2, 0), (0, 2)),
          'use_lll_primitive_cell': False,
          'nspins': None,
          'n_electrons': None,
          'n_electrons_up': None,
          'n_electrons_down': None,
          'spin_polarized': False,
          # Geometry & continuum model physics knobs:
          'unstrained': False,
          'twist_angle_deg': tmose2_continuum.DEFAULT_TWIST_ANGLE_DEG,
          'crystal_rotation_deg': tmose2_continuum.DEFAULT_CRYSTAL_ROTATION_DEG,
          'mass_model': 'hse',          # 'hse', 'pbe', or 'isotropic'
          'deformation_model': 'soc',   # 'soc' (+64.2, -1352.5 meV), 'spinless', or 'none'
          'delta_t_mev': 0.0,           # first-harmonic tunneling split on +-Q_1 (Ferreira idx 1)
          'include_heterostrain_deformation': True,
          'include_piezo': True,
          'include_ferroelectric': True,
          'ez_use_cell_area': None,     # None -> False (strained) / True (unstrained)
          'deterministic_valley_order': None,
          'eta_pinning': 0.0,           # 0.0 for symmetric bilayer, in [-1, 1] for asymmetric
          'ez_phys_mv_per_angstrom': None,  # None -> two-sheet Gauss law
          'ez_top_layer_fraction': 0.5,
          'ez_nu_offset': 0.0,
          'ez_resolved_mv_per_angstrom': 0.0,
          'fourier_cutoff': 12,         # |m_1|, |m_2| <= 12 -> 312 cosine modes (<0.5 meV max err)
          'eps_homo_uniaxial': 0.0,
          'phi_homo_deg': 167.256,
          'a1_nm': tmose2_continuum.STM_A1_NM,
          'a2_nm': tmose2_continuum.STM_A2_NM,
          '_auto_lattice_vecs': True,
          'resolved_a1_nm': tmose2_continuum.STM_A1_NM,
          'resolved_a2_nm': tmose2_continuum.STM_A2_NM,
          'grid_n': 64,
          'valley_mean_energy_mev': 0.0,
          'valley_mean_energy_hartree': 0.0,
          'valley_relative_offset_hartree': 0.0,
          'total_valley_mean_offset_hartree': 0.0,
          # Effective mass parameters (mass_e used when mass_model='isotropic';
          # explicit 2x2 override supported via moire.mass_tensor_e):
          'mass_e': 0.78,
          'mass_h': 0.78,
          'mass_tensor': None,
          'mass_tensor_e': None,
          'mass_tensor_h': None,
          '_auto_mass_tensor_e': '__UNINIT__',
          '_auto_pair_lattice': '__UNINIT__',
          'interlayer_distance': 0.0,
          'lattice_constant': 355.56,
          'primitive_lattice': None,
          'supercell_lattice': None,
          'moire_lattice_e': None,
          'moire_lattice_h': None,
          'v_e': 0.0,
          'v_h': 0.0,
          'phi_e': 0.0,
          'phi_h': 0.0,
          'offset_e': (0.0, 0.0),
          'offset_h': (0.0, 0.0),
          # Electrostatics (phenomenological open-surface tMoSe2 on hBN/vacuum, Ge et al. 2026):
          'dielectric_constant': tmose2_continuum.DEFAULT_DIELECTRIC_CONSTANT,
          'ewald_gmax': 100,
          'nlatvec': 1,
          'ewald_alpha': None,
          'include_capacitor': False,
          'allow_charged_capacitor': True,
      },
      type_safe=False,
  )

  cfg = _configure_tmose2_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.tmose2_0deg_cb_wigner'
  return cfg


def build_preset_config(
    *,
    nu: float,
    supercell_matrix: Tuple[Tuple[int, int], Tuple[int, int]],
    config_module: str,
    valley_index: int = 2,
    delta_t_mev: float = 9.0,
    unstrained: bool = False,
    spin_polarized: bool = False,
) -> ml_collections.ConfigDict:
  """Builds a production 0-deg tMoSe2 preset config (strained or unstrained)."""
  cfg = get_config()
  mcfg = cfg.system.moire
  with geometry._mutable_cfg(cfg):
    mcfg.nu = float(nu)
    mcfg._auto_nu = float(nu)
    mcfg.effective_nu = float(nu)
    mcfg.supercell_matrix = tuple(
        tuple(int(x) for x in row) for row in supercell_matrix
    )
    mcfg.spin_polarized = bool(spin_polarized)
    mcfg.valley_index = int(valley_index)
    mcfg.delta_t_mev = float(delta_t_mev)
    if unstrained:
      mcfg.unstrained = True
      mcfg.a1_nm = None
      mcfg.a2_nm = None
      mcfg.deformation_model = 'none'
      mcfg.include_heterostrain_deformation = False
      mcfg.phi_homo_deg = 0.0
      mcfg.ez_use_cell_area = True
      mcfg.deterministic_valley_order = True
    else:
      mcfg.unstrained = False
      mcfg.a1_nm = tmose2_continuum.STM_A1_NM
      mcfg.a2_nm = tmose2_continuum.STM_A2_NM
      mcfg.deformation_model = 'soc'
      mcfg.include_heterostrain_deformation = True
      mcfg.phi_homo_deg = 167.256
      mcfg.ez_use_cell_area = False
      mcfg.deterministic_valley_order = False

  cfg = _configure_tmose2_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = str(config_module)
  return cfg
