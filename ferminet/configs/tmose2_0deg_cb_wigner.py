"""Strained 0-deg P-MoSe2 conduction-band Q-valley Wigner/Kagome configuration.

Configures neural quantum state (FermiNet / PsiFormer) VMC simulations for
electron-doped parallel-stacked 0-deg twisted bilayer MoSe2 (``theta ~ 1.01 deg``,
``epsilon_het ~ 0.44%``) calibrated to open-surface STM (``nu = 1 .. 7``).

Default settings model the emergent Kagome state at ``nu = 3`` electrons per
moire unit cell on a ``2 x 2`` simulation supercell (``12`` electrons,
``nspins = (6, 6, 0, 0)``) in the heterostrain-selected Kramers valley pair
``+-Q_*`` (aligned with ``DW_1 = A_2 - A_1``):
  * Zero-free-parameter relaxed continuum potential ``V_eff(r)`` expanded in 60
    exact 2D Fourier cosine modes (``|m_1|, |m_2| <= 5``, RMS error ``1.38 meV``),
    incorporating continuum elasticity, signed out-of-plane corrugation
    ``Z(r) = d(r) - 6.70 A``, piezoelectric charge (Enaldiev PRL 2020), P-MoSe2
    Q-valley tunneling/stacking harmonics (Ferreira APL 2021 Table S6), 22-band
    SOC Q-valley deformation potentials ``(+64.2, -1352.5) meV`` (Fang PRB 2018),
    and Gauss-law internal displacement field ``E_z^phys(nu)``.
  * Rotated 2x2 HSE06 Q-valley effective mass tensor ``M_Q = ((0.5092, 0.1291),
    (0.1291, 1.0508)) m_0`` (light-mass axis ``167.256 deg`` in STM coordinates;
    Kormanyos 2D Mater. 2015 Table 7).
  * Open-surface hBN/vacuum Coulomb screening ``dielectric_constant = 3.73``
    (Ge et al., Nature 2026), with optional metallic back-gate image screening
    via ``gate_distance`` (e.g. ``65.0`` with ``gate_distance_units = 'nm'``).
  * Symmetric intra-valley spin-exchange MCMC moves (``spin_swap_prob = 0.25``)
    to equilibrate spin singlets/triplets and Kagome magnetism across ~10 nm
    Wigner-molecule lobe separations.

Example CLI overrides:
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=1``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=2``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.nu=5 --config.system.moire.two_valley=True``
  * ``--config=ferminet/configs/tmose2_0deg_cb_wigner.py --config.system.moire.gate_distance=65.0``
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
  """Splits ``n_total`` carriers into ``(n_up, n_down)``."""
  n_val = geometry.validate_nonnegative_int(n_total, 'n_total')
  if spin_polarized:
    return n_val, 0
  return n_val - (n_val // 2), n_val // 2


def _resolve_supercell_count(
    cfg: ml_collections.ConfigDict,
    prim_lattice: np.ndarray,
    sim_lattice: np.ndarray,
) -> int:
  """Returns integer number of 1x1 primitive moire cells inside ``sim_lattice``."""
  del cfg
  ratio = geometry.validate_commensurate(prim_lattice, sim_lattice)
  n_cells = abs(round(float(np.linalg.det(ratio))))
  if n_cells <= 0:
    raise ValueError(f'Invalid supercell cell count {n_cells}.')
  return int(n_cells)


def _same_2x2_tensor(a: object, b: object) -> bool:
  """Returns whether ``a`` and ``b`` represent the same 2x2 numeric matrix."""
  if a is None or b is None or isinstance(a, str) or isinstance(b, str):
    return False
  arr_a = np.asarray(a, dtype=np.float64)
  arr_b = np.asarray(b, dtype=np.float64)
  return arr_a.shape == (2, 2) and arr_b.shape == (2, 2) and np.allclose(arr_a, arr_b)


def _resolve_auto_or_user_mass_tensor(
    mcfg: ml_collections.ConfigDict,
    layer: str,
    default_tensor: object,
    two_valley: bool = False,
) -> object:
  """Resolves per-layer 2x2 mass tensor while honoring user/CLI overrides."""
  mt_key = f'mass_tensor_{layer}'
  auto_key = f'_auto_mass_tensor_{layer}'
  cur_val = mcfg.get(mt_key) if mt_key in mcfg else None
  auto_marker = mcfg.get(auto_key, '__UNINIT__')
  allow_generic = layer == 'e' or two_valley

  if auto_marker in ('__UNINIT__', '__AUTO_NONE__'):
    if cur_val is not None:
      return geometry._as_nested_float_tuple(
          geometry.validate_mass_tensor_2d(cur_val, name=f'moire.{mt_key}')
      )
    if allow_generic and mcfg.get('mass_tensor') is not None:
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
        geometry.validate_mass_tensor_2d(cur_val, name=f'moire.{mt_key}')
    )

  # Previously auto-populated 2x2 tuple:
  if cur_val is None:
    return None
  if _same_2x2_tensor(cur_val, auto_marker):
    if allow_generic and mcfg.get('mass_tensor') is not None:
      return geometry._as_nested_float_tuple(
          geometry.validate_mass_tensor_2d(
              mcfg.get('mass_tensor'), name='moire.mass_tensor'
          )
      )
    return geometry._as_nested_float_tuple(default_tensor)
  return geometry._as_nested_float_tuple(
      geometry.validate_mass_tensor_2d(cur_val, name=f'moire.{mt_key}')
  )


def _resolve_tmose2_nspins(
    mcfg: ml_collections.ConfigDict,
    n_cells: int,
    two_valley: bool,
) -> Tuple[int, int, int, int]:
  """Resolves 4-channel ``nspins`` from ``nu``, ``n_layer*``, ``n_electrons``, or ``nspins``."""
  spin_polarized = bool(mcfg.get('spin_polarized', False))

  has_layer_aliases = any(
      mcfg.get(k) is not None
      for k in (
          'n_layer1',
          'n_layer2',
          'n_layer1_up',
          'n_layer1_down',
          'n_layer2_up',
          'n_layer2_down',
      )
  )
  has_explicit_counts = has_layer_aliases or (
      mcfg.get('n_electrons') is not None
      or mcfg.get('n_holes') is not None
      or mcfg.get('n_electrons_up') is not None
      or mcfg.get('n_electrons_down') is not None
      or mcfg.get('n_holes_up') is not None
      or mcfg.get('n_holes_down') is not None
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
    return nspins  # type: ignore[return-value]

  if has_explicit_counts:
    if mcfg.get('n_layer1') is not None:
      n1_up, n1_dn = _split_spin_counts(mcfg.n_layer1, spin_polarized)
    elif (
        mcfg.get('n_layer1_up') is not None
        or mcfg.get('n_layer1_down') is not None
    ):
      n1_up = geometry.validate_nonnegative_int(
          mcfg.get('n_layer1_up')
          if mcfg.get('n_layer1_up') is not None
          else (mcfg.get('n_electrons_up') or 0),
          'moire.n_layer1_up',
      )
      n1_dn = geometry.validate_nonnegative_int(
          mcfg.get('n_layer1_down')
          if mcfg.get('n_layer1_down') is not None
          else (mcfg.get('n_electrons_down') or 0),
          'moire.n_layer1_down',
      )
      if spin_polarized:
        n1_up, n1_dn = n1_up + n1_dn, 0
    elif mcfg.get('n_electrons') is not None:
      n1_up, n1_dn = _split_spin_counts(mcfg.n_electrons, spin_polarized)
    else:
      n1_up = geometry.validate_nonnegative_int(
          mcfg.get('n_electrons_up') or 0, 'moire.n_electrons_up'
      )
      n1_dn = geometry.validate_nonnegative_int(
          mcfg.get('n_electrons_down') or 0, 'moire.n_electrons_down'
      )
      if spin_polarized:
        n1_up, n1_dn = n1_up + n1_dn, 0

    if mcfg.get('n_layer2') is not None:
      n2_up, n2_dn = _split_spin_counts(mcfg.n_layer2, spin_polarized)
    elif (
        mcfg.get('n_layer2_up') is not None
        or mcfg.get('n_layer2_down') is not None
    ):
      n2_up = geometry.validate_nonnegative_int(
          mcfg.get('n_layer2_up')
          if mcfg.get('n_layer2_up') is not None
          else (mcfg.get('n_holes_up') or 0),
          'moire.n_layer2_up',
      )
      n2_dn = geometry.validate_nonnegative_int(
          mcfg.get('n_layer2_down')
          if mcfg.get('n_layer2_down') is not None
          else (mcfg.get('n_holes_down') or 0),
          'moire.n_layer2_down',
      )
      if spin_polarized:
        n2_up, n2_dn = n2_up + n2_dn, 0
    elif mcfg.get('n_holes') is not None:
      n2_up, n2_dn = _split_spin_counts(mcfg.n_holes, spin_polarized)
    else:
      n2_up = geometry.validate_nonnegative_int(
          mcfg.get('n_holes_up') or 0, 'moire.n_holes_up'
      )
      n2_dn = geometry.validate_nonnegative_int(
          mcfg.get('n_holes_down') or 0, 'moire.n_holes_down'
      )
      if spin_polarized:
        n2_up, n2_dn = n2_up + n2_dn, 0
    return (n1_up, n1_dn, n2_up, n2_dn)

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

  if not two_valley:
    n1_up, n1_dn = _split_spin_counts(n_total, spin_polarized)
    return (n1_up, n1_dn, 0, 0)

  # In two-valley mode (e.g. nu >= 5), fill up to 4 electrons per cell in the
  # heterostrain-selected +-Q_* valley (DW_1 dimer + MX' + XM') and place the
  # remaining carriers in the runner-up valley, unless valley1_max_per_cell is
  # customized.
  v1_cap_per_cell = float(mcfg.get('valley1_max_per_cell', 4.0))
  n_v1_cap = max(0, int(round(v1_cap_per_cell * float(n_cells))))
  n_v1 = min(n_total, n_v1_cap) if n_v1_cap > 0 else (n_total - n_total // 2)
  n_v2 = n_total - n_v1
  n1_up, n1_dn = _split_spin_counts(n_v1, spin_polarized)
  n2_up, n2_dn = _split_spin_counts(n_v2, spin_polarized)
  return (n1_up, n1_dn, n2_up, n2_dn)


def _configure_tmose2_system(
    cfg: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Resolves strained 0-deg tMoSe2 continuum Hamiltonian and network kwargs."""
  mcfg = cfg.system.moire
  two_valley = bool(mcfg.get('two_valley', False))
  mass_model = str(mcfg.get('mass_model', 'hse')).strip().lower()

  model = tmose2_continuum.build_tmose2_continuum_model(
      nu_filling=float(mcfg.get('nu', 3.0)),
      eta_pinning=float(mcfg.get('eta_pinning', 1.0)),
      ez_phys_mv_per_angstrom=mcfg.get('ez_phys_mv_per_angstrom', None),
      fourier_cutoff=int(mcfg.get('fourier_cutoff', 5)),
      mass_model=mass_model,
      deformation_model=str(mcfg.get('deformation_model', 'soc')),
      eps_homo_uniaxial=float(mcfg.get('eps_homo_uniaxial', 0.0)),
      phi_homo_deg=float(mcfg.get('phi_homo_deg', 167.256)),
      a1_nm=mcfg.get('a1_nm', tmose2_continuum.STM_A1_NM),
      a2_nm=mcfg.get('a2_nm', tmose2_continuum.STM_A2_NM),
      grid_n=int(mcfg.get('grid_n', 64)),
  )

  if mcfg.get('primitive_lattice') is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
    )
  elif bool(mcfg.get('use_lll_primitive_cell', False)):
    prim_lattice = geometry.lll_reduced_basis(model.primitive_lattice_bohr)
    if float(np.linalg.det(prim_lattice)) < 0.0:
      prim_lattice = prim_lattice.copy()
      prim_lattice[:, 1] = -prim_lattice[:, 1]
  else:
    prim_lattice = model.primitive_lattice_bohr.copy()

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

  n_cells = _resolve_supercell_count(cfg, prim_lattice, sim_lattice)
  nspins = _resolve_tmose2_nspins(mcfg, n_cells=n_cells, two_valley=two_valley)
  n_e, n_h = geometry.layer_particle_counts(nspins)
  if n_e + n_h <= 0:
    raise ValueError(f'Total carrier count must be > 0, got nspins={nspins}.')
  if n_h > 0:
    two_valley = True

  v0 = model.valleys[int(mcfg.get('valley1_index', 0))]
  v1 = model.valleys[int(mcfg.get('valley2_index', 1))]

  use_isotropic_mass = mass_model == 'isotropic'
  default_mt_e = (
      None
      if use_isotropic_mass
      else geometry._as_nested_float_tuple(v0.mass_tensor_m0)
  )
  default_mt_h = (
      None
      if (use_isotropic_mass or not two_valley)
      else geometry._as_nested_float_tuple(v1.mass_tensor_m0)
  )
  mt_e = _resolve_auto_or_user_mass_tensor(
      mcfg, 'e', default_mt_e, two_valley=two_valley
  )
  mt_h = _resolve_auto_or_user_mass_tensor(
      mcfg, 'h', default_mt_h, two_valley=two_valley
  )
  eff_me = (
      geometry.effective_scalar_mass(mt_e, 'mass_tensor_e')
      if mt_e is not None
      else float(mcfg.get('mass_e', 0.78))
  )
  eff_mh = (
      geometry.effective_scalar_mass(mt_h, 'mass_tensor_h')
      if mt_h is not None
      else float(mcfg.get('mass_h', 0.78))
  )

  fg_e = tuple(tuple(float(x) for x in row) for row in v0.fourier_gvectors_bohr_inv)
  fa_e = tuple(float(x) for x in v0.fourier_amps_hartree)
  fp_e = tuple(float(x) for x in v0.fourier_phases_rad)

  if two_valley:
    fg_h = tuple(tuple(float(x) for x in row) for row in v1.fourier_gvectors_bohr_inv)
    fa_h = tuple(float(x) for x in v1.fourier_amps_hartree)
    fp_h = tuple(float(x) for x in v1.fourier_phases_rad)
  else:
    fg_h, fa_h, fp_h = None, None, None

  interlayer_d = float(mcfg.get('interlayer_distance', 0.0))
  distance_scale = geometry.average_lattice_period(prim_lattice)

  with geometry._mutable_cfg(cfg):
    cfg.system.anisotropic = True
    cfg.system.electrons = nspins
    cfg.system.ndim = 2
    cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]
    cfg.system.lattice = sim_lattice
    mcfg.lattice_constant = float(distance_scale)

    for layer_tag, mt_val, def_val in (
        ('e', mt_e, default_mt_e),
        ('h', mt_h, default_mt_h),
    ):
      auto_key = f'_auto_mass_tensor_{layer_tag}'
      if mt_val is None:
        mcfg[auto_key] = '__AUTO_NONE__' if def_val is None else '__USER__'
      elif (
          def_val is not None
          and _same_2x2_tensor(mt_val, def_val)
          and mcfg.get('mass_tensor') is None
      ):
        mcfg[auto_key] = mt_val
      else:
        mcfg[auto_key] = '__USER__'
    mcfg.mass_tensor_e = mt_e
    mcfg.mass_tensor_h = mt_h
    mcfg.mass_tensor = None

    if two_valley and mcfg.get('charges') is None:
      mcfg.charges = 'ee'

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
        'gate_distance',
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
        'layer_charges': (-1, -1),
        'charge_e': -1,
        'charge_h': -1,
        'anisotropic': True,
        'two_layer_anisotropic': True,
        'gvector_units': 'bohr^-1',
        'mass_tensor_e': mt_e,
        'mass_tensor_h': mt_h,
        'fourier_gvectors_e': fg_e,
        'fourier_amps_e': fa_e,
        'fourier_phases_e': fp_e,
    })
    if two_valley:
      le_kwargs.update({
          'fourier_gvectors_h': fg_h,
          'fourier_amps_h': fa_h,
          'fourier_phases_h': fp_h,
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

    jastrow_choice = str(cfg.network.get('jastrow', 'default')).lower()
    if jastrow_choice == 'default':
      cfg.network.jastrow = 'bilayer'
    elif jastrow_choice == 'simple_ee':
      raise ValueError(
          "cfg.network.jastrow='simple_ee' is incompatible with 2D moire "
          'configs. Use none or bilayer.'
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
        'interlayer_distance': interlayer_d,
        'charges': (-1, -1),
        'charge_e': -1,
        'charge_h': -1,
    })
    cfg.network.jastrow_kwargs = jastrow_kwargs

  cfg = geometry.apply_moire_gate_config(cfg)
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
  cfg.mcmc.spin_swap_prob = 0.25
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
          'two_layer_anisotropic': True,
          # Moire filling factor (electrons per 1x1 moire cell, nu = 1 .. 7):
          'nu': 3.0,
          'two_valley': False,
          'valley1_index': 0,
          'valley2_index': 1,
          'valley1_max_per_cell': 4.0,
          'supercell_matrix': ((2, 0), (0, 2)),
          'use_lll_primitive_cell': False,
          'nspins': None,
          'n_electrons': None,
          'n_holes': None,
          'n_electrons_up': None,
          'n_electrons_down': None,
          'n_holes_up': None,
          'n_holes_down': None,
          'n_layer1': None,
          'n_layer2': None,
          'n_layer1_up': None,
          'n_layer1_down': None,
          'n_layer2_up': None,
          'n_layer2_down': None,
          'spin_polarized': False,
          'charges': 'ee',
          # Continuum model physics knobs:
          'mass_model': 'hse',          # 'hse', 'pbe', or 'isotropic'
          'deformation_model': 'soc',   # 'soc' (+64.2, -1352.5 meV), 'spinless', or 'none'
          'eta_pinning': 1.0,           # 1.0 for bottom-layer hBN clamping, 0.0 for symmetric
          'ez_phys_mv_per_angstrom': None,  # None -> Gauss-law 0.811 * (nu + 0.89) mV/A
          'fourier_cutoff': 5,          # |m_1|, |m_2| <= 5 -> 60 cosine modes
          'eps_homo_uniaxial': 0.0,
          'phi_homo_deg': 167.256,
          'a1_nm': tmose2_continuum.STM_A1_NM,
          'a2_nm': tmose2_continuum.STM_A2_NM,
          'grid_n': 64,
          # Fallback scalar masses (used when mass_model='isotropic'; explicit
          # 2x2 overrides can also be set via moire.mass_tensor_e / mass_tensor_h):
          'mass_e': 0.78,
          'mass_h': 0.78,
          'mass_tensor': None,
          'mass_tensor_e': None,
          'mass_tensor_h': None,
          '_auto_mass_tensor_e': '__UNINIT__',
          '_auto_mass_tensor_h': '__UNINIT__',
          # Coplanar layer-hybridized bonding CB state => d_interlayer = 0:
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
          # Electrostatics:
          'dielectric_constant': tmose2_continuum.DEFAULT_DIELECTRIC_CONSTANT,
          'gate_distance': None,        # Optional hBN gate distance (e.g. 65.0 with 'nm')
          'gate_distance_units': 'nm',
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
