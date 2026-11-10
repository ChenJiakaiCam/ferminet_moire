"""2D periodic geometry utilities for moire bilayer systems.

All lattice matrices follow FermiNet's column-vector convention: a 2D lattice
matrix ``L`` has shape ``(2, 2)`` where ``L[:, 0]`` and ``L[:, 1]`` are the two
direct lattice vectors ``a_1`` and ``a_2``. Consequently, for a row vector
``r = (x, y)`` or array of positions with trailing dimension ``2``:

  - Fractional coordinates: ``s = r @ inv(L).T``
  - Cartesian coordinates: ``r = s @ L.T``
  - Reciprocal lattice vectors (rows): ``B = 2 * pi * inv(L)`` so that
    ``B @ L = 2 * pi * I``.
"""


import math
from typing import Any, Optional, Sequence, Tuple, Union

from ferminet import mcmc
import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray] #i.e. either numpy or JAX array
CarrierSpec = Union[int, float, str]

def validate_nonnegative_int(value: object, name: str = 'count') -> int:
  if isinstance(value, (bool, np.bool_)):
    raise ValueError(f'{name} must be a non-negative integer, got {value!r}.')
  if isinstance(value, (int, np.integer)):
    int_val = int(value)
  elif isinstance(value, (float, np.floating)):
    if not np.isfinite(value) or not float(value).is_integer():
      raise ValueError(f'{name} must be an integer, got {value!r}.')
    int_val = int(value)
  else:
    raise ValueError(f'{name} must be a non-negative integer, got {value!r}.')
  if int_val < 0:
    raise ValueError(f'{name} must be >= 0, got {value!r}.')
  return int_val


def resolve_carrier_charge(value: object, name: str = 'charge') -> int:
  """Resolves a single layer's carrier charge sign in ``{-1, +1}``.

  Accepts either unit charges ``-1`` (electrons) / ``+1`` (holes) or string
  carrier labels such as ``'e'``, ``'electron'``, ``'electrons'`` (``-> -1``)
  and ``'h'``, ``'hole'``, ``'holes'`` (``-> +1``).

  Args:
    value: Carrier charge or label for one layer.
    name: Parameter name used in error messages.

  Returns:
    ``-1`` for electrons or ``+1`` for holes.

  Raises:
    ValueError: If ``value`` does not represent a unit electron or hole charge.
  """
  if isinstance(value, (bool, np.bool_)):
    raise ValueError(
        f"{name} must be -1/+1 or 'e'/'h', got boolean {value!r}."
    )
  if isinstance(value, (int, np.integer)):
    q = int(value)
    if q in (-1, 1):
      return q
  elif isinstance(value, (float, np.floating)):
    if np.isfinite(value) and float(value).is_integer():
      q = int(value)
      if q in (-1, 1):
        return q
  elif isinstance(value, str):
    key = value.strip().lower()
    if key in ('e', 'elec', 'electron', 'electrons', '-', '-1', '-1.0', '-e'):
      return -1
    if key in (
        'h',
        'hole',
        'holes',
        '+',
        '+1',
        '1',
        '+1.0',
        '1.0',
        '+e',
    ):
      return 1
  raise ValueError(
      f"{name} must be -1 (electron) or +1 (hole) (or 'e'/'h'), got {value!r}."
  )


def resolve_layer_charges(
    charges: Optional[Union[Sequence[object], str]] = None,
    charge_e: Optional[object] = None,
    charge_h: Optional[object] = None,
) -> Tuple[int, int]:
  """Resolves the per-layer carrier charges ``(q_1, q_2)`` in ``{-1, +1}^2``.

  Each layer is restricted to a single carrier type (either electrons with
  ``q = -1`` or holes with ``q = +1``), supporting:
    - electron-hole bilayers: ``(-1, +1)`` (default) or ``'eh'`` / ``('e', 'h')``
    - electron-electron bilayers: ``(-1, -1)`` or ``'ee'`` / ``('e', 'e')``
    - hole-electron bilayers: ``(+1, -1)`` or ``'he'`` / ``('h', 'e')``
    - hole-hole bilayers: ``(+1, +1)`` or ``'hh'`` / ``('h', 'h')``

  A 4-element sequence ``(q_1_up, q_1_down, q_2_up, q_2_down)`` is also
  accepted provided ``q_1_up == q_1_down`` and ``q_2_up == q_2_down``.

  Args:
    charges: Optional 2-tuple (or 4-tuple) of per-layer charges/labels, or a
      shorthand string in ``{'eh', 'ee', 'he', 'hh', 'e-h', 'e-e', 'h-e', 'h-h'}``.
    charge_e: Optional carrier charge/label for layer 1 (defaults to ``-1`` when
      ``charges`` is None).
    charge_h: Optional carrier charge/label for layer 2 (defaults to ``+1`` when
      ``charges`` is None).

  Returns:
    Tuple ``(q_1, q_2)`` with ``q_1, q_2 in {-1, +1}``.
  """
  if charges is not None:
    if isinstance(charges, str):
      key = (
          charges.strip()
          .lower()
          .replace('-', '')
          .replace('_', '')
          .replace(' ', '')
      )
      shorthand = {
          'eh': (-1, 1),
          'ee': (-1, -1),
          'he': (1, -1),
          'hh': (1, 1),
      }
      if key not in shorthand:
        raise ValueError(
            "String 'charges' must be one of 'eh', 'ee', 'he', 'hh' "
            f"(or 'e-h', 'e-e', 'h-e', 'h-h'), got {charges!r}."
        )
      q1, q2 = shorthand[key]
    elif isinstance(charges, (Sequence, np.ndarray)):
      items = list(charges)
      if len(items) == 2:
        q1 = resolve_carrier_charge(items[0], 'charges[0]')
        q2 = resolve_carrier_charge(items[1], 'charges[1]')
      elif len(items) == 4:
        q1_up = resolve_carrier_charge(items[0], 'charges[0]')
        q1_dn = resolve_carrier_charge(items[1], 'charges[1]')
        q2_up = resolve_carrier_charge(items[2], 'charges[2]')
        q2_dn = resolve_carrier_charge(items[3], 'charges[3]')
        if q1_up != q1_dn or q2_up != q2_dn:
          raise ValueError(
              'Each layer must contain only a single carrier type (either '
              f'electrons or holes); got 4-channel charges {charges!r} with '
              f'layer 1 ({q1_up}, {q1_dn}) and layer 2 ({q2_up}, {q2_dn}).'
          )
        q1, q2 = q1_up, q2_up
      else:
        raise ValueError(
            'Expected charges to have length 2 (q_layer1, q_layer2) or '
            f'length 4, got {charges!r}.'
        )
    else:
      raise ValueError(
          'Expected charges to be a 2-tuple of (-1, +1) or carrier type '
          f'strings, got {charges!r}.'
      )
    if charge_e is not None:
      q1_override = resolve_carrier_charge(charge_e, 'charge_e')
      if q1_override != q1:
        raise ValueError(
            f'Conflicting layer 1 charge specifications: charges={charges!r} '
            f'vs charge_e={charge_e!r}.'
        )
    if charge_h is not None:
      q2_override = resolve_carrier_charge(charge_h, 'charge_h')
      if q2_override != q2:
        raise ValueError(
            f'Conflicting layer 2 charge specifications: charges={charges!r} '
            f'vs charge_h={charge_h!r}.'
        )
    return q1, q2

  q1 = -1 if charge_e is None else resolve_carrier_charge(charge_e, 'charge_e')
  q2 = 1 if charge_h is None else resolve_carrier_charge(charge_h, 'charge_h')
  return q1, q2


def resolve_moire_charges_from_config(
    mcfg: Any,
    syscfg: Any = None,
) -> Tuple[int, int]:
  """Resolves per-layer carrier charges ``(charge_e, charge_h)`` from config."""
  charges_spec = None
  charge_e = None
  charge_h = None

  if syscfg is not None:
    charges_spec = syscfg.get('layer_charges', None)
    if charges_spec is None:
      charges_spec = syscfg.get('carrier_types', None)
    if charges_spec is None:
      charges_spec = syscfg.get('charges', None)
    charge_e = syscfg.get('charge_e', None)
    if charge_e is None:
      charge_e = syscfg.get('charge_1', None)
    charge_h = syscfg.get('charge_h', None)
    if charge_h is None:
      charge_h = syscfg.get('charge_2', None)

  if mcfg is not None:
    if charges_spec is None:
      charges_spec = mcfg.get('layer_charges', None)
    if charges_spec is None:
      charges_spec = mcfg.get('carrier_types', None)
    if charges_spec is None:
      charges_spec = mcfg.get('charges', None)
    if charge_e is None:
      charge_e = mcfg.get('charge_e', None)
    if charge_e is None:
      charge_e = mcfg.get('charge_1', None)
    if charge_h is None:
      charge_h = mcfg.get('charge_h', None)
    if charge_h is None:
      charge_h = mcfg.get('charge_2', None)

  # Allow charge_e / charge_h to override the default charges=(-1, 1) when
  # layer_charges / carrier_types were not explicitly set.
  is_default_charges = (
      isinstance(charges_spec, (tuple, list))
      and len(charges_spec) == 2
      and tuple(charges_spec) == (-1, 1)
  )
  has_explicit_layer_charges = (
      (syscfg is not None and (
          syscfg.get('layer_charges', None) is not None
          or syscfg.get('carrier_types', None) is not None
      ))
      or (mcfg is not None and (
          mcfg.get('layer_charges', None) is not None
          or mcfg.get('carrier_types', None) is not None
      ))
  )
  if (
      (charge_e is not None or charge_h is not None)
      and not has_explicit_layer_charges
      and is_default_charges
  ):
    charges_spec = None

  return resolve_layer_charges(
      charges_spec, charge_e=charge_e, charge_h=charge_h
  )


def apply_moire_charges_config(cfg: Any) -> Any:
  """Synchronizes per-layer carrier charges and layer count aliases on ``cfg``."""
  if cfg is None or 'system' not in cfg or cfg.system is None:
    return cfg
  syscfg = cfg.system
  mcfg = syscfg.get('moire') if 'moire' in syscfg else None
  if mcfg is None:
    return cfg

  # 1. Support n_layer1 / n_layer2 aliases on cfg.system.moire when nspins is
  # not explicitly set.
  has_layer_count_aliases = any(
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
  if has_layer_count_aliases and mcfg.get('nspins') is None:
    spin_polarized = bool(mcfg.get('spin_polarized', False))
    n1_total_spec = (
        mcfg.get('n_layer1')
        if mcfg.get('n_layer1') is not None
        else mcfg.get('n_electrons')
    )
    if n1_total_spec is not None:
      n1_total = validate_nonnegative_int(n1_total_spec, 'moire.n_electrons')
      n1_up = n1_total if spin_polarized else n1_total - (n1_total // 2)
      n1_down = 0 if spin_polarized else n1_total // 2
    else:
      n1_up = validate_nonnegative_int(
          mcfg.get('n_layer1_up')
          if mcfg.get('n_layer1_up') is not None
          else mcfg.get('n_electrons_up', 2),
          'moire.n_electrons_up',
      )
      n1_down = validate_nonnegative_int(
          mcfg.get('n_layer1_down')
          if mcfg.get('n_layer1_down') is not None
          else mcfg.get('n_electrons_down', 2),
          'moire.n_electrons_down',
      )
      if spin_polarized:
        n1_up, n1_down = n1_up + n1_down, 0

    n2_total_spec = (
        mcfg.get('n_layer2')
        if mcfg.get('n_layer2') is not None
        else mcfg.get('n_holes')
    )
    if n2_total_spec is not None:
      n2_total = validate_nonnegative_int(n2_total_spec, 'moire.n_holes')
      n2_up = n2_total if spin_polarized else n2_total - (n2_total // 2)
      n2_down = 0 if spin_polarized else n2_total // 2
    else:
      n2_up = validate_nonnegative_int(
          mcfg.get('n_layer2_up')
          if mcfg.get('n_layer2_up') is not None
          else mcfg.get('n_holes_up', 2),
          'moire.n_holes_up',
      )
      n2_down = validate_nonnegative_int(
          mcfg.get('n_layer2_down')
          if mcfg.get('n_layer2_down') is not None
          else mcfg.get('n_holes_down', 2),
          'moire.n_holes_down',
      )
      if spin_polarized:
        n2_up, n2_down = n2_up + n2_down, 0

    nspins = (n1_up, n1_down, n2_up, n2_down)
    n_e, n_h = layer_particle_counts(nspins)
    if n_e < 0 or n_h < 0 or (n_e + n_h) <= 0:
      raise ValueError(
          f'Carrier counts must be non-negative with N_e + N_h > 0, got '
          f'nspins={nspins} (n_electrons={n_e}, n_holes={n_h}).'
      )
    with cfg.ignore_type():
      cfg.system.electrons = nspins

  # 2. Synchronize per-layer carrier charges when specified on cfg.system,
  # cfg.system.moire, or cfg.system.make_local_energy_kwargs. When unset, leave
  # cfg completely unchanged for 100% backward compatibility.
  charge_keys = (
      'charges',
      'layer_charges',
      'carrier_types',
      'charge_e',
      'charge_h',
      'charge_1',
      'charge_2',
  )
  has_sys_or_mcfg_charge_spec = any(
      mcfg.get(k) is not None for k in charge_keys
  ) or any(syscfg.get(k) is not None for k in charge_keys)
  le_kwargs = (
      dict(cfg.system.make_local_energy_kwargs)
      if cfg.system.get('make_local_energy_kwargs') is not None
      else {}
  )
  if not has_sys_or_mcfg_charge_spec and 'layer_charges' not in le_kwargs:
    return cfg

  if has_sys_or_mcfg_charge_spec:
    charge_e, charge_h = resolve_moire_charges_from_config(mcfg, syscfg=syscfg)
  else:
    charge_e, charge_h = resolve_layer_charges(
        le_kwargs.get('layer_charges'),
        charge_e=le_kwargs.get('charge_e'),
        charge_h=le_kwargs.get('charge_h'),
    )
  layer_charges = (int(charge_e), int(charge_h))
  allow_charged_capacitor = bool(mcfg.get('allow_charged_capacitor', False))
  if mcfg.get('include_capacitor') is None:
    include_capacitor = bool(charge_e * charge_h < 0)
  elif charge_e * charge_h > 0 and not allow_charged_capacitor:
    include_capacitor = False
  else:
    include_capacitor = bool(mcfg.include_capacitor)

  with cfg.ignore_type():
    le_kwargs.update({
        'include_capacitor': include_capacitor,
        'allow_charged_capacitor': allow_charged_capacitor,
        'layer_charges': layer_charges,
        'charge_e': int(charge_e),
        'charge_h': int(charge_h),
    })
    cfg.system.make_local_energy_kwargs = le_kwargs
    if 'network' in cfg and cfg.network is not None:
      jastrow_kwargs = (
          dict(cfg.network.jastrow_kwargs)
          if cfg.network.get('jastrow_kwargs') is not None
          else {}
      )
      jastrow_kwargs.update({
          'charges': layer_charges,
          'charge_e': int(charge_e),
          'charge_h': int(charge_h),
      })
      cfg.network.jastrow_kwargs = jastrow_kwargs
  return cfg

def layer_particle_counts(nspins: Sequence[int]) -> Tuple[int, int]:
  """Resolves total (n_electrons, n_holes) from 4- or 2-channel ``nspins``.

  Supports either:
    - 4 fermion channels ``(n_e_up, n_e_down, n_h_up, n_h_down)`` (spinful
      electrons and spinful holes), or
    - 2 fermion channels ``(n_electrons, n_holes)`` (spin-polarized per layer).

  Args:
    nspins: Sequence of non-negative integer particle counts per channel.

  Returns:
    Tuple ``(n_electrons, n_holes)`` of total carrier counts per layer.

  Raises:
    ValueError: If ``len(nspins)`` is not 2 or 4, or if any count is negative.
  """
  counts = [validate_nonnegative_int(n, 'nspins entry') for n in nspins]
  if any(n < 0 for n in counts):
    raise ValueError(f'All entries in nspins must be >= 0, got {nspins}.')
  if len(counts) == 4:
    return counts[0] + counts[1], counts[2] + counts[3]
  if len(counts) == 2:
    return counts[0], counts[1]
  raise ValueError(
      'Expected nspins to have length 4 (n_e_up, n_e_down, n_h_up, n_h_down) '
      f'or length 2 (n_e, n_h), got {nspins}.'
  )
  
  

def triangular_lattice(lattice_constant: float = 1.0) -> np.ndarray:
  """Returns a 2x2 triangular (equilateral) primitive lattice matrix (columns are vectors) given the lattice constant.

  The primitive vectors are ``a_1 = a_M * (sqrt(3)/2, -1/2)`` and
  ``a_2 = a_M * (0, 1)``, matching the canonical 120-degree hexagonal/triangular
  moire basis.

  Args:
    lattice_constant: Moire lattice constant ``a_M`` in Bohr (or dimensionless
      units).

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are ``a_1`` and ``a_2``.
  """
  if lattice_constant <= 0.0:
    raise ValueError(
        f'lattice_constant must be positive, got {lattice_constant}.'
    )
  basis = np.array( # Apparently in here and the other functions below that just construct the lattice, which only happens once, we can just use np and not jnp
      [
          [np.sqrt(3.0) / 2.0, 0.0],
          [-0.5, 1.0],
      ],
      dtype=np.float64,
  )
  return float(lattice_constant) * basis


def validate_lattice_2d(lattice: Array, name: str = 'lattice') -> np.ndarray:
  """Validates that ``lattice`` is a non-singular 2x2 matrix."""
  lat = np.asarray(lattice, dtype=np.float64)
  if lat.shape != (2, 2):
    raise ValueError(f'{name} must have shape (2, 2), got {lat.shape}.')
  if not np.all(np.isfinite(lat)):
    raise ValueError(f'{name} must have finite entries.')
  area = abs(np.linalg.det(lat))
  if area <= 1e-12:
    raise ValueError(f'{name} must be non-singular, got det={area}.')
  return lat


def supercell_lattice(
    primitive_lattice: Array,
    supercell_matrix: Array,
) -> np.ndarray:
  """Constructs a 2D simulation supercell from a primitive lattice.

  L_prime @ S is just matrix multiplication of the two 2x2 matrices, to get the supercell lattice vectors in terms of the same coordinates used to express the primitive lattice vectors. 
  The rest are just checks that sc is a 2x2 matrix of integers.
  
  Args:
    primitive_lattice: Shape ``(2, 2)`` matrix whose columns are primitive
      lattice vectors.
    supercell_matrix: Shape ``(2, 2)`` integer matrix with non-zero determinant.

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are the supercell lattice
    vectors.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sc = np.asarray(supercell_matrix, dtype=np.float64)
  if sc.shape != (2, 2):
    raise ValueError(
        f'supercell_matrix must have shape (2, 2), got {sc.shape}.'
    )
  if not np.allclose(sc, np.round(sc), atol=1e-8):
    raise ValueError('supercell_matrix must have integer entries.')
  sc_int = np.round(sc)
  det = abs(round(float(np.linalg.det(sc_int))))
  if det <= 0:
    raise ValueError('supercell_matrix must have non-zero determinant.')
  return prim @ sc_int


def validate_commensurate(
    primitive_lattice: Array, # Here primitive_lattice is basically just the moire primitive cell. So we're checking the supercell is an integer number of moire cells on both layers. No moire potential on either layer (infinite period) is fine too by setting moire_lattice_e = None and v_e = 0.0 in config
    simulation_lattice: Array,
    atol: float = 1e-5,
) -> np.ndarray:
  """Checks that ``simulation_lattice`` is an integer supercell of ``primitive_lattice``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).
    simulation_lattice: Shape ``(2, 2)`` simulation supercell lattice (columns).
    atol: Tolerance for integer check on ``inv(L_prim) @ L_sim``.

  Returns:
    The rounded ``(2, 2)`` integer supercell matrix ``S = inv(L_prim) @ L_sim``.

  Raises:
    ValueError: If ``simulation_lattice`` is not commensurate with
      ``primitive_lattice``.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sim = validate_lattice_2d(simulation_lattice, name='simulation_lattice')
  ratio = np.linalg.inv(prim) @ sim # This gives the coefficients of the primitive cell lattice vectors used to form the simulation cell, which we then check to be integers.
  rounded = np.round(ratio)
  if not np.allclose(ratio, rounded, atol=atol):
    raise ValueError(
        'Simulation supercell is not commensurate with primitive moire '
        f'lattice: inv(L_prim) @ L_sim = {ratio.tolist()}.'
    )
  return rounded.astype(int)


def sqrt3_unit_cell_matrix(angle_deg: int = 120) -> np.ndarray:
  """Returns the 2x2 integer matrix for the sqrt(3) x sqrt(3) R30 moire unit cell.

  In a triangular moire lattice with primitive basis ``a_1 = a_M (sqrt(3)/2, -1/2)``
  and ``a_2 = a_M (0, 1)`` (120-degree angle, as returned by
  ``triangular_lattice``), the enlarged ``sqrt(3) x sqrt(3)`` Wigner-crystal
  unit cell of area ``3 * Area(L_prim)`` (used e.g. at filling ``nu = 1/3`` and
  ``2/3`` in arXiv:2406.11134) is spanned by:
    - ``angle_deg == 120``: ``A_1 = 2 a_1 + a_2 = a_M (sqrt(3), 0)`` and
      ``A_2 = -a_1 + a_2 = a_M (-sqrt(3)/2, 3/2)``, giving integer matrix
      ``[[2, -1], [1, 1]]`` (``det = 3``, ``|A_1| = |A_2| = sqrt(3) a_M``,
      120-degree angle).
    - ``angle_deg == 60``: ``A_1 = 2 a_1 + a_2 = a_M (sqrt(3), 0)`` and
      ``A_2 = a_1 + 2 a_2 = a_M (sqrt(3)/2, 3/2)``, giving integer matrix
      ``[[2, 1], [1, 2]]`` (``det = 3``, ``|A_1| = |A_2| = sqrt(3) a_M``,
      60-degree angle, spanning the identical 2D Bravais lattice).

  Args:
    angle_deg: Angle in degrees (``120`` or ``60``) between the two basis
      vectors of the enlarged unit cell.

  Returns:
    NumPy integer array of shape ``(2, 2)`` with determinant ``3``.
  """
  if angle_deg == 120:
    return np.array([[2, -1], [1, 1]], dtype=int)
  if angle_deg == 60:
    return np.array([[2, 1], [1, 2]], dtype=int)
  raise ValueError(
      f'angle_deg must be 120 or 60 for sqrt3_unit_cell_matrix, got {angle_deg}.'
  )


def parse_unit_cell_matrix(unit_cell_matrix: Union[str, Array]) -> np.ndarray:
  """Parses a 2x2 integer matrix or preset string for an enlarged moire unit cell.

  Supported string presets (case-insensitive):
    - ``'sqrt3'``, ``'sqrt3xsqrt3'``, ``'sqrt3_120'``: ``[[2, -1], [1, 1]]``
      (120-degree ``sqrt(3) x sqrt(3)`` unit cell of area 3).
    - ``'sqrt3_60'``: ``[[2, 1], [1, 2]]`` (60-degree ``sqrt(3) x sqrt(3)``
      unit cell of area 3, as written in arXiv:2406.11134 Table S4).
    - ``'NxM'`` (e.g. ``'1x1'``, ``'2x2'``, ``'5x5'``, ``'6x6'``, ``'2x1'``):
      diagonal integer matrix ``[[N, 0], [0, M]]``.

  Args:
    unit_cell_matrix: Either a shape ``(2, 2)`` integer array/sequence or a
      preset string.

  Returns:
    NumPy integer array of shape ``(2, 2)`` with non-zero determinant.
  """
  if isinstance(unit_cell_matrix, str):
    key = unit_cell_matrix.strip().lower().replace(' ', '')
    if key in ('sqrt3', 'sqrt3xsqrt3', 'sqrt3_120'):
      return sqrt3_unit_cell_matrix(120)
    if key == 'sqrt3_60':
      return sqrt3_unit_cell_matrix(60)
    if 'x' in key:
      parts = key.split('x')
      if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        n1, n2 = int(parts[0]), int(parts[1])
        if n1 > 0 and n2 > 0:
          return np.array([[n1, 0], [0, n2]], dtype=int)
    raise ValueError(
        f'Unrecognized unit_cell_matrix preset string: {unit_cell_matrix!r}. '
        "Expected 'sqrt3', 'sqrt3_120', 'sqrt3_60', 'NxM' (e.g. '2x2', '5x5'), "
        'or a (2, 2) integer matrix.'
    )

  sc = np.asarray(unit_cell_matrix, dtype=np.float64)
  if sc.shape != (2, 2):
    raise ValueError(
        f'unit_cell_matrix must have shape (2, 2), got {sc.shape}.'
    )
  if not np.allclose(sc, np.round(sc), atol=1e-8):
    raise ValueError('unit_cell_matrix must have integer entries.')
  sc_int = np.round(sc).astype(int)
  if abs(round(float(np.linalg.det(sc_int)))) <= 0:
    raise ValueError('unit_cell_matrix must have non-zero determinant.')
  return sc_int


def enlarged_unit_cell_lattice(
    primitive_lattice: Array,
    unit_cell_matrix: Optional[Union[str, Array]] = None,
    unit_cell_lattice: Optional[Array] = None,
) -> np.ndarray:
  """Resolves a (possibly enlarged) crystal unit cell commensurate with ``primitive_lattice``.

  At fractional moire fillings (e.g. ``nu = 1/3, 2/3, 1/2, 2/5, 1/6`` in
  arXiv:2406.11134), generalized Wigner crystals break the ``1 x 1`` moire
  translation symmetry down to an enlarged crystal unit cell
  ``L_unit = L_prim @ M_unit`` containing ``|det(M_unit)|`` primitive moire
  cells and an integer number of particles. Single-particle periodic network
  features use ``L_unit`` while the external moire potential in the
  Hamiltonian retains the ``1 x 1`` primitive moire lattice ``L_prim``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` matrix whose columns are the ``1 x 1``
      primitive moire lattice vectors.
    unit_cell_matrix: Optional ``(2, 2)`` integer matrix ``M_unit`` (or preset
      string such as ``'sqrt3'``, ``'2x2'``, ``'5x5'``) with non-zero
      determinant defining ``L_unit = L_prim @ M_unit``.
    unit_cell_lattice: Optional explicit ``(2, 2)`` matrix whose columns are
      the enlarged unit cell vectors; validated to be commensurate with
      ``primitive_lattice``.

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are the crystal unit cell
    vectors (equals ``primitive_lattice`` when both optional arguments are
    ``None``).
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  if unit_cell_matrix is not None and unit_cell_lattice is not None:
    raise ValueError(
        'Specify at most one of unit_cell_matrix or unit_cell_lattice, not both.'
    )
  if unit_cell_matrix is not None:
    m_unit = parse_unit_cell_matrix(unit_cell_matrix)
    return supercell_lattice(prim, m_unit)
  if unit_cell_lattice is not None:
    unit = validate_lattice_2d(unit_cell_lattice, name='unit_cell_lattice')
    validate_commensurate(prim, unit)
    return unit
  return prim


def resolve_moire_lattices(
    lattice_constant: float = 1.0,
    primitive_lattice: Optional[Array] = None,
    unit_cell_matrix: Optional[Union[str, Array]] = None,
    unit_cell_lattice: Optional[Array] = None,
    supercell_matrix: Array = ((1, 0), (0, 1)),
    supercell_lattice_override: Optional[Array] = None,
    supercell_in_unit_cells: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Resolves ``(prim_lattice, unit_lattice, sim_lattice)`` for a moire system.

  Args:
    lattice_constant: Primitive moire lattice constant ``a_M`` (used when
      ``primitive_lattice`` is ``None``).
    primitive_lattice: Optional explicit ``(2, 2)`` ``1 x 1`` primitive moire
      lattice matrix.
    unit_cell_matrix: Optional ``(2, 2)`` integer matrix (or preset string)
      defining an enlarged crystal unit cell ``L_unit = L_prim @ M_unit``.
    unit_cell_lattice: Optional explicit ``(2, 2)`` enlarged crystal unit cell
      lattice matrix commensurate with ``L_prim``.
    supercell_matrix: ``(2, 2)`` integer matrix tiling ``L_unit`` (when
      ``supercell_in_unit_cells=True``, the default matching arXiv:2406.11134)
      or tiling ``L_prim`` (when ``supercell_in_unit_cells=False``).
    supercell_lattice_override: Optional explicit ``(2, 2)`` simulation
      supercell lattice matrix overriding ``supercell_matrix``.
    supercell_in_unit_cells: If ``True`` (default), ``supercell_matrix``
      multiplies ``unit_lattice`` (``L_sim = L_unit @ S``); if ``False``,
      ``supercell_matrix`` multiplies ``prim_lattice`` (``L_sim = L_prim @ S``).

  Returns:
    Tuple ``(prim_lattice, unit_lattice, sim_lattice)`` of ``(2, 2)`` float64
    NumPy arrays, where ``sim_lattice`` is validated to be commensurate with
    both ``unit_lattice`` and ``prim_lattice`` whenever an enlarged unit cell is
    specified.
  """
  if primitive_lattice is not None:
    prim_lattice = validate_lattice_2d(
        primitive_lattice, name='moire.primitive_lattice'
    )
  else:
    prim_lattice = triangular_lattice(float(lattice_constant))

  unit_lattice = enlarged_unit_cell_lattice(
      prim_lattice,
      unit_cell_matrix=unit_cell_matrix,
      unit_cell_lattice=unit_cell_lattice,
  )

  if supercell_lattice_override is not None:
    sim_lattice = validate_lattice_2d(
        supercell_lattice_override, name='moire.supercell_lattice'
    )
  elif supercell_in_unit_cells:
    sim_lattice = supercell_lattice(unit_lattice, np.asarray(supercell_matrix))
  else:
    sim_lattice = supercell_lattice(prim_lattice, np.asarray(supercell_matrix))

  if unit_cell_matrix is not None or unit_cell_lattice is not None:
    validate_commensurate(unit_lattice, sim_lattice)
    validate_commensurate(prim_lattice, sim_lattice)

  return prim_lattice, unit_lattice, sim_lattice


def apply_moire_unit_cell_config(cfg: Any) -> Any:
  """Applies optional enlarged moire unit-cell settings to a ConfigDict.

  When ``unit_cell_matrix`` or ``unit_cell_lattice`` is set on ``cfg.system``,
  ``cfg.system.moire``, or ``cfg.network.make_feature_layer_kwargs``, resolves
  the enlarged crystal unit cell ``L_unit`` (e.g. ``sqrt(3) x sqrt(3)`` as in
  arXiv:2406.11134) while keeping the ``1 x 1`` primitive moire lattice
  ``L_prim`` for the Hamiltonian moire potentials and ``distance_scale``.
  When no enlarged unit-cell settings are specified, ``cfg`` is returned
  completely unchanged for 100% backward compatibility.

  Args:
    cfg: ``ml_collections.ConfigDict`` for a FermiNet calculation.

  Returns:
    The updated ``cfg`` (also modified in place).
  """
  if cfg is None or 'system' not in cfg:
    return cfg

  mcfg = cfg.system.get('moire') if 'moire' in cfg.system else None
  fl_kwargs = (
      dict(cfg.network.make_feature_layer_kwargs)
      if 'network' in cfg and cfg.network.get('make_feature_layer_kwargs')
      else {}
  )

  sys_uc_mat = cfg.system.get('unit_cell_matrix')
  if sys_uc_mat is None and mcfg is not None:
    sys_uc_mat = mcfg.get('unit_cell_matrix')
  sys_uc_lat = cfg.system.get('unit_cell_lattice')
  if sys_uc_lat is None and mcfg is not None:
    sys_uc_lat = mcfg.get('unit_cell_lattice')

  fl_uc_mat = fl_kwargs.get('unit_cell_matrix')
  fl_uc_lat = fl_kwargs.get('unit_cell_lattice')

  has_sys_uc = sys_uc_mat is not None or sys_uc_lat is not None
  has_fl_uc = fl_uc_mat is not None or fl_uc_lat is not None
  if not has_sys_uc and not has_fl_uc:
    return cfg

  if mcfg is not None and has_sys_uc:
    sc_in_uc = cfg.system.get('supercell_in_unit_cells', True)
    if mcfg.get('supercell_in_unit_cells') is not None:
      sc_in_uc = bool(mcfg.get('supercell_in_unit_cells'))
    prim_lattice, unit_lattice, sim_lattice = resolve_moire_lattices(
        lattice_constant=float(mcfg.lattice_constant),
        primitive_lattice=mcfg.primitive_lattice,
        unit_cell_matrix=sys_uc_mat,
        unit_cell_lattice=sys_uc_lat,
        supercell_matrix=np.asarray(mcfg.supercell_matrix),
        supercell_lattice_override=mcfg.supercell_lattice,
        supercell_in_unit_cells=bool(sc_in_uc),
    )
    lat_e = (
        prim_lattice
        if mcfg.moire_lattice_e is None
        else validate_lattice_2d(
            mcfg.moire_lattice_e, name='moire.moire_lattice_e'
        )
    )
    lat_h = (
        prim_lattice
        if mcfg.moire_lattice_h is None
        else validate_lattice_2d(
            mcfg.moire_lattice_h, name='moire.moire_lattice_h'
        )
    )
    if np.any(np.asarray(mcfg.v_e, dtype=np.float64) != 0.0):
      validate_commensurate(lat_e, sim_lattice)
    if np.any(np.asarray(mcfg.v_h, dtype=np.float64) != 0.0):
      validate_commensurate(lat_h, sim_lattice)

    with cfg.ignore_type():
      cfg.system.lattice = sim_lattice
      le_kwargs = dict(cfg.system.make_local_energy_kwargs)
      le_kwargs.update({
          'lattice': sim_lattice,
          'moire_lattice': prim_lattice,
          'moire_lattice_e': lat_e,
          'moire_lattice_h': lat_h,
      })
      cfg.system.make_local_energy_kwargs = le_kwargs
      fl_kwargs['lattice'] = sim_lattice
      fl_kwargs['primitive_lattice'] = unit_lattice
      fl_kwargs.pop('unit_cell_matrix', None)
      fl_kwargs.pop('unit_cell_lattice', None)
      cfg.network.make_feature_layer_kwargs = fl_kwargs
    return cfg

  # Either mcfg is None or enlarged unit cell was set only on feature_layer_kwargs.
  uc_mat = sys_uc_mat if has_sys_uc else fl_uc_mat
  uc_lat = sys_uc_lat if has_sys_uc else fl_uc_lat
  if mcfg is not None:
    base_prim = (
        validate_lattice_2d(mcfg.primitive_lattice, name='moire.primitive_lattice')
        if mcfg.primitive_lattice is not None
        else triangular_lattice(float(mcfg.lattice_constant))
    )
  else:
    le_kwargs = dict(cfg.system.get('make_local_energy_kwargs', {}))
    base_prim = le_kwargs.get(
        'moire_lattice',
        fl_kwargs.get('primitive_lattice', cfg.system.lattice),
    )
    if base_prim is None:
      raise ValueError(
          'Cannot resolve enlarged unit cell without primitive_lattice or '
          'cfg.system.lattice.'
      )

  unit_lattice = enlarged_unit_cell_lattice(
      base_prim,
      unit_cell_matrix=uc_mat,
      unit_cell_lattice=uc_lat,
  )
  sim_lattice = cfg.system.lattice
  if sim_lattice is not None:
    validate_commensurate(unit_lattice, sim_lattice)
  with cfg.ignore_type():
    fl_kwargs['primitive_lattice'] = unit_lattice
    fl_kwargs.pop('unit_cell_matrix', None)
    fl_kwargs.pop('unit_cell_lattice', None)
    cfg.network.make_feature_layer_kwargs = fl_kwargs
  return cfg


def _validate_positive_float(value: object, name: str) -> float:
  """Validates that ``value`` is a finite positive real scalar (not a bool)."""
  if isinstance(value, (bool, np.bool_)):
    raise ValueError(f'{name} must be a positive float, got {value!r}.')
  try:
    fval = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError) as exc:
    raise ValueError(f'{name} must be a positive float, got {value!r}.') from exc
  if not math.isfinite(fval) or fval <= 0.0:
    raise ValueError(f'{name} must be finite and positive, got {value!r}.')
  return fval


def moire_energy_scale(
    mass_e: Optional[float] = None,
    mass_h: Optional[float] = None,
    moire_period: Optional[float] = None,
    primitive_lattice: Optional[Array] = None,
    masses: Optional[Sequence[float]] = None,
) -> float:
  r"""Returns the characteristic moire kinetic energy scale in Hartree.

  Evaluates

  .. math::
      E_M = \frac{\hbar^2}{2 (m_e + m_h) a_M^2}

  in Hartree atomic units (``\hbar = m_0 = 1``, lengths in Bohr, energies in
  Hartree), where ``m_e, m_h`` are the two layer carrier effective masses in
  units of ``m_0`` and ``a_M`` is the ``1 x 1`` primitive moire lattice period
  in Bohr.

  Args:
    mass_e: Electron (or lower-layer) effective mass ``m_e^* / m_0 > 0``.
    mass_h: Hole (or upper-layer) effective mass ``m_h^* / m_0 > 0``.
    moire_period: Primitive moire lattice constant ``a_M > 0`` in Bohr (used
      when ``primitive_lattice`` is ``None``).
    primitive_lattice: Optional ``(2, 2)`` primitive moire lattice matrix whose
      first column vector norm defines ``a_M = ||primitive_lattice[:, 0]||_2``.
    masses: Optional sequence of effective masses overriding ``(mass_e, mass_h)``.
      Supports either 2-layer ``(mass_e, mass_h)`` or 4-channel
      ``(mass_e_up, mass_e_down, mass_h_up, mass_h_down)``.

  Returns:
    Positive float giving ``1 / (2 * (mass_e + mass_h) * a_M^2)`` in Hartree.
  """
  if masses is not None:
    if isinstance(masses, (str, bytes)) or not hasattr(masses, '__len__'):
      raise ValueError(f'masses must be a sequence of length 2 or 4, got {masses!r}.')
    masses_list = list(masses)
    if len(masses_list) == 2:
      m_e = _validate_positive_float(masses_list[0], 'masses[0]')
      m_h = _validate_positive_float(masses_list[1], 'masses[1]')
    elif len(masses_list) == 4:
      m_e_up = _validate_positive_float(masses_list[0], 'masses[0]')
      m_e_down = _validate_positive_float(masses_list[1], 'masses[1]')
      m_h_up = _validate_positive_float(masses_list[2], 'masses[2]')
      m_h_down = _validate_positive_float(masses_list[3], 'masses[3]')
      if not np.isclose(m_e_up, m_e_down, rtol=1e-12, atol=0.0):
        raise ValueError(
            '4-channel masses must have equal intra-layer spin masses '
            f'masses[0]==masses[1], got {masses_list}.'
        )
      if not np.isclose(m_h_up, m_h_down, rtol=1e-12, atol=0.0):
        raise ValueError(
            '4-channel masses must have equal intra-layer spin masses '
            f'masses[2]==masses[3], got {masses_list}.'
        )
      m_e = m_e_up
      m_h = m_h_up
    else:
      raise ValueError(
          f'masses must have length 2 or 4, got {masses_list}.'
      )
  else:
    if mass_e is None or mass_h is None:
      raise ValueError(
          'Both mass_e and mass_h (or masses) must be provided to compute '
          'moire_energy_scale.'
      )
    m_e = _validate_positive_float(mass_e, 'mass_e')
    m_h = _validate_positive_float(mass_h, 'mass_h')

  if primitive_lattice is not None:
    prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
    a_m = _validate_positive_float(
        float(np.linalg.norm(prim[:, 0])), 'norm(primitive_lattice[:, 0])'
    )
  elif moire_period is not None:
    a_m = _validate_positive_float(moire_period, 'moire_period')
  else:
    raise ValueError(
        'Either moire_period or primitive_lattice must be provided to compute '
        'moire_energy_scale.'
    )

  scale = 1.0 / (2.0 * (m_e + m_h) * (a_m ** 2))
  if not math.isfinite(scale) or scale <= 0.0:
    raise ValueError(
        f'Computed moire_energy_scale must be finite and positive, got {scale}.'
    )
  return float(scale)


def _is_moire_system_config(cfg: Any) -> bool:
  """Returns whether ``cfg`` describes a moire bilayer calculation."""
  if cfg is None or 'system' not in cfg or cfg.system is None:
    return False
  sys_cfg = cfg.system
  if 'moire' in sys_cfg and sys_cfg.get('moire') is not None:
    return True
  if (
      sys_cfg.get('make_local_energy_fn')
      == 'ferminet.moire.hamiltonian.local_energy'
  ):
    return True
  le_kwargs = sys_cfg.get('make_local_energy_kwargs')
  if le_kwargs and 'moire_lattice' in le_kwargs:
    return True
  return False


def resolve_moire_energy_scale(cfg: Any) -> float:
  """Resolves the optimization objective energy scale (in Hartree) from ``cfg``.

  Precedence:
    1. If ``cfg.optim.energy_scale`` is explicitly set (not ``None``), validates
       and returns ``float(cfg.optim.energy_scale)``.
    2. If ``cfg.optim.scale_by_moire_energy`` is ``False`` (the default),
       returns ``1.0``.
    3. If ``cfg.optim.scale_by_moire_energy`` is ``True`` and ``cfg`` is a moire
       bilayer configuration, computes and returns
       ``hbar^2 / (2 * (mass_e + mass_h) * a_M^2)`` using the ``1 x 1``
       primitive moire period ``a_M`` (even when an enlarged crystal unit cell
       ``L_unit`` is used for single-particle PBC features).
    4. Otherwise (non-moire systems such as molecules, atoms, or HEG), returns
       ``1.0``.

  Args:
    cfg: ``ml_collections.ConfigDict`` for a FermiNet calculation.

  Returns:
    Positive float energy scale in Hartree.
  """
  if cfg is None:
    return 1.0

  optim = cfg.get('optim') if 'optim' in cfg else None
  scale_flag = False
  if optim is not None:
    explicit_scale = optim.get('energy_scale')
    if explicit_scale is not None:
      return _validate_positive_float(explicit_scale, 'optim.energy_scale')
    scale_flag = bool(optim.get('scale_by_moire_energy', False))

  if not scale_flag or not _is_moire_system_config(cfg):
    return 1.0

  sys_cfg = cfg.system
  mcfg = sys_cfg.get('moire') if 'moire' in sys_cfg else None
  le_kwargs = (
      dict(sys_cfg.get('make_local_energy_kwargs'))
      if sys_cfg.get('make_local_energy_kwargs') is not None
      else {}
  )
  fl_kwargs = (
      dict(cfg.network.get('make_feature_layer_kwargs'))
      if ('network' in cfg and cfg.network is not None and cfg.network.get('make_feature_layer_kwargs') is not None)
      else {}
  )

  masses = None
  mass_e = None
  mass_h = None
  if mcfg is not None:
    masses = mcfg.get('masses') if 'masses' in mcfg else None
    mass_e = mcfg.get('mass_e') if 'mass_e' in mcfg else None
    mass_h = mcfg.get('mass_h') if 'mass_h' in mcfg else None
  if masses is None:
    masses = le_kwargs.get('masses')
  if mass_e is None:
    mass_e = le_kwargs.get('mass_e')
  if mass_h is None:
    mass_h = le_kwargs.get('mass_h')

  prim_lat = None
  moire_period = None
  if mcfg is not None and mcfg.get('primitive_lattice') is not None:
    prim_lat = mcfg.get('primitive_lattice')
  elif mcfg is not None and mcfg.get('lattice_constant') is not None:
    moire_period = mcfg.get('lattice_constant')
  elif le_kwargs.get('moire_lattice') is not None:
    prim_lat = le_kwargs.get('moire_lattice')
  elif fl_kwargs.get('distance_scale') is not None:
    moire_period = fl_kwargs.get('distance_scale')
  elif le_kwargs.get('lattice') is not None:
    prim_lat = le_kwargs.get('lattice')
  elif sys_cfg.get('lattice') is not None:
    prim_lat = sys_cfg.get('lattice')

  return moire_energy_scale(
      mass_e=mass_e,
      mass_h=mass_h,
      moire_period=moire_period,
      primitive_lattice=prim_lat,
      masses=masses,
  )


def apply_moire_energy_scale_config(cfg: Any) -> Any:
  """Populates ``cfg.optim.energy_scale`` for moire bilayer configurations.

  When ``cfg.optim.scale_by_moire_energy`` is ``True`` in a moire bilayer
  system (or when ``cfg.optim.energy_scale`` is explicitly provided), resolves
  and validates the characteristic energy scale
  ``hbar^2 / (2 * (mass_e + mass_h) * a_M^2)``. When
  ``cfg.optim.scale_by_moire_energy`` is ``False`` (the default) and
  ``cfg.optim.energy_scale`` is ``None``, ``cfg`` is returned completely
  unchanged for 100% backward compatibility.

  Args:
    cfg: ``ml_collections.ConfigDict`` for a FermiNet calculation.

  Returns:
    The updated ``cfg`` (also modified in place).
  """
  if cfg is None or 'optim' not in cfg or cfg.optim is None:
    return cfg
  has_explicit = cfg.optim.get('energy_scale') is not None
  scale_flag = bool(cfg.optim.get('scale_by_moire_energy', False))
  if not has_explicit and not (scale_flag and _is_moire_system_config(cfg)):
    return cfg
  scale = resolve_moire_energy_scale(cfg)
  with cfg.ignore_type():
    cfg.optim.energy_scale = float(scale)
  return cfg



def cell_area(lattice: Array) -> jnp.ndarray:
  """Returns the area ``|det(lattice)|`` of a 2D cell."""
  return jnp.abs(jnp.linalg.det(jnp.asarray(lattice)))


def reciprocal_lattice(lattice: Array) -> jnp.ndarray:
  """Returns the reciprocal lattice matrix whose columns are ``b_1, b_2``.

  Satisfies ``lattice.T @ reciprocal_lattice(lattice) = 2 * pi * I``.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix with direct vectors as columns.

  Returns:
    Array of shape ``(ndim, ndim)`` with reciprocal vectors as columns.
  """
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat).T


def reciprocal_vectors(lattice: Array) -> jnp.ndarray:
  """ The same as above just transposed, so we follow the same convention as when writing the lattice vectors.
  """
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat)


def _triangular_reciprocal_basis_rows(primitive_lattice: Array) -> np.ndarray:
  """Returns validated (b_1, b_2) reciprocal basis rows for a triangular lattice. 
  Just some checks that either the reciprocal lattice or some linear combinations of them is a triangular lattice. 
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  rec_rows = 2.0 * np.pi * np.linalg.inv(prim)

  def _is_triangular_pair(rows: np.ndarray) -> bool:
    norms = np.linalg.norm(rows, axis=1)
    if not np.isclose(norms[0], norms[1], rtol=1e-5, atol=1e-8):
      return False
    cos_theta = np.dot(rows[0], rows[1]) / (norms[0] * norms[1])
    return bool(np.isclose(abs(cos_theta), 0.5, rtol=1e-5, atol=1e-6)) #Note abs so the vectors can be 60/120 degrees apart

  if not _is_triangular_pair(rec_rows):
    # Check whether an unimodular LLL reduction of the reciprocal basis yields
    # a standard triangular pair.
    rec_rows = lll_reduced_basis(rec_rows.T).T #Some algorithm to see if a triangular lattice can be formed from linear combinations of the reciprocal lattice vectors. 
    if not _is_triangular_pair(rec_rows):
      raise ValueError(
          'primitive_lattice must define a 2D triangular lattice with six '
          f'equal-length first-shell reciprocal vectors, got {prim.tolist()}.'
      )
  return rec_rows


def lll_reduced_basis(lattice: Array, delta: float = 0.75) -> np.ndarray:
  """LLL-reduces (Lenstra–Lenstra–Lovász) a column-vector lattice basis so 1-shell MIC search is exact.
  Basically looks for first shell version of the (reciprocal) lattice vectors.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix whose columns are lattice vectors.
    delta: Lovasz reduction parameter in ``(0.25, 1.0)``.

  Returns:
    NumPy array of shape ``(ndim, ndim)`` whose columns span the same lattice.
  """
  # Work with rows internally, then transpose back to columns.
  basis = np.asarray(lattice, dtype=np.float64).T.copy()
  ndim = basis.shape[0]
  if not np.all(np.isfinite(basis)):
    return np.asarray(lattice, dtype=np.float64)
  norm_scale = np.linalg.norm(basis, axis=1).max()
  if abs(np.linalg.det(basis)) <= np.finfo(np.float64).eps * norm_scale**ndim:
    return np.asarray(lattice, dtype=np.float64)

  def gram_schmidt() -> Tuple[np.ndarray, np.ndarray]:
    _, r = np.linalg.qr(basis.T)
    b_star_norm = np.diag(r)
    mu = np.zeros((ndim, ndim), dtype=np.float64)
    for i in range(ndim):
      mu[i, :i] = r[:i, i] / b_star_norm[:i]
    return b_star_norm, mu

  k = 1
  while k < ndim:
    b_star_norm, mu = gram_schmidt()
    for j in range(k - 1, -1, -1):
      if abs(mu[k, j]) > 0.5:
        basis[k] = basis[k] - np.round(mu[k, j]) * basis[j]
        b_star_norm, mu = gram_schmidt()
    lovasz = (delta - mu[k, k - 1] ** 2) * b_star_norm[k - 1] ** 2
    if b_star_norm[k] ** 2 >= lovasz:
      k += 1
    else:
      basis[[k, k - 1]] = basis[[k - 1, k]]
      k = max(k - 1, 1)
  return basis.T



def moire_first_shell_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the six first-shell moire reciprocal vectors ``g_1, ..., g_6``.

  Supports both the 120-degree direct triangular basis (where ``b_1 . b_2 > 0``)
  and the 60-degree direct triangular basis (where ``b_1 . b_2 < 0``), ordering
  the six equal-length vectors of norm ``G_M = 4*pi / (sqrt(3)*a_M)`` in
  60-degree steps so that ``first_shell[0:5:2]`` always forms a C3-symmetric
  triad summing to zero.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).

  Returns:
    Array of shape ``(6, 2)`` containing the six first-shell G-vectors as rows.

  Raises:
    ValueError: If ``primitive_lattice`` is not a 2D triangular lattice.
  """
  rec_rows = _triangular_reciprocal_basis_rows(primitive_lattice)
  b1 = jnp.asarray(rec_rows[0])
  b2 = jnp.asarray(rec_rows[1])
  if float(np.dot(rec_rows[0], rec_rows[1])) > 0.0:
    # 120-degree direct basis -> 60-degree reciprocal basis (|b_2 - b_1| = G_M).
    return jnp.stack([b1, b2, b2 - b1, -b1, -b2, b1 - b2], axis=0)
  # 60-degree direct basis -> 120-degree reciprocal basis (|b_1 + b_2| = G_M).
  return jnp.stack([b1, b1 + b2, b2, -b1, -(b1 + b2), -b2], axis=0)



def moire_harmonic_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the three C3-related first-shell moire reciprocal vectors used for constructing effective moire potential.

  Returns three equal-length first-shell reciprocal vectors ``(g_1, g_2, g_3)``
  separated by 120 degrees satisfying ``g_1 + g_2 + g_3 = 0``, defining the
  three-harmonic continuum moire potential
  ``V(r) = 2 V_0 sum_{m=1}^3 cos(g_m . (r - r_0) + phi)``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).

  Returns:
    Array of shape ``(3, 2)`` containing the three C3 harmonic G-vectors.
  """
  first_shell = moire_first_shell_gvectors(primitive_lattice)
  return first_shell[0:5:2]


def moire_harmonic_shells_from_first_shell(
    first_shell_triad: Array,
    num_shells: int = 1,
) -> jnp.ndarray:
  """Constructs ``num_shells`` C3-symmetric reciprocal vector triads from the 1st shell.

  Given the first C3-symmetric triad ``(g_1, g_2, g_3)``, we generate new lattice vectors indexed by (p, q) which corresponds to p*q_1 - q*q_3 and its C3 rotations by doing cyclic permutations of q_1 and q_3.
  By cosine rule the length of (p,q) is sqrt(p^2 + p*q + q^2) * |g_1|, so we can sort the (p,q) pairs by increasing p^2 + p*q + q^2 to get the first num_shells shells.
  

  Args:
    first_shell_triad: Shape ``(3, 2)`` array containing ``(g_1, g_2, g_3)``.
    num_shells: Number of C3-symmetric harmonic triads to return (``>= 1``).

  Returns:
    JAX array of shape ``(num_shells, 3, 2)`` containing the C3-symmetric
    reciprocal vector triads ordered by increasing norm.
  """
  n_shells = validate_nonnegative_int(num_shells, 'num_shells')
  if n_shells < 1:
    raise ValueError(f'num_shells must be >= 1, got {num_shells}.')
  g_triad = jnp.asarray(first_shell_triad)
  if g_triad.shape != (3, 2):
    raise ValueError(
        f'first_shell_triad must have shape (3, 2), got {g_triad.shape}.'
    )
  g1, g2, g3 = g_triad[0], g_triad[1], g_triad[2]

  # i.e. Radius of nth shell is at least n*g_1
  max_idx = int(np.ceil(np.sqrt(float(n_shells)))) + 2
  candidates = []
  # This range of p,q basically selects the first pi/3 angle wedge of the triangular lattice, and then we can get the other two wedges by cyclic permutations of g_1, g_2, g_3.
  # This makes sure we only select three vectors per shell.
  for p in range(1, max_idx + 1):
    for q in range(0, max_idx + 1):
      q_metric = p * p + p * q + q * q
      candidates.append((q_metric, q, p))
  candidates.sort()
  selected = candidates[:n_shells]

  shells = []
  for _, q, p in selected:
    pf = float(p)
    qf = float(q)
    shells.append(
        jnp.stack(
            [pf * g1 - qf * g3, pf * g2 - qf * g1, pf * g3 - qf * g2],
            axis=0,
        )
    )
  return jnp.stack(shells, axis=0)


def moire_harmonic_shells(
    primitive_lattice: Array,
    num_shells: int = 1,
) -> jnp.ndarray:
  """Returns the first ``num_shells`` C3-symmetric moire reciprocal triads.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).
    num_shells: Number of C3-symmetric harmonic shells/triads (``>= 1``).

  Returns:
    JAX array of shape ``(num_shells, 3, 2)`` where ``shells[0]`` equals
    ``moire_harmonic_gvectors(primitive_lattice)``.
  """
  first_shell_triad = moire_harmonic_gvectors(primitive_lattice)
  return moire_harmonic_shells_from_first_shell(
      first_shell_triad, num_shells=num_shells
  )


def cartesian_to_fractional(
    positions: Array, #Flattened 1D array, like [x1, y1, x2, y2, ......, xn, yn] for 2D
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts Cartesian coordinates to fractional coordinates in the cell.
  Note fractional here just means coordinates measured along the lattice vectors. They can be greater than 1 or negative.

  Supports both reshaped coordinates ``(..., ndim)`` and flattened walker
  arrays ``(..., nelec * ndim)``.

  Args:
    positions: Cartesian coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Fractional coordinates with the same shape as ``positions``.
  """
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  inv_lat = jnp.linalg.inv(lat)
  orig_shape = pos.shape
  coords = jnp.reshape(pos, (-1, ndim)) #Get (N_particles, 2) matrix where each row is length 2 vector position of a particle
  fractional = coords @ inv_lat.T # Transform into coefficients of the triangular lattice vector
  return jnp.reshape(fractional, orig_shape) #Another (N_particles, 2) matrix


def fractional_to_cartesian(
    fractional: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts fractional cell coordinates to Cartesian coordinates.

  Supports both reshaped coordinates ``(..., ndim)`` and flattened walker
  arrays ``(..., nelec * ndim)``.

  Args:
    fractional: Fractional coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Cartesian coordinates with the same shape as ``fractional``.
  """
  frac = jnp.asarray(fractional)
  lat = jnp.asarray(lattice, dtype=frac.dtype)
  orig_shape = frac.shape
  coords = jnp.reshape(frac, (-1, ndim))
  cartesian = coords @ lat.T
  return jnp.reshape(cartesian, orig_shape)



def wrap_positions(
    positions: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Wraps Cartesian positions into the primary simulation cell ``[0, 1)^ndim``.

  Delegates to ``ferminet.mcmc.map_to_simulation_cell`` so coordinate wrapping
  is identical across MCMC sampling, Ewald summation, and geometry utilities.

  Args:
    positions: Particle positions in Cartesian coordinates.
    lattice: Shape ``(ndim, ndim)`` matrix with lattice vectors as columns.
    ndim: Spatial dimension (default 2).

  Returns:
    Wrapped Cartesian coordinates with the same shape as ``positions``.
  """
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  rec = 2.0 * jnp.pi * jnp.linalg.inv(lat)
  return mcmc.map_to_simulation_cell(pos, lat, rec, ndim)




def make_minimum_image_distance_fn(lattice: Array):
  """Builds a minimum-image displacement and distance function for a 2D cell.
  So when there's particles A and B in the simulation cell, there may be periodic image of B that's closer to A, and we want to find shortest such distance among all periodic images of A and B. 
  This is for Ewald summation later
  
  Args:
    lattice: Shape ``(ndim, ndim)`` matrix whose columns are lattice vectors.

  Returns:
    Callable ``dist_fn(pos_a, pos_b) -> (disp, dist)`` where ``pos_a`` has shape
    ``(..., n_a, ndim)``, ``pos_b`` has shape ``(..., n_b, ndim)``, ``disp`` has
    shape ``(..., n_a, n_b, ndim)`` giving minimum-image ``pos_a - pos_b``, and
    ``dist`` has shape ``(..., n_a, n_b)``.
  """
  reduced_np = lll_reduced_basis(lattice)
  ndim = reduced_np.shape[0]
  inv_reduced_np = np.linalg.inv(reduced_np)
  grid = np.meshgrid(*[np.array([-1, 0, 1]) for _ in range(ndim)], indexing='ij')
  point_list = np.stack([g.ravel() for g in grid], axis=-1)  # (3^ndim, ndim)
  shifts_np = point_list @ reduced_np.T  # (3^ndim, ndim)

  def distance_fn(
      pos_a: jnp.ndarray, pos_b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dtype = pos_a.dtype
    reduced = jnp.asarray(reduced_np, dtype=dtype)
    inv_reduced = jnp.asarray(inv_reduced_np, dtype=dtype)
    shifts = jnp.asarray(shifts_np, dtype=dtype)

    diff = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    frac_diff = diff @ inv_reduced.T
    wrapped = diff - jnp.round(frac_diff) @ reduced.T
    diff_all = wrapped[..., None, :] + shifts
    # dists_all = jnp.linalg.norm(diff_all, axis=-1)
    # min_idx = jnp.argmin(dists_all, axis=-1)
    dists_sq_all = jnp.sum(diff_all**2, axis=-1) #Avoid linalg.norm which can have reverse AD NaNs
    min_idx = jnp.argmin(dists_sq_all, axis=-1)
    best_disp = jnp.take_along_axis(
        diff_all, min_idx[..., None, None], axis=-2
    )[..., 0, :]
    # best_dist = jnp.linalg.norm(best_disp, axis=-1)
    best_dist_sq = jnp.sum(best_disp**2, axis=-1)
    zero_mask = best_dist_sq == 0.0
    best_dist = jnp.where(
        zero_mask,
        0.0,
        jnp.sqrt(jnp.where(zero_mask, 1.0, best_dist_sq)),
    )
    return best_disp, best_dist # Displacement vector and distance from the nearest periodic image of pos_b to pos_a

  return distance_fn
