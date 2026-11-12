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
from ferminet.utils import units
import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray] #i.e. either numpy or JAX array

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
    le_kwargs = dict(cfg.system.make_local_energy_kwargs)
    is_aniso = _is_anisotropic_moire_config(cfg)
    sc_in_uc = cfg.system.get('supercell_in_unit_cells', True)
    if mcfg.get('supercell_in_unit_cells') is not None:
      sc_in_uc = bool(mcfg.get('supercell_in_unit_cells'))
    prim_override = mcfg.primitive_lattice
    if prim_override is None and is_aniso:
      prim_override = le_kwargs.get('moire_lattice')
    prim_lattice, unit_lattice, sim_lattice = resolve_moire_lattices(
        lattice_constant=float(mcfg.lattice_constant),
        primitive_lattice=prim_override,
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
      if is_aniso and le_kwargs.get('moire_gvectors_e') is not None:
        validate_gvectors_commensurate(
            le_kwargs['moire_gvectors_e'], sim_lattice
        )
    if np.any(np.asarray(mcfg.v_h, dtype=np.float64) != 0.0):
      validate_commensurate(lat_h, sim_lattice)
      if is_aniso and le_kwargs.get('moire_gvectors_h') is not None:
        validate_gvectors_commensurate(
            le_kwargs['moire_gvectors_h'], sim_lattice
        )

    with cfg.ignore_type():
      cfg.system.lattice = sim_lattice
      le_kwargs.update({
          'lattice': sim_lattice,
          'moire_lattice': prim_lattice,
          'moire_lattice_e': lat_e,
          'moire_lattice_h': lat_h,
      })
      cfg.system.make_local_energy_kwargs = le_kwargs
      fl_kwargs['lattice'] = sim_lattice
      fl_kwargs['primitive_lattice'] = unit_lattice
      if is_aniso:
        fl_kwargs['distance_scale'] = average_lattice_period(prim_lattice)
      fl_kwargs.pop('unit_cell_matrix', None)
      fl_kwargs.pop('unit_cell_lattice', None)
      cfg.network.make_feature_layer_kwargs = fl_kwargs
    return cfg

  # Either mcfg is None or enlarged unit cell was set only on
  # feature_layer_kwargs.
  uc_mat = sys_uc_mat if has_sys_uc else fl_uc_mat
  uc_lat = sys_uc_lat if has_sys_uc else fl_uc_lat
  le_kwargs = dict(cfg.system.get('make_local_energy_kwargs', {}))
  is_aniso = _is_anisotropic_moire_config(cfg)
  if mcfg is not None:
    if mcfg.primitive_lattice is not None:
      base_prim = validate_lattice_2d(
          mcfg.primitive_lattice, name='moire.primitive_lattice'
      )
    elif is_aniso and le_kwargs.get('moire_lattice') is not None:
      base_prim = validate_lattice_2d(
          le_kwargs['moire_lattice'], name='moire_lattice'
      )
    else:
      base_prim = triangular_lattice(float(mcfg.lattice_constant))
  else:
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


def _normalize_gvector_units(gvector_units: str = 'bohr^-1') -> str:
  """Normalizes reciprocal wavevector unit strings."""
  if not isinstance(gvector_units, str):
    raise ValueError(
        f'gvector_units must be a string, got {gvector_units!r}.'
    )
  key = gvector_units.strip().lower().replace(' ', '')
  if key in ('bohr^-1', 'bohr**-1', '1/bohr', 'bohr_inv', 'au', 'a.u.'):
    return 'bohr^-1'
  if key in ('nm^-1', 'nm**-1', '1/nm', 'nm_inv'):
    return 'nm^-1'
  if key in (
      'angstrom^-1',
      'angstrom**-1',
      '1/angstrom',
      'angstrom_inv',
      'a^-1',
  ):
    return 'angstrom^-1'
  raise ValueError(
      f'Unsupported gvector_units={gvector_units!r}. '
      "Expected 'bohr^-1', 'nm^-1', or 'angstrom^-1'."
  )


def convert_gvectors_to_bohr_inv(
    gvectors: Array,
    gvector_units: str = 'bohr^-1',
) -> np.ndarray:
  """Converts reciprocal lattice vectors into inverse Bohr (``Bohr^-1``).

  Uses ``ferminet.utils.units.ANGSTROM_BOHR`` (``1 Bohr = 0.52917721067 A``),
  so ``1 A^-1 = ANGSTROM_BOHR Bohr^-1`` and
  ``1 nm^-1 = 0.1 * ANGSTROM_BOHR Bohr^-1``.

  Args:
    gvectors: Array-like of reciprocal wavevectors.
    gvector_units: Input wavevector units (``'bohr^-1'``, ``'nm^-1'``, or
      ``'angstrom^-1'``).

  Returns:
    Float64 NumPy array in ``Bohr^-1``.
  """
  unit_key = _normalize_gvector_units(gvector_units)
  raw = np.asarray(gvectors)
  if raw.dtype == np.bool_:
    raise ValueError('gvectors must be numeric, not boolean.')
  arr = np.asarray(gvectors, dtype=np.float64)
  if not np.all(np.isfinite(arr)):
    raise ValueError('gvectors must have finite entries.')
  if unit_key == 'bohr^-1':
    return arr.copy()
  if unit_key == 'nm^-1':
    return arr * (0.1 * units.ANGSTROM_BOHR)
  return arr * units.ANGSTROM_BOHR


def validate_anisotropic_gvectors(
    gvectors: Array,
    gvector_units: str = 'bohr^-1',
    closure_rtol: float = 1e-2,
) -> np.ndarray:
  """Validates a first-shell reciprocal triad ``(G_1, G_2, G_3)`` in ``Bohr^-1``.

  Checks that ``gvectors`` has shape ``(3, 2)`` with non-zero finite vectors,
  linearly independent ``(G_1, G_2)``, and ``||G_1 + G_2 + G_3||_2 <=
  closure_rtol * max_m ||G_m||_2`` (tolerating e.g. 3-decimal rounded table
  inputs). Enforces exact triad closure ``G_3 = -(G_1 + G_2)`` so the resulting
  real-space moire potential is strictly periodic across all supercell
  boundaries.

  Args:
    gvectors: Shape ``(3, 2)`` array containing ``(G_1, G_2, G_3)`` as rows.
    gvector_units: Units of ``gvectors`` (``'bohr^-1'``, ``'nm^-1'``, or
      ``'angstrom^-1'``).
    closure_rtol: Maximum relative norm residual for ``G_1 + G_2 + G_3``.

  Returns:
    Shape ``(3, 2)`` float64 NumPy array in ``Bohr^-1`` with exact closure
    ``G_3 == -(G_1 + G_2)``.
  """
  if closure_rtol < 0.0 or not math.isfinite(float(closure_rtol)):
    raise ValueError(
        f'closure_rtol must be finite and >= 0, got {closure_rtol!r}.'
    )
  g_bohr = convert_gvectors_to_bohr_inv(gvectors, gvector_units=gvector_units)
  if g_bohr.shape != (3, 2):
    raise ValueError(
        'Anisotropic first-shell moire_gvectors must have shape (3, 2) '
        f'containing (G_1, G_2, G_3), got shape {g_bohr.shape}.'
    )
  norms = np.linalg.norm(g_bohr, axis=1)
  if np.any(norms <= 1e-12):
    raise ValueError(
        'All three moire_gvectors must be non-zero, got norms '
        f'{norms.tolist()}.'
    )
  det_12 = abs(float(np.linalg.det(g_bohr[:2])))
  if det_12 <= 1e-10 * float(norms[0] * norms[1]):
    raise ValueError(
        'First two reciprocal vectors G_1 and G_2 must be linearly '
        f'independent, got det={det_12:.3e} for {g_bohr[:2].tolist()}.'
    )
  max_norm = float(np.max(norms))
  closure_residual = float(np.linalg.norm(g_bohr[0] + g_bohr[1] + g_bohr[2]))
  if closure_residual > float(closure_rtol) * max_norm:
    raise ValueError(
        'Anisotropic first-shell reciprocal triad must sum to zero '
        '(G_1 + G_2 + G_3 = 0) within relative tolerance '
        f'{closure_rtol}: got ||G_1 + G_2 + G_3|| = {closure_residual:.6e} '
        f'({closure_residual / max_norm:.4%} of max ||G_m||).'
    )
  closed = g_bohr.copy()
  closed[2] = -(closed[0] + closed[1])
  return closed


def primitive_lattice_from_gvectors(
    gvectors: Array,
    gvector_units: str = 'bohr^-1',
    closure_rtol: float = 1e-2,
) -> np.ndarray:
  """Derives a right-handed LLL-reduced 2D primitive cell from ``(G_1, G_2, G_3)``.

  Constructs ``L_0 = 2 * pi * inv([G_1; G_2])`` so ``[G_1; G_2] @ L_0 = 2*pi*I``
  and applies unimodular LLL basis reduction to obtain the two shortest
  primitive moire lattice vectors ``(a_1, a_2)`` as columns with right-handed
  orientation (``det(L_prim) > 0``).

  Args:
    gvectors: Shape ``(3, 2)`` reciprocal triad ``(G_1, G_2, G_3)`` as rows.
    gvector_units: Units of ``gvectors`` (``'bohr^-1'``, ``'nm^-1'``, or
      ``'angstrom^-1'``).
    closure_rtol: Relative tolerance for ``G_1 + G_2 + G_3 == 0``.

  Returns:
    Shape ``(2, 2)`` float64 NumPy array in Bohr whose columns are ``a_1, a_2``.
  """
  g_closed = validate_anisotropic_gvectors(
      gvectors, gvector_units=gvector_units, closure_rtol=closure_rtol
  )
  raw_prim = 2.0 * np.pi * np.linalg.inv(g_closed[:2])
  prim = lll_reduced_basis(raw_prim)
  if float(np.linalg.det(prim)) < 0.0:
    prim = prim.copy()
    prim[:, 1] = -prim[:, 1]
  return validate_lattice_2d(prim, name='primitive_lattice')


def anisotropic_first_shell_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns a closed first-shell reciprocal triad ``(G_1, G_2, G_3)``."""
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  rec_rows = lll_reduced_basis((2.0 * np.pi * np.linalg.inv(prim)).T).T
  b1 = rec_rows[0]
  b2 = rec_rows[1]
  if float(np.dot(b1, b2)) > 0.0:
    triad = np.stack([b1, b2 - b1, -b2], axis=0)
  else:
    triad = np.stack([b1, b2, -(b1 + b2)], axis=0)
  return jnp.asarray(triad)


def validate_gvectors_commensurate(
    gvectors: Array,
    simulation_lattice: Array,
    atol: float = 1e-5,
) -> np.ndarray:
  """Checks ``(G @ L_sim) / (2 * pi)`` is an integer matrix for periodic BC."""
  g_arr = np.asarray(gvectors, dtype=np.float64)
  if g_arr.shape != (3, 2) or not np.all(np.isfinite(g_arr)):
    raise ValueError(
        f'gvectors must be a finite (3, 2) array, got shape {g_arr.shape}.'
    )
  sim = validate_lattice_2d(simulation_lattice, name='simulation_lattice')
  windings = (g_arr @ sim) / (2.0 * np.pi)
  rounded = np.round(windings)
  if not np.allclose(windings, rounded, atol=atol):
    raise ValueError(
        'Anisotropic moire reciprocal vectors are not commensurate with '
        f'simulation_lattice: (G @ L_sim) / (2*pi) = {windings.tolist()}.'
    )
  return rounded.astype(int)


def average_lattice_period(primitive_lattice: Array) -> float:
  """Returns the average length ``0.5 * (||a_1|| + ||a_2||)`` of a 2D cell."""
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  a1 = float(np.linalg.norm(prim[:, 0]))
  a2 = float(np.linalg.norm(prim[:, 1]))
  return _validate_positive_float(0.5 * (a1 + a2), 'average_lattice_period')


def validate_mass_tensor_2d(
    mass_tensor: Array,
    name: str = 'mass_tensor',
    sym_atol: float = 1e-10,
) -> np.ndarray:
  """Validates that ``mass_tensor`` is a 2x2 SPD matrix."""
  raw = np.asarray(mass_tensor)
  if raw.dtype == np.bool_:
    raise ValueError(
        f'{name} must be a numeric 2x2 matrix, got {mass_tensor!r}.'
    )
  mat = np.asarray(mass_tensor, dtype=np.float64)
  if mat.shape != (2, 2):
    raise ValueError(f'{name} must have shape (2, 2), got {mat.shape}.')
  if not np.all(np.isfinite(mat)):
    raise ValueError(f'{name} must have finite entries, got {mat.tolist()}.')
  if not np.allclose(mat, mat.T, rtol=1e-10, atol=sym_atol):
    raise ValueError(
        f'{name} must be symmetric (M == M.T), got {mat.tolist()}.'
    )
  mat = 0.5 * (mat + mat.T)
  eigvals = np.linalg.eigvalsh(mat)
  if np.any(eigvals <= 0.0):
    raise ValueError(
        f'{name} must be symmetric positive-definite, got eigenvalues '
        f'{eigvals.tolist()} for {mat.tolist()}.'
    )
  return mat


def effective_scalar_mass(mass: object, name: str = 'mass') -> float:
  """Returns scalar mass or ``0.5 * Tr(M)`` for a 2x2 SPD mass tensor."""
  if isinstance(mass, (bool, np.bool_)):
    raise ValueError(f'{name} must be positive numeric, got {mass!r}.')
  arr = np.asarray(mass)
  if arr.ndim == 0:
    return _validate_positive_float(mass, name)
  mat = validate_mass_tensor_2d(arr, name=name)
  return _validate_positive_float(
      0.5 * float(np.trace(mat)), f'0.5*Tr({name})'
  )


def _validate_positive_float(value: object, name: str) -> float:
  """Validates that ``value`` is a finite positive real scalar (not a bool)."""
  if isinstance(value, (bool, np.bool_)):
    raise ValueError(f'{name} must be a positive float, got {value!r}.')
  try:
    fval = float(value)  # type: ignore[arg-type]
  except (TypeError, ValueError) as exc:
    raise ValueError(
        f'{name} must be a positive float, got {value!r}.'
    ) from exc
  if not math.isfinite(fval) or fval <= 0.0:
    raise ValueError(f'{name} must be finite and positive, got {value!r}.')
  return fval


def _resolve_scalar_layer_masses(
    mass_e: Optional[Union[float, Array]],
    mass_h: Optional[Union[float, Array]],
    masses: Optional[Sequence[Any]],
    anisotropic: bool,
    mass_tensor_e: Optional[Array],
    mass_tensor_h: Optional[Array],
) -> Tuple[float, float]:
  """Resolves scalar effective masses ``(m_e, m_h)`` for energy scaling."""
  validate_fn = (
      effective_scalar_mass if anisotropic else _validate_positive_float
  )
  if masses is not None:
    if isinstance(masses, (str, bytes)) or not hasattr(masses, '__len__'):
      raise ValueError(
          f'masses must be a sequence of length 2 or 4, got {masses!r}.'
      )
    masses_list = list(masses)
    if len(masses_list) == 2:
      m_e = (
          1.0
          if (mass_tensor_e is not None and masses_list[0] is None)
          else validate_fn(masses_list[0], 'masses[0]')
      )
      m_h = (
          1.0
          if (mass_tensor_h is not None and masses_list[1] is None)
          else validate_fn(masses_list[1], 'masses[1]')
      )
    elif len(masses_list) == 4:
      if mass_tensor_e is None:
        m_e_up = validate_fn(masses_list[0], 'masses[0]')
        m_e_down = validate_fn(masses_list[1], 'masses[1]')
        if not np.isclose(m_e_up, m_e_down, rtol=1e-12, atol=0.0):
          raise ValueError(
              '4-channel masses must have equal intra-layer spin masses '
              f'masses[0]==masses[1], got {masses_list}.'
          )
        m_e = m_e_up
      else:
        m_e = 1.0
      if mass_tensor_h is None:
        m_h_up = validate_fn(masses_list[2], 'masses[2]')
        m_h_down = validate_fn(masses_list[3], 'masses[3]')
        if not np.isclose(m_h_up, m_h_down, rtol=1e-12, atol=0.0):
          raise ValueError(
              '4-channel masses must have equal intra-layer spin masses '
              f'masses[2]==masses[3], got {masses_list}.'
          )
        m_h = m_h_up
      else:
        m_h = 1.0
    else:
      raise ValueError(f'masses must have length 2 or 4, got {masses_list}.')
  else:
    if (mass_e is None and mass_tensor_e is None) or (
        mass_h is None and mass_tensor_h is None
    ):
      raise ValueError(
          'Both mass_e and mass_h (or masses) must be provided to compute '
          'moire_energy_scale.'
      )
    m_e = (
        validate_fn(mass_e, 'mass_e') if mass_e is not None else 1.0
    )
    m_h = (
        validate_fn(mass_h, 'mass_h') if mass_h is not None else 1.0
    )

  if mass_tensor_e is not None:
    m_e = effective_scalar_mass(mass_tensor_e, 'mass_tensor_e')
  if mass_tensor_h is not None:
    m_h = effective_scalar_mass(mass_tensor_h, 'mass_tensor_h')
  return m_e, m_h


def moire_energy_scale(
    mass_e: Optional[Union[float, Array]] = None,
    mass_h: Optional[Union[float, Array]] = None,
    moire_period: Optional[float] = None,
    primitive_lattice: Optional[Array] = None,
    masses: Optional[Sequence[Any]] = None,
    anisotropic: bool = False,
    mass_tensor_e: Optional[Array] = None,
    mass_tensor_h: Optional[Array] = None,
) -> float:
  r"""Returns the characteristic moire kinetic energy scale in Hartree.

  Evaluates

  .. math::
      E_M = \frac{\hbar^2}{2 (m_e + m_h) a_M^2}

  in Hartree atomic units (``\hbar = m_0 = 1``, lengths in Bohr, energies in
  Hartree), where ``m_e, m_h`` are the two layer carrier effective masses in
  units of ``m_0`` (with ``0.5 * Tr(M)`` used for a 2x2 anisotropic mass tensor
  ``M`` when ``anisotropic=True``) and ``a_M`` is the ``1 x 1`` primitive moire
  lattice period in Bohr (using ``0.5 * (||a_1|| + ||a_2||)`` when
  ``anisotropic=True``).

  Args:
    mass_e: Electron (or lower-layer) effective mass ``m_e^* / m_0 > 0`` (or
      2x2 SPD mass tensor when ``anisotropic=True``).
    mass_h: Hole (or upper-layer) effective mass ``m_h^* / m_0 > 0`` (or 2x2 SPD
      mass tensor when ``anisotropic=True``).
    moire_period: Primitive moire lattice constant ``a_M > 0`` in Bohr (used
      when ``primitive_lattice`` is ``None``).
    primitive_lattice: Optional ``(2, 2)`` primitive moire lattice matrix.
    masses: Optional sequence of effective masses overriding
      ``(mass_e, mass_h)``. Supports either 2-layer ``(mass_e, mass_h)`` or
      4-channel ``(mass_e_up, mass_e_down, mass_h_up, mass_h_down)``.
    anisotropic: If ``True``, uses ``0.5 * (||a_1|| + ||a_2||)`` for ``a_M`` and
      supports 2x2 SPD mass tensors via ``0.5 * Tr(M)``.
    mass_tensor_e: Optional ``(2, 2)`` electron mass tensor (requires
      ``anisotropic=True``).
    mass_tensor_h: Optional ``(2, 2)`` hole mass tensor (requires
      ``anisotropic=True``).

  Returns:
    Positive float giving ``1 / (2 * (mass_e + mass_h) * a_M^2)`` in Hartree.
  """
  if not anisotropic and (
      mass_tensor_e is not None or mass_tensor_h is not None
  ):
    raise ValueError(
        'mass_tensor_e and mass_tensor_h require anisotropic=True.'
    )
  if mass_tensor_e is not None and mass_tensor_h is not None:
    raise ValueError(
        'Anisotropic mass tensor is supported on at most one layer.'
    )

  m_e, m_h = _resolve_scalar_layer_masses(
      mass_e=mass_e,
      mass_h=mass_h,
      masses=masses,
      anisotropic=bool(anisotropic),
      mass_tensor_e=mass_tensor_e,
      mass_tensor_h=mass_tensor_h,
  )

  if primitive_lattice is not None:
    prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
    if anisotropic:
      a_m = average_lattice_period(prim)
    else:
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


def _is_anisotropic_moire_config(cfg: Any) -> bool:
  """Returns whether anisotropic moire mode is enabled in ``cfg``."""
  if cfg is None or 'system' not in cfg or cfg.system is None:
    return False
  sys_cfg = cfg.system
  if bool(sys_cfg.get('anisotropic', False)):
    return True
  mcfg = sys_cfg.get('moire') if 'moire' in sys_cfg else None
  if mcfg is not None and bool(mcfg.get('anisotropic', False)):
    return True
  le_kwargs = sys_cfg.get('make_local_energy_kwargs')
  if le_kwargs and bool(le_kwargs.get('anisotropic', False)):
    return True
  return False


def _lookup_aniso_field(
    field: str,
    mcfg: Any,
    sys_cfg: Any,
    le_kwargs: dict[str, Any],
) -> Any:
  """Looks up an anisotropic field across ``moire``, ``system``, and kwargs."""
  if mcfg is not None and field in mcfg and mcfg.get(field) is not None:
    return mcfg.get(field)
  if (
      sys_cfg is not None
      and field in sys_cfg
      and sys_cfg.get(field) is not None
  ):
    return sys_cfg.get(field)
  if mcfg is None or field not in mcfg:
    return le_kwargs.get(field)
  return None


def _validate_no_anisotropic_fields_when_disabled(cfg: Any) -> None:
  """Raises ``ValueError`` if anisotropic fields are set while disabled."""
  if cfg is None or 'system' not in cfg or cfg.system is None:
    return
  sys_cfg = cfg.system
  mcfg = sys_cfg.get('moire') if 'moire' in sys_cfg else None
  le_kwargs = (
      dict(sys_cfg.get('make_local_energy_kwargs'))
      if sys_cfg.get('make_local_energy_kwargs') is not None
      else {}
  )
  aniso_fields = (
      'moire_gvectors',
      'moire_gvectors_e',
      'moire_gvectors_h',
      'mass_tensor',
      'mass_tensor_e',
      'mass_tensor_h',
  )
  for name in aniso_fields:
    if _lookup_aniso_field(name, mcfg, sys_cfg, le_kwargs) is not None:
      raise ValueError(
          f'{name} is set, but anisotropic=False. Set anisotropic=True to use '
          'anisotropic moire reciprocal vectors or mass tensors.'
      )
  for m_key in ('mass_e', 'mass_h'):
    if mcfg is not None and m_key in mcfg and mcfg.get(m_key) is not None:
      if np.asarray(mcfg.get(m_key)).ndim > 0:
        raise ValueError(
            f'moire.{m_key} is a matrix/sequence, which requires '
            'anisotropic=True.'
        )


def _route_single_layer_quantity(
    generic_val: Any,
    val_e: Any,
    val_h: Any,
    has_v_e: bool,
    has_v_h: bool,
    label: str,
    equal_fn: Any,
    fallback_layer: Optional[str] = None,
) -> Tuple[Any, Any]:
  """Routes a generic or layer-specific quantity onto the active moire layer."""
  if val_e is not None and val_h is not None:
    raise ValueError(
        f'When anisotropic=True, {label} may only be set on one layer '
        f'(not both {label}_e and {label}_h).'
    )
  if has_v_e and val_h is not None:
    raise ValueError(
        f'Cannot set {label}_h when moire potential is active on the electron '
        'layer (v_e != 0).'
    )
  if has_v_h and val_e is not None:
    raise ValueError(
        f'Cannot set {label}_e when moire potential is active on the hole '
        'layer (v_h != 0).'
    )
  if generic_val is None:
    return val_e, val_h

  target = (
      'e'
      if has_v_e
      else ('h' if has_v_h else fallback_layer)
  )
  if target == 'e':
    if val_e is not None and not equal_fn(generic_val, val_e):
      raise ValueError(f'Conflicting {label} and {label}_e values.')
    return generic_val, None
  if target == 'h':
    if val_h is not None and not equal_fn(generic_val, val_h):
      raise ValueError(f'Conflicting {label} and {label}_h values.')
    return None, generic_val
  if val_e is not None:
    if not equal_fn(generic_val, val_e):
      raise ValueError(f'Conflicting {label} and {label}_e values.')
    return val_e, None
  if val_h is not None:
    if not equal_fn(generic_val, val_h):
      raise ValueError(f'Conflicting {label} and {label}_h values.')
    return None, val_h
  raise ValueError(
      f'Ambiguous generic {label} when both v_e == 0 and v_h == 0; specify '
      f'{label}_e or {label}_h (or set non-zero v_e / v_h).'
  )


def _normalize_aniso_harmonic_param(
    value: Any,
    name: str,
) -> Union[float, Tuple[float, float, float]]:
  """Validates scalar or 3-element first-shell harmonic parameters."""
  arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
  if arr.ndim != 1 or arr.size not in (1, 3):
    raise ValueError(
        f'When anisotropic=True, {name} must be a scalar or a length-3 '
        'first-shell sequence for (G_1, G_2, G_3), got shape '
        f'{np.asarray(value).shape}.'
    )
  if not np.all(np.isfinite(arr)):
    raise ValueError(f'{name} must have finite entries.')
  if arr.size == 1:
    return float(arr[0])
  return (float(arr[0]), float(arr[1]), float(arr[2]))


def _resolve_gvecs_with_units(
    mcfg: Any,
    sys_cfg: Any,
    le_kwargs: dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
  """Extracts explicit validated ``Bohr^-1`` G-vector triads from config."""
  gvec_keys = ('moire_gvectors', 'moire_gvectors_e', 'moire_gvectors_h')
  has_direct = any(
      (mcfg is not None and k in mcfg and mcfg.get(k) is not None)
      or (sys_cfg is not None and k in sys_cfg and sys_cfg.get(k) is not None)
      for k in gvec_keys
  )

  def _extract_one(key: str) -> Optional[np.ndarray]:
    if mcfg is not None and key in mcfg and mcfg.get(key) is not None:
      u = (
          mcfg.get('gvector_units')
          or sys_cfg.get('gvector_units')
          or 'bohr^-1'
      )
      return validate_anisotropic_gvectors(mcfg.get(key), gvector_units=u)
    if (
        sys_cfg is not None
        and key in sys_cfg
        and sys_cfg.get(key) is not None
    ):
      u = sys_cfg.get('gvector_units') or 'bohr^-1'
      return validate_anisotropic_gvectors(sys_cfg.get(key), gvector_units=u)
    if (
        not has_direct
        and (mcfg is None or key not in mcfg)
        and le_kwargs.get(key) is not None
    ):
      u = le_kwargs.get('gvector_units', 'bohr^-1')
      return validate_anisotropic_gvectors(le_kwargs[key], gvector_units=u)
    return None

  return (
      _extract_one('moire_gvectors'),
      _extract_one('moire_gvectors_e'),
      _extract_one('moire_gvectors_h'),
  )


def prepare_moire_anisotropic_config(cfg: Any) -> Any:
  """Pre-normalizes anisotropic moire fields before ``cfg.system.set_molecule``.

  When ``anisotropic`` is ``False`` (the default), returns ``cfg`` completely
  untouched. When ``anisotropic`` is ``True`` on an existing moire config whose
  ``set_molecule`` callback expects scalar ``v_e``/``v_h`` or
  ``mass_e``/``mass_h``, temporarily stashes sequence/matrix fields so
  ``set_molecule`` succeeds before ``apply_moire_anisotropic_config`` finalizes
  all anisotropic settings.
  """
  if not _is_anisotropic_moire_config(cfg):
    return cfg
  sys_cfg = cfg.system
  mcfg = sys_cfg.get('moire') if 'moire' in sys_cfg else None
  if mcfg is None or not callable(sys_cfg.get('set_molecule')):
    return cfg

  le_kwargs = (
      dict(sys_cfg.get('make_local_energy_kwargs'))
      if sys_cfg.get('make_local_energy_kwargs') is not None
      else {}
  )
  stashed: dict[str, Any] = {}
  prev_moire_lat = le_kwargs.get('moire_lattice')
  cur_pair_lat = (
      cfg.mcmc.get('pair_lattice')
      if ('mcmc' in cfg and cfg.mcmc is not None)
      else None
  )
  if (
      cur_pair_lat is not None
      and prev_moire_lat is not None
      and np.asarray(cur_pair_lat).shape == (2, 2)
      and np.asarray(prev_moire_lat).shape == (2, 2)
      and np.allclose(cur_pair_lat, prev_moire_lat)
  ):
    stashed['pair_lattice_was_auto'] = True

  with cfg.ignore_type():
    for layer in ('e', 'h'):
      m_key = f'mass_{layer}'
      mt_key = f'mass_tensor_{layer}'
      if mcfg.get(m_key) is not None and np.asarray(mcfg.get(m_key)).ndim > 0:
        mat = validate_mass_tensor_2d(mcfg.get(m_key), name=f'moire.{m_key}')
        if mcfg.get(mt_key) is not None and not np.allclose(
            validate_mass_tensor_2d(mcfg.get(mt_key), name=f'moire.{mt_key}'),
            mat,
        ):
          raise ValueError(
              f'Conflicting moire.{m_key} and moire.{mt_key} matrices.'
          )
        mcfg[mt_key] = tuple(tuple(float(x) for x in row) for row in mat)
        mcfg[m_key] = effective_scalar_mass(mat, name=f'moire.{m_key}')

    for p_key in ('v_e', 'v_h', 'phi_e', 'phi_h'):
      val = mcfg.get(p_key)
      if val is not None and np.asarray(val).ndim > 0:
        norm_val = _normalize_aniso_harmonic_param(val, f'moire.{p_key}')
        stashed[p_key] = norm_val
        arr = np.atleast_1d(np.asarray(norm_val, dtype=np.float64))
        idx = int(np.argmax(np.abs(arr)))
        mcfg[p_key] = float(arr[idx])

    g_gen, g_e, g_h = _resolve_gvecs_with_units(mcfg, sys_cfg, le_kwargs)
    active_g = g_gen if g_gen is not None else (g_e if g_e is not None else g_h)
    if active_g is not None and mcfg.get('primitive_lattice') is None:
      stashed['primitive_lattice_was_none'] = True
      mcfg.primitive_lattice = primitive_lattice_from_gvectors(
          active_g, gvector_units='bohr^-1'
      )

    if stashed:
      mcfg['_aniso_stashed'] = stashed
  return cfg


def _extract_aniso_mass_tensors(
    mcfg: Any,
    sys_cfg: Any,
    le_kwargs: dict[str, Any],
    has_v_e: bool,
    has_v_h: bool,
    fallback_layer: Optional[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """Extracts and validates single-layer 2x2 mass tensors."""
  mt_keys = ('mass_tensor', 'mass_tensor_e', 'mass_tensor_h')
  has_direct_mt = any(
      (mcfg is not None and k in mcfg and mcfg.get(k) is not None)
      or (sys_cfg is not None and k in sys_cfg and sys_cfg.get(k) is not None)
      for k in mt_keys
  ) or (
      mcfg is not None
      and (
          (
              mcfg.get('mass_e') is not None
              and np.asarray(mcfg.get('mass_e')).ndim > 0
          )
          or (
              mcfg.get('mass_h') is not None
              and np.asarray(mcfg.get('mass_h')).ndim > 0
          )
      )
  )
  eff_le = {} if has_direct_mt else le_kwargs
  raw_gen = _lookup_aniso_field('mass_tensor', mcfg, sys_cfg, eff_le)
  raw_e = _lookup_aniso_field('mass_tensor_e', mcfg, sys_cfg, eff_le)
  raw_h = _lookup_aniso_field('mass_tensor_h', mcfg, sys_cfg, eff_le)
  if mcfg is not None:
    if (
        mcfg.get('mass_e') is not None
        and np.asarray(mcfg.get('mass_e')).ndim > 0
    ):
      if raw_e is not None and not np.allclose(raw_e, mcfg.get('mass_e')):
        raise ValueError('Conflicting moire.mass_e and mass_tensor_e.')
      raw_e = mcfg.get('mass_e')
    if (
        mcfg.get('mass_h') is not None
        and np.asarray(mcfg.get('mass_h')).ndim > 0
    ):
      if raw_h is not None and not np.allclose(raw_h, mcfg.get('mass_h')):
        raise ValueError('Conflicting moire.mass_h and mass_tensor_h.')
      raw_h = mcfg.get('mass_h')

  mt_gen = (
      validate_mass_tensor_2d(raw_gen, name='mass_tensor')
      if raw_gen is not None
      else None
  )
  mt_e = (
      validate_mass_tensor_2d(raw_e, name='mass_tensor_e')
      if raw_e is not None
      else None
  )
  mt_h = (
      validate_mass_tensor_2d(raw_h, name='mass_tensor_h')
      if raw_h is not None
      else None
  )
  return _route_single_layer_quantity(
      mt_gen,
      mt_e,
      mt_h,
      has_v_e=has_v_e,
      has_v_h=has_v_h,
      label='mass_tensor',
      equal_fn=np.allclose,
      fallback_layer=fallback_layer,
  )


def _as_nested_float_tuple(
    mat: Optional[np.ndarray],
) -> Optional[Tuple[Tuple[float, ...], ...]]:
  """Converts a 2D NumPy array to nested Python float tuples for ConfigDict."""
  if mat is None:
    return None
  return tuple(tuple(float(x) for x in row) for row in mat)


def apply_moire_anisotropic_config(cfg: Any) -> Any:
  """Applies opt-in single-layer moire anisotropy settings to ``cfg``.

  When ``anisotropic`` is ``False`` (the default), validates that no
  anisotropic-only parameters were set and returns ``cfg`` completely unchanged
  for 100% backward compatibility. When ``anisotropic`` is ``True``, validates:
    1. Moire potential is active on at most one carrier layer (1st shell only).
    2. First-shell reciprocal vectors ``(G_1, G_2, G_3)`` (if provided) close
       within tolerance and define an LLL-reduced primitive lattice ``L_prim``.
    3. Anisotropic 2x2 SPD mass tensor (if provided) is on the moire-active
       layer.
    4. Periodic feature ``distance_scale`` uses ``0.5 * (||a_1|| + ||a_2||)``.

  Args:
    cfg: ``ml_collections.ConfigDict`` for a FermiNet calculation.

  Returns:
    The updated ``cfg`` (also modified in place).
  """
  if cfg is None or 'system' not in cfg or cfg.system is None:
    return cfg
  if not _is_anisotropic_moire_config(cfg):
    _validate_no_anisotropic_fields_when_disabled(cfg)
    return cfg
  if not _is_moire_system_config(cfg):
    raise ValueError(
        'anisotropic=True is only supported for moire bilayer configurations.'
    )

  sys_cfg = cfg.system
  mcfg = sys_cfg.get('moire') if 'moire' in sys_cfg else None
  le_kwargs = (
      dict(sys_cfg.get('make_local_energy_kwargs'))
      if sys_cfg.get('make_local_energy_kwargs') is not None
      else {}
  )

  stashed_auto_pair = False
  if mcfg is not None and '_aniso_stashed' in mcfg:
    with cfg.ignore_type():
      stashed = dict(mcfg['_aniso_stashed'])
      del mcfg['_aniso_stashed']
      stashed_auto_pair = bool(stashed.get('pair_lattice_was_auto', False))
      for p_key in ('v_e', 'v_h', 'phi_e', 'phi_h'):
        if p_key in stashed:
          mcfg[p_key] = stashed[p_key]
      if stashed.get('primitive_lattice_was_none'):
        mcfg.primitive_lattice = None

  raw_v_e = mcfg.v_e if mcfg is not None else le_kwargs.get('v_e', 0.0)
  raw_v_h = mcfg.v_h if mcfg is not None else le_kwargs.get('v_h', 0.0)
  raw_phi_e = mcfg.phi_e if mcfg is not None else le_kwargs.get('phi_e', 0.0)
  raw_phi_h = mcfg.phi_h if mcfg is not None else le_kwargs.get('phi_h', 0.0)

  v_e = _normalize_aniso_harmonic_param(raw_v_e, 'v_e')
  v_h = _normalize_aniso_harmonic_param(raw_v_h, 'v_h')
  phi_e = _normalize_aniso_harmonic_param(raw_phi_e, 'phi_e')
  phi_h = _normalize_aniso_harmonic_param(raw_phi_h, 'phi_h')
  for amp, ph, layer_name in ((v_e, phi_e, 'e'), (v_h, phi_h, 'h')):
    if isinstance(amp, float) and isinstance(ph, tuple):
      amp_3 = (amp, amp, amp)
      if layer_name == 'e':
        v_e = amp_3
      else:
        v_h = amp_3

  has_v_e = bool(np.any(np.asarray(v_e, dtype=np.float64) != 0.0))
  has_v_h = bool(np.any(np.asarray(v_h, dtype=np.float64) != 0.0))
  if has_v_e and has_v_h:
    raise ValueError(
        'When anisotropic=True, moire potential may only be active on one '
        'layer (either v_e != 0 or v_h != 0, not both).'
    )

  g_gen, g_e_in, g_h_in = _resolve_gvecs_with_units(mcfg, sys_cfg, le_kwargs)
  fallback_layer = (
      'e' if g_e_in is not None else ('h' if g_h_in is not None else 'h')
  )
  gvecs_e, gvecs_h = _route_single_layer_quantity(
      g_gen,
      g_e_in,
      g_h_in,
      has_v_e=has_v_e,
      has_v_h=has_v_h,
      label='moire_gvectors',
      equal_fn=np.allclose,
      fallback_layer=fallback_layer,
  )
  active_gvecs = gvecs_e if gvecs_e is not None else gvecs_h

  mt_fallback = (
      'e'
      if (has_v_e or gvecs_e is not None)
      else ('h' if (has_v_h or gvecs_h is not None) else None)
  )
  mass_tensor_e, mass_tensor_h = _extract_aniso_mass_tensors(
      mcfg=mcfg,
      sys_cfg=sys_cfg,
      le_kwargs=le_kwargs,
      has_v_e=has_v_e,
      has_v_h=has_v_h,
      fallback_layer=mt_fallback,
  )

  if mcfg is not None:
    if active_gvecs is not None:
      derived_prim = primitive_lattice_from_gvectors(
          active_gvecs, gvector_units='bohr^-1'
      )
      if mcfg.primitive_lattice is not None:
        prim_lattice = validate_lattice_2d(
            mcfg.primitive_lattice, name='moire.primitive_lattice'
        )
        validate_gvectors_commensurate(active_gvecs, prim_lattice)
        ratio = validate_commensurate(derived_prim, prim_lattice)
        if abs(round(float(np.linalg.det(ratio)))) != 1:
          raise ValueError(
              'moire.primitive_lattice must span the same 1x1 moire cell as '
              f'moire_gvectors, got det(ratio)={np.linalg.det(ratio)}.'
          )
      else:
        prim_lattice = derived_prim
    elif mcfg.primitive_lattice is not None:
      prim_lattice = validate_lattice_2d(
          mcfg.primitive_lattice, name='moire.primitive_lattice'
      )
    else:
      prim_lattice = triangular_lattice(float(mcfg.lattice_constant))

    if mcfg.supercell_lattice is not None:
      sim_lattice = validate_lattice_2d(
          mcfg.supercell_lattice, name='moire.supercell_lattice'
      )
    else:
      sim_lattice = supercell_lattice(
          prim_lattice, np.asarray(mcfg.supercell_matrix)
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
  else:
    sim_lattice = validate_lattice_2d(
        le_kwargs.get('lattice', sys_cfg.lattice), name='lattice'
    )
    if active_gvecs is not None and le_kwargs.get('moire_lattice') is None:
      prim_lattice = primitive_lattice_from_gvectors(
          active_gvecs, gvector_units='bohr^-1'
      )
    else:
      prim_lattice = validate_lattice_2d(
          le_kwargs.get('moire_lattice', sim_lattice), name='moire_lattice'
      )
    lat_e = validate_lattice_2d(
        le_kwargs.get('moire_lattice_e', prim_lattice), name='moire_lattice_e'
    )
    lat_h = validate_lattice_2d(
        le_kwargs.get('moire_lattice_h', prim_lattice), name='moire_lattice_h'
    )

  if has_v_e:
    check_g_e = (
        gvecs_e
        if gvecs_e is not None
        else np.asarray(anisotropic_first_shell_gvectors(lat_e))
    )
    validate_commensurate(lat_e, sim_lattice)
    validate_gvectors_commensurate(check_g_e, sim_lattice)
  if has_v_h:
    check_g_h = (
        gvecs_h
        if gvecs_h is not None
        else np.asarray(anisotropic_first_shell_gvectors(lat_h))
    )
    validate_commensurate(lat_h, sim_lattice)
    validate_gvectors_commensurate(check_g_h, sim_lattice)

  raw_masses = (
      mcfg.get('masses')
      if (mcfg is not None and 'masses' in mcfg)
      else (le_kwargs.get('masses') if mcfg is None else None)
  )
  if raw_masses is not None:
    eff_mass_e, eff_mass_h = _resolve_scalar_layer_masses(
        mass_e=None,
        mass_h=None,
        masses=raw_masses,
        anisotropic=True,
        mass_tensor_e=mass_tensor_e,
        mass_tensor_h=mass_tensor_h,
    )
  else:
    raw_me = (
        mcfg.get('mass_e')
        if (mcfg is not None and mcfg.get('mass_e') is not None)
        else le_kwargs.get('mass_e', 1.0)
    )
    raw_mh = (
        mcfg.get('mass_h')
        if (mcfg is not None and mcfg.get('mass_h') is not None)
        else le_kwargs.get('mass_h', 1.0)
    )
    eff_mass_e, eff_mass_h = _resolve_scalar_layer_masses(
        mass_e=raw_me,
        mass_h=raw_mh,
        masses=None,
        anisotropic=True,
        mass_tensor_e=mass_tensor_e,
        mass_tensor_h=mass_tensor_h,
    )
  dist_scale = average_lattice_period(prim_lattice)
  prev_moire_lat = le_kwargs.get('moire_lattice')

  with cfg.ignore_type():
    cfg.system.lattice = sim_lattice

    if 'mcmc' in cfg and cfg.mcmc is not None:
      cur_pair_lat = cfg.mcmc.get('pair_lattice')
      should_update_pair_lat = cur_pair_lat is None or stashed_auto_pair
      if (
          not should_update_pair_lat
          and prev_moire_lat is not None
          and np.asarray(cur_pair_lat).shape == (2, 2)
          and np.asarray(prev_moire_lat).shape == (2, 2)
          and np.allclose(cur_pair_lat, prev_moire_lat)
      ):
        should_update_pair_lat = True
      if should_update_pair_lat:
        cfg.mcmc.pair_lattice = prim_lattice

    le_kwargs.update({
        'lattice': sim_lattice,
        'moire_lattice': prim_lattice,
        'moire_lattice_e': lat_e,
        'moire_lattice_h': lat_h,
        'masses': (eff_mass_e, eff_mass_h),
        'mass_e': eff_mass_e,
        'mass_h': eff_mass_h,
        'v_e': v_e,
        'v_h': v_h,
        'phi_e': phi_e,
        'phi_h': phi_h,
        'anisotropic': True,
        'moire_gvectors_e': _as_nested_float_tuple(gvecs_e),
        'moire_gvectors_h': _as_nested_float_tuple(gvecs_h),
        'gvector_units': 'bohr^-1',
        'mass_tensor_e': _as_nested_float_tuple(mass_tensor_e),
        'mass_tensor_h': _as_nested_float_tuple(mass_tensor_h),
    })
    le_kwargs.pop('moire_gvectors', None)
    le_kwargs.pop('mass_tensor', None)
    cfg.system.make_local_energy_kwargs = le_kwargs

    if 'network' in cfg and cfg.network is not None:
      fl_kwargs = (
          dict(cfg.network.make_feature_layer_kwargs)
          if cfg.network.get('make_feature_layer_kwargs') is not None
          else {}
      )
      fl_kwargs['lattice'] = sim_lattice
      fl_kwargs['distance_scale'] = dist_scale
      cfg.network.make_feature_layer_kwargs = fl_kwargs

      if cfg.network.get('jastrow_kwargs') is not None:
        j_kwargs = dict(cfg.network.jastrow_kwargs)
        j_kwargs['mass_e'] = eff_mass_e
        j_kwargs['mass_h'] = eff_mass_h
        cfg.network.jastrow_kwargs = j_kwargs

  return cfg


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
       primitive moire period ``a_M`` (with ``a_M = 0.5 * (||a_1|| + ||a_2||)``
       and ``0.5 * Tr(M)`` when ``anisotropic=True``).
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
      if (
          'network' in cfg
          and cfg.network is not None
          and cfg.network.get('make_feature_layer_kwargs') is not None
      )
      else {}
  )

  is_aniso = _is_anisotropic_moire_config(cfg)
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

  if not is_aniso:
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

  raw_v_e = mcfg.v_e if mcfg is not None else le_kwargs.get('v_e', 0.0)
  raw_v_h = mcfg.v_h if mcfg is not None else le_kwargs.get('v_h', 0.0)
  has_v_e = bool(np.any(np.asarray(raw_v_e, dtype=np.float64) != 0.0))
  has_v_h = bool(np.any(np.asarray(raw_v_h, dtype=np.float64) != 0.0))
  g_gen, g_e_in, g_h_in = _resolve_gvecs_with_units(mcfg, sys_cfg, le_kwargs)
  active_g = (
      g_gen if g_gen is not None else (g_e_in if g_e_in is not None else g_h_in)
  )
  mt_fallback = (
      'e'
      if (has_v_e or g_e_in is not None)
      else ('h' if (has_v_h or g_h_in is not None) else None)
  )
  mt_e, mt_h = _extract_aniso_mass_tensors(
      mcfg=mcfg,
      sys_cfg=sys_cfg,
      le_kwargs=le_kwargs,
      has_v_e=has_v_e,
      has_v_h=has_v_h,
      fallback_layer=mt_fallback,
  )

  prim_lat = None
  moire_period = None
  if mcfg is not None and mcfg.get('primitive_lattice') is not None:
    prim_lat = mcfg.get('primitive_lattice')
  elif active_g is not None:
    prim_lat = primitive_lattice_from_gvectors(
        active_g, gvector_units='bohr^-1'
    )
  elif le_kwargs.get('moire_lattice') is not None:
    prim_lat = le_kwargs.get('moire_lattice')
  elif mcfg is not None and mcfg.get('lattice_constant') is not None:
    moire_period = mcfg.get('lattice_constant')
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
      anisotropic=True,
      mass_tensor_e=mt_e,
      mass_tensor_h=mt_h,
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
