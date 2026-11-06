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


from typing import Any, Optional, Sequence, Tuple, Union

from ferminet import mcmc
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
