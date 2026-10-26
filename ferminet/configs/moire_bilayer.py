# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from typing import Tuple
from ferminet import base_config
from ferminet.moire import geometry
from ferminet.utils import system
import ml_collections
import numpy as np


def _resolve_moire_nspins(
    mcfg: ml_collections.ConfigDict,
) -> Tuple[int, ...]:
  """Resolves the 4-channel ``(n_e_up, n_e_down, n_h_up, n_h_down)`` tuple."""
  if mcfg.get('nspins') is not None:
    nspins = tuple(int(x) for x in mcfg.nspins)
    geometry.layer_particle_counts(nspins)
    return nspins

  spin_polarized = bool(mcfg.get('spin_polarized', False))

  if mcfg.get('n_electrons') is not None:
    n_e_total = int(mcfg.n_electrons)
    if spin_polarized:
      n_e_up, n_e_down = n_e_total, 0
    else:
      n_e_up = n_e_total - (n_e_total // 2)
      n_e_down = n_e_total // 2
  else:
    n_e_up = int(mcfg.get('n_electrons_up', 2))
    n_e_down = int(mcfg.get('n_electrons_down', 2))
    if spin_polarized:
      n_e_up, n_e_down = n_e_up + n_e_down, 0

  if mcfg.get('n_holes') is not None:
    n_h_total = int(mcfg.n_holes)
    if spin_polarized:
      n_h_up, n_h_down = n_h_total, 0
    else:
      n_h_up = n_h_total - (n_h_total // 2)
      n_h_down = n_h_total // 2
  else:
    n_h_up = int(mcfg.get('n_holes_up', 2))
    n_h_down = int(mcfg.get('n_holes_down', 2))
    if spin_polarized:
      n_h_up, n_h_down = n_h_up + n_h_down, 0

  return (n_e_up, n_e_down, n_h_up, n_h_down)


def _configure_moire_system(
    cfg: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Resolves moire geometry, carrier counts, and Hamiltonian/network kwargs."""
  mcfg = cfg.system.moire

  nspins = _resolve_moire_nspins(mcfg)
  n_e, n_h = geometry.layer_particle_counts(nspins)
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

  if mcfg.primitive_lattice is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
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

  # Both moire potentials must be commensurate with the simulation supercell.
  geometry.validate_commensurate(lat_e, sim_lattice)
  geometry.validate_commensurate(lat_h, sim_lattice)

  with cfg.ignore_type():
    cfg.system.lattice = sim_lattice
    cfg.system.make_local_energy_fn = 'ferminet.moire.hamiltonian.local_energy'
    cfg.system.make_local_energy_kwargs = {
        'lattice': sim_lattice,
        'moire_lattice': prim_lattice,
        'moire_lattice_e': lat_e,
        'moire_lattice_h': lat_h,
        'masses': (float(mcfg.mass_e), float(mcfg.mass_h)),
        'mass_e': float(mcfg.mass_e),
        'mass_h': float(mcfg.mass_h),
        'interlayer_distance': float(mcfg.interlayer_distance),
        'dielectric_constant': float(mcfg.dielectric_constant),
        'v_e': float(mcfg.v_e),
        'v_h': float(mcfg.v_h),
        'phi_e': float(mcfg.phi_e),
        'phi_h': float(mcfg.phi_h),
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
    cfg.network.make_feature_layer_fn = (
        'ferminet.pbc.feature_layer.make_pbc_feature_layer'
    )
    cfg.network.make_feature_layer_kwargs = {
        'lattice': sim_lattice,
        'include_r_ae': False,
        'interlayer_distance': float(mcfg.interlayer_distance),
    }
    cfg.network.jastrow_kwargs = {
        'mass_e': float(mcfg.mass_e),
        'mass_h': float(mcfg.mass_h),
        'dielectric_constant': float(mcfg.dielectric_constant),
        'interlayer_distance': float(mcfg.interlayer_distance),
    }
    cfg.network.make_envelope_fn = 'ferminet.envelopes.make_null_envelope'
    cfg.network.make_envelope_kwargs = {}

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
  cfg.mcmc.scale_by_nuclear_distance = False
  cfg.network.full_det = False
  cfg.network.jastrow = 'none'
  cfg.optim.spin_energy = 0.0
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
      'n_electrons_up': 2,
      'n_electrons_down': 2,
      'n_holes_up': 2,
      'n_holes_down': 2,
      'nspins': None,
      'n_electrons': None,
      'n_holes': None,
      'spin_polarized': False,
      # Effective masses m_e^* / m_0 and m_h^* / m_0.
      'mass_e': 0.5,
      'mass_h': 0.6,
      # Out-of-plane separation d between electron (z=0) and hole (z=d) layers.
      'interlayer_distance': 10.0,
      # Primitive moire lattice constant a_M (Bohr) and optional explicit (2, 2)
      # column-vector lattice matrices.
      'lattice_constant': 100.0,
      'primitive_lattice': None,
      'supercell_matrix': ((2, 0), (0, 2)),
      'supercell_lattice': None,
      'moire_lattice_e': None,
      'moire_lattice_h': None,
      # Separate electron and hole moire potential parameters:
      # V_alpha(r) = 2 * v_alpha * sum_{m=1}^3 cos(g_m . (r - offset_alpha) + phi_alpha)
      'v_e': 5.0e-4,
      'v_h': 5.0e-4,
      'phi_e': 0.0,
      'phi_h': 0.0,
      'offset_e': (0.0, 0.0),
      'offset_h': (0.0, 0.0),
      # Screened Coulomb interaction and 2D Ewald convergence settings.
      'dielectric_constant': 10.0,
      'ewald_gmax': 100,
      'nlatvec': 1,
      'ewald_alpha': None,
      'include_capacitor': True,
      'allow_charged_capacitor': False,
  })

  cfg = _configure_moire_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.moire_bilayer'
  return cfg
