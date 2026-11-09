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

"""Tests for enlarged moire unit cells (arXiv:2406.11134) and PBC features."""

import copy
import os
import tempfile
from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import loss as qmc_loss_functions
from ferminet import networks
from ferminet import train
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
from ferminet.pbc import envelopes as pbc_envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class MoireUnitCellTest(parameterized.TestCase):

  def test_default_config_resolve_is_100_percent_backward_compatible(self):
    """When enlarged unit-cell options are unset, resolve(cfg) is unchanged."""
    cfg_raw = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config()
    expected_sim_lat = np.asarray(cfg_raw.system.lattice).copy()
    expected_le_kwargs = copy.deepcopy(
        dict(cfg_raw.system.make_local_energy_kwargs)
    )
    expected_fl_kwargs = copy.deepcopy(
        dict(cfg_raw.network.make_feature_layer_kwargs)
    )

    cfg_resolved = base_config.resolve(cfg_raw)
    np.testing.assert_allclose(cfg_resolved.system.lattice, expected_sim_lat)
    self.assertEqual(
        set(cfg_resolved.network.make_feature_layer_kwargs.keys()),
        set(expected_fl_kwargs.keys()),
    )
    self.assertNotIn(
        'primitive_lattice', cfg_resolved.network.make_feature_layer_kwargs
    )
    np.testing.assert_allclose(
        cfg_resolved.system.make_local_energy_kwargs['moire_lattice'],
        expected_le_kwargs['moire_lattice'],
    )

  def test_sqrt3_unit_cell_matrices_120_and_60_span_same_bravais_lattice(self):
    """Tests 120-deg and 60-deg sqrt(3) x sqrt(3) enlarged unit cells."""
    a_m = 151.18
    prim = geometry.triangular_lattice(a_m)
    m_120 = geometry.sqrt3_unit_cell_matrix(120)
    m_60 = geometry.sqrt3_unit_cell_matrix(60)
    self.assertEqual(round(float(np.linalg.det(m_120))), 3)
    self.assertEqual(round(float(np.linalg.det(m_60))), 3)

    unit_120 = geometry.enlarged_unit_cell_lattice(prim, unit_cell_matrix=m_120)
    unit_60 = geometry.enlarged_unit_cell_lattice(prim, unit_cell_matrix=m_60)

    # Both have basis vector norms equal to sqrt(3) * a_M and area 3 * Area(prim).
    for unit_lat, expected_cos in ((unit_120, -0.5), (unit_60, 0.5)):
      norms = np.linalg.norm(unit_lat, axis=0)
      np.testing.assert_allclose(norms, np.sqrt(3.0) * a_m, rtol=1e-6)
      cos_angle = np.dot(unit_lat[:, 0], unit_lat[:, 1]) / (norms[0] * norms[1])
      np.testing.assert_allclose(cos_angle, expected_cos, atol=1e-6)
      np.testing.assert_allclose(
          abs(np.linalg.det(unit_lat)), 3.0 * abs(np.linalg.det(prim)), rtol=1e-6
      )

    # Both span the exact same 2D Bravais lattice (unimodular integer ratio).
    ratio = geometry.validate_commensurate(unit_120, unit_60)
    self.assertEqual(abs(round(float(np.linalg.det(ratio)))), 1)

  @parameterized.parameters(
      ('sqrt3', 3),
      ('sqrt3_120', 3),
      ('sqrt3_60', 3),
      ('2x2', 4),
      ('5x5', 25),
      ('6x6', 36),
      ('2x1', 2),
  )
  def test_string_presets_match_arxiv_2406_11134_table_s4(
      self, preset, expected_det
  ):
    """Tests string presets for enlarged unit cells in arXiv:2406.11134."""
    prim = geometry.triangular_lattice(100.0)
    unit = geometry.enlarged_unit_cell_lattice(prim, unit_cell_matrix=preset)
    ratio = geometry.validate_commensurate(prim, unit)
    self.assertEqual(abs(round(float(np.linalg.det(ratio)))), expected_det)

  def test_resolve_moire_lattices_supercell_in_unit_cells_vs_primitive(self):
    """Tests tiling enlarged unit cell vs tiling 1x1 primitive cell."""
    a_m = 151.18
    # 1. Tiling the sqrt(3) x sqrt(3) unit cell by 2 x 3 gives 6 unit cells = 18 moire cells.
    prim, unit, sim_uc = geometry.resolve_moire_lattices(
        lattice_constant=a_m,
        unit_cell_matrix='sqrt3',
        supercell_matrix=((2, 0), (0, 3)),
        supercell_in_unit_cells=True,
    )
    ratio_prim = geometry.validate_commensurate(prim, sim_uc)
    self.assertEqual(abs(round(float(np.linalg.det(ratio_prim)))), 18)

    # 2. Tiling the 1x1 primitive cell by 6 x 3 (supercell_in_unit_cells=False)
    # is ALSO commensurate with the 120-deg sqrt(3) x sqrt(3) unit cell!
    _, unit_2, sim_prim = geometry.resolve_moire_lattices(
        lattice_constant=a_m,
        unit_cell_matrix=((2, -1), (1, 1)),
        supercell_matrix=((6, 0), (0, 3)),
        supercell_in_unit_cells=False,
    )
    np.testing.assert_allclose(unit, unit_2)
    ratio_uc = geometry.validate_commensurate(unit_2, sim_prim)
    np.testing.assert_array_equal(ratio_uc, np.array([[2, 1], [-2, 2]]))

    # 3. An incommensurate 4 x 3 primitive supercell must be rejected for sqrt(3) x sqrt(3).
    with self.assertRaises(ValueError):
      geometry.resolve_moire_lattices(
          lattice_constant=a_m,
          unit_cell_matrix='sqrt3',
          supercell_matrix=((4, 0), (0, 3)),
          supercell_in_unit_cells=False,
      )

  def test_enlarged_unit_cell_preserves_1x1_moire_potential_and_breaks_1x1_ae_symmetry(
      self,
  ):
    """Tests that Hamiltonian keeps 1x1 moire period while ae features use enlarged cell."""
    cfg = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config()
    cfg.system.unit_cell_matrix = 'sqrt3'
    cfg.system.supercell_in_unit_cells = False  # Keep 6x3 primitive supercell (18 cells)
    cfg = base_config.resolve(cfg)

    a_m = float(cfg.system.moire.lattice_constant)
    prim_lattice = geometry.triangular_lattice(a_m)
    unit_lattice = geometry.enlarged_unit_cell_lattice(
        prim_lattice, unit_cell_matrix='sqrt3'
    )
    sim_lattice = geometry.supercell_lattice(prim_lattice, ((6, 0), (0, 3)))

    # 1. Verify resolved config lattices.
    np.testing.assert_allclose(cfg.system.lattice, sim_lattice)
    np.testing.assert_allclose(
        cfg.system.make_local_energy_kwargs['moire_lattice'], prim_lattice
    )
    np.testing.assert_allclose(
        cfg.system.make_local_energy_kwargs['moire_lattice_h'], prim_lattice
    )
    np.testing.assert_allclose(
        cfg.network.make_feature_layer_kwargs['primitive_lattice'], unit_lattice
    )
    np.testing.assert_allclose(
        cfg.network.make_feature_layer_kwargs['lattice'], sim_lattice
    )
    self.assertAlmostEqual(
        float(cfg.network.make_feature_layer_kwargs['distance_scale']), a_m
    )

    # 2. Verify Hamiltonian moire potential still has the 1x1 moire period a_1.
    pot_fn = moire_hamiltonian.make_moire_potential(
        nspins=cfg.system.electrons,
        simulation_lattice=cfg.system.lattice,
        moire_lattice=cfg.system.make_local_energy_kwargs['moire_lattice'],
        moire_lattice_e=cfg.system.make_local_energy_kwargs['moire_lattice_e'],
        moire_lattice_h=cfg.system.make_local_energy_kwargs['moire_lattice_h'],
        v_e=cfg.system.make_local_energy_kwargs['v_e'],
        v_h=cfg.system.make_local_energy_kwargs['v_h'],
        phi_e=cfg.system.make_local_energy_kwargs['phi_e'],
        phi_h=cfg.system.make_local_energy_kwargs['phi_h'],
    )
    nelec = sum(cfg.system.electrons)
    rng = np.random.default_rng(42)
    pos = jnp.asarray(rng.uniform(-100.0, 100.0, size=(nelec, 2)), dtype=jnp.float32)
    a1 = jnp.asarray(prim_lattice[:, 0], dtype=jnp.float32)
    pos_shift_a1 = pos.at[-1].add(a1)
    np.testing.assert_allclose(
        pot_fn(pos.reshape(-1)),
        pot_fn(pos_shift_a1.reshape(-1)),
        atol=1e-5,
        rtol=1e-5,
    )

    # 3. Verify feature layer periodicity:
    # - Shifting by 1x1 moire vector a_1 MUST change ae_features (distinguishes
    #   the 3 moire sublattice sites inside the sqrt(3) x sqrt(3) unit cell!).
    # - Shifting by enlarged unit cell vector A_1 = 2 a_1 + a_2 MUST leave
    #   ae_features invariant while changing ee_features.
    # - Shifting by supercell vector L_sim[:, 0] MUST leave both ae_features and
    #   ee_features invariant.
    fl = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=cfg.system.electrons,
        ndim=2,
        **cfg.network.make_feature_layer_kwargs,
    )
    _, fl_params = fl.init()
    atoms = jnp.zeros((1, 2), dtype=jnp.float32)

    ae0, ee0, r_ae0, r_ee0 = networks.construct_input_features(pos, atoms, ndim=2)
    ae_feat_0, ee_feat_0 = fl.apply(
        ae=ae0, r_ae=r_ae0, ee=ee0, r_ee=r_ee0, **fl_params
    )

    # Shift particle 0 by 1x1 moire vector a_1 -> ae_features must change!
    pos_a1 = pos.at[0].add(a1)
    ae1, ee1, r_ae1, r_ee1 = networks.construct_input_features(
        pos_a1, atoms, ndim=2
    )
    ae_feat_a1, _ = fl.apply(
        ae=ae1, r_ae=r_ae1, ee=ee1, r_ee=r_ee1, **fl_params
    )
    self.assertGreater(
        float(jnp.max(jnp.abs(ae_feat_0[0] - ae_feat_a1[0]))), 0.1
    )

    # Shift particle 0 by enlarged unit-cell vector A_1 -> ae_features invariant!
    cap_a1 = jnp.asarray(unit_lattice[:, 0], dtype=jnp.float32)
    pos_uc = pos.at[0].add(cap_a1)
    ae_u, ee_u, r_ae_u, r_ee_u = networks.construct_input_features(
        pos_uc, atoms, ndim=2
    )
    ae_feat_uc, ee_feat_uc = fl.apply(
        ae=ae_u, r_ae=r_ae_u, ee=ee_u, r_ee=r_ee_u, **fl_params
    )
    np.testing.assert_allclose(ae_feat_0, ae_feat_uc, atol=1e-4, rtol=1e-4)
    self.assertGreater(
        float(jnp.max(jnp.abs(ee_feat_0 - ee_feat_uc))), 1e-2
    )

    # Shift particle 0 by supercell vector L_sim[:, 0] -> both invariant!
    sim_vec = jnp.asarray(sim_lattice[:, 0], dtype=jnp.float32)
    pos_sc = pos.at[0].add(sim_vec)
    ae_s, ee_s, r_ae_s, r_ee_s = networks.construct_input_features(
        pos_sc, atoms, ndim=2
    )
    ae_feat_sc, ee_feat_sc = fl.apply(
        ae=ae_s, r_ae=r_ae_s, ee=ee_s, r_ee=r_ee_s, **fl_params
    )
    np.testing.assert_allclose(ae_feat_0, ae_feat_sc, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ee_feat_0, ee_feat_sc, atol=1e-4, rtol=1e-4)

  def test_ferminet_forward_pass_with_enlarged_unit_cell_tiling(self):
    """Tests FermiNet ansatz with a 2x2 supercell of sqrt(3)xsqrt(3) unit cells."""
    cfg = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config()
    cfg.system.moire.nspins = (2, 2, 2, 2)
    cfg.system.moire.supercell_matrix = ((2, 0), (0, 2))
    cfg.system.unit_cell_matrix = ((2, -1), (1, 1))
    cfg.system.supercell_in_unit_cells = True
    cfg = base_config.resolve(cfg)

    fl = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=cfg.system.electrons,
        ndim=2,
        **cfg.network.make_feature_layer_kwargs,
    )
    kpoints = pbc_envelopes.make_kpoints(
        cfg.system.lattice, cfg.system.electrons, ndim=2
    )
    envelope = pbc_envelopes.make_multiwave_envelope(kpoints)
    charges = jnp.zeros((1,), dtype=jnp.float32)
    atoms = jnp.zeros((1, 2), dtype=jnp.float32)
    net = networks.make_fermi_net(
        nspins=cfg.system.electrons,
        charges=charges,
        ndim=2,
        determinants=2,
        hidden_dims=((16, 8), (16, 8)),
        feature_layer=fl,
        envelope=envelope,
        full_det=True,
    )
    params = net.init(jax.random.PRNGKey(0))
    nelec = sum(cfg.system.electrons)
    pos = jax.random.uniform(
        jax.random.PRNGKey(1), shape=(nelec * 2,), minval=-80.0, maxval=80.0
    )
    spins = jnp.zeros((nelec,), dtype=jnp.float32)
    sign_0, logpsi_0 = net.apply(params, pos, spins, atoms, charges)
    self.assertTrue(bool(jnp.isfinite(logpsi_0)))

    sim_shift = jnp.asarray(cfg.system.lattice[:, 0], dtype=jnp.float32)
    pos_shifted = pos.at[:2].add(sim_shift)
    sign_s, logpsi_s = net.apply(params, pos_shifted, spins, atoms, charges)
    np.testing.assert_allclose(sign_0, sign_s, atol=1e-5)
    np.testing.assert_allclose(logpsi_0, logpsi_s, atol=1e-4, rtol=1e-4)

  def test_moire_energy_scale_value_and_validation(self):
    """Tests hbar^2 / [2 * (m_e + m_h) * a_M^2] in Hartree atomic units."""
    mass_e = 0.70
    mass_h = 0.45
    a_m = 151.18
    expected = 1.0 / (2.0 * (mass_e + mass_h) * (a_m ** 2))
    self.assertAlmostEqual(expected, 1.902319597434049e-5, places=18)

    scale_period = geometry.moire_energy_scale(
        mass_e=mass_e, mass_h=mass_h, moire_period=a_m
    )
    self.assertAlmostEqual(scale_period, expected, places=18)

    prim = geometry.triangular_lattice(a_m)
    scale_prim = geometry.moire_energy_scale(
        mass_e=mass_e, mass_h=mass_h, primitive_lattice=prim
    )
    self.assertAlmostEqual(scale_prim, expected, places=18)

    scale_masses_2 = geometry.moire_energy_scale(
        masses=(mass_e, mass_h), moire_period=a_m
    )
    self.assertAlmostEqual(scale_masses_2, expected, places=18)

    scale_masses_4 = geometry.moire_energy_scale(
        masses=(mass_e, mass_e, mass_h, mass_h), moire_period=a_m
    )
    self.assertAlmostEqual(scale_masses_4, expected, places=18)

    # Check invalid inputs raise ValueError.
    for bad_val in (0.0, -0.5, float('nan'), float('inf'), True, False):
      with self.assertRaises(ValueError):
        geometry.moire_energy_scale(
            mass_e=bad_val, mass_h=mass_h, moire_period=a_m
        )
      with self.assertRaises(ValueError):
        geometry.moire_energy_scale(
            mass_e=mass_e, mass_h=bad_val, moire_period=a_m
        )
      with self.assertRaises(ValueError):
        geometry.moire_energy_scale(
            mass_e=mass_e, mass_h=mass_h, moire_period=bad_val
        )
    with self.assertRaises(ValueError):
      geometry.moire_energy_scale(
          masses=(0.70, 0.65, 0.45, 0.45), moire_period=a_m
      )
    with self.assertRaises(ValueError):
      geometry.moire_energy_scale(masses=(0.70, 0.45, 0.45), moire_period=a_m)
    with self.assertRaises(ValueError):
      geometry.moire_energy_scale(mass_e=mass_e, mass_h=mass_h)

  def test_resolve_moire_energy_scale_across_configs(self):
    """Tests resolution of cfg.optim.energy_scale on moire configs."""
    expected = 1.0 / (2.0 * (0.70 + 0.45) * (151.18 ** 2))

    # 1. Default moire bilayer configs keep scale_by_moire_energy=False and
    # energy_scale=None (1.0) for 100% backward compatibility.
    for cfg_mod in (
        WSe2WS2_MoSe2_vh13_ve13_3shell_scaled,
        WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice,
    ):
      cfg_default = cfg_mod.get_config()
      self.assertFalse(cfg_default.optim.scale_by_moire_energy)
      cfg_default = base_config.resolve(cfg_default)
      self.assertIsNone(cfg_default.optim.energy_scale)
      self.assertEqual(geometry.resolve_moire_energy_scale(cfg_default), 1.0)

      # Opting in via scale_by_moire_energy=True resolves to E_M.
      cfg = cfg_mod.get_config()
      cfg.optim.scale_by_moire_energy = True
      cfg = base_config.resolve(cfg)
      self.assertAlmostEqual(float(cfg.optim.energy_scale), expected, places=18)

    # 2. Enlarged sqrt(3) x sqrt(3) crystal unit cell still uses 1x1 moire period a_M.
    cfg_uc = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config()
    cfg_uc.optim.scale_by_moire_energy = True
    cfg_uc.system.unit_cell_matrix = 'sqrt3'
    cfg_uc.system.supercell_in_unit_cells = False
    cfg_uc = base_config.resolve(cfg_uc)
    self.assertAlmostEqual(float(cfg_uc.optim.energy_scale), expected, places=18)

    # 3. Updating mass_e, mass_h, or lattice_constant before resolve updates E_M.
    cfg_custom = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config()
    cfg_custom.optim.scale_by_moire_energy = True
    cfg_custom.system.moire.mass_e = 0.50
    cfg_custom.system.moire.mass_h = 0.50
    cfg_custom.system.moire.lattice_constant = 100.0
    cfg_custom = base_config.resolve(cfg_custom)
    self.assertAlmostEqual(
        float(cfg_custom.optim.energy_scale),
        1.0 / (2.0 * 1.0 * (100.0 ** 2)),
        places=18,
    )

    # 4. Disabling scale_by_moire_energy leaves energy_scale = None (resolves to 1.0).
    cfg_off = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config()
    cfg_off.optim.scale_by_moire_energy = False
    cfg_off = base_config.resolve(cfg_off)
    self.assertIsNone(cfg_off.optim.energy_scale)
    self.assertEqual(geometry.resolve_moire_energy_scale(cfg_off), 1.0)

    # 5. Explicit cfg.optim.energy_scale override is preserved.
    cfg_exp = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config()
    cfg_exp.optim.energy_scale = 2.5e-4
    cfg_exp = base_config.resolve(cfg_exp)
    self.assertAlmostEqual(float(cfg_exp.optim.energy_scale), 2.5e-4, places=18)

    # 6. Non-moire default config leaves energy_scale as None and resolves to 1.0.
    cfg_mol = base_config.default()
    cfg_mol.system.molecule = []
    cfg_mol = base_config.resolve(cfg_mol)
    self.assertIsNone(cfg_mol.optim.energy_scale)
    self.assertEqual(geometry.resolve_moire_energy_scale(cfg_mol), 1.0)

  @parameterized.parameters(
      (False, 0.0, False),
      (False, 2.0, False),
      (False, 2.0, True),
      (True, 0.0, False),
      (True, 2.0, False),
  )
  def test_make_loss_scales_objective_variance_and_jvp_gradients(
      self, complex_output, clip_local_energy, clip_from_median
  ):
    """Tests that make_loss scales loss, aux_data, and custom JVP gradients by 1/E_M."""
    e_m = geometry.moire_energy_scale(
        mass_e=0.70, mass_h=0.45, moire_period=151.18
    )
    batch_size = 8
    ndim = 2
    nelec = 4

    def mock_network(params, pos, spins, atoms, charges):
      del spins, atoms, charges
      val = jnp.sum(params['w'] * pos) + 0.05 * jnp.sum((params['w'] * pos) ** 2)
      if complex_output:
        phase = jnp.sum(params['w_phase'] * pos)
        return val + 1.0j * phase
      return val

    def mock_local_energy(params, key, data):
      del key
      # Synthetic local energies around -0.011124 Ha with walker variation.
      base_e = -0.011124 + 1e-4 * jnp.sum(params['w'] * data.positions)
      if complex_output:
        imag_e = 2e-5 * jnp.sum(params['w_phase'] * data.positions)
        return base_e + 1.0j * imag_e, None
      return base_e, None

    loss_unscaled_fn = qmc_loss_functions.make_loss(
        network=mock_network,
        local_energy=mock_local_energy,
        clip_local_energy=clip_local_energy,
        clip_from_median=clip_from_median,
        center_at_clipped_energy=True,
        complex_output=complex_output,
        energy_scale=1.0,
    )
    loss_scaled_fn = qmc_loss_functions.make_loss(
        network=mock_network,
        local_energy=mock_local_energy,
        clip_local_energy=clip_local_energy,
        clip_from_median=clip_from_median,
        center_at_clipped_energy=True,
        complex_output=complex_output,
        energy_scale=e_m,
    )

    rng = jax.random.PRNGKey(123)
    k_pos, k_eval = jax.random.split(rng)
    positions = jax.random.normal(
        k_pos, shape=(batch_size, nelec * ndim), dtype=jnp.float32
    )
    data = networks.FermiNetData(
        positions=positions,
        spins=jnp.zeros((batch_size, nelec), dtype=jnp.float32),
        atoms=jnp.zeros((batch_size, 1, ndim), dtype=jnp.float32),
        charges=jnp.zeros((batch_size, 1), dtype=jnp.float32),
    )
    params = {
        'w': jnp.linspace(0.1, 0.8, nelec * ndim, dtype=jnp.float32),
        'w_phase': jnp.linspace(-0.3, 0.4, nelec * ndim, dtype=jnp.float32),
    }

    val_grad_unscaled = jax.value_and_grad(
        loss_unscaled_fn, argnums=0, has_aux=True
    )
    val_grad_scaled = jax.value_and_grad(
        loss_scaled_fn, argnums=0, has_aux=True
    )

    (loss_u, aux_u), grad_u = val_grad_unscaled(params, k_eval, data)
    (loss_s, aux_s), grad_s = val_grad_scaled(params, k_eval, data)

    self.assertEqual(loss_u.dtype, loss_s.dtype)
    np.testing.assert_allclose(loss_s, loss_u / e_m, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        aux_s.energy, aux_u.energy / e_m, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        aux_s.local_energy, aux_u.local_energy / e_m, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        aux_s.clipped_energy, aux_u.clipped_energy / e_m, rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        aux_s.variance, aux_u.variance / (e_m ** 2), rtol=1e-4, atol=1e-4
    )
    for k in params:
      self.assertEqual(grad_u[k].dtype, grad_s[k].dtype)
      np.testing.assert_allclose(
          grad_s[k], grad_u[k] / e_m, rtol=1e-5, atol=1e-4
      )

    # Check invalid energy_scale raises ValueError.
    for bad_scale in (0.0, -1.0, float('nan'), float('inf'), True):
      with self.assertRaises(ValueError):
        qmc_loss_functions.make_loss(
            network=mock_network,
            local_energy=mock_local_energy,
            energy_scale=bad_scale,
        )

  def test_make_wqmc_and_overlap_loss_energy_scale(self):
    """Tests that make_wqmc_loss and make_energy_overlap_loss support energy_scale."""
    e_m = geometry.moire_energy_scale(
        mass_e=0.70, mass_h=0.45, moire_period=151.18
    )
    batch_size = 4
    ndim = 2
    nelec = 2

    def mock_net(params, pos, spins, atoms, charges):
      del spins, atoms, charges
      return jnp.sum(params['w'] * pos)

    def mock_el(params, key, data):
      del key
      return -0.01 + 1e-4 * jnp.sum(params['w'] * data.positions), None

    wqmc_u = qmc_loss_functions.make_wqmc_loss(
        mock_net, mock_el, clip_local_energy=2.0, vmc_weight=0.5, energy_scale=1.0
    )
    wqmc_s = qmc_loss_functions.make_wqmc_loss(
        mock_net, mock_el, clip_local_energy=2.0, vmc_weight=0.5, energy_scale=e_m
    )
    rng = jax.random.PRNGKey(7)
    pos = jax.random.normal(rng, shape=(batch_size, nelec * ndim), dtype=jnp.float32)
    data = networks.FermiNetData(
        positions=pos,
        spins=jnp.zeros((batch_size, nelec), dtype=jnp.float32),
        atoms=jnp.zeros((batch_size, 1, ndim), dtype=jnp.float32),
        charges=jnp.zeros((batch_size, 1), dtype=jnp.float32),
    )
    params = {'w': jnp.linspace(0.1, 0.4, nelec * ndim, dtype=jnp.float32)}
    loss_u, aux_u = wqmc_u(params, rng, data)
    loss_s, aux_s = wqmc_s(params, rng, data)
    np.testing.assert_allclose(loss_s, loss_u / e_m, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(
        aux_s.variance, aux_u.variance / (e_m ** 2), rtol=1e-4, atol=1e-4
    )

  def test_moire_train_step_with_energy_scale(self):
    """Tests end-to-end moire KFAC training step with scaled objective."""
    with tempfile.TemporaryDirectory() as tmpdir_scaled, tempfile.TemporaryDirectory() as tmpdir_unscaled:
      def _make_test_cfg(save_path, scale_by_moire_energy):
        cfg = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config()
        cfg.system.moire.nspins = (1, 1, 1, 1)
        cfg.system.moire.supercell_matrix = ((2, 0), (0, 2))
        cfg.system.moire.ewald_gmax = 8
        cfg.network.ferminet.hidden_dims = ((8, 4),)
        cfg.network.determinants = 1
        cfg.batch_size = 8
        cfg.mcmc.burn_in = 1
        cfg.mcmc.steps = 1
        cfg.optim.iterations = 1
        cfg.optim.scale_by_moire_energy = scale_by_moire_energy
        cfg.debug.deterministic = True
        cfg.debug.check_nan = True
        cfg.log.save_path = save_path
        return base_config.resolve(cfg)

      cfg_scaled = _make_test_cfg(tmpdir_scaled, True)
      cfg_unscaled = _make_test_cfg(tmpdir_unscaled, False)

      expected_e_m = 1.0 / (2.0 * (0.70 + 0.45) * (151.18 ** 2))
      self.assertAlmostEqual(float(cfg_scaled.optim.energy_scale), expected_e_m, places=18)
      self.assertIsNone(cfg_unscaled.optim.energy_scale)
      self.assertAlmostEqual(geometry.resolve_moire_energy_scale(cfg_unscaled), 1.0, places=12)

      train.train(cfg_scaled)
      train.train(cfg_unscaled)

      with open(os.path.join(tmpdir_scaled, 'train_stats.csv')) as f_s:
        lines_s = [line.strip().split(',') for line in f_s if line.strip()]
      with open(os.path.join(tmpdir_unscaled, 'train_stats.csv')) as f_u:
        lines_u = [line.strip().split(',') for line in f_u if line.strip()]

      # Step 0 evaluates the identical initial wave function and MCMC walkers.
      energy_s = float(lines_s[1][1])
      energy_u = float(lines_u[1][1])
      var_s = float(lines_s[1][3])
      var_u = float(lines_u[1][3])
      np.testing.assert_allclose(energy_s, energy_u / expected_e_m, rtol=1e-4)
      np.testing.assert_allclose(var_s, var_u / (expected_e_m ** 2), rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
