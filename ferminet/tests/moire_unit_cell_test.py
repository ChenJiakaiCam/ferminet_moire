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
from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import networks
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled
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


if __name__ == '__main__':
  absltest.main()
