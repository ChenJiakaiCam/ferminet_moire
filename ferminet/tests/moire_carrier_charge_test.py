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

"""Tests for per-layer carrier charges and electron-electron moire bilayers."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import jastrows
from ferminet import networks
from ferminet import train
from ferminet.configs import moire_bilayer_filling1
from ferminet.configs import moire_bilayer_vh13_ve0
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled
from ferminet.moire import ewald
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
import jax
import jax.numpy as jnp
import numpy as np


class BilayerCarrierChargeTest(parameterized.TestCase):

  @parameterized.parameters(
      moire_bilayer_filling1,
      WSe2WS2_MoSe2_vh13_ve13_3shell_scaled,
  )
  def test_default_moire_configs_unchanged_without_charge_options(
      self, cfg_module
  ):
    cfg_raw = cfg_module.get_config()
    expected_sim_lat = np.asarray(cfg_raw.system.lattice, dtype=np.float64)
    expected_le_keys = set(cfg_raw.system.make_local_energy_kwargs.keys())
    expected_fl_keys = set(cfg_raw.network.make_feature_layer_kwargs.keys())

    cfg_resolved = base_config.resolve(cfg_raw)
    np.testing.assert_allclose(cfg_resolved.system.lattice, expected_sim_lat)
    self.assertEqual(
        set(cfg_resolved.system.make_local_energy_kwargs.keys()),
        expected_le_keys,
    )
    self.assertNotIn(
        'layer_charges', cfg_resolved.system.make_local_energy_kwargs
    )
    self.assertEqual(
        set(cfg_resolved.network.make_feature_layer_kwargs.keys()),
        expected_fl_keys,
    )

  def test_resolve_layer_charges_formats_and_validation(self):
    self.assertEqual(geometry.resolve_layer_charges(), (-1, 1))
    self.assertEqual(geometry.resolve_layer_charges((-1, -1)), (-1, -1))
    self.assertEqual(geometry.resolve_layer_charges((-1, 1)), (-1, 1))
    self.assertEqual(geometry.resolve_layer_charges((1, -1)), (1, -1))
    self.assertEqual(geometry.resolve_layer_charges((1, 1)), (1, 1))
    self.assertEqual(geometry.resolve_layer_charges('ee'), (-1, -1))
    self.assertEqual(geometry.resolve_layer_charges('e-e'), (-1, -1))
    self.assertEqual(geometry.resolve_layer_charges('eh'), (-1, 1))
    self.assertEqual(geometry.resolve_layer_charges('he'), (1, -1))
    self.assertEqual(geometry.resolve_layer_charges('hh'), (1, 1))
    self.assertEqual(geometry.resolve_layer_charges(('e', 'e')), (-1, -1))
    self.assertEqual(
        geometry.resolve_layer_charges(('electron', 'hole')), (-1, 1)
    )
    self.assertEqual(
        geometry.resolve_layer_charges((-1, -1, -1, -1)), (-1, -1)
    )
    self.assertEqual(geometry.resolve_layer_charges((-1, -1, 1, 1)), (-1, 1))
    self.assertEqual(
        geometry.resolve_layer_charges(charge_e='e', charge_h='e'), (-1, -1)
    )
    self.assertEqual(
        geometry.resolve_moire_charges_from_config(
            {'charges': np.array([-1, -1]), 'charge_h': -1}
        ),
        (-1, -1),
    )
    self.assertEqual(
        geometry.resolve_moire_charges_from_config(
            {'charges': (-1, 1), 'charge_2': -1}
        ),
        (-1, -1),
    )
    self.assertEqual(
        geometry.resolve_moire_charges_from_config(
            {'charges': np.array([-1, 1]), 'charge_2': -1}
        ),
        (-1, -1),
    )

    # Reject mixed carrier types within a single layer.
    with self.assertRaises(ValueError):
      geometry.resolve_layer_charges((-1, 1, -1, -1))
    with self.assertRaises(ValueError):
      geometry.resolve_layer_charges((-1, -1, -1, 1))
    # Reject invalid charge values and conflicting specifications.
    with self.assertRaises(ValueError):
      geometry.resolve_layer_charges((0, 1))
    with self.assertRaises(ValueError):
      geometry.resolve_layer_charges((True, -1))
    with self.assertRaises(ValueError):
      geometry.resolve_layer_charges((-1, 1), charge_h=-1)
    with self.assertRaises(ValueError):
      geometry.resolve_moire_charges_from_config(
          {'layer_charges': 'ee', 'carrier_types': 'eh'}
      )

  @parameterized.parameters(
      {'charges': (-1, 1), 'd': 0.0},
      {'charges': (-1, -1), 'd': 0.0},
      {'charges': (1, -1), 'd': 3.5},
      {'charges': (-1, -1), 'd': 3.5},
      {'charges': (1, 1), 'd': 3.5},
  )
  def test_ewald_alpha_invariance_and_interlayer_sign_for_all_charges(
      self, charges, d
  ):
    lat = geometry.supercell_lattice(
        geometry.triangular_lattice(12.0), [[2, 0], [0, 2]]
    )
    nspins = (2, 1, 1, 2)
    n_e, _ = geometry.layer_particle_counts(nspins)
    key = jax.random.PRNGKey(23)
    frac = jax.random.uniform(key, shape=(sum(nspins), 2))
    pos = (frac @ jnp.asarray(lat).T).reshape(-1)
    pos_e = pos.reshape(-1, 2)[:n_e]
    pos_h = pos.reshape(-1, 2)[n_e:]

    ew1 = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=4.5,
        ewald_gmax=80,
        nlatvec=2,
        alpha=0.22,
        charges=charges,
    )
    ew2 = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=4.5,
        ewald_gmax=80,
        nlatvec=2,
        alpha=0.44,
        charges=charges,
    )
    c1 = ew1.components(pos_e, pos_h)
    c2 = ew2.components(pos_e, pos_h)
    for u1, u2 in zip(c1, c2):
      np.testing.assert_allclose(u1, u2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        ew1.energy(pos), ew2.energy(pos), atol=1e-4, rtol=1e-4
    )

    # Compare interlayer interaction sign against opposite-charge bilayer with
    # per-layer neutralizing backgrounds (include_capacitor=False).
    ew_eh_nocap = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=4.5,
        ewald_gmax=80,
        nlatvec=2,
        include_capacitor=False,
        charges=(-1, 1),
    )
    ew_cur_nocap = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=4.5,
        ewald_gmax=80,
        nlatvec=2,
        include_capacitor=False,
        charges=charges,
    )
    _, _, u_eh_ref = ew_eh_nocap.components(pos_e, pos_h)
    _, _, u_eh_cur = ew_cur_nocap.components(pos_e, pos_h)
    expected_sign = -float(charges[0] * charges[1])
    np.testing.assert_allclose(
        u_eh_cur, expected_sign * u_eh_ref, atol=1e-5, rtol=1e-5
    )

  def test_electron_electron_bilayer_d0_matches_single_layer_coplanar_energy(
      self,
  ):
    lat = geometry.triangular_lattice(18.0)
    nspins = (2, 1, 1, 1)
    n_total = sum(nspins)
    key = jax.random.PRNGKey(31)
    coords = jax.random.uniform(key, shape=(n_total, 2)) @ jnp.asarray(lat).T

    ew_ee_bilayer = ewald.BilayerEwaldSum(
        lat,
        nspins=nspins,
        interlayer_distance=0.0,
        dielectric_constant=5.0,
        charges=(-1, -1),
    )
    ew_single_layer = ewald.BilayerEwaldSum(
        lat,
        nspins=(n_total, 0),
        interlayer_distance=0.0,
        dielectric_constant=5.0,
        include_capacitor=False,
        charges=(-1, -1),
    )
    np.testing.assert_allclose(
        ew_ee_bilayer.energy(coords.reshape(-1)),
        ew_single_layer.energy(coords.reshape(-1)),
        atol=1e-5,
        rtol=1e-5,
    )

  def test_bilayer_jastrow_and_config_for_electron_electron_bilayer(self):
    lat = jnp.asarray(geometry.triangular_lattice(20.0))
    # At d = 0, interlayer cusp must flip sign between e-h (-1, 1) and e-e (-1, -1).
    init_eh, apply_eh = jastrows.make_bilayer_jastrow(
        lattice=lat,
        ndim=2,
        mass_e=0.5,
        mass_h=0.5,
        dielectric_constant=4.0,
        interlayer_distance=0.0,
        charges=(-1, 1),
    )
    init_ee, apply_ee = jastrows.make_bilayer_jastrow(
        lattice=lat,
        ndim=2,
        mass_e=0.5,
        mass_h=0.5,
        dielectric_constant=4.0,
        interlayer_distance=0.0,
        charges=(-1, -1),
    )
    params_eh = init_eh()
    params_ee = init_ee()
    pos = jnp.array([[1.0, 2.0], [4.5, -1.5]])
    ee_disp = pos[:, None, :] - pos[None, :, :]
    val_eh = float(apply_eh(ee_disp, params_eh, nspins=(1, 1)))
    val_ee = float(apply_ee(ee_disp, params_ee, nspins=(1, 1)))
    self.assertAlmostEqual(val_ee, -val_eh, places=6)
    self.assertLess(val_ee, 0.0)  # Repulsive Jastrow factor u(r) = -c a^2 / (a + r) < 0

    # Verify moire bilayer configs and local_energy with electrons in both layers.
    for cfg_mod in (
        moire_bilayer_filling1,
        WSe2WS2_MoSe2_vh13_ve13_3shell_scaled,
    ):
      cfg = cfg_mod.get_config()
      cfg.system.moire.charges = (-1, -1)
      cfg.system.moire.n_layer1_up = 1
      cfg.system.moire.n_layer1_down = 0
      cfg.system.moire.n_layer2_up = 1
      cfg.system.moire.n_layer2_down = 0
      cfg.system.moire.supercell_matrix = ((1, 0), (0, 1))
      cfg = base_config.resolve(cfg)

      kwargs = cfg.system.make_local_energy_kwargs
      self.assertEqual(kwargs['layer_charges'], (-1, -1))
      self.assertFalse(kwargs['include_capacitor'])
      self.assertEqual(cfg.network.jastrow_kwargs['charges'], (-1, -1))

      def dummy_wf(params, pos_flat, spins, atoms, charges):
        del params, spins, atoms, charges
        return jnp.ones((), dtype=pos_flat.dtype), -0.001 * jnp.sum(pos_flat**2)

      e_l_ee = moire_hamiltonian.local_energy(
          f=dummy_wf,
          charges=jnp.zeros((1,)),
          nspins=cfg.system.electrons,
          use_scan=False,
          complex_output=False,
          **kwargs,
      )
      data = networks.FermiNetData(
          positions=jnp.array([10.0, -5.0, -12.0, 8.0], dtype=jnp.float32),
          spins=jnp.array([0.0, 2.0], dtype=jnp.float32),
          atoms=jnp.zeros((1, 2), dtype=jnp.float32),
          charges=jnp.zeros((1,), dtype=jnp.float32),
      )
      total_ee, _ = e_l_ee({}, jax.random.PRNGKey(0), data)
      self.assertTrue(bool(jnp.isfinite(total_ee)))

  def test_charged_capacitor_validation_and_end_to_end_ee_train_step(self):
    lat = geometry.triangular_lattice(15.0)
    # Opposite-charge bilayer with imbalanced counts and d > 0 raises ValueError
    # when include_capacitor=True and allow_charged_capacitor=False.
    with self.assertRaises(ValueError):
      ewald.BilayerEwaldSum(
          lat,
          nspins=(2, 1, 1, 1),
          interlayer_distance=3.0,
          charges=(-1, 1),
      )
    # Same-charge bilayer defaults to include_capacitor=False, so equal or
    # unequal layer counts work without error; forcing include_capacitor=True
    # without allow_charged_capacitor=True raises ValueError.
    ew_ee_default = ewald.BilayerEwaldSum(
        lat,
        nspins=(1, 1, 1, 1),
        interlayer_distance=3.0,
        charges=(-1, -1),
    )
    self.assertFalse(ew_ee_default.include_capacitor)
    with self.assertRaises(ValueError):
      ewald.BilayerEwaldSum(
          lat,
          nspins=(1, 1, 1, 1),
          interlayer_distance=3.0,
          charges=(-1, -1),
          include_capacitor=True,
      )

    # End-to-end training step with electron-electron bilayer ('ee'), enlarged
    # sqrt(3) x sqrt(3) unit cell, pair MCMC, and moire energy objective scaling.
    with tempfile.TemporaryDirectory() as tmpdir_ee:
      cfg = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config()
      cfg.system.moire.charges = 'ee'
      cfg.system.moire.n_layer1 = 2
      cfg.system.moire.n_layer2 = 2
      cfg.system.moire.supercell_matrix = ((1, 0), (0, 1))
      cfg.system.unit_cell_matrix = 'sqrt3'
      cfg.system.supercell_in_unit_cells = True
      cfg.system.moire.ewald_gmax = 8
      cfg.network.ferminet.hidden_dims = ((8, 4),)
      cfg.network.determinants = 1
      cfg.network.jastrow = 'bilayer'
      cfg.batch_size = 8
      cfg.mcmc.burn_in = 1
      cfg.mcmc.steps = 2
      cfg.mcmc.pair_prob = 0.5
      cfg.mcmc.pair_min_fraction = 0.0
      cfg.optim.iterations = 1
      cfg.optim.scale_by_moire_energy = True
      cfg.debug.deterministic = True
      cfg.debug.check_nan = True
      cfg.log.save_path = tmpdir_ee
      cfg = base_config.resolve(cfg)

      self.assertEqual(cfg.system.electrons, (1, 1, 1, 1))
      self.assertEqual(
          cfg.system.make_local_energy_kwargs['layer_charges'], (-1, -1)
      )
      self.assertFalse(
          cfg.system.make_local_energy_kwargs['include_capacitor']
      )
      train.train(cfg)
      with open(os.path.join(tmpdir_ee, 'train_stats.csv')) as f_ee:
        lines_ee = [line.strip().split(',') for line in f_ee if line.strip()]
      self.assertGreaterEqual(len(lines_ee), 2)
      self.assertNotIn('pmove_pair', lines_ee[0])
      self.assertTrue(np.isfinite(float(lines_ee[1][1])))

  def test_cli_placeholders_locked_config_and_repurposed_configs(self):
    # 1. Repurposing moire_bilayer_vh13_ve0 (which has allow_charged_capacitor=True
    # and n_electrons=0) with n_holes=0 and n_layer1=4, n_layer2=4, layer_charges='ee'.
    cfg = moire_bilayer_vh13_ve0.get_config()
    cfg.system.layer_charges = 'ee'
    cfg.system.moire.n_holes = 0
    cfg.system.moire.n_layer1 = 4
    cfg.system.moire.n_layer2 = 4
    cfg.lock()
    cfg = base_config.resolve(cfg)
    self.assertTrue(cfg.is_locked)
    self.assertEqual(cfg.system.electrons, (2, 2, 2, 2))
    self.assertEqual(
        cfg.system.make_local_energy_kwargs['layer_charges'], (-1, -1)
    )
    self.assertFalse(cfg.system.make_local_energy_kwargs['include_capacitor'])

    # 2. Reject mixed intra-layer spin masses in 4-channel kinetic energy.
    with self.assertRaises(ValueError):
      moire_hamiltonian.local_kinetic_energy(
          f=lambda p, x, s, a, c: (jnp.ones(()), jnp.zeros(())),
          nspins=(1, 1, 1, 1),
          masses=(0.7, 0.45, 0.7, 0.45),
      )

    # 3. Custom make_local_energy_fn without layer_charges: allowed when only
    # one layer is occupied, raises ValueError when both layers are occupied
    # with non-default charges.
    cfg_single = moire_bilayer_vh13_ve0.get_config()
    cfg_single.system.make_local_energy_fn = 'ferminet.hamiltonian.local_energy'
    cfg_single.system.layer_charges = 'ee'
    cfg_single = base_config.resolve(cfg_single)
    self.assertNotIn(
        'layer_charges', cfg_single.system.make_local_energy_kwargs
    )

    cfg_two = moire_bilayer_filling1.get_config()
    cfg_two.system.make_local_energy_fn = 'ferminet.hamiltonian.local_energy'
    cfg_two.system.layer_charges = 'ee'
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_two)


if __name__ == '__main__':
  absltest.main()
