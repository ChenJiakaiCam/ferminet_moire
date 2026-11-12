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

"""Tests for single-layer anisotropic moire potential, mass, and scaling."""

import copy
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import networks
from ferminet import train
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled
from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice
from ferminet.configs import WSe2WS2_MoSe2_vh1_ve0
from ferminet.configs import anisotropic_moire_bilayer
from ferminet.configs import moire_bilayer_filling1
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
from ferminet.pbc import envelopes as pbc_envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
from ferminet.utils import system
from ferminet.utils import units
import jax
import jax.numpy as jnp
import numpy as np


SCREENSHOT_GVECS_NM_INV = (
    (-0.487, 0.204),
    (0.344, 0.269),
    (0.143, -0.474),
)


class AnisotropicMoireTest(parameterized.TestCase):

  def test_backward_compatibility_all_existing_configs(self):
    """When anisotropic=False (default), existing configs resolve identically."""
    base_cfg = base_config.default()
    self.assertFalse(base_cfg.system.anisotropic)

    config_factories = (
        WSe2WS2_MoSe2_vh13_ve13_3shell_scaled_pairlattice.get_config,
        WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config,
        WSe2WS2_MoSe2_vh1_ve0.get_config,
        moire_bilayer_filling1.get_config,
    )
    for get_cfg in config_factories:
      cfg_raw = get_cfg()
      self.assertFalse(cfg_raw.system.anisotropic)
      expected_lat = np.asarray(cfg_raw.system.lattice).copy()
      expected_le = copy.deepcopy(dict(cfg_raw.system.make_local_energy_kwargs))
      expected_fl = copy.deepcopy(
          dict(cfg_raw.network.make_feature_layer_kwargs)
      )
      expected_j = copy.deepcopy(dict(cfg_raw.network.jastrow_kwargs))
      expected_pair = (
          None
          if cfg_raw.mcmc.pair_lattice is None
          else np.asarray(cfg_raw.mcmc.pair_lattice).copy()
      )

      cfg_res = base_config.resolve(cfg_raw)
      self.assertFalse(cfg_res.system.anisotropic)
      np.testing.assert_allclose(cfg_res.system.lattice, expected_lat)
      self.assertEqual(
          set(cfg_res.system.make_local_energy_kwargs.keys()),
          set(expected_le.keys()),
      )
      self.assertNotIn('anisotropic', cfg_res.system.make_local_energy_kwargs)
      self.assertNotIn('mass_tensor_h', cfg_res.system.make_local_energy_kwargs)
      self.assertEqual(
          set(cfg_res.network.make_feature_layer_kwargs.keys()),
          set(expected_fl.keys()),
      )
      self.assertEqual(dict(cfg_res.network.jastrow_kwargs), expected_j)
      if expected_pair is None:
        self.assertIsNone(cfg_res.mcmc.pair_lattice)
      else:
        np.testing.assert_allclose(cfg_res.mcmc.pair_lattice, expected_pair)

  def test_anisotropic_options_rejected_when_flag_false(self):
    """Setting anisotropic fields while anisotropic=False raises ValueError."""
    cfg = WSe2WS2_MoSe2_vh1_ve0.get_config()
    with cfg.ignore_type():
      cfg.system.moire.moire_gvectors = SCREENSHOT_GVECS_NM_INV
    with self.assertRaises(ValueError):
      base_config.resolve(cfg)

    cfg_mt = WSe2WS2_MoSe2_vh1_ve0.get_config()
    with cfg_mt.ignore_type():
      cfg_mt.system.moire.mass_tensor_h = ((0.4, 0.0), (0.0, 0.6))
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_mt)

    with self.assertRaises(ValueError):
      moire_hamiltonian.make_moire_potential(
          nspins=(1, 1),
          simulation_lattice=np.eye(2) * 100.0,
          v_h=-1e-4,
          anisotropic=False,
          moire_gvectors=SCREENSHOT_GVECS_NM_INV,
          gvector_units='nm^-1',
      )

    with self.assertRaises(ValueError):
      geometry.moire_energy_scale(
          mass_e=0.7,
          mass_h=0.45,
          moire_period=150.0,
          anisotropic=False,
          mass_tensor_h=((0.4, 0.0), (0.0, 0.5)),
      )

  def test_screenshot_gvectors_geometry_and_periodicity(self):
    """Tests unit conversion, rounded triad closure, LLL cell, and PBC."""
    g_bohr = geometry.validate_anisotropic_gvectors(
        SCREENSHOT_GVECS_NM_INV, gvector_units='nm^-1'
    )
    self.assertEqual(g_bohr.shape, (3, 2))
    # First two rows convert via 0.1 * ANGSTROM_BOHR; G_3 closes triad.
    scale_nm = 0.1 * units.ANGSTROM_BOHR
    np.testing.assert_allclose(
        g_bohr[:2],
        np.asarray(SCREENSHOT_GVECS_NM_INV[:2]) * scale_nm,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        g_bohr[0] + g_bohr[1] + g_bohr[2], np.zeros(2), atol=1e-16
    )

    # Angstrom^-1 conversion is 10x larger than nm^-1 for the same numbers.
    g_ang = geometry.convert_gvectors_to_bohr_inv(
        SCREENSHOT_GVECS_NM_INV, gvector_units='angstrom^-1'
    )
    np.testing.assert_allclose(
        g_ang,
        np.asarray(SCREENSHOT_GVECS_NM_INV) * units.ANGSTROM_BOHR,
        rtol=1e-14,
    )

    # Large closure error (> 1%) or collinear (G_1, G_2) must fail.
    with self.assertRaises(ValueError):
      geometry.validate_anisotropic_gvectors(
          ((-0.487, 0.204), (0.344, 0.269), (0.200, -0.474)),
          gvector_units='nm^-1',
      )
    with self.assertRaises(ValueError):
      geometry.validate_anisotropic_gvectors(
          ((1.0, 0.0), (2.0, 0.0), (-3.0, 0.0)),
          gvector_units='bohr^-1',
      )

    prim = geometry.primitive_lattice_from_gvectors(
        SCREENSHOT_GVECS_NM_INV, gvector_units='nm^-1'
    )
    self.assertGreater(float(np.linalg.det(prim)), 0.0)
    norms = np.linalg.norm(prim, axis=0)
    self.assertNotAlmostEqual(float(norms[0]), float(norms[1]), places=2)

    sim = geometry.supercell_lattice(prim, np.array([[3, 0], [0, 2]]))
    windings = geometry.validate_gvectors_commensurate(g_bohr, sim)
    self.assertEqual(windings.shape, (3, 2))

    # Test V(r) = -(v_q / 4) sum_{i=1}^3 cos(G_i . r) via v_h = -v_q / 8.
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      v_q = 4.0e-3
      pot_fn = moire_hamiltonian.make_moire_potential(
          nspins=(1, 2),
          simulation_lattice=sim,
          moire_lattice=prim,
          v_e=0.0,
          v_h=-v_q / 8.0,
          phi_h=0.0,
          anisotropic=True,
          moire_gvectors_h=SCREENSHOT_GVECS_NM_INV,
          gvector_units='nm^-1',
      )
      rng = np.random.default_rng(42)
      pos = rng.normal(scale=50.0, size=(3, 2))
      e_base = float(pot_fn(jnp.asarray(pos.reshape(-1), dtype=jnp.float64)))

      # Direct analytical check on the 2 hole coordinates.
      expected = float(
          -(v_q / 4.0) * np.sum(np.cos(pos[1:] @ g_bohr.T))
      )
      self.assertAlmostEqual(e_base, expected, places=14)

      # Exact periodicity under primitive and supercell hole shifts.
      for shift in (
          prim[:, 0],
          prim[:, 1],
          2.0 * prim[:, 0] - 3.0 * prim[:, 1],
          sim[:, 0],
          sim[:, 1],
      ):
        pos_shifted = pos.copy()
        pos_shifted[1] += shift
        e_shifted = float(
            pot_fn(jnp.asarray(pos_shifted.reshape(-1), dtype=jnp.float64))
        )
        self.assertAlmostEqual(e_base, e_shifted, places=14)

      # Per-harmonic 3-tuple amplitudes and phases.
      v_tuple = (-1.1e-4, -2.3e-4, -0.7e-4)
      phi_tuple = (0.2, -0.5, 1.1)
      pot_tuple_fn = moire_hamiltonian.make_moire_potential(
          nspins=(0, 2),
          simulation_lattice=sim,
          moire_lattice=prim,
          v_h=v_tuple,
          phi_h=phi_tuple,
          offset_h=(3.0, -4.0),
          anisotropic=True,
          moire_gvectors=g_bohr,
      )
      holes_only = pos[1:]
      val_tuple = float(
          pot_tuple_fn(jnp.asarray(holes_only.reshape(-1), dtype=jnp.float64))
      )
      proj = (holes_only - np.array([[3.0, -4.0]])) @ g_bohr.T
      cos_sum = np.sum(np.cos(proj + phi_tuple), axis=0)
      expected_tuple = float(2.0 * np.sum(np.asarray(v_tuple) * cos_sum))
      self.assertAlmostEqual(val_tuple, expected_tuple, places=14)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_isotropic_limit_of_anisotropic_mode_matches_exactly(self):
    """Equilateral G-vectors and M = m*I_2 reproduce isotropic mode."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      a_m = 151.18
      prim_iso = geometry.triangular_lattice(a_m)
      g_iso = np.asarray(geometry.moire_harmonic_gvectors(prim_iso))

      prim_from_g = geometry.primitive_lattice_from_gvectors(g_iso)
      self.assertAlmostEqual(
          geometry.average_lattice_period(prim_from_g), a_m, places=12
      )
      ratio = geometry.validate_commensurate(prim_iso, prim_from_g)
      self.assertEqual(abs(round(float(np.linalg.det(ratio)))), 1)

      sim = geometry.supercell_lattice(prim_iso, np.array([[2, 0], [0, 2]]))
      pot_iso = moire_hamiltonian.make_moire_potential(
          nspins=(1, 1),
          simulation_lattice=sim,
          moire_lattice=prim_iso,
          v_h=-5.512e-4,
          phi_h=0.4,
          anisotropic=False,
      )
      pot_aniso = moire_hamiltonian.make_moire_potential(
          nspins=(1, 1),
          simulation_lattice=sim,
          moire_lattice=prim_iso,
          v_h=-5.512e-4,
          phi_h=0.4,
          anisotropic=True,
          moire_gvectors_h=g_iso,
      )
      rng = np.random.default_rng(7)
      pos = jnp.asarray(rng.normal(scale=40.0, size=(4,)), dtype=jnp.float64)
      np.testing.assert_allclose(
          pot_iso(pos), pot_aniso(pos), rtol=1e-13, atol=1e-14
      )

      e_scale_iso = geometry.moire_energy_scale(
          mass_e=0.70,
          mass_h=0.45,
          primitive_lattice=prim_iso,
          anisotropic=False,
      )
      e_scale_aniso = geometry.moire_energy_scale(
          mass_e=0.70,
          primitive_lattice=prim_from_g,
          anisotropic=True,
          mass_tensor_h=((0.45, 0.0), (0.0, 0.45)),
      )
      self.assertAlmostEqual(e_scale_iso, e_scale_aniso, places=18)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_single_layer_and_spd_mass_tensor_constraints(self):
    """Validates single-layer potential/mass rules and SPD mass checks."""
    # Non-symmetric or non-SPD mass tensors must fail.
    with self.assertRaises(ValueError):
      geometry.validate_mass_tensor_2d([[0.4, 0.1], [0.2, 0.5]])
    with self.assertRaises(ValueError):
      geometry.validate_mass_tensor_2d([[0.4, 0.5], [0.5, 0.4]])  # det < 0
    with self.assertRaises(ValueError):
      geometry.validate_mass_tensor_2d([[0.0, 0.0], [0.0, 0.5]])  # eigval == 0

    # Default config represents electrons-only strained monolayer 1T'-ReSe2.
    cfg_def = base_config.resolve(anisotropic_moire_bilayer.get_config())
    self.assertEqual(tuple(cfg_def.system.electrons), (3, 3, 0, 0))
    le_def = cfg_def.system.make_local_energy_kwargs
    self.assertAlmostEqual(float(le_def['v_e']), -5.512e-4, places=12)
    self.assertAlmostEqual(float(le_def['v_h']), 0.0, places=12)
    self.assertAlmostEqual(
        float(le_def['dielectric_constant']), 3.0, places=12
    )
    self.assertTrue(bool(le_def['allow_charged_capacitor']))
    self.assertIsNotNone(le_def['moire_gvectors_e'])
    self.assertIsNone(le_def['moire_gvectors_h'])
    self.assertIsNone(le_def['mass_tensor_h'])
    mt_rese2 = np.asarray(le_def['mass_tensor_e'], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(mt_rese2)
    np.testing.assert_allclose(eigvals, (0.55, 3.52), rtol=1e-12, atol=1e-12)
    self.assertAlmostEqual(float(le_def['mass_e']), 2.035, places=12)
    prim_def = np.asarray(le_def['moire_lattice'], dtype=np.float64)
    a_short = prim_def[:, int(np.argmin(np.linalg.norm(prim_def, axis=0)))]
    theta_short = np.arctan2(a_short[1], a_short[0])
    theta_expect = theta_short + np.deg2rad(-7.4)
    e_light_expect = np.array([np.cos(theta_expect), np.sin(theta_expect)])
    self.assertAlmostEqual(
        abs(float(np.dot(eigvecs[:, 0], e_light_expect))), 1.0, places=12
    )

    # Both v_e and v_h non-zero when anisotropic=True must fail.
    cfg_both = anisotropic_moire_bilayer.get_config()
    cfg_both.system.moire.v_e = -1.0e-4
    cfg_both.system.moire.v_h = -1.0e-4
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_both)

    # Multi-shell length (e.g. 2 or 4) when anisotropic=True must fail.
    cfg_ms = anisotropic_moire_bilayer.get_config()
    cfg_ms.system.moire.v_e = (-1.0e-4, -2.0e-5)
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_ms)

    # Mass tensor on hole layer when moire potential is on electron layer fails.
    cfg_wrong_layer = anisotropic_moire_bilayer.get_config()
    cfg_wrong_layer.system.moire.mass_tensor_e = None
    cfg_wrong_layer.system.moire.mass_tensor_h = ((0.6, 0.0), (0.0, 0.8))
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_wrong_layer)

    # Hole-layer anisotropic configuration (v_e == 0, v_h != 0) also succeeds.
    cfg_h = anisotropic_moire_bilayer.get_config()
    cfg_h.system.moire.v_e = 0.0
    cfg_h.system.moire.v_h = -3.0e-4
    cfg_h.system.moire.mass_tensor_e = None
    cfg_h.system.moire.mass_tensor = ((0.60, 0.05), (0.05, 0.80))
    cfg_h = base_config.resolve(cfg_h)
    le_h = cfg_h.system.make_local_energy_kwargs
    self.assertIsNone(le_h['moire_gvectors_e'])
    self.assertIsNotNone(le_h['moire_gvectors_h'])
    self.assertIsNone(le_h['mass_tensor_e'])
    np.testing.assert_allclose(
        le_h['mass_tensor_h'], ((0.60, 0.05), (0.05, 0.80))
    )
    self.assertAlmostEqual(le_h['mass_h'], 0.70, places=12)

  def test_anisotropic_plane_wave_kinetic_energy_analytic(self):
    """Checks -1/2 del_i (M^{-1})_{ij} del_j on an exact plane-wave state."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      k_e = jnp.array([0.13, -0.09], dtype=jnp.float64)
      k_h = jnp.array([-0.21, 0.17], dtype=jnp.float64)
      alpha_e = jnp.array([0.03, -0.02], dtype=jnp.float64)
      alpha_h = jnp.array([-0.04, 0.05], dtype=jnp.float64)
      m_e = 0.70
      m_h_mat = np.array([[0.42, 0.09], [0.09, 0.68]], dtype=np.float64)
      inv_mh = np.linalg.inv(m_h_mat)

      def plane_wave_f(params, pos, spins, atoms, charges):
        del params, spins, atoms, charges
        r_e = pos[:2]
        r_h = pos[2:]
        phase = jnp.dot(k_e, r_e) + jnp.dot(k_h, r_h)
        logabs = jnp.dot(alpha_e, r_e) + jnp.dot(alpha_h, r_h)
        return phase, logabs

      grad_logpsi_e = np.asarray(alpha_e) + 1.0j * np.asarray(k_e)
      grad_logpsi_h = np.asarray(alpha_h) + 1.0j * np.asarray(k_h)
      expected_ke = -0.5 * (
          (1.0 / m_e) * np.sum(grad_logpsi_e ** 2)
          + grad_logpsi_h @ inv_mh @ grad_logpsi_h
      )

      data = networks.FermiNetData(
          positions=jnp.array([1.2, -0.7, 0.5, 2.1], dtype=jnp.float64),
          spins=jnp.array([1, -1]),
          atoms=jnp.zeros((1, 2), dtype=jnp.float64),
          charges=jnp.zeros((1,), dtype=jnp.float64),
      )
      for method, use_scan in (
          ('folx', False),
          ('default', False),
          ('default', True),
      ):
        ke_fn = moire_hamiltonian.local_kinetic_energy(
            plane_wave_f,
            nspins=(1, 1),
            masses=(m_e, 0.55),
            ndim=2,
            use_scan=use_scan,
            complex_output=True,
            laplacian_method=method,
            mass_tensor_h=m_h_mat,
        )
        val = complex(ke_fn({}, data))
        np.testing.assert_allclose(val, expected_ke, rtol=1e-13, atol=1e-13)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  @parameterized.parameters(
      {'complex_output': False, 'nspins': (2, 2), 'aniso_layer': 'h'},
      {'complex_output': True, 'nspins': (1, 1, 1, 1), 'aniso_layer': 'h'},
      {'complex_output': True, 'nspins': (1, 1, 1, 1), 'aniso_layer': 'e'},
  )
  def test_anisotropic_ferminet_kinetic_energy_matches_exact_hessian(
      self, complex_output, nspins, aniso_layer
  ):
    """Checks folx and default kinetic energy against exact jax.hessian."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      prim = geometry.primitive_lattice_from_gvectors(
          SCREENSHOT_GVECS_NM_INV, gvector_units='nm^-1'
      )
      sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))
      dist_scale = geometry.average_lattice_period(prim)
      charges = jnp.zeros((1,), dtype=jnp.float64)
      atoms = jnp.zeros((1, 2), dtype=jnp.float64)

      feature_layer = pbc_feature_layer.make_pbc_feature_layer(
          natoms=1,
          nspins=nspins,
          ndim=2,
          lattice=sim,
          include_r_ae=False,
          interlayer_distance=37.79,
          distance_scale=dist_scale,
      )
      kpoints = pbc_envelopes.make_kpoints(sim, nspins, ndim=2)
      envelope = pbc_envelopes.make_multiwave_envelope(kpoints)
      net = networks.make_fermi_net(
          nspins,
          charges,
          ndim=2,
          lattice=sim,
          determinants=2,
          envelope=envelope,
          feature_layer=feature_layer,
          jastrow='bilayer',
          jastrow_kwargs={
              'mass_e': 0.70,
              'mass_h': 0.55,
              'dielectric_constant': 5.7,
              'interlayer_distance': 37.79,
          },
          full_det=True,
          complex_output=complex_output,
          hidden_dims=((16, 4), (16, 4)),
      )
      key = jax.random.PRNGKey(123)
      k_init, k_pos = jax.random.split(key)
      params = net.init(k_init)
      n_tot = sum(nspins)
      pos = jax.random.normal(k_pos, (2 * n_tot,), dtype=jnp.float64) * 30.0
      spins = jnp.ones((n_tot,), dtype=jnp.float64)
      data = networks.FermiNetData(
          positions=pos, spins=spins, atoms=atoms, charges=charges
      )

      m_mat = np.array([[0.42, 0.09], [0.09, 0.68]], dtype=np.float64)
      mt_e = m_mat if aniso_layer == 'e' else None
      mt_h = m_mat if aniso_layer == 'h' else None
      m_e_scalar = 0.70
      m_h_scalar = 0.45

      ke_folx = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(m_e_scalar, m_h_scalar),
          ndim=2,
          complex_output=complex_output,
          laplacian_method='folx',
          mass_tensor_e=mt_e,
          mass_tensor_h=mt_h,
      )(params, data)
      ke_def = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(m_e_scalar, m_h_scalar),
          ndim=2,
          use_scan=False,
          complex_output=complex_output,
          laplacian_method='default',
          mass_tensor_e=mt_e,
          mass_tensor_h=mt_h,
      )(params, data)
      ke_scan = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(m_e_scalar, m_h_scalar),
          ndim=2,
          use_scan=True,
          complex_output=complex_output,
          laplacian_method='default',
          mass_tensor_e=mt_e,
          mass_tensor_h=mt_h,
      )(params, data)

      # Exact jax.hessian reference: -1/2 sum_p Tr(M_p^{-1} (H_logpsi + g g^T)).
      n_e, _ = geometry.layer_particle_counts(nspins)
      w_e = (
          np.linalg.inv(m_mat)
          if aniso_layer == 'e'
          else np.eye(2) / m_e_scalar
      )
      w_h = (
          np.linalg.inv(m_mat)
          if aniso_layer == 'h'
          else np.eye(2) / m_h_scalar
      )

      def logabs_only(r):
        return net.apply(params, r, spins, atoms, charges)[1]

      grad_u = jax.grad(logabs_only)(pos)
      hess_u = jax.hessian(logabs_only)(pos)
      if complex_output:
        def phase_only(r):
          return net.apply(params, r, spins, atoms, charges)[0]
        grad_th = jax.grad(phase_only)(pos)
        hess_th = jax.hessian(phase_only)(pos)
        grad_c = grad_u + 1.0j * grad_th
        hess_c = hess_u + 1.0j * hess_th
      else:
        grad_c = grad_u
        hess_c = hess_u

      ref = 0.0j if complex_output else 0.0
      for p in range(n_tot):
        w_p = w_e if p < n_e else w_h
        sl = slice(2 * p, 2 * p + 2)
        g_p = grad_c[sl]
        h_p = hess_c[sl, sl]
        ref = ref - 0.5 * (jnp.trace(w_p @ h_p) + g_p @ (w_p @ g_p))

      np.testing.assert_allclose(ke_folx, ref, rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(ke_def, ref, rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(ke_scan, ref, rtol=1e-12, atol=1e-12)

      # Isotropic mass tensor M = m * I_2 matches scalar local_kinetic_energy.
      ke_iso_vec = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(m_e_scalar, m_h_scalar),
          ndim=2,
          complex_output=complex_output,
          laplacian_method='folx',
      )(params, data)
      ke_iso_mat = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(m_e_scalar, m_h_scalar),
          ndim=2,
          complex_output=complex_output,
          laplacian_method='folx',
          mass_tensor_h=m_h_scalar * np.eye(2),
      )(params, data)
      np.testing.assert_allclose(
          ke_iso_vec, ke_iso_mat, rtol=1e-13, atol=1e-13
      )
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_anisotropic_distance_scale_energy_scale_and_enlarged_unit_cell(self):
    """Tests average lattice period scaling, Tr(M)/2 mass, and idempotency."""
    cfg = anisotropic_moire_bilayer.get_config()
    cfg.system.moire.mass_tensor_e = ((0.42, 0.08), (0.08, 0.58))
    cfg.system.moire.mass_h = 0.70
    cfg.network.jastrow = 'bilayer'
    cfg = base_config.resolve(cfg)

    expected_prim = geometry.primitive_lattice_from_gvectors(
        SCREENSHOT_GVECS_NM_INV, gvector_units='nm^-1'
    )
    a1 = float(np.linalg.norm(expected_prim[:, 0]))
    a2 = float(np.linalg.norm(expected_prim[:, 1]))
    expected_a_avg = 0.5 * (a1 + a2)
    expected_me_eff = 0.5 * (0.42 + 0.58)  # 0.50
    expected_e_scale = 1.0 / (
        2.0 * (expected_me_eff + 0.70) * (expected_a_avg ** 2)
    )

    self.assertAlmostEqual(
        float(cfg.network.make_feature_layer_kwargs['distance_scale']),
        expected_a_avg,
        places=12,
    )
    self.assertAlmostEqual(
        float(cfg.optim.energy_scale), expected_e_scale, places=18
    )
    self.assertAlmostEqual(
        float(cfg.network.jastrow_kwargs['mass_e']), expected_me_eff, places=12
    )
    np.testing.assert_allclose(cfg.mcmc.pair_lattice, expected_prim)

    # Idempotency under repeated apply_moire_anisotropic_config / resolve.
    cfg_again = geometry.apply_moire_anisotropic_config(copy.deepcopy(cfg))
    np.testing.assert_allclose(cfg_again.system.lattice, cfg.system.lattice)
    np.testing.assert_allclose(
        cfg_again.system.make_local_energy_kwargs['moire_gvectors_e'],
        cfg.system.make_local_energy_kwargs['moire_gvectors_e'],
    )
    self.assertAlmostEqual(
        float(cfg_again.network.make_feature_layer_kwargs['distance_scale']),
        expected_a_avg,
        places=12,
    )

    # Enlarged 2x1 stripe unit cell preserves 1x1 average moire period scaling.
    cfg_uc = anisotropic_moire_bilayer.get_config()
    cfg_uc.system.moire.mass_tensor_e = ((0.42, 0.08), (0.08, 0.58))
    cfg_uc.system.moire.mass_h = 0.70
    cfg_uc.system.unit_cell_matrix = '2x1'
    cfg_uc.system.moire.supercell_matrix = ((2, 0), (0, 2))
    cfg_uc = base_config.resolve(cfg_uc)
    np.testing.assert_allclose(
        cfg_uc.network.make_feature_layer_kwargs['primitive_lattice'],
        expected_prim @ np.array([[2, 0], [0, 1]]),
    )
    self.assertAlmostEqual(
        float(cfg_uc.network.make_feature_layer_kwargs['distance_scale']),
        expected_a_avg,
        places=12,
    )
    self.assertAlmostEqual(
        float(cfg_uc.optim.energy_scale), expected_e_scale, places=18
    )

    # Turning anisotropic=False on anisotropic_moire_bilayer falls back cleanly.
    cfg_fallback = anisotropic_moire_bilayer.get_config()
    cfg_fallback.system.moire.anisotropic = False
    cfg_fallback.system.moire.moire_gvectors = None
    cfg_fallback.system.moire.mass_tensor_e = None
    cfg_fallback = base_config.resolve(cfg_fallback)
    self.assertNotIn(
        'anisotropic', cfg_fallback.system.make_local_energy_kwargs
    )
    self.assertAlmostEqual(
        float(cfg_fallback.network.make_feature_layer_kwargs['distance_scale']),
        274.69,
        places=12,
    )

    # Opting into anisotropic=True on an existing 1-shell config (3-tuple v_h).
    cfg_existing = WSe2WS2_MoSe2_vh1_ve0.get_config()
    with cfg_existing.ignore_type():
      cfg_existing.system.anisotropic = True
      cfg_existing.system.moire.moire_gvectors = SCREENSHOT_GVECS_NM_INV
      cfg_existing.system.moire.gvector_units = 'nm^-1'
      cfg_existing.system.moire.v_h = (-1.0e-4, -1.5e-4, -2.0e-4)
      cfg_existing.system.moire.mass_h = ((0.40, 0.05), (0.05, 0.60))
      cfg_existing.optim.scale_by_moire_energy = True
    cfg_existing = base_config.resolve(cfg_existing)
    self.assertEqual(
        cfg_existing.system.make_local_energy_kwargs['v_h'],
        (-1.0e-4, -1.5e-4, -2.0e-4),
    )
    np.testing.assert_allclose(
        cfg_existing.system.make_local_energy_kwargs['mass_tensor_h'],
        ((0.40, 0.05), (0.05, 0.60)),
    )
    self.assertAlmostEqual(
        float(cfg_existing.network.make_feature_layer_kwargs['distance_scale']),
        expected_a_avg,
        places=12,
    )

  def test_kwargs_only_masses_preserved_without_moire_block(self):
    """Configuring via make_local_energy_kwargs preserves masses tuple."""
    prim = geometry.primitive_lattice_from_gvectors(
        SCREENSHOT_GVECS_NM_INV, gvector_units='nm^-1'
    )
    sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))

    for masses_in in ((0.70, 0.45), (0.70, 0.70, 0.45, 0.45)):
      cfg = base_config.default()
      cfg.system.ndim = 2
      cfg.system.electrons = (1, 0, 1, 0)
      cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]
      cfg.system.anisotropic = True
      cfg.system.make_local_energy_fn = (
          'ferminet.moire.hamiltonian.local_energy'
      )
      with cfg.ignore_type():
        cfg.system.lattice = sim
        cfg.system.make_local_energy_kwargs = {
            'lattice': sim,
            'moire_lattice': prim,
            'masses': masses_in,
            'mass_tensor_h': ((0.40, 0.0), (0.0, 0.55)),
            'v_h': -5.512e-4,
            'moire_gvectors_h': SCREENSHOT_GVECS_NM_INV,
            'gvector_units': 'nm^-1',
        }
        cfg.optim.scale_by_moire_energy = True

      cfg = base_config.resolve(cfg)
      cfg = geometry.apply_moire_anisotropic_config(cfg)
      le = cfg.system.make_local_energy_kwargs
      self.assertAlmostEqual(float(le['mass_e']), 0.70, places=12)
      self.assertAlmostEqual(float(le['mass_h']), 0.475, places=12)
      np.testing.assert_allclose(le['masses'], (0.70, 0.475), rtol=1e-12)

  def test_reapplying_config_without_explicit_gvectors_preserves_supercell(
      self,
  ):
    """Re-running apply_moire_anisotropic_config keeps triangular supercell."""
    cfg = WSe2WS2_MoSe2_vh1_ve0.get_config()
    with cfg.ignore_type():
      cfg.system.anisotropic = True
      cfg.system.moire.supercell_matrix = ((3, 0), (0, 2))
      cfg.system.moire.mass_tensor_h = ((0.40, 0.0), (0.0, 0.55))

    cfg1 = base_config.resolve(cfg)
    expected_prim = geometry.triangular_lattice(151.18)
    expected_sim = geometry.supercell_lattice(
        expected_prim, np.array([[3, 0], [0, 2]])
    )
    np.testing.assert_allclose(cfg1.system.lattice, expected_sim, atol=1e-12)
    self.assertIsNone(
        cfg1.system.make_local_energy_kwargs['moire_gvectors_h']
    )

    # Simulate train.train(cfg) calling apply_moire_anisotropic_config again.
    cfg2 = geometry.apply_moire_anisotropic_config(copy.deepcopy(cfg1))
    np.testing.assert_allclose(cfg2.system.lattice, cfg1.system.lattice)
    np.testing.assert_allclose(
        cfg2.system.make_local_energy_kwargs['moire_lattice'],
        expected_prim,
    )
    self.assertIsNone(
        cfg2.system.make_local_energy_kwargs['moire_gvectors_h']
    )

    # Mutating explicit G-vectors on an already-resolved config also updates
    # cleanly without conflicting against stale derived kwargs.
    cfg_mut = WSe2WS2_MoSe2_vh1_ve0.get_config()
    with cfg_mut.ignore_type():
      cfg_mut.system.anisotropic = True
      cfg_mut.system.moire.moire_gvectors = SCREENSHOT_GVECS_NM_INV
      cfg_mut.system.moire.gvector_units = 'nm^-1'
    cfg_mut = base_config.resolve(cfg_mut)
    half_gvecs = tuple(
        (0.5 * gx, 0.5 * gy) for gx, gy in SCREENSHOT_GVECS_NM_INV
    )
    with cfg_mut.ignore_type():
      cfg_mut.system.moire.moire_gvectors = half_gvecs
    cfg_mut2 = geometry.apply_moire_anisotropic_config(cfg_mut)
    expected_prim2 = geometry.primitive_lattice_from_gvectors(
        half_gvecs, gvector_units='nm^-1'
    )
    np.testing.assert_allclose(
        cfg_mut2.system.make_local_energy_kwargs['moire_lattice'],
        expected_prim2,
    )

  def test_editing_gvectors_updates_auto_pair_lattice_but_preserves_override(
      self,
  ):
    """Editing G-vectors updates auto pair_lattice while keeping overrides."""
    cfg = anisotropic_moire_bilayer.get_config()
    orig_prim = np.asarray(cfg.mcmc.pair_lattice).copy()
    cfg.system.moire.moire_gvectors = tuple(
        (0.5 * gx, 0.5 * gy) for gx, gy in SCREENSHOT_GVECS_NM_INV
    )
    cfg = base_config.resolve(cfg)
    np.testing.assert_allclose(cfg.mcmc.pair_lattice, 2.0 * orig_prim)
    np.testing.assert_allclose(
        cfg.mcmc.pair_lattice,
        cfg.system.make_local_energy_kwargs['moire_lattice'],
    )

    # Explicit user override of cfg.mcmc.pair_lattice must be preserved.
    cfg_custom = anisotropic_moire_bilayer.get_config()
    custom_pair = np.array([[195.0, 12.0], [-8.0, 175.0]])
    with cfg_custom.ignore_type():
      cfg_custom.mcmc.pair_lattice = custom_pair.copy()
    cfg_custom.system.moire.moire_gvectors = tuple(
        (0.5 * gx, 0.5 * gy) for gx, gy in SCREENSHOT_GVECS_NM_INV
    )
    cfg_custom = base_config.resolve(cfg_custom)
    cfg_custom = geometry.apply_moire_anisotropic_config(cfg_custom)
    np.testing.assert_allclose(cfg_custom.mcmc.pair_lattice, custom_pair)

  def test_anisotropic_moire_train_step_end_to_end(self):
    """Runs an end-to-end KFAC VMC step on strained ReSe2 anisotropic moire."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg = anisotropic_moire_bilayer.get_config()
      cfg.system.moire.nspins = (1, 1, 0, 0)
      cfg.system.moire.supercell_matrix = ((2, 0), (0, 2))
      cfg.system.moire.ewald_gmax = 8
      cfg.network.ferminet.hidden_dims = ((8, 4),)
      cfg.network.determinants = 1
      cfg.batch_size = 8
      cfg.mcmc.burn_in = 1
      cfg.mcmc.steps = 1
      cfg.optim.iterations = 1
      cfg.optim.scale_by_moire_energy = True
      cfg.debug.deterministic = True
      cfg.debug.check_nan = True
      cfg.log.save_path = tmpdir
      cfg = base_config.resolve(cfg)

      train.train(cfg)

      stats_csv = os.path.join(tmpdir, 'train_stats.csv')
      self.assertTrue(os.path.exists(stats_csv))
      with open(stats_csv, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
      self.assertGreaterEqual(len(lines), 2)
      energy_val = float(lines[-1].split(',')[1])
      self.assertTrue(np.isfinite(energy_val))


if __name__ == '__main__':
  absltest.main()
