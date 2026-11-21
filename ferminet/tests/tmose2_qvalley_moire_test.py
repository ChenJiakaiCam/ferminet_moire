"""Tests for strained 0-deg tMoSe2 Q-valley continuum model, Fourier potentials,
dual-layer mass tensors, metallic gate Ewald screening, and spin-swap MCMC."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import mcmc
from ferminet import networks
from ferminet import train
from ferminet.configs import tmose2_0deg_cb_wigner
from ferminet.moire import ewald
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
from ferminet.moire import tmose2_continuum
from ferminet.pbc import envelopes as pbc_envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class TMoSe2QValleyMoireTest(parameterized.TestCase):

  def test_tmose2_continuum_model_physics_and_fourier_accuracy(self):
    """Checks corrugation, site ordering, mass tensor, Fourier RMS, and spinors."""
    model = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0,
        eta_pinning=1.0,
        fourier_cutoff=5,
        mass_model='hse',
        deformation_model='soc',
        grid_n=64,
    )
    # Signed out-of-plane corrugation Z(r) = d(r) - 6.70 A in [-0.225, +0.449] A:
    z_min = float(np.min(model.z_corrugation_angstrom))
    z_max = float(np.max(model.z_corrugation_angstrom))
    self.assertAlmostEqual(z_min, -0.225, delta=0.005)
    self.assertAlmostEqual(z_max, +0.449, delta=0.005)

    # Selected lowest-energy valley is Ferreira index 3 (+-Q_* along DW_1):
    v0 = model.valleys[0]
    self.assertEqual(v0.ferreira_index, 3)
    self.assertAlmostEqual(v0.light_mass_angle_deg, 167.256, delta=0.02)

    mt0 = np.asarray(v0.mass_tensor_m0, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(mt0)
    np.testing.assert_allclose(eigvals, (0.48, 1.08), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        mt0, ((0.5092, 0.1291), (0.1291, 1.0508)), atol=2e-4
    )

    # Site energy hierarchy DW_1 < MX' < XM' < DW_3 < DW_2 < XX' at nu=3, eta=1:
    n = v0.v_eff_mev.shape[0]
    v_shift = v0.v_eff_mev - np.min(v0.v_eff_mev)
    e_dw1 = float(v_shift[n // 2, n // 2])
    e_mx = float(v_shift[(2 * n) // 3, (2 * n) // 3])
    e_xm = float(v_shift[n // 3, n // 3])
    e_dw3 = float(v_shift[n // 2, 0])
    e_dw2 = float(v_shift[0, n // 2])
    e_xx = float(v_shift[0, 0])
    self.assertLess(e_dw1, 0.5)
    self.assertLess(e_dw1, e_mx)
    self.assertLess(e_mx, e_xm)
    self.assertLess(e_xm, e_dw3)
    self.assertLess(e_dw3, e_dw2)
    self.assertLess(e_dw2, e_xx)
    self.assertAlmostEqual(e_xx, 73.61, delta=0.5)

    # 60 independent half-space cosine modes for |m_1|, |m_2| <= 5:
    self.assertEqual(v0.fourier_amps_hartree.shape, (60,))
    self.assertEqual(v0.fourier_gvectors_bohr_inv.shape, (60, 2))
    self.assertLess(v0.fourier_rms_error_mev, 1.5)

    # Layer spinor weights satisfy 0 <= w_top <= 1, w_top + w_bot == 1, and PBC:
    rng = np.random.default_rng(11)
    pts = rng.normal(scale=200.0, size=(10, 2))
    w_top, w_bot = tmose2_continuum.evaluate_layer_weights(model, pts)
    np.testing.assert_allclose(w_top + w_bot, np.ones(10), atol=1e-14)
    self.assertTrue(np.all((w_top >= 0.0) & (w_top <= 1.0)))
    w_top_shift, _ = tmose2_continuum.evaluate_layer_weights(
        model, pts + model.primitive_lattice_bohr[:, 0]
    )
    np.testing.assert_allclose(w_top, w_top_shift, atol=1e-12)

  def test_multi_harmonic_fourier_potential_pbc_and_incommensurate_guard(self):
    """Checks 60-mode Fourier potential PBC invariance and analytic formula."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      model = tmose2_continuum.build_tmose2_continuum_model(
          nu_filling=3.0, fourier_cutoff=5, grid_n=64
      )
      prim = model.primitive_lattice_bohr
      sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))
      v0, v1 = model.valleys[0], model.valleys[1]

      pot_fn = moire_hamiltonian.make_moire_potential(
          nspins=(2, 1),
          simulation_lattice=sim,
          moire_lattice=prim,
          anisotropic=True,
          two_layer_anisotropic=True,
          fourier_gvectors_e=v0.fourier_gvectors_bohr_inv,
          fourier_amps_e=v0.fourier_amps_hartree,
          fourier_phases_e=v0.fourier_phases_rad,
          fourier_gvectors_h=v1.fourier_gvectors_bohr_inv,
          fourier_amps_h=v1.fourier_amps_hartree,
          fourier_phases_h=v1.fourier_phases_rad,
      )
      rng = np.random.default_rng(29)
      pos = rng.normal(scale=120.0, size=(3, 2))
      val0 = float(pot_fn(jnp.asarray(pos.reshape(-1), dtype=jnp.float64)))

      # Direct analytical check:
      ref_e = np.sum(
          v0.fourier_amps_hartree[None, :]
          * np.cos(pos[:2] @ v0.fourier_gvectors_bohr_inv.T + v0.fourier_phases_rad[None, :])
      )
      ref_h = np.sum(
          v1.fourier_amps_hartree[None, :]
          * np.cos(pos[2:] @ v1.fourier_gvectors_bohr_inv.T + v1.fourier_phases_rad[None, :])
      )
      self.assertAlmostEqual(val0, float(ref_e + ref_h), places=14)

      for shift in (prim[:, 0], prim[:, 1], sim[:, 0], sim[:, 1]):
        pos_s = pos.copy()
        pos_s[0] += shift
        pos_s[2] -= shift
        val_s = float(pot_fn(jnp.asarray(pos_s.reshape(-1), dtype=jnp.float64)))
        self.assertAlmostEqual(val0, val_s, places=13)

      # Incommensurate wavevector must raise ValueError:
      bad_g = v0.fourier_gvectors_bohr_inv.copy()
      bad_g[0, 0] *= 1.07
      with self.assertRaises(ValueError):
        moire_hamiltonian.make_moire_potential(
            nspins=(1, 0),
            simulation_lattice=sim,
            moire_lattice=prim,
            anisotropic=True,
            fourier_gvectors_e=bad_g,
            fourier_amps_e=v0.fourier_amps_hartree,
        )
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  @parameterized.parameters(False, True)
  def test_dual_layer_anisotropic_mass_tensors_match_exact_hessian(
      self, complex_output
  ):
    """Checks simultaneous mass_tensor_e != mass_tensor_h across folx/default/hessian."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      model = tmose2_continuum.build_tmose2_continuum_model(nu_filling=5.0)
      prim = model.primitive_lattice_bohr
      sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 1]]))
      mt_e = np.asarray(model.valleys[0].mass_tensor_m0, dtype=np.float64)
      mt_h = np.asarray(model.valleys[1].mass_tensor_m0, dtype=np.float64)
      self.assertFalse(np.allclose(mt_e, mt_h))

      nspins = (1, 1, 1, 0)
      charges = jnp.zeros((1,), dtype=jnp.float64)
      atoms = jnp.zeros((1, 2), dtype=jnp.float64)
      fl = pbc_feature_layer.make_pbc_feature_layer(
          natoms=1,
          nspins=nspins,
          ndim=2,
          lattice=sim,
          include_r_ae=False,
          interlayer_distance=0.0,
          distance_scale=geometry.average_lattice_period(prim),
      )
      env = pbc_envelopes.make_multiwave_envelope(
          pbc_envelopes.make_kpoints(sim, nspins, ndim=2)
      )
      net = networks.make_fermi_net(
          nspins,
          charges,
          ndim=2,
          lattice=sim,
          determinants=2,
          envelope=env,
          feature_layer=fl,
          jastrow='none',
          full_det=True,
          complex_output=complex_output,
          hidden_dims=((12, 4), (12, 4)),
      )
      key = jax.random.PRNGKey(77)
      k1, k2 = jax.random.split(key)
      params = net.init(k1)
      pos = jax.random.normal(k2, (6,), dtype=jnp.float64) * 40.0
      spins = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float64)
      data = networks.FermiNetData(
          positions=pos, spins=spins, atoms=atoms, charges=charges
      )

      ke_folx = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(0.78, 0.78),
          ndim=2,
          complex_output=complex_output,
          laplacian_method='folx',
          mass_tensor_e=mt_e,
          mass_tensor_h=mt_h,
      )(params, data)
      ke_def = moire_hamiltonian.local_kinetic_energy(
          net.apply,
          nspins=nspins,
          masses=(0.78, 0.78),
          ndim=2,
          use_scan=False,
          complex_output=complex_output,
          laplacian_method='default',
          mass_tensor_e=mt_e,
          mass_tensor_h=mt_h,
      )(params, data)

      w_e = np.linalg.inv(mt_e)
      w_h = np.linalg.inv(mt_h)
      logabs_fn = lambda r: net.apply(params, r, spins, atoms, charges)[1]
      grad_u = jax.grad(logabs_fn)(pos)
      hess_u = jax.hessian(logabs_fn)(pos)
      if complex_output:
        phase_fn = lambda r: net.apply(params, r, spins, atoms, charges)[0]
        grad_c = grad_u + 1.0j * jax.grad(phase_fn)(pos)
        hess_c = hess_u + 1.0j * jax.hessian(phase_fn)(pos)
      else:
        grad_c, hess_c = grad_u, hess_u

      ref = 0.0j if complex_output else 0.0
      for p in range(3):
        w_p = w_e if p < 2 else w_h
        sl = slice(2 * p, 2 * p + 2)
        g_p = grad_c[sl]
        h_p = hess_c[sl, sl]
        ref = ref - 0.5 * (jnp.trace(w_p @ h_p) + g_p @ (w_p @ g_p))

      np.testing.assert_allclose(ke_folx, ref, rtol=1e-12, atol=1e-12)
      np.testing.assert_allclose(ke_def, ref, rtol=1e-12, atol=1e-12)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_gate_screened_ewald_alpha_invariance_and_limits(self):
    """Checks exact Ewald alpha-independence, d_g -> inf, and self-image limit."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      lat = geometry.triangular_lattice(40.0)
      rng = np.random.default_rng(19)
      pos = jnp.asarray(rng.uniform(-10.0, 10.0, size=(8,)), dtype=jnp.float64)

      for charges_spec, d_inter in (('ee', 0.0), ('ee', 3.5), ('eh', 4.0)):
        for inc_cap in (False, True):
          vals = []
          for alpha in (0.08, 0.14, 0.22):
            ew = ewald.BilayerEwaldSum(
                lattice=lat,
                nspins=(1, 1, 1, 1),
                interlayer_distance=d_inter,
                dielectric_constant=3.73,
                ewald_gmax=60,
                nlatvec=3,
                alpha=alpha,
                include_capacitor=inc_cap,
                charges=charges_spec,
                gate_distance=12.0,
            )
            vals.append(float(ew.energy(pos)))
          self.assertAlmostEqual(vals[0], vals[1], delta=1e-11)
          self.assertAlmostEqual(vals[1], vals[2], delta=1e-11)

      # Large gate_distance with include_capacitor=False recovers ungated Ewald:
      ew_ungated = ewald.BilayerEwaldSum(
          lattice=lat,
          nspins=(2, 1, 1, 0),
          interlayer_distance=0.0,
          dielectric_constant=3.73,
          charges='ee',
          include_capacitor=False,
      )
      ew_far_gate = ewald.BilayerEwaldSum(
          lattice=lat,
          nspins=(2, 1, 1, 0),
          interlayer_distance=0.0,
          dielectric_constant=3.73,
          charges='ee',
          include_capacitor=False,
          gate_distance=1.0e6,
      )
      self.assertAlmostEqual(
          float(ew_ungated.energy(pos)),
          float(ew_far_gate.energy(pos)),
          places=9,
      )

      # Single charge on huge cell with include_capacitor=True approaches self-image
      # energy -1 / (4 * eps_r * d_g):
      huge_lat = geometry.triangular_lattice(4000.0)
      d_g = 4.0
      eps_r = 3.73
      ew_single = ewald.BilayerEwaldSum(
          lattice=huge_lat,
          nspins=(1, 0, 0, 0),
          dielectric_constant=eps_r,
          ewald_gmax=80,
          nlatvec=2,
          include_capacitor=True,
          gate_distance=d_g,
      )
      e_single = float(ew_single.energy(jnp.zeros((2,), dtype=jnp.float64)))
      expected_self_img = -1.0 / (4.0 * eps_r * d_g)
      self.assertAlmostEqual(e_single, expected_self_img, delta=5e-4)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_spin_swap_mcmc_detailed_balance_and_pmove_isolation(self):
    """Verifies exact Boltzmann ratio across isolated wells and clean pmove."""
    # Two wells at x_A = -100, x_B = +100 with move_width = 1.0 (zero Gaussian
    # tunneling). Construct |psi(r_up, r_dn)|^2 proportional to:
    #   4 * G_A(r_up) G_B(r_dn) + 1 * G_B(r_up) G_A(r_dn),
    # so exact equilibrium P(r_up in A, r_dn in B) = 4 / (4 + 1) = 0.80.
    r_a = jnp.array([-100.0, 0.0], dtype=jnp.float32)
    r_b = jnp.array([+100.0, 0.0], dtype=jnp.float32)
    sigma = 2.0

    def single_logabs(params, pos, spins, atoms, charges):
      del params, spins, atoms, charges
      r_up = pos[:2]
      r_dn = pos[2:]
      log_ab = (
          jnp.log(2.0)
          - 0.25 * jnp.sum((r_up - r_a) ** 2 + (r_dn - r_b) ** 2) / (sigma**2)
      )
      log_ba = -0.25 * jnp.sum((r_up - r_b) ** 2 + (r_dn - r_a) ** 2) / (
          sigma**2
      )
      return jnp.logaddexp(log_ab, log_ba)

    batch_net = jax.vmap(single_logabs, in_axes=(None, 0, 0, 0, 0))
    batch_size = 512
    # Start all walkers in the unfavorable (B, A) configuration:
    init_pos = jnp.tile(
        jnp.concatenate([r_b, r_a])[None, :], (batch_size, 1)
    )
    data_init = networks.FermiNetData(
        positions=init_pos,
        spins=jnp.tile(jnp.array([[1.0, -1.0]]), (batch_size, 1)),
        atoms=jnp.zeros((batch_size, 1, 2)),
        charges=jnp.zeros((batch_size, 1)),
    )

    step_no_swap = jax.jit(
        mcmc.make_mcmc_step(
            batch_net,
            batch_per_device=batch_size,
            steps=20,
            ndim=2,
            nspins=(1, 1, 0, 0),
            spin_swap_prob=0.0,
        )
    )
    data_trapped, _ = step_no_swap(
        {}, data_init, jax.random.PRNGKey(0), jnp.asarray(1.0)
    )
    frac_ab_trapped = float(jnp.mean(data_trapped.positions[:, 0] < 0.0))
    self.assertEqual(frac_ab_trapped, 0.0)

    step_with_swap = jax.jit(
        mcmc.make_mcmc_step(
            batch_net,
            batch_per_device=batch_size,
            steps=30,
            ndim=2,
            nspins=(1, 1, 0, 0),
            spin_swap_prob=0.5,
            return_stats=True,
        )
    )
    data_eq, stats = step_with_swap(
        {}, data_init, jax.random.PRNGKey(1), jnp.asarray(1.0)
    )
    frac_ab = float(jnp.mean(data_eq.positions[:, 0] < 0.0))
    self.assertAlmostEqual(frac_ab, 0.80, delta=0.05)
    # Spins array is untouched:
    np.testing.assert_array_equal(data_eq.spins, data_init.spins)
    # Spatial pmove equals pmove_normal and is not polluted by discrete swaps:
    self.assertAlmostEqual(
        float(stats['pmove']), float(stats['pmove_normal']), places=6
    )
    self.assertGreater(float(stats['n_try_spin_swap']), 0.0)

  def test_tmose2_config_variants_and_end_to_end_train_steps(self):
    """Checks config resolution across fillings/modes and runs 1- & 2-valley VMC."""
    # Default nu = 3 on 2x2 => 12 electrons, (6, 6, 0, 0):
    cfg3 = base_config.resolve(tmose2_0deg_cb_wigner.get_config())
    self.assertEqual(tuple(cfg3.system.electrons), (6, 6, 0, 0))
    le3 = cfg3.system.make_local_energy_kwargs
    self.assertEqual(len(le3['fourier_amps_e']), 60)
    self.assertIsNotNone(le3['mass_tensor_e'])
    self.assertIsNone(le3['mass_tensor_h'])
    self.assertNotIn('gate_distance', le3)

    # Two-valley nu = 5 on 2x2 => 20 electrons, (8, 8, 2, 2) with dual tensors:
    cfg5 = tmose2_0deg_cb_wigner.get_config()
    cfg5.system.moire.nu = 5.0
    cfg5.system.moire.two_valley = True
    cfg5.system.moire.gate_distance = 65.0
    cfg5 = base_config.resolve(cfg5)
    self.assertEqual(tuple(cfg5.system.electrons), (8, 8, 2, 2))
    le5 = cfg5.system.make_local_energy_kwargs
    self.assertEqual(le5['layer_charges'], (-1, -1))
    self.assertIsNotNone(le5['mass_tensor_e'])
    self.assertIsNotNone(le5['mass_tensor_h'])
    self.assertEqual(len(le5['fourier_amps_h']), 60)
    self.assertAlmostEqual(
        float(le5['gate_distance']),
        65.0 * tmose2_continuum.NM_TO_BOHR,
        places=8,
    )

    # Locked-ConfigDict CLI override tests (n_layer1, mass_tensor_e, mass_model,
    # and double-resolution idempotency):
    cfg_cli = tmose2_0deg_cb_wigner.get_config()
    cfg_cli.lock()
    cfg_cli.system.moire.n_layer1 = 8
    cfg_cli.system.moire.two_valley = True
    cfg_cli.system.moire.n_layer2 = 4
    cfg_cli.system.moire.mass_tensor_e = ((0.60, 0.05), (0.05, 0.90))
    cfg_cli = base_config.resolve(cfg_cli)
    cfg_cli = base_config.resolve(cfg_cli)
    self.assertTrue(cfg_cli.is_locked)
    self.assertEqual(tuple(cfg_cli.system.electrons), (4, 4, 2, 2))
    le_cli = cfg_cli.system.make_local_energy_kwargs
    np.testing.assert_allclose(
        le_cli['mass_tensor_e'], ((0.60, 0.05), (0.05, 0.90))
    )
    # Valley 2 keeps its auto-computed HSE mass tensor without being clobbered:
    self.assertIsNotNone(le_cli['mass_tensor_h'])

    cfg_l1_only = tmose2_0deg_cb_wigner.get_config()
    cfg_l1_only.lock()
    cfg_l1_only.system.moire.n_layer1 = 8
    cfg_l1_only.system.moire.mass_model = 'isotropic'
    cfg_l1_only = base_config.resolve(cfg_l1_only)
    self.assertEqual(tuple(cfg_l1_only.system.electrons), (4, 4, 0, 0))
    self.assertIsNone(
        cfg_l1_only.system.make_local_energy_kwargs['mass_tensor_e']
    )

    # End-to-end 1-valley VMC step (nu=2 on 1x1, gated, spin_swap_prob=0.5):
    with tempfile.TemporaryDirectory() as tmp1:
      cfg_run1 = tmose2_0deg_cb_wigner.get_config()
      cfg_run1.system.moire.nu = 2.0
      cfg_run1.system.moire.supercell_matrix = ((1, 0), (0, 1))
      cfg_run1.system.moire.use_lll_primitive_cell = True
      cfg_run1.system.moire.gate_distance = 65.0
      cfg_run1.system.moire.ewald_gmax = 8
      cfg_run1.network.ferminet.hidden_dims = ((8, 4),)
      cfg_run1.network.determinants = 1
      cfg_run1.batch_size = 8
      cfg_run1.mcmc.burn_in = 1
      cfg_run1.mcmc.steps = 2
      cfg_run1.mcmc.spin_swap_prob = 0.5
      cfg_run1.optim.iterations = 1
      cfg_run1.debug.deterministic = True
      cfg_run1.debug.check_nan = True
      cfg_run1.log.save_path = tmp1
      cfg_run1 = base_config.resolve(cfg_run1)
      self.assertEqual(tuple(cfg_run1.system.electrons), (1, 1, 0, 0))
      train.train(cfg_run1)
      self.assertTrue(os.path.exists(os.path.join(tmp1, 'train_stats.csv')))

    # End-to-end 2-valley 'ee' VMC step with dual mass tensors & dual Fourier:
    with tempfile.TemporaryDirectory() as tmp2:
      cfg_run2 = tmose2_0deg_cb_wigner.get_config()
      cfg_run2.system.moire.nspins = (1, 1, 1, 0)
      cfg_run2.system.moire.two_valley = True
      cfg_run2.system.moire.supercell_matrix = ((1, 0), (0, 1))
      cfg_run2.system.moire.ewald_gmax = 8
      cfg_run2.network.ferminet.hidden_dims = ((8, 4),)
      cfg_run2.network.determinants = 1
      cfg_run2.batch_size = 8
      cfg_run2.mcmc.burn_in = 1
      cfg_run2.mcmc.steps = 2
      cfg_run2.mcmc.spin_swap_prob = 0.5
      cfg_run2.optim.iterations = 1
      cfg_run2.debug.deterministic = True
      cfg_run2.debug.check_nan = True
      cfg_run2.log.save_path = tmp2
      cfg_run2 = base_config.resolve(cfg_run2)
      self.assertEqual(tuple(cfg_run2.system.electrons), (1, 1, 1, 0))
      train.train(cfg_run2)
      self.assertTrue(os.path.exists(os.path.join(tmp2, 'train_stats.csv')))


if __name__ == '__main__':
  absltest.main()
