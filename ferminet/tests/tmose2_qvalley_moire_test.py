"""Tests for strained 0-deg tMoSe2 Q-valley continuum model, Fourier moire potential, and anisotropic kinetic energy."""

import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import networks
from ferminet import train
from ferminet.configs import tmose2_0deg_cb_wigner
from ferminet.moire import fang_tmd_wannier
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
from ferminet.moire import tmose2_continuum
from ferminet.pbc import envelopes as pbc_envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class TMoSe2QValleyMoireTest(parameterized.TestCase):

  def test_fang_22band_q_valley_deformation_potentials_and_fd_checks(self):
    """Verifies Fang 11/22-band Wannier Q-valley deformation potentials & FD."""
    # 1. Benchmark K_+ k.p parameters against Fang (2018) Table IV (p. 10):
    kp = fang_tmd_wannier.compute_mose2_k_valley_kp_parameters()
    self.assertAlmostEqual(kp['f0'], -4.586, delta=0.01)
    self.assertAlmostEqual(kp['f1'], 1.550, delta=0.01)
    self.assertAlmostEqual(kp['f2'], 0.880, delta=0.01)
    self.assertAlmostEqual(kp['f3'], -5.004, delta=0.01)
    self.assertAlmostEqual(kp['f4'], -2.277, delta=0.01)
    self.assertAlmostEqual(kp['f5'], 1.839, delta=0.01)

    # 2. Spinless 11-band (6-band M_1-even) Q-valley CBM & FD cross-checks:
    hf_sp = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=False, method='hellmann_feynman'
    )
    fd_sp = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=False, method='fixed_k_fd', fd_eps=1.0e-5
    )
    re_sp = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=False, method='reminimized_fd', fd_eps=1.0e-4
    )
    self.assertAlmostEqual(hf_sp.k_q_over_k, 0.578149, places=5)
    self.assertAlmostEqual(hf_sp.d_par_mev, 93.917461, delta=1.0e-3)
    self.assertAlmostEqual(hf_sp.d_perp_mev, -1204.205058, delta=1.0e-3)
    self.assertAlmostEqual(hf_sp.d_shear_mev, 0.0, delta=1.0e-9)
    self.assertAlmostEqual(fd_sp.d_par_mev, hf_sp.d_par_mev, delta=1.0e-5)
    self.assertAlmostEqual(fd_sp.d_perp_mev, hf_sp.d_perp_mev, delta=1.0e-5)
    self.assertAlmostEqual(re_sp.d_par_mev, hf_sp.d_par_mev, delta=5.0e-3)
    self.assertAlmostEqual(re_sp.d_perp_mev, hf_sp.d_perp_mev, delta=5.0e-3)

    # 3. 22-band Wannier + atomic SOC Q^(2) lower CBM: Hellmann-Feynman vs.
    #    fixed-k FD vs. 2D k-reminimized FD in crystal [r_0] and lab [(I+u)r_0]:
    hf_soc = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=True, method='hellmann_feynman'
    )
    fd_soc_1e5 = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=True, method='fixed_k_fd', fd_eps=1.0e-5
    )
    fd_soc_1e4 = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=True, method='fixed_k_fd', fd_eps=1.0e-4
    )
    re_soc_crys = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=True,
        method='reminimized_fd',
        fd_eps=1.0e-4,
        lab_frame_k=False,
    )
    re_soc_lab = fang_tmd_wannier.compute_mose2_q_deformation_potentials_mev(
        include_soc=True,
        method='reminimized_fd',
        fd_eps=1.0e-4,
        lab_frame_k=True,
    )
    self.assertAlmostEqual(hf_soc.k_q_over_k, 0.588016, places=5)
    self.assertAlmostEqual(
        hf_soc.spin_splitting_2delta_q_mev, 36.674, delta=0.01
    )
    self.assertAlmostEqual(hf_soc.d_par_mev, 64.208653, delta=1.0e-3)
    self.assertAlmostEqual(hf_soc.d_perp_mev, -1352.455453, delta=1.0e-3)
    self.assertAlmostEqual(hf_soc.d_shear_mev, 0.0, delta=1.0e-9)

    for chk in (fd_soc_1e5, fd_soc_1e4, re_soc_crys, re_soc_lab):
      self.assertAlmostEqual(chk.d_par_mev, hf_soc.d_par_mev, delta=5.0e-3)
      self.assertAlmostEqual(chk.d_perp_mev, hf_soc.d_perp_mev, delta=5.0e-3)
      self.assertAlmostEqual(chk.d_shear_mev, 0.0, delta=1.0e-6)

    # 4. C3-rotated valley k_{Q_2} = R_{2pi/3} k_{Q_1} under rotated principal
    #    strains (tests all 2*u_xy shear parameters beta_5..beta_8) and exact
    #    antiunitary M_yz*T invariance E_Q(+u_xy) == E_Q(-u_xy) at finite shear:
    k_q1 = np.array([hf_soc.k_q_ang_inv, 0.0])
    e_pos_sh = np.linalg.eigvalsh(
        fang_tmd_wannier.build_mose2_22band_hamiltonian(
            k_q1, uxx=0.01, uyy=-0.005, two_uxy=+0.04
        )
    )[14]
    e_neg_sh = np.linalg.eigvalsh(
        fang_tmd_wannier.build_mose2_22band_hamiltonian(
            k_q1, uxx=0.01, uyy=-0.005, two_uxy=-0.04
        )
    )[14]
    self.assertAlmostEqual(float(e_pos_sh), float(e_neg_sh), places=13)

    th_120 = 2.0 * np.pi / 3.0
    e_par = np.array([np.cos(th_120), np.sin(th_120)])
    e_perp = np.array([-np.sin(th_120), np.cos(th_120)])
    k_q2 = hf_soc.k_q_ang_inv * e_par
    eps_rot = 1.0e-5
    u_par_mat = np.outer(e_par, e_par)
    u_perp_mat = np.outer(e_perp, e_perp)
    d_par_q2 = (
        np.linalg.eigvalsh(
            fang_tmd_wannier.build_mose2_22band_hamiltonian(
                k_q2,
                uxx=eps_rot * u_par_mat[0, 0],
                uyy=eps_rot * u_par_mat[1, 1],
                two_uxy=2.0 * eps_rot * u_par_mat[0, 1],
            )
        )[14]
        - np.linalg.eigvalsh(
            fang_tmd_wannier.build_mose2_22band_hamiltonian(
                k_q2,
                uxx=-eps_rot * u_par_mat[0, 0],
                uyy=-eps_rot * u_par_mat[1, 1],
                two_uxy=-2.0 * eps_rot * u_par_mat[0, 1],
            )
        )[14]
    ) / (2.0 * eps_rot) * 1.0e3
    d_perp_q2 = (
        np.linalg.eigvalsh(
            fang_tmd_wannier.build_mose2_22band_hamiltonian(
                k_q2,
                uxx=eps_rot * u_perp_mat[0, 0],
                uyy=eps_rot * u_perp_mat[1, 1],
                two_uxy=2.0 * eps_rot * u_perp_mat[0, 1],
            )
        )[14]
        - np.linalg.eigvalsh(
            fang_tmd_wannier.build_mose2_22band_hamiltonian(
                k_q2,
                uxx=-eps_rot * u_perp_mat[0, 0],
                uyy=-eps_rot * u_perp_mat[1, 1],
                two_uxy=-2.0 * eps_rot * u_perp_mat[0, 1],
            )
        )[14]
    ) / (2.0 * eps_rot) * 1.0e3
    self.assertAlmostEqual(float(d_par_q2), hf_soc.d_par_mev, delta=1.0e-5)
    self.assertAlmostEqual(float(d_perp_q2), hf_soc.d_perp_mev, delta=1.0e-5)

    # 5. Continuum module dictionary matches exact 22-band and spinless solvers:
    np.testing.assert_allclose(
        tmose2_continuum.DEFORMATION_POTENTIALS_MEV['soc'],
        (hf_soc.d_par_mev, hf_soc.d_perp_mev),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tmose2_continuum.DEFORMATION_POTENTIALS_MEV['spinless'],
        (hf_sp.d_par_mev, hf_sp.d_perp_mev),
        rtol=0.0,
        atol=1.0e-12,
    )

  def test_tmose2_continuum_model_physics_and_fourier_convergence(self):
    """Checks corrugation, miniband valley ordering, Fourier cutoff 12, and spinors."""
    model = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0,
        eta_pinning=0.0,
        fourier_cutoff=12,
        mass_model='hse',
        deformation_model='soc',
        grid_n=64,
    )
    # Signed out-of-plane corrugation Z(r) = d(r) - 6.70 A in [-0.225, +0.449] A:
    z_min = float(np.min(model.z_corrugation_angstrom))
    z_max = float(np.max(model.z_corrugation_angstrom))
    self.assertAlmostEqual(z_min, -0.225, delta=0.005)
    self.assertAlmostEqual(z_max, +0.449, delta=0.005)

    # At symmetric eta=0, common piezo potential ephi_plus vanishes identically:
    np.testing.assert_allclose(model.ephi_plus_piezo_mev, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        model.ephi_top_piezo_mev, model.ephi_minus_piezo_mev, atol=1e-14
    )
    self.assertAlmostEqual(
        model.ez_phys_mv_per_angstrom, 0.4055 * 3.0, places=6
    )

    # Valleys are sorted unconditionally by lowest Gamma-point miniband energy:
    g0 = [v.gamma_energies_mev[0] for v in model.valleys]
    self.assertEqual(g0, sorted(g0))
    v0 = model.valleys[0]
    self.assertEqual(v0.ferreira_index, 3)
    self.assertAlmostEqual(v0.light_mass_angle_deg, 167.256, delta=0.02)

    mt0 = np.asarray(v0.mass_tensor_m0, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(mt0)
    np.testing.assert_allclose(eigvals, (0.48, 1.08), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        mt0, ((0.5092, 0.1291), (0.1291, 1.0508)), atol=2e-4
    )

    # Site hierarchy at eta=0, nu=3: active DWs < MX' < XM' < DW_2 < XX':
    n = v0.v_eff_mev.shape[0]
    v_shift = v0.v_eff_mev - np.min(v0.v_eff_mev)
    e_dw1 = float(v_shift[n // 2, n // 2])
    e_dw3 = float(v_shift[n // 2, 0])
    e_mx = float(v_shift[(2 * n) // 3, (2 * n) // 3])
    e_xm = float(v_shift[n // 3, n // 3])
    e_dw2 = float(v_shift[0, n // 2])
    e_xx = float(v_shift[0, 0])
    self.assertLess(max(e_dw1, e_dw3), e_mx)
    self.assertLess(e_mx, e_xm)
    self.assertLess(e_xm, e_dw2)
    self.assertLess(e_dw2, e_xx)

    # 312 independent cosine modes for |m_1|, |m_2| <= 12 converge XX' barrier:
    self.assertEqual(v0.fourier_amps_hartree.shape, (312,))
    self.assertEqual(v0.fourier_gvectors_bohr_inv.shape, (312, 2))
    self.assertLess(v0.fourier_rms_error_mev, 0.05)
    self.assertLess(v0.fourier_max_error_mev, 0.55)

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
    r_xm_bohr = (
        model.primitive_lattice_bohr[:, 0] + model.primitive_lattice_bohr[:, 1]
    ) / 3.0
    r_mx_bohr = 2.0 * r_xm_bohr
    w_xm_mx, _ = tmose2_continuum.evaluate_layer_weights(
        model, np.stack([r_xm_bohr, r_mx_bohr], axis=0)
    )
    self.assertGreater(float(w_xm_mx[0]), 0.5)
    self.assertLess(float(w_xm_mx[1]), 0.5)

  def test_eta_pinning_and_heterostrain_self_consistency(self):
    """Checks consistent eta elasticity/piezo and homogeneous heterostrain."""
    m_eta0 = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0, eta_pinning=0.0, fourier_cutoff=8, grid_n=64
    )
    m_eta1 = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0, eta_pinning=1.0, fourier_cutoff=8, grid_n=64
    )
    # Stiffer (1 + eta^2 = 2) relaxation at eta=1 weakens domain reconstruction,
    # changing phi(r) and z_corrugation(r) relative to eta=0:
    self.assertGreater(
        float(
            np.max(
                np.abs(
                    m_eta1.z_corrugation_angstrom
                    - m_eta0.z_corrugation_angstrom
                )
            )
        ),
        0.01,
    )
    # At eta=1, bottom sheet is unstrained (rho_b = 0 => phi_t != -phi_b), so
    # symmetric piezo potential e*phi_+ = 0.5*e*(phi_t + phi_b) is substantial:
    rms_plus = float(np.sqrt(np.mean(m_eta1.ephi_plus_piezo_mev**2)))
    self.assertGreater(rms_plus, 25.0)

    # Including homogeneous heterostrain shifts both V_def^- and V_def^+:
    m_no_hom = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0,
        eta_pinning=0.0,
        include_heterostrain_deformation=False,
        fourier_cutoff=8,
        grid_n=64,
    )
    diff_hom = np.max(
        np.abs(
            m_eta0.valleys[0].v_eff_mev - m_no_hom.valleys[0].v_eff_mev
        )
    )
    self.assertGreater(float(diff_hom), 0.1)

  def test_multi_harmonic_fourier_potential_pbc_offset_and_guard(self):
    """Checks Fourier potential PBC invariance, constant offset, and guard."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      model = tmose2_continuum.build_tmose2_continuum_model(
          nu_filling=3.0, fourier_cutoff=8, grid_n=64
      )
      prim = model.primitive_lattice_bohr
      sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))
      v1 = model.valleys[1]
      const_per_e = 1.25e-4

      pot_fn = moire_hamiltonian.make_moire_potential(
          nspins=(2, 1, 0, 0),
          simulation_lattice=sim,
          moire_lattice=prim,
          anisotropic=True,
          fourier_gvectors_e=v1.fourier_gvectors_bohr_inv,
          fourier_amps_e=v1.fourier_amps_hartree,
          fourier_phases_e=v1.fourier_phases_rad,
          fourier_const_e=const_per_e,
      )
      rng = np.random.default_rng(29)
      pos = rng.normal(scale=120.0, size=(3, 2))
      val0 = float(pot_fn(jnp.asarray(pos.reshape(-1), dtype=jnp.float64)))

      ref = 3.0 * const_per_e + np.sum(
          v1.fourier_amps_hartree[None, :]
          * np.cos(
              pos @ v1.fourier_gvectors_bohr_inv.T
              + v1.fourier_phases_rad[None, :]
          )
      )
      self.assertAlmostEqual(val0, float(ref), places=14)

      for shift in (prim[:, 0], prim[:, 1], sim[:, 0], sim[:, 1]):
        pos_s = pos.copy()
        pos_s[0] += shift
        pos_s[2] -= shift
        val_s = float(pot_fn(jnp.asarray(pos_s.reshape(-1), dtype=jnp.float64)))
        self.assertAlmostEqual(val0, val_s, places=13)

      # Incommensurate wavevector must raise ValueError:
      bad_g = v1.fourier_gvectors_bohr_inv.copy()
      bad_g[0, 0] *= 1.07
      with self.assertRaises(ValueError):
        moire_hamiltonian.make_moire_potential(
            nspins=(1, 0),
            simulation_lattice=sim,
            moire_lattice=prim,
            anisotropic=True,
            fourier_gvectors_e=bad_g,
            fourier_amps_e=v1.fourier_amps_hartree,
        )
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  @parameterized.parameters(False, True)
  def test_anisotropic_mass_tensor_matches_exact_hessian(self, complex_output):
    """Checks rotated 2x2 mass_tensor_e across folx, default, and exact Hessian."""
    x64_prev = jax.config.jax_enable_x64
    jax.config.update('jax_enable_x64', True)
    try:
      model = tmose2_continuum.build_tmose2_continuum_model(nu_filling=3.0)
      prim = model.primitive_lattice_bohr
      sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 1]]))
      mt_e = np.asarray(model.valleys[0].mass_tensor_m0, dtype=np.float64)

      nspins = (1, 1, 0, 0)
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
      pos = jax.random.normal(k2, (4,), dtype=jnp.float64) * 40.0
      spins = jnp.array([1.0, -1.0], dtype=jnp.float64)
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
      )(params, data)

      w_e = np.linalg.inv(mt_e)
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
      for p in range(2):
        sl = slice(2 * p, 2 * p + 2)
        g_p = grad_c[sl]
        h_p = hess_c[sl, sl]
        ref -= 0.5 * (np.sum(w_e * h_p) + g_p @ w_e @ g_p)

      np.testing.assert_allclose(ke_folx, ref, rtol=1e-11, atol=1e-12)
      np.testing.assert_allclose(ke_def, ref, rtol=1e-11, atol=1e-12)
    finally:
      jax.config.update('jax_enable_x64', x64_prev)

  def test_config_filling_resolution_energy_scale_and_valley_offsets(self):
    """Checks explicit particle counts, fresh energy_scale, and valley offsets."""
    cfg3 = base_config.resolve(tmose2_0deg_cb_wigner.get_config())
    self.assertEqual(tuple(cfg3.system.electrons), (6, 6, 0, 0))
    self.assertEqual(cfg3.network.jastrow, 'none')
    le3 = cfg3.system.make_local_energy_kwargs
    self.assertEqual(len(le3['fourier_amps_e']), 312)
    self.assertAlmostEqual(float(le3['fourier_const_e']), 0.0, places=15)
    self.assertAlmostEqual(
        float(cfg3.optim.energy_scale), 2.535306e-6, delta=5e-11
    )

    # 1. Explicit electron count updates physical filling and E_z electrostatics:
    cfg_ne4 = tmose2_0deg_cb_wigner.get_config()
    cfg_ne4.system.moire.n_electrons = 4
    cfg_ne4 = base_config.resolve(cfg_ne4)
    self.assertEqual(tuple(cfg_ne4.system.electrons), (2, 2, 0, 0))
    self.assertAlmostEqual(float(cfg_ne4.system.moire.effective_nu), 1.0)
    self.assertAlmostEqual(
        float(cfg_ne4.system.moire.ez_resolved_mv_per_angstrom),
        0.4055,
        places=6,
    )

    # Conflicting explicit n_electrons and explicit nu raises ValueError:
    cfg_conflict = tmose2_0deg_cb_wigner.get_config()
    cfg_conflict.system.moire.n_electrons = 4
    cfg_conflict.system.moire.nu = 2.0
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_conflict)

    # High filling nu > 4 in single valley raises ValueError unless opted in:
    cfg_nu7 = tmose2_0deg_cb_wigner.get_config()
    cfg_nu7.system.moire.nu = 7.0
    with self.assertRaises(ValueError):
      base_config.resolve(cfg_nu7)

    # 2. Automatic energy_scale (pre-populated in get_config()) refreshes after
    #    mass, lattice, or disable overrides:
    cfg_pbe = tmose2_0deg_cb_wigner.get_config()
    self.assertAlmostEqual(
        float(cfg_pbe.optim.energy_scale), 2.535306e-6, delta=5e-11
    )
    cfg_pbe.system.moire.mass_model = 'pbe'
    cfg_pbe = base_config.resolve(cfg_pbe)
    self.assertAlmostEqual(
        float(cfg_pbe.optim.energy_scale), 2.4566e-6, delta=5e-10
    )

    cfg_lat = tmose2_0deg_cb_wigner.get_config()
    cfg_lat.system.moire.a1_nm = tuple(
        1.1 * x for x in tmose2_continuum.STM_A1_NM
    )
    cfg_lat.system.moire.a2_nm = tuple(
        1.1 * x for x in tmose2_continuum.STM_A2_NM
    )
    cfg_lat = base_config.resolve(cfg_lat)
    self.assertAlmostEqual(
        float(cfg_lat.optim.energy_scale), 2.0953e-6, delta=5e-10
    )

    cfg_noscale = tmose2_0deg_cb_wigner.get_config()
    cfg_noscale.optim.scale_by_moire_energy = False
    cfg_noscale = base_config.resolve(cfg_noscale)
    self.assertIsNone(cfg_noscale.optim.energy_scale)
    self.assertEqual(geometry.resolve_moire_energy_scale(cfg_noscale), 1.0)

    cfg_manual_es = tmose2_0deg_cb_wigner.get_config()
    cfg_manual_es.optim.energy_scale = 5.0e-5
    cfg_manual_es = base_config.resolve(cfg_manual_es)
    self.assertAlmostEqual(float(cfg_manual_es.optim.energy_scale), 5.0e-5)
    self.assertAlmostEqual(
        geometry.resolve_moire_energy_scale(cfg_manual_es), 5.0e-5
    )

    # 3. Secondary valley_index=1 preserves relative mean offset (bar_V_1 - bar_V_0):
    cfg_v1 = tmose2_0deg_cb_wigner.get_config()
    cfg_v1.system.moire.valley_index = 1
    cfg_v1 = base_config.resolve(cfg_v1)
    fc_v1 = float(cfg_v1.system.make_local_energy_kwargs['fourier_const_e'])
    self.assertNotAlmostEqual(fc_v1, 0.0, places=9)
    self.assertAlmostEqual(
        fc_v1,
        float(cfg_v1.system.moire.valley_relative_offset_hartree),
        places=15,
    )

  def test_delta_t_unstrained_geometry_and_preset_configs(self):
    """Checks delta_t_mev=+9 meV, unstrained 60-deg geometry, and nu=1..4 presets."""
    import importlib

    # 1. Strained delta_t_mev = +9 meV on +-Q_1 (valley_index=2, Ferreira idx 1):
    m_str_dt9 = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0, delta_t_mev=9.0, grid_n=64
    )
    self.assertEqual(
        tuple(v.ferreira_index for v in m_str_dt9.valleys), (3, 2, 1)
    )
    vq1_str = m_str_dt9.valleys[2]
    n = vq1_str.v_eff_mev.shape[0]
    v_str_shift = vq1_str.v_eff_mev - np.min(vq1_str.v_eff_mev)
    self.assertLess(
        float(v_str_shift[n // 2, n // 2]),
        float(v_str_shift[n // 2, 0]) - 20.0,
    )

    # 2. Unstrained equilateral 60-deg geometry (epsilon_het = 0%, L_M = 18.6709 nm):
    m_unstr = tmose2_continuum.build_tmose2_continuum_model(
        nu_filling=3.0,
        deformation_model='none',
        include_heterostrain_deformation=False,
        phi_homo_deg=0.0,
        a1_nm=tmose2_continuum.UNSTRAINED_A1_NM,
        a2_nm=tmose2_continuum.UNSTRAINED_A2_NM,
        delta_t_mev=9.0,
        ez_use_cell_area=True,
        deterministic_valley_order=True,
        grid_n=64,
    )
    np.testing.assert_allclose(m_unstr.eps_tb_hom_lab, 0.0, atol=1e-15)
    self.assertAlmostEqual(
        m_unstr.ez_phys_mv_per_angstrom, 1.233267, places=5
    )
    vq1_unstr = m_unstr.valleys[2]
    np.testing.assert_allclose(
        vq1_unstr.mass_tensor_m0,
        ((1.08, 0.0), (0.0, 0.48)),
        rtol=0.0,
        atol=1e-15,
    )
    v_unstr_shift = vq1_unstr.v_eff_mev - np.min(vq1_unstr.v_eff_mev)
    e_dw1 = float(v_unstr_shift[n // 2, n // 2])
    e_mx = float(v_unstr_shift[(2 * n) // 3, (2 * n) // 3])
    e_xm = float(v_unstr_shift[n // 3, n // 3])
    e_dw3 = float(v_unstr_shift[n // 2, 0])
    self.assertAlmostEqual(e_dw1, 0.0, places=9)
    self.assertLess(e_dw1, e_mx)
    self.assertLess(e_mx, e_xm)
    self.assertLess(e_xm, e_dw3)

    # 3. All 8 strained & unstrained production presets resolve cleanly:
    expected = (
        ('tmose2_0deg_cb_nu1_unpol_4x4_valley2_dt9', (8, 8, 0, 0), False),
        ('tmose2_0deg_cb_nu2_unpol_3x3_valley2_dt9', (9, 9, 0, 0), False),
        ('tmose2_0deg_cb_nu3_unpol_2x3_valley2_dt9', (9, 9, 0, 0), False),
        ('tmose2_0deg_cb_nu4_unpol_2x3_valley2_dt9', (12, 12, 0, 0), False),
        ('tmose2_0deg_cb_nu1_unpol_4x4_valley2_dt9_unstrained', (8, 8, 0, 0), True),
        ('tmose2_0deg_cb_nu2_unpol_3x3_valley2_dt9_unstrained', (9, 9, 0, 0), True),
        ('tmose2_0deg_cb_nu3_unpol_2x3_valley2_dt9_unstrained', (9, 9, 0, 0), True),
        ('tmose2_0deg_cb_nu4_unpol_2x3_valley2_dt9_unstrained', (12, 12, 0, 0), True),
    )
    for mod_stem, nspins_exp, is_unstr in expected:
      mod = importlib.import_module(f'ferminet.configs.{mod_stem}')
      cfg = base_config.resolve(mod.get_config())
      self.assertEqual(tuple(cfg.system.electrons), nspins_exp)
      self.assertEqual(bool(cfg.system.moire.unstrained), is_unstr)
      self.assertEqual(int(cfg.system.moire.valley_index), 2)
      self.assertAlmostEqual(float(cfg.system.moire.delta_t_mev), 9.0)
      self.assertEqual(
          len(cfg.system.make_local_energy_kwargs['fourier_amps_e']), 312
      )

  def test_tmose2_end_to_end_train_step(self):
    """Runs end-to-end VMC step (nu=2 on 1x1 LLL cell, jastrow='none')."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_run = tmose2_0deg_cb_wigner.get_config()
      cfg_run.system.moire.nu = 2.0
      cfg_run.system.moire.fourier_cutoff = 8
      cfg_run.system.moire.supercell_matrix = ((1, 0), (0, 1))
      cfg_run.system.moire.use_lll_primitive_cell = True
      cfg_run.system.moire.ewald_gmax = 8
      cfg_run.network.ferminet.hidden_dims = ((8, 4),)
      cfg_run.network.determinants = 1
      cfg_run.batch_size = 8
      cfg_run.mcmc.burn_in = 1
      cfg_run.mcmc.steps = 2
      cfg_run.optim.iterations = 1
      cfg_run.debug.deterministic = True
      cfg_run.debug.check_nan = True
      cfg_run.log.save_path = tmpdir
      cfg_run = base_config.resolve(cfg_run)
      self.assertEqual(tuple(cfg_run.system.electrons), (1, 1, 0, 0))
      train.train(cfg_run)
      self.assertTrue(os.path.exists(os.path.join(tmpdir, 'train_stats.csv')))


if __name__ == '__main__':
  absltest.main()
