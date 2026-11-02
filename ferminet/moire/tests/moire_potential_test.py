"""Tests for multi-shell and reconstructed scalar moire potentials."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import networks
from ferminet.configs import moire_bilayer
from ferminet.moire import geometry
from ferminet.moire import hamiltonian
import jax
import jax.numpy as jnp
import numpy as np


class MoirePotentialTest(parameterized.TestCase):

  def test_moire_harmonic_shells_geometry_and_c3_symmetry(self):
    a_m = 151.18
    prim = geometry.triangular_lattice(a_m)
    g_first = np.asarray(geometry.moire_harmonic_gvectors(prim))
    shells = np.asarray(geometry.moire_harmonic_shells(prim, num_shells=10))
    self.assertEqual(shells.shape, (10, 3, 2))

    # Shell 0 must be identical to moire_harmonic_gvectors.
    np.testing.assert_allclose(shells[0], g_first, rtol=1e-6, atol=1e-6)

    # Expected squared-norm multipliers Q(p, q) = p^2 + p*q + q^2.
    expected_q = [1, 3, 4, 7, 7, 9, 12, 13, 13, 16]
    g_m_norm = 4.0 * np.pi / (np.sqrt(3.0) * a_m)

    # 120-degree counterclockwise rotation matrix.
    c, s = -0.5, np.sqrt(3.0) / 2.0
    r_c3 = np.array([[c, -s], [s, c]], dtype=np.float64)

    seen_vectors = []
    for idx, q_mult in enumerate(expected_q):
      triad = shells[idx]
      # Sum of C3 triad vectors must vanish.
      np.testing.assert_allclose(
          np.sum(triad, axis=0), np.zeros(2), atol=1e-5
      )
      # Norms must match sqrt(Q_s) * G_M.
      norms = np.linalg.norm(triad, axis=1)
      np.testing.assert_allclose(
          norms, np.full(3, np.sqrt(q_mult) * g_m_norm), rtol=1e-5
      )
      # Each triad must cycle under 120-degree rotation: R_C3 @ G_m = G_{m+1}.
      np.testing.assert_allclose((r_c3 @ triad[0]), triad[1], rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose((r_c3 @ triad[1]), triad[2], rtol=1e-5, atol=1e-6)
      np.testing.assert_allclose((r_c3 @ triad[2]), triad[0], rtol=1e-5, atol=1e-6)

      # Reciprocal lattice commensurability: G . a_i / (2*pi) must be integers.
      integer_proj = (triad @ prim) / (2.0 * np.pi)
      np.testing.assert_allclose(
          integer_proj, np.round(integer_proj), atol=1e-5
      )

      # Check no duplicate +-G vectors across all triads.
      for g_vec in triad:
        for prev in seen_vectors:
          self.assertGreater(np.linalg.norm(g_vec - prev), 1e-4)
          self.assertGreater(np.linalg.norm(g_vec + prev), 1e-4)
        seen_vectors.append(g_vec)

  def test_single_species_backward_compatibility_and_shell_additivity(self):
    prim = geometry.triangular_lattice(151.18)
    g1 = geometry.moire_harmonic_gvectors(prim)
    g3 = geometry.moire_harmonic_shells(prim, num_shells=3)
    coords = jnp.array([[12.3, -45.6], [78.9, 23.4], [-10.0, 50.0]])
    offset = (5.0, -7.5)

    v1 = -1.344658e-4
    phi1 = 0.888198
    pot_scalar = hamiltonian.single_species_moire_potential(
        coords, g1, amplitude=v1, phase=phi1, offset=offset
    )
    pot_tuple1 = hamiltonian.single_species_moire_potential(
        coords, g1, amplitude=(v1,), phase=(phi1,), offset=offset
    )
    pot_padded3 = hamiltonian.single_species_moire_potential(
        coords,
        g3,
        amplitude=(v1, 0.0, 0.0),
        phase=(phi1, 0.5, -1.2),
        offset=offset,
    )
    np.testing.assert_allclose(pot_scalar, pot_tuple1, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(pot_scalar, pot_padded3, rtol=1e-6, atol=1e-10)

    # Multi-shell evaluation with 2D (3, 2) g1 must auto-expand shells and match
    # explicit (3, 3, 2) g3, as well as the sum of individual shell evaluations.
    v_3shell = (-1.344658e-4, 1.025306e-5, -2.215984e-5)
    phi_3shell = (0.888198, 0.185005, -1.923353)
    pot_from_g1 = hamiltonian.single_species_moire_potential(
        coords, g1, amplitude=v_3shell, phase=phi_3shell, offset=offset
    )
    pot_from_g3 = hamiltonian.single_species_moire_potential(
        coords, g3, amplitude=v_3shell, phase=phi_3shell, offset=offset
    )
    pot_sum = sum(
        hamiltonian.single_species_moire_potential(
            coords, g3[s], amplitude=v_3shell[s], phase=phi_3shell[s], offset=offset
        )
        for s in range(3)
    )
    np.testing.assert_allclose(pot_from_g1, pot_from_g3, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(pot_from_g3, pot_sum, rtol=1e-6, atol=1e-10)

  @parameterized.parameters(
      *list(hamiltonian.MOIRE_POTENTIAL_PRESETS.keys())
  )
  def test_preset_periodicity_c3_symmetry_and_zero_cell_mean(self, preset_name):
    a_m = 151.18
    prim = geometry.triangular_lattice(a_m)
    sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))
    offset = np.array([11.0, -19.0])

    pot_fn = hamiltonian.make_moire_potential(
        nspins=(0, 1),
        simulation_lattice=sim,
        moire_lattice=prim,
        offset_h=offset,
        potential_model_h=preset_name,
    )

    rng = np.random.default_rng(42)
    pts = rng.uniform(-100.0, 100.0, size=(8, 2))
    c, s = -0.5, np.sqrt(3.0) / 2.0
    r_c3 = np.array([[c, -s], [s, c]], dtype=np.float64)

    for pt in pts:
      val_0 = float(pot_fn(jnp.asarray(pt)))
      # Lattice periodicity under primitive vectors a_1 and a_2.
      val_a1 = float(pot_fn(jnp.asarray(pt + prim[:, 0])))
      val_a2 = float(pot_fn(jnp.asarray(pt + prim[:, 1])))
      self.assertAlmostEqual(val_0, val_a1, places=7)
      self.assertAlmostEqual(val_0, val_a2, places=7)

      # C3 rotational invariance around offset.
      pt_rot = offset + r_c3 @ (pt - offset)
      val_rot = float(pot_fn(jnp.asarray(pt_rot)))
      self.assertAlmostEqual(val_0, val_rot, places=7)

    # Zero mean over a uniform grid covering the primitive moire unit cell.
    n_grid = 18
    u = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    uu, vv = np.meshgrid(u, u, indexing='ij')
    frac = np.stack([uu.ravel(), vv.ravel()], axis=-1)
    cart = frac @ prim.T
    v_arr, phi_arr = hamiltonian.resolve_moire_potential_params(
        model=preset_name
    )
    shells = geometry.moire_harmonic_shells(prim, num_shells=len(v_arr))
    total_grid_sum = float(
        hamiltonian.single_species_moire_potential(
            jnp.asarray(cart), shells, amplitude=v_arr, phase=phi_arr
        )
    )
    self.assertAlmostEqual(total_grid_sum / len(cart), 0.0, places=9)

  def test_luskin_table_s1_wyckoff_values(self):
    # Verify that Luskin et al. (2026) Table S1 SCAN-rVV10 3-shell preset
    # reproduces the exact valence-band values (+10.954, -25.638, +9.748) meV
    # -> hole potential (-10.954, +25.638, -9.748) meV at the 3 Wyckoff sites.
    prim = geometry.triangular_lattice(151.18)
    pot_fn = hamiltonian.make_moire_potential(
        nspins=(0, 1),
        simulation_lattice=prim,
        moire_lattice=prim,
        potential_model_h='luskin_2026_3shell_scan',
    )
    wyckoff_frac = [
        np.array([0.0, 0.0]),
        np.array([1.0 / 3.0, 2.0 / 3.0]),
        np.array([2.0 / 3.0, 1.0 / 3.0]),
    ]
    expected_hole_mev = [-10.9539, +25.6382, -9.7483]
    for frac, exp_mev in zip(wyckoff_frac, expected_hole_mev):
      r = prim @ frac
      val_mev = float(pot_fn(jnp.asarray(r))) * hamiltonian.HARTREE_TO_MEV
      self.assertAlmostEqual(val_mev, exp_mev, places=3)

  def test_luskin_continuum_relaxation_solver_matches_presets(self):
    for func, preset_key in [
        ('scan', 'luskin_2026_relaxed_scan'),
        ('dftd3', 'luskin_2026_relaxed_dftd3'),
    ]:
      v_sol, phi_sol = hamiltonian.solve_luskin_continuum_relaxed_potential(
          theta_deg=0.0,
          delta=0.04,
          functional=func,
          include_deformation_potential=True,
          num_shells=6,
      )
      v_pre, phi_pre = hamiltonian.resolve_moire_potential_params(
          model=preset_key
      )
      np.testing.assert_allclose(v_sol, v_pre, rtol=1e-4, atol=1e-9)
      np.testing.assert_allclose(phi_sol, phi_pre, rtol=1e-4, atol=1e-5)

  def test_wang_2026_3shell_fit_matches_19_point_dft_diagonal(self):
    # Extracted 19 DFT points (meV) from Wang, Gao, and Tong (2026) Fig. 6d/7d.
    v_r_dft = np.array([
        17.16, 17.84, 11.08, 9.41, 5.14, 1.62, -0.05, 3.11, 14.86, 31.84,
        53.00, 68.78, 73.24, 73.14, 65.73, 49.32, 32.43, 19.19, 17.78,
    ])
    v_h_dft = np.array([
        6.61, 9.10, 13.97, 23.83, 32.82, 40.59, 41.59, 40.20, 34.40, 25.13,
        13.75, 4.44, 2.38, 2.38, 1.99, 0.02, 2.38, 4.16, 5.96,
    ])

    prim = geometry.triangular_lattice(155.90)  # 8.25 nm in Bohr
    # Long diagonal direction d = -(2 a_1 + a_2) so t=1/3 is Wyckoff A and t=2/3 is Wyckoff B.
    diag_vec = -(2.0 * prim[:, 0] + prim[:, 1])
    t_vals = np.linspace(0.0, 1.0, 18, endpoint=False)
    coords = np.outer(t_vals, diag_vec)

    for preset_name, dft_curve, max_rmse_mev in [
        ('wang_2026_rtype_3shell', v_r_dft, 1.55),
        ('wang_2026_htype_3shell', v_h_dft, 0.90),
    ]:
      v_arr, phi_arr = hamiltonian.resolve_moire_potential_params(
          model=preset_name
      )
      shells = geometry.moire_harmonic_shells(prim, num_shells=3)
      pred_mev = np.array([
          float(
              hamiltonian.single_species_moire_potential(
                  jnp.asarray(pt[None, :]), shells, amplitude=v_arr, phase=phi_arr
              )
          )
          * hamiltonian.HARTREE_TO_MEV
          for pt in coords
      ])
      y_dft = dft_curve[:18].copy()
      y_dft[0] = 0.5 * (dft_curve[0] + dft_curve[18])
      u_hole_dft = -(y_dft - np.mean(y_dft))
      rmse = float(np.sqrt(np.mean((pred_mev - u_hole_dft) ** 2)))
      self.assertLess(rmse, max_rmse_mev)

  def test_local_energy_and_config_with_multishell_and_presets(self):
    cfg = moire_bilayer.get_config()
    cfg.system.moire.n_electrons_up = 1
    cfg.system.moire.n_electrons_down = 0
    cfg.system.moire.n_holes_up = 1
    cfg.system.moire.n_holes_down = 0
    cfg.system.moire.supercell_matrix = ((1, 0), (0, 1))
    cfg.system.moire.potential_model_h = 'luskin_2026_relaxed_scan'
    cfg = moire_bilayer.finalize(cfg)

    kwargs = cfg.system.make_local_energy_kwargs
    self.assertEqual(len(kwargs['v_h']), 6)
    self.assertEqual(len(kwargs['phi_h']), 6)

    # Dummy Gaussian wavefunction f(params, pos, spins, atoms, charges) -> (sign, logabs).
    def dummy_wf(params, pos, spins, atoms, charges):
      del params, spins, atoms, charges
      return jnp.ones((), dtype=pos.dtype), -0.001 * jnp.sum(pos**2)

    e_l_fn = hamiltonian.local_energy(
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
    total_e, _ = e_l_fn({}, jax.random.PRNGKey(0), data)
    self.assertTrue(bool(jnp.isfinite(total_e)))

    # Verify folx laplacian_method matches default laplacian_method.
    e_l_folx = hamiltonian.local_energy(
        f=dummy_wf,
        charges=jnp.zeros((1,)),
        nspins=cfg.system.electrons,
        use_scan=False,
        complex_output=False,
        laplacian_method='folx',
        **kwargs,
    )
    total_e_folx, _ = e_l_folx({}, jax.random.PRNGKey(0), data)
    np.testing.assert_allclose(total_e, total_e_folx, rtol=1e-5, atol=1e-6)

  def test_explicit_tuple_config_matches_preset_and_validates_inputs(self):
    cfg_preset = moire_bilayer.get_config()
    cfg_preset.system.moire.potential_model_h = 'luskin_2026_3shell_scan'
    cfg_preset = moire_bilayer.finalize(cfg_preset)

    cfg_tuple = moire_bilayer.get_config()
    cfg_tuple.system.moire.v_h = (-1.344658e-4, +1.025306e-5, -2.215984e-5)
    cfg_tuple.system.moire.phi_h = (0.888198, 0.185005, -1.923353)
    cfg_tuple = moire_bilayer.finalize(cfg_tuple)

    np.testing.assert_allclose(
        cfg_preset.system.make_local_energy_kwargs['v_h'],
        cfg_tuple.system.make_local_energy_kwargs['v_h'],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        cfg_preset.system.make_local_energy_kwargs['phi_h'],
        cfg_tuple.system.make_local_energy_kwargs['phi_h'],
        rtol=1e-7,
    )

    with self.assertRaises(ValueError):
      hamiltonian.resolve_moire_potential_params(
          amplitude=(1.0e-4, 2.0e-4), phase=(0.1, 0.2, 0.3)
      )
    with self.assertRaises(ValueError):
      hamiltonian.resolve_moire_potential_params(model='nonexistent_preset')


if __name__ == '__main__':
  absltest.main()
