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

"""Tests for ferminet.moire package and moire_bilayer config."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from ferminet import base_config
from ferminet import envelopes
from ferminet import jastrows
from ferminet import mcmc
from ferminet import networks
from ferminet import train
from ferminet.configs import moire_bilayer
from ferminet.moire import ewald
from ferminet.moire import geometry
from ferminet.moire import hamiltonian
from ferminet.pbc import envelopes as pbc_envelopes
from ferminet.pbc import feature_layer as pbc_feature_layer
from ferminet.utils import system
import jax
import jax.numpy as jnp
import numpy as np

FLAGS = flags.FLAGS
FLAGS.mark_as_parsed()


class MoireGeometryTest(parameterized.TestCase):

  def test_triangular_lattice_and_reciprocal(self):
    a_m = 25.0
    g_m = 4.0 * np.pi / (np.sqrt(3.0) * a_m)
    lat_120 = geometry.triangular_lattice(a_m)
    self.assertEqual(lat_120.shape, (2, 2))
    np.testing.assert_allclose(np.linalg.norm(lat_120, axis=0), [a_m, a_m])

    rec_rows = geometry.reciprocal_vectors(lat_120)
    np.testing.assert_allclose(
        rec_rows @ lat_120, 2.0 * np.pi * np.eye(2), atol=1e-6
    )

    # 60-degree primitive basis spanning the exact same triangular lattice.
    lat_60 = np.column_stack([lat_120[:, 0] + lat_120[:, 1], lat_120[:, 1]])
    np.testing.assert_allclose(np.linalg.norm(lat_60, axis=0), [a_m, a_m])

    for lat in (lat_120, lat_60):
      g_shell = geometry.moire_first_shell_gvectors(lat)
      self.assertEqual(g_shell.shape, (6, 2))
      np.testing.assert_allclose(
          np.linalg.norm(g_shell, axis=-1), np.full(6, g_m), atol=1e-6
      )

      g_harm = geometry.moire_harmonic_gvectors(lat)
      self.assertEqual(g_harm.shape, (3, 2))
      np.testing.assert_allclose(
          np.linalg.norm(g_harm, axis=-1), np.full(3, g_m), atol=1e-6
      )
      np.testing.assert_allclose(np.sum(g_harm, axis=0), [0.0, 0.0], atol=1e-6)

    # Non-triangular lattice must be rejected.
    with self.assertRaises(ValueError):
      geometry.moire_first_shell_gvectors(np.diag([10.0, 14.0]))

    self.assertEqual(geometry.layer_particle_counts((2, 1, 1, 2)), (3, 3))
    self.assertEqual(geometry.layer_particle_counts((3, 2)), (3, 2))
    with self.assertRaises(ValueError):
      geometry.layer_particle_counts((1, 1, 1))

  def test_supercell_and_commensurability(self):
    prim = geometry.triangular_lattice(10.0)
    sc = geometry.supercell_lattice(prim, [[2, 0], [0, 3]])
    s_mat = geometry.validate_commensurate(prim, sc)
    np.testing.assert_array_equal(s_mat, [[2, 0], [0, 3]])

    with self.assertRaises(ValueError):
      geometry.validate_commensurate(prim, prim * 1.35)

  def test_wrap_positions_shapes_and_periodicity(self):
    lat = geometry.triangular_lattice(12.0)
    rng = np.random.default_rng(0)
    frac = rng.uniform(-2.5, 3.5, size=(4, 5, 2))
    cart = geometry.fractional_to_cartesian(frac, lat, ndim=2)

    wrapped = geometry.wrap_positions(cart, lat, ndim=2)
    wrapped_frac = geometry.cartesian_to_fractional(wrapped, lat, ndim=2)
    np.testing.assert_allclose(wrapped_frac, frac % 1.0, atol=1e-5)

    # Also test flattened (batch, nelec * 2) representation.
    cart_flat = jnp.reshape(cart, (4, 10))
    wrapped_flat = geometry.wrap_positions(cart_flat, lat, ndim=2)
    np.testing.assert_allclose(
        wrapped_flat, jnp.reshape(wrapped, (4, 10)), atol=1e-5
    )


class BilayerEwaldTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'nspins': (2, 1, 1, 2),
          'd': 0.0,
          'include_capacitor': True,
          'allow_charged_capacitor': False,
      },
      {
          'nspins': (2, 1, 1, 2),
          'd': 3.5,
          'include_capacitor': True,
          'allow_charged_capacitor': False,
      },
      {
          'nspins': (2, 1, 1, 1),
          'd': 3.5,
          'include_capacitor': False,
          'allow_charged_capacitor': False,
      },
      {
          'nspins': (3, 2),
          'd': 3.5,
          'include_capacitor': True,
          'allow_charged_capacitor': True,
      },
  )
  def test_alpha_invariance(
      self, nspins, d, include_capacitor, allow_charged_capacitor
  ):
    lat = geometry.supercell_lattice(
        geometry.triangular_lattice(10.0), [[2, 0], [0, 2]]
    )
    n_e, _ = geometry.layer_particle_counts(nspins)
    key = jax.random.PRNGKey(17)
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
        alpha=0.25,
        include_capacitor=include_capacitor,
        allow_charged_capacitor=allow_charged_capacitor,
    )
    ew2 = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=4.5,
        ewald_gmax=80,
        nlatvec=2,
        alpha=0.45,
        include_capacitor=include_capacitor,
        allow_charged_capacitor=allow_charged_capacitor,
    )
    c1 = ew1.components(pos_e, pos_h)
    c2 = ew2.components(pos_e, pos_h)
    for u1, u2 in zip(c1, c2):
      np.testing.assert_allclose(u1, u2, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(
        ew1.energy(pos), ew2.energy(pos), atol=1e-4, rtol=1e-4
    )

  def test_charged_capacitor_requires_explicit_opt_in(self):
    lat = geometry.triangular_lattice(12.0)
    with self.assertRaises(ValueError):
      ewald.BilayerEwaldSum(
          lat,
          nspins=(3, 2),
          interlayer_distance=4.0,
          include_capacitor=True,
      )

  def test_capacitor_term_and_periodicity(self):
    lat = geometry.triangular_lattice(15.0)
    nspins = (2, 2)
    d = 4.0
    eps = 5.0
    key = jax.random.PRNGKey(99)
    pos = jax.random.uniform(key, shape=(sum(nspins), 2)) @ jnp.asarray(lat).T

    ew_cap = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=eps,
        include_capacitor=True,
    )
    ew_nocap = ewald.BilayerEwaldSum(
        lat,
        nspins,
        interlayer_distance=d,
        dielectric_constant=eps,
        include_capacitor=False,
    )
    e_cap = ew_cap.energy(pos.reshape(-1))
    e_nocap = ew_nocap.energy(pos.reshape(-1))
    expected_cap = (
        2.0 * np.pi * d * nspins[0] * nspins[1] / (eps * ew_cap.area)
    )
    np.testing.assert_allclose(e_cap - e_nocap, expected_cap, atol=1e-5, rtol=1e-5)

    # Displace a hole by a random supercell lattice vector (2*a_1 - 3*a_2).
    shift = (2.0 * lat[:, 0] - 3.0 * lat[:, 1]).astype(np.float32)
    pos_shifted = pos.at[3].add(shift)
    e_shifted = ew_cap.energy(pos_shifted.reshape(-1))
    np.testing.assert_allclose(e_cap, e_shifted, atol=1e-4, rtol=1e-4)


class MoireHamiltonianTest(parameterized.TestCase):

  @parameterized.parameters(
      {'laplacian_method': 'default', 'use_scan': False, 'complex_output': False},
      {'laplacian_method': 'default', 'use_scan': True, 'complex_output': False},
      {'laplacian_method': 'folx', 'use_scan': False, 'complex_output': False},
      {'laplacian_method': 'default', 'use_scan': False, 'complex_output': True},
      {'laplacian_method': 'folx', 'use_scan': False, 'complex_output': True},
  )
  def test_mass_weighted_kinetic_energy(
      self, laplacian_method, use_scan, complex_output
  ):
    nspins = (1, 1, 1, 0)
    masses = (0.4, 0.8)
    ndim = 2
    # Test wavefunction: log|psi(x)| = -0.5 * sum_k c_k x_k^2,
    # phase(x) = sum_k p_k x_k^2 (if complex_output).
    ncoord = sum(nspins) * ndim
    c_coeff = jnp.arange(1.0, ncoord + 1.0) * 0.1
    p_coeff = jnp.arange(1.0, ncoord + 1.0) * 0.05

    def fake_net(params, pos, spins, atoms, charges):
      del params, spins, atoms, charges
      logabs = -0.5 * jnp.sum(c_coeff * (pos**2))
      if complex_output:
        phase = jnp.sum(p_coeff * (pos**2))
      else:
        phase = jnp.ones_like(logabs)
      return phase, logabs

    ke_fn = hamiltonian.local_kinetic_energy(
        fake_net,
        nspins=nspins,
        masses=masses,
        ndim=ndim,
        use_scan=use_scan,
        complex_output=complex_output,
        laplacian_method=laplacian_method,
    )
    pos = jnp.linspace(0.2, 1.2, ncoord)
    data = networks.FermiNetData(
        positions=pos,
        spins=jnp.array([0.0, 1.0, 2.0]),
        atoms=jnp.zeros((1, 2)),
        charges=jnp.zeros((1,)),
    )
    val = ke_fn({}, data)

    inv_m = jnp.array([1.0 / 0.4] * 4 + [1.0 / 0.8] * 2)
    grad_log = -c_coeff * pos
    lap_log = -c_coeff
    expected = -0.5 * jnp.sum(inv_m * (lap_log + grad_log**2))
    if complex_output:
      grad_p = 2.0 * p_coeff * pos
      lap_p = 2.0 * p_coeff
      expected = (
          expected
          - 0.5j * jnp.sum(inv_m * lap_p)
          + 0.5 * jnp.sum(inv_m * (grad_p**2))
          - 1.0j * jnp.sum(inv_m * grad_log * grad_p)
      )
    np.testing.assert_allclose(val, expected, atol=1e-5, rtol=1e-5)

  def test_local_energy_periodicity_and_2d_kpoints(self):
    prim = geometry.triangular_lattice(20.0)
    sim_lat = geometry.supercell_lattice(prim, [[2, 0], [0, 2]])
    nspins = (1, 1, 1, 1)
    atoms = jnp.zeros((1, 2))
    charges = jnp.zeros((1,))
    spins = jnp.array([0.0, 1.0, 2.0, 3.0])

    # Verify 2D make_kpoints works with upstream ndim=2 keyword argument and
    # produces reciprocal lattice vectors satisfying (k . a_j) / (2*pi) in Z.
    kpts_2d = pbc_envelopes.make_kpoints(sim_lat, nspins, ndim=2)
    self.assertEqual(kpts_2d.shape[1], 2)
    reduced_k = np.asarray(kpts_2d) @ sim_lat / (2.0 * np.pi)
    np.testing.assert_allclose(reduced_k, np.round(reduced_k), atol=1e-5)

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=jnp.asarray(sim_lat),
        include_r_ae=False,
        interlayer_distance=5.0,
    )
    net = networks.make_fermi_net(
        nspins,
        charges,
        ndim=2,
        lattice=jnp.asarray(sim_lat),
        determinants=2,
        envelope=envelopes.make_null_envelope(),
        feature_layer=feature_layer,
        jastrow='bilayer',
        jastrow_kwargs={
            'mass_e': 0.5,
            'mass_h': 0.7,
            'dielectric_constant': 8.0,
            'interlayer_distance': 5.0,
        },
        full_det=False,
        hidden_dims=((16, 4), (16, 4)),
    )
    key = jax.random.PRNGKey(7)
    key, init_key, pos_key = jax.random.split(key, 3)
    params = net.init(init_key)
    xs = jax.random.uniform(pos_key, shape=(sum(nspins), 2)) @ jnp.asarray(
        sim_lat
    ).T

    el_fn = hamiltonian.local_energy(
        f=net.apply,
        charges=charges,
        nspins=nspins,
        ndim=2,
        lattice=sim_lat,
        moire_lattice=prim,
        masses=(0.5, 0.7),
        interlayer_distance=5.0,
        dielectric_constant=8.0,
        v_e=1e-3,
        v_h=-8e-4,
        phi_e=0.3,
        phi_h=-0.5,
        offset_e=(1.0, -0.5),
        offset_h=(0.0, 2.0),
    )

    data1 = networks.FermiNetData(
        positions=xs.reshape(-1), spins=spins, atoms=atoms, charges=charges
    )
    e1, _ = el_fn(params, key, data1)

    # Displace one electron and one hole by supercell lattice vectors.
    xs2 = xs.at[0].add(jnp.asarray(sim_lat[:, 0])).at[2].add(
        -2.0 * jnp.asarray(sim_lat[:, 1])
    )
    data2 = networks.FermiNetData(
        positions=xs2.reshape(-1), spins=spins, atoms=atoms, charges=charges
    )
    e2, _ = el_fn(params, key, data2)
    np.testing.assert_allclose(e1, e2, atol=2e-3, rtol=2e-3)

  def test_coincident_eh_feature_derivatives_and_4flavour_antisymmetry(self):
    lat = geometry.triangular_lattice(20.0)
    d = 4.0
    atoms = jnp.zeros((1, 2))
    charges = jnp.zeros((1,))

    # 1. Verify smooth finite gradient and Hessian at coincident e-h positions.
    fl = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=(1, 0, 1, 0),
        ndim=2,
        lattice=jnp.asarray(lat),
        include_r_ae=False,
        interlayer_distance=d,
    )
    _, fl_params = fl.init()

    def eh_distance(r_e):
      pos = jnp.concatenate([r_e, jnp.array([2.5, -1.5])])
      ae, ee, r_ae, r_ee = networks.construct_input_features(
          pos, atoms, ndim=2
      )
      _, ee_feat = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **fl_params)
      return ee_feat[0, 1, 0]

    r_coincident = jnp.array([2.5, -1.5])
    val = eh_distance(r_coincident)
    grad = jax.grad(eh_distance)(r_coincident)
    hess = jax.hessian(eh_distance)(r_coincident)
    np.testing.assert_allclose(val, d, atol=1e-6)
    np.testing.assert_allclose(grad, jnp.zeros(2), atol=1e-6)
    np.testing.assert_allclose(hess, jnp.eye(2) / d, atol=1e-5)

    # 2. Verify 4-flavour exchange antisymmetry with >1 particle per flavour.
    nspins = (2, 1, 1, 2)
    spins = jnp.repeat(jnp.arange(4, dtype=jnp.float32), jnp.asarray(nspins))
    fl_many = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=jnp.asarray(lat),
        include_r_ae=False,
        interlayer_distance=d,
    )
    net = networks.make_fermi_net(
        nspins,
        charges,
        ndim=2,
        lattice=jnp.asarray(lat),
        determinants=2,
        envelope=envelopes.make_null_envelope(),
        feature_layer=fl_many,
        jastrow='none',
        full_det=False,
        hidden_dims=((16, 4), (16, 4)),
    )
    key = jax.random.PRNGKey(42)
    params = net.init(key)
    coords = (
        jax.random.uniform(key, shape=(sum(nspins), 2))
        @ jnp.asarray(lat).T
    )
    sign_orig, log_orig = net.apply(
        params, coords.reshape(-1), spins, atoms, charges
    )
    # Swap the two e_up particles (0 and 1).
    coords_swap_e = coords.at[jnp.array([0, 1])].set(
        coords[jnp.array([1, 0])]
    )
    sign_e, log_e = net.apply(
        params, coords_swap_e.reshape(-1), spins, atoms, charges
    )
    np.testing.assert_allclose(sign_e, -sign_orig, atol=1e-5)
    np.testing.assert_allclose(log_e, log_orig, atol=1e-5)

    # Swap the two h_down particles (4 and 5).
    coords_swap_h = coords.at[jnp.array([4, 5])].set(
        coords[jnp.array([5, 4])]
    )
    sign_h, log_h = net.apply(
        params, coords_swap_h.reshape(-1), spins, atoms, charges
    )
    np.testing.assert_allclose(sign_h, -sign_orig, atol=1e-5)
    np.testing.assert_allclose(log_h, log_orig, atol=1e-5)

  def test_bilayer_jastrow_cusps(self):
    mass_e = 0.5
    mass_h = 0.6
    eps = 10.0
    j_init, j_apply = jastrows.make_bilayer_jastrow(
        lattice=None,
        ndim=2,
        mass_e=mass_e,
        mass_h=mass_h,
        dielectric_constant=eps,
        interlayer_distance=5.0,
    )
    j_params = j_init()

    def pair_slope(nspins):
      def f(r):
        pos = jnp.array([[0.0, 0.0], [r, 0.0]])
        ee = pos[None, :, :] - pos[:, None, :]
        return j_apply(ee, j_params, nspins)

      return float(jax.grad(f)(1e-6))

    # Same-flavour electron pair: m_e / (3 * eps).
    np.testing.assert_allclose(
        pair_slope((2, 0, 0, 0)), mass_e / (3.0 * eps), rtol=1e-4
    )
    # Different-flavour electron pair: m_e / eps.
    np.testing.assert_allclose(
        pair_slope((1, 1, 0, 0)), mass_e / eps, rtol=1e-4
    )
    # Same-flavour hole pair: m_h / (3 * eps).
    np.testing.assert_allclose(
        pair_slope((0, 0, 2, 0)), mass_h / (3.0 * eps), rtol=1e-4
    )
    # Different-flavour hole pair: m_h / eps.
    np.testing.assert_allclose(
        pair_slope((0, 0, 1, 1)), mass_h / eps, rtol=1e-4
    )
    # Interlayer e-h pair with d > 0: 0 cusp.
    np.testing.assert_allclose(
        pair_slope((1, 0, 1, 0)), 0.0, atol=1e-7
    )

    # Verify full network with jastrow='bilayer' has finite value, gradient, and
    # Hessian when an electron and hole have identical in-plane coordinates.
    lat = geometry.triangular_lattice(20.0)
    nspins = (1, 1, 1, 1)
    spins = jnp.array([0.0, 1.0, 2.0, 3.0])
    atoms = jnp.zeros((1, 2))
    charges = jnp.zeros((1,))
    fl = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=jnp.asarray(lat),
        include_r_ae=False,
        interlayer_distance=4.0,
    )
    net = networks.make_fermi_net(
        nspins,
        charges,
        ndim=2,
        lattice=jnp.asarray(lat),
        determinants=2,
        envelope=envelopes.make_null_envelope(),
        feature_layer=fl,
        jastrow='bilayer',
        jastrow_kwargs={
            'mass_e': mass_e,
            'mass_h': mass_h,
            'dielectric_constant': eps,
            'interlayer_distance': 4.0,
        },
        full_det=False,
        hidden_dims=((16, 4), (16, 4)),
    )
    params = net.init(jax.random.PRNGKey(9))
    # Electron 0 and hole 0 share the exact same 2D coordinate [1.5, -0.5].
    coincident_pos = jnp.array([
        1.5, -0.5,
        4.0, 2.0,
        1.5, -0.5,
        -3.0, 3.5,
    ])
    logpsi_fn = lambda x: net.apply(params, x, spins, atoms, charges)[1]
    val = logpsi_fn(coincident_pos)
    grad = jax.grad(logpsi_fn)(coincident_pos)
    hess = jax.hessian(logpsi_fn)(coincident_pos)
    self.assertTrue(bool(jnp.isfinite(val)))
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(hess))))


class MoireMcmcAndTrainTest(parameterized.TestCase):

  @parameterized.parameters([1, 2])
  def test_mcmc_wrapping(self, blocks):
    lat = geometry.triangular_lattice(10.0)
    nspins = (1, 1, 1, 1)
    batch_size = 8
    key = jax.random.PRNGKey(12)
    pos, spins = train.init_electrons(
        key,
        molecule=[system.Atom('X', (0.0, 0.0))],
        electrons=nspins,
        batch_size=batch_size,
        init_width=1.0,
        ndim=2,
        lattice=lat,
    )
    self.assertEqual(pos.shape, (batch_size, sum(nspins) * 2))
    self.assertEqual(spins.shape, (batch_size, sum(nspins)))
    np.testing.assert_array_equal(spins[0], [0.0, 1.0, 2.0, 3.0])
    frac_init = geometry.cartesian_to_fractional(pos, lat, ndim=2)
    self.assertTrue(np.all((frac_init >= 0.0) & (frac_init < 1.0)))

    atoms = jnp.zeros((batch_size, 1, 2))
    charges = jnp.zeros((batch_size, 1))
    data = networks.FermiNetData(
        positions=pos, spins=spins, atoms=atoms, charges=charges
    )
    batch_net = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))
    step_fn = mcmc.make_mcmc_step(
        batch_net,
        batch_per_device=batch_size,
        steps=5,
        atoms=None,
        ndim=2,
        blocks=blocks,
        lattice=lat,
    )
    new_data, _ = step_fn({}, data, key, width=50.0)
    frac_after = geometry.cartesian_to_fractional(new_data.positions, lat, ndim=2)
    self.assertTrue(np.all((frac_after >= 0.0) & (frac_after < 1.0)))

  @parameterized.parameters(
      ('adam', 'default'),
      ('adam', 'folx'),
      ('kfac', 'default'),
  )
  def test_moire_bilayer_train_step(self, optimizer, laplacian):
    cfg = moire_bilayer.get_config()
    cfg.system.moire.n_electrons_up = 1
    cfg.system.moire.n_electrons_down = 1
    cfg.system.moire.n_holes_up = 1
    cfg.system.moire.n_holes_down = 1
    cfg.system.moire.mass_e = 0.45
    cfg.system.moire.mass_h = 0.65
    cfg.system.moire.interlayer_distance = 8.0
    cfg.system.moire.dielectric_constant = 6.0
    cfg.system.moire.ewald_gmax = 30
    cfg.network.ferminet.hidden_dims = ((16, 4), (16, 4))
    cfg.network.determinants = 2
    cfg.batch_size = 8
    cfg.mcmc.burn_in = 2
    cfg.mcmc.steps = 2
    cfg.optim.optimizer = optimizer
    cfg.optim.laplacian = laplacian
    cfg.optim.iterations = 2
    cfg.debug.check_nan = True
    cfg.log.save_path = self.create_tempdir().full_path
    cfg = base_config.resolve(cfg)
    self.assertEqual(cfg.system.electrons, (1, 1, 1, 1))
    # Also verify train.train populates 'lattice' when not pre-set in kwargs.
    del cfg.network.make_feature_layer_kwargs['lattice']
    del cfg.system.make_local_energy_kwargs['lattice']
    train.train(cfg)


if __name__ == '__main__':
  absltest.main()
