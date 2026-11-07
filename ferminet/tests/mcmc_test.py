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

"""Tests for ferminet.mcmc, including joint electron-hole pair MCMC sampling."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import mcmc
from ferminet import networks
from ferminet.moire import geometry
import jax
from jax import numpy as jnp
import numpy as np


def _make_dummy_data(positions: jnp.ndarray, nlec: int) -> networks.FermiNetData:
  batch = positions.shape[0]
  return networks.FermiNetData(
      positions=positions,
      spins=jnp.ones((batch, nlec), dtype=jnp.int32),
      atoms=jnp.zeros((batch, 1, 2), dtype=positions.dtype),
      charges=jnp.zeros((batch, 1), dtype=positions.dtype),
  )


class McmcTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 3)
  def test_backward_compatibility_default_and_disabled(self, blocks):
    """pair_prob=0.0 and monolayer nspins must match legacy make_mcmc_step."""
    lattice = geometry.supercell_lattice(
        geometry.triangular_lattice(150.0), np.diag([3, 3])
    )
    batch_size = 8
    nspins = (2, 1, 2, 1)
    nelec = sum(nspins)
    ndim = 2

    key = jax.random.PRNGKey(42)
    key_init, key_run = jax.random.split(key)
    pos = jax.random.uniform(
        key_init, (batch_size, nelec * ndim), minval=0.0, maxval=100.0
    )
    data = _make_dummy_data(pos, nelec)

    def log_psi(params, x, spins, atoms, charges):
      del params, spins, atoms, charges
      return -0.001 * jnp.sum(x**2, axis=-1)

    step_legacy = jax.pmap(
        mcmc.make_mcmc_step(
            log_psi,
            batch_per_device=batch_size,
            steps=5,
            ndim=ndim,
            blocks=blocks,
            lattice=lattice,
        ),
        in_axes=(None, 0, 0, None),
    )
    step_disabled = jax.pmap(
        mcmc.make_mcmc_step(
            log_psi,
            batch_per_device=batch_size,
            steps=5,
            ndim=ndim,
            blocks=blocks,
            lattice=lattice,
            nspins=nspins,
            pair_prob=0.0,
            pair_radius=50.0,
        ),
        in_axes=(None, 0, 0, None),
    )
    step_monolayer = jax.pmap(
        mcmc.make_mcmc_step(
            log_psi,
            batch_per_device=batch_size,
            steps=5,
            ndim=ndim,
            blocks=blocks,
            lattice=lattice,
            nspins=(0, 0, 3, 3),
            pair_prob=0.5,
            pair_radius=50.0,
        ),
        in_axes=(None, 0, 0, None),
    )

    sharded_data = jax.tree_util.tree_map(lambda x: x[None, ...], data)
    sharded_key = key_run[None, ...]
    width = jnp.asarray(2.0)

    out_legacy, pmove_legacy = step_legacy({}, sharded_data, sharded_key, width)
    out_disabled, pmove_disabled = step_disabled(
        {}, sharded_data, sharded_key, width
    )
    out_mono, pmove_mono = step_monolayer({}, sharded_data, sharded_key, width)

    np.testing.assert_array_equal(out_legacy.positions, out_disabled.positions)
    np.testing.assert_array_equal(pmove_legacy, pmove_disabled)
    np.testing.assert_array_equal(out_legacy.positions, out_mono.positions)
    np.testing.assert_array_equal(pmove_legacy, pmove_mono)

  def test_minimum_image_dist_sq_matches_geometry(self):
    """PBC minimum-image squared distance matches geometry.make_minimum_image_distance_fn."""
    prim = geometry.triangular_lattice(151.18)
    lattice = geometry.supercell_lattice(prim, np.array([[6, 0], [0, 3]]))
    dist_sq_fn = mcmc._make_minimum_image_dist_sq_fn(lattice)
    _, geom_dist_fn = None, geometry.make_minimum_image_distance_fn(lattice)

    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    pos_e = jax.random.uniform(k1, (16, 6, 2), minval=-200.0, maxval=800.0)
    pos_h = jax.random.uniform(k2, (16, 6, 2), minval=-200.0, maxval=800.0)

    d_sq_mcmc = dist_sq_fn(pos_e, pos_h)
    _, d_geom = geom_dist_fn(pos_e, pos_h)
    np.testing.assert_allclose(d_sq_mcmc, d_geom**2, rtol=1e-5, atol=1e-5)

  def test_find_mutual_eh_pairs_with_pbc_and_crowding(self):
    """Mutual nearest-neighbor pairing respects cutoff, PBC, and 1-to-1 matching."""
    lattice = np.diag([100.0, 100.0])
    dist_sq_fn = mcmc._make_minimum_image_dist_sq_fn(lattice)
    pair_radius = 20.0

    # Batch of 2 walkers, n_e = 3, n_h = 3:
    # Walker 0:
    #   e0 at (2, 2) and h1 at (98, 99) -> MIC distance sqrt(4^2 + 3^2) = 5 < 20 (paired: e0 <-> h1)
    #   e1 at (50, 50) and h0 at (55, 50) -> distance 5 < 20 (paired: e1 <-> h0)
    #   e2 at (58, 50) -> also within 20 of h0 (dist 3), so e2 is closer to h0 than e1!
    #     Thus e2 <-> h0 is mutual, while e1 must remain unpaired (-1)!
    #   h2 at (10, 80) -> far from all electrons (unpaired).
    pos_e_w0 = np.array([[2.0, 2.0], [50.0, 50.0], [58.0, 50.0]])
    pos_h_w0 = np.array([[55.0, 50.0], [98.0, 99.0], [10.0, 80.0]])

    # Walker 1: all pairs separated by > pair_radius = 20.0 -> all -1
    pos_e_w1 = np.array([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]])
    pos_h_w1 = np.array([[0.0, 25.0], [30.0, 25.0], [60.0, 25.0]])

    pos_e = jnp.asarray(np.stack([pos_e_w0, pos_e_w1], axis=0))
    pos_h = jnp.asarray(np.stack([pos_h_w0, pos_h_w1], axis=0))

    partner_h, is_paired = mcmc._find_mutual_eh_pairs(
        pos_e, pos_h, pair_radius, dist_sq_fn
    )
    np.testing.assert_array_equal(partner_h[0], np.array([1, -1, 0]))
    np.testing.assert_array_equal(is_paired[0], np.array([True, False, True]))
    np.testing.assert_array_equal(partner_h[1], np.array([-1, -1, -1]))
    np.testing.assert_array_equal(is_paired[1], np.array([False, False, False]))

  def test_mh_pair_update_preserves_bound_pair_relative_displacement(self):
    """Bound (e, h) pairs shift by the exact same vector across PBC boundaries."""
    lattice = geometry.supercell_lattice(
        geometry.triangular_lattice(150.0), np.diag([4, 4])
    )
    rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
    map_to_cell = lambda p: mcmc.map_to_simulation_cell(p, lattice, rec, 2)
    dist_fn = geometry.make_minimum_image_distance_fn(lattice)
    dist_sq_fn = mcmc._make_minimum_image_dist_sq_fn(lattice)

    batch_size = 4
    n_e, n_h = 3, 3
    nelec = n_e + n_h
    # Place 3 well-separated excitons at sites far apart, with small internal e-h offset
    centers = jnp.array([
        [20.0, 20.0],
        [220.0, 40.0],
        [120.0, 260.0],
    ])
    offsets = jnp.array([
        [3.0, -2.0],
        [-4.0, 1.5],
        [2.5, 3.5],
    ])
    pos_h_init = jnp.broadcast_to(centers[None, ...], (batch_size, n_h, 2))
    pos_e_init = pos_h_init + offsets[None, ...]
    x_init = map_to_cell(
        jnp.reshape(
            jnp.concatenate([pos_e_init, pos_h_init], axis=1), (batch_size, -1)
        )
    )
    data = _make_dummy_data(x_init, nelec)

    # Constant log_psi = 0 so every reversible move is accepted with probability 1
    log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],), dtype=x.dtype)
    lp_1 = jnp.zeros((batch_size,))

    new_data, _, _, num_accepts = mcmc.mh_pair_update(
        params={},
        f=log_psi,
        data=data,
        key=jax.random.PRNGKey(123),
        lp_1=lp_1,
        num_accepts=0.0,
        stddev=15.0,
        atoms=None,
        ndim=2,
        map_to_cell=map_to_cell,
        blocks=1,
        i=0,
        layer_counts=(n_e, n_h),
        pair_radius=40.0,
        dist_sq_fn=dist_sq_fn,
    )

    self.assertEqual(int(num_accepts), batch_size)
    self.assertFalse(jnp.allclose(new_data.positions, data.positions))

    pos_after = jnp.reshape(new_data.positions, (batch_size, nelec, 2))
    disp_after, _ = dist_fn(pos_after[:, :n_e, :], pos_after[:, n_e:, :])
    # Diagonal entries i == j are the 3 bound pairs; their MIC relative vectors must equal offsets
    for i in range(n_e):
      np.testing.assert_allclose(
          disp_after[:, i, i, :],
          jnp.broadcast_to(offsets[i], (batch_size, 2)),
          atol=1e-4,
      )

  def test_reversibility_rejects_broken_or_swapped_pairs(self):
    """Moves that alter the mutual-nearest-neighbor pairing map are rejected."""
    dist_sq_fn = mcmc._make_minimum_image_dist_sq_fn(None)
    # 1 electron, 1 hole initially paired at distance 10 < pair_radius = 15,
    # plus a second hole at distance 16 (just outside). A large step that moves
    # hole 1 closer to electron 0 than hole 0 changes the pairing and must be rejected.
    pos_e = jnp.array([[[0.0, 0.0]]])  # (1, 1, 2)
    pos_h = jnp.array([[[10.0, 0.0], [11.0, 0.0]]])  # (1, 2, 2)
    x1 = jnp.reshape(jnp.concatenate([pos_e, pos_h], axis=1), (1, -1))
    data = _make_dummy_data(x1, 3)

    # Even with a huge positive log_psi reward for moving, broken pairing must reject
    log_psi = lambda params, x, s, a, c: 100.0 * jnp.ones((x.shape[0],))
    lp_1 = jnp.zeros((1,))

    accepted_any_invalid = False
    key = jax.random.PRNGKey(99)
    for _ in range(25):
      key, subkey = jax.random.split(key)
      new_data, _, _, _ = mcmc.mh_pair_update(
          params={},
          f=log_psi,
          data=data,
          key=subkey,
          lp_1=lp_1,
          num_accepts=0.0,
          stddev=20.0,
          atoms=None,
          ndim=2,
          layer_counts=(1, 2),
          pair_radius=15.0,
          dist_sq_fn=dist_sq_fn,
      )
      pos_new = jnp.reshape(new_data.positions, (1, 3, 2))
      partner_new, _ = mcmc._find_mutual_eh_pairs(
          pos_new[:, :1, :], pos_new[:, 1:, :], 15.0, dist_sq_fn
      )
      if int(partner_new[0, 0]) != 0:
        accepted_any_invalid = True
    self.assertFalse(accepted_any_invalid)

  def test_detailed_balance_and_barrier_crossing_in_bimodal_potential(self):
    """Mixed pair MCMC crosses inter-well barriers and samples the exact target density."""
    # 2D system: 1 electron and 1 hole strongly bound (sigma_bind = 0.5 Bohr)
    # in a double-well potential with minima at R = (-6, 0) and R = (+6, 0).
    batch_size = 256
    ndim = 2
    sigma_bind = 0.5
    well_sep = 6.0
    sigma_well = 2.0

    def log_psi(params, x, spins, atoms, charges):
      del params, spins, atoms, charges
      pos = jnp.reshape(x, (-1, 2, 2))
      r_e = pos[:, 0, :]
      r_h = pos[:, 1, :]
      r_rel_sq = jnp.sum((r_e - r_h) ** 2, axis=-1)
      # Double well on hole position: minima at (-well_sep, 0) and (+well_sep, 0)
      d1_sq = jnp.sum(
          (r_h - jnp.array([-well_sep, 0.0])) ** 2, axis=-1
      )
      d2_sq = jnp.sum(
          (r_h - jnp.array([+well_sep, 0.0])) ** 2, axis=-1
      )
      log_well = 0.5 * jnp.logaddexp(
          -0.5 * d1_sq / (sigma_well**2),
          -0.5 * d2_sq / (sigma_well**2),
      )
      log_bind = -0.25 * r_rel_sq / (sigma_bind**2)
      return log_well + log_bind

    # Start all walkers in the LEFT well (-well_sep, 0) with COINCIDENT (e, h)
    # positions (initial relative variance = 0.0) so the test verifies both
    # inter-well barrier crossing AND internal relative-coordinate equilibration.
    key = jax.random.PRNGKey(2026)
    key, k_h = jax.random.split(key)
    r_h0 = jnp.array([-well_sep, 0.0]) + 0.5 * jax.random.normal(
        k_h, (batch_size, 2)
    )
    r_e0 = r_h0
    x0 = jnp.reshape(jnp.stack([r_e0, r_h0], axis=1), (batch_size, 4))
    data0 = _make_dummy_data(x0, 2)

    # With step width = 1.0 Bohr and pair_width_scale = 6.0:
    # - Single-particle moves (pair_prob=0.0) cannot cross the well barrier at x=0
    # - Mixed pair moves (pair_prob=0.5) equilibrate internal separation (from 0
    #   to sigma_bind^2) via ordinary steps AND hop between (-6, 0) and (+6, 0)
    #   via pair steps.
    step_single = jax.pmap(
        mcmc.make_mcmc_step(
            log_psi,
            batch_per_device=batch_size,
            steps=150,
            ndim=ndim,
            nspins=(1, 1),
            pair_prob=0.0,
        ),
        in_axes=(None, 0, 0, None),
    )
    step_paired = jax.pmap(
        mcmc.make_mcmc_step(
            log_psi,
            batch_per_device=batch_size,
            steps=150,
            ndim=ndim,
            nspins=(1, 1),
            pair_prob=0.5,
            pair_radius=5.0,
            pair_width_scale=6.0,
            pair_min_fraction=0.0,
        ),
        in_axes=(None, 0, 0, None),
    )

    sharded_data = jax.tree_util.tree_map(lambda x: x[None, ...], data0)
    k_single, k_pair = jax.random.split(key)
    width = jnp.asarray(1.0)

    data_single, _ = step_single({}, sharded_data, k_single[None, ...], width)
    data_paired, _ = step_paired({}, sharded_data, k_pair[None, ...], width)

    pos_single = jnp.reshape(data_single.positions[0], (batch_size, 2, 2))
    pos_paired = jnp.reshape(data_paired.positions[0], (batch_size, 2, 2))

    frac_right_single = float(jnp.mean(pos_single[:, 1, 0] > 0.0))
    frac_right_paired = float(jnp.mean(pos_paired[:, 1, 0] > 0.0))

    # Single-particle MCMC is trapped in the left well, whereas paired MCMC equilibrates ~50/50
    self.assertLess(frac_right_single, 0.05)
    self.assertAlmostEqual(frac_right_paired, 0.5, delta=0.1)

    # Verify internal relative variance equilibrated from 0.0 up to sigma_bind^2 = 0.25
    rel_var = float(jnp.var(pos_paired[:, 0, :] - pos_paired[:, 1, :]))
    self.assertAlmostEqual(rel_var, sigma_bind**2, delta=0.08)

  def test_pair_mcmc_blocks_and_excited_states(self):
    """mh_pair_update supports blocks > 1 and multi-state packed configurations."""
    batch_size = 4
    n_e, n_h = 2, 2
    nelec = n_e + n_h
    num_states = 2
    blocks_per_state = 2
    blocks = blocks_per_state * num_states

    # Create 2 states of 2 tightly bound pairs each
    centers = jnp.array([[10.0, 10.0], [60.0, 60.0]])
    offsets = jnp.array([[1.0, -1.0], [-1.5, 0.5]])
    pos_h = jnp.broadcast_to(
        centers[None, ...], (batch_size * num_states, n_h, 2)
    )
    pos_e = pos_h + offsets[None, ...]
    x0 = jnp.reshape(
        jnp.concatenate([pos_e, pos_h], axis=1),
        (batch_size, num_states * nelec * 2),
    )
    data = _make_dummy_data(x0, num_states * nelec)
    log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],), dtype=x.dtype)
    lp_1 = jnp.zeros((batch_size,))

    key = jax.random.PRNGKey(11)
    cur_data = data
    total_accepts = 0.0
    for step_i in range(blocks):
      cur_data, key, lp_1, total_accepts = mcmc.mh_pair_update(
          params={},
          f=log_psi,
          data=cur_data,
          key=key,
          lp_1=lp_1,
          num_accepts=total_accepts,
          stddev=2.0,
          atoms=None,
          ndim=2,
          blocks=blocks,
          i=step_i,
          layer_counts=(n_e, n_h),
          pair_radius=10.0,
      )
    self.assertEqual(int(total_accepts), batch_size * blocks)
    pos_out = jnp.reshape(
        cur_data.positions, (batch_size * num_states, nelec, 2)
    )
    rel_out = pos_out[:, :n_e, :] - pos_out[:, n_e:, :]
    np.testing.assert_allclose(
        rel_out,
        jnp.broadcast_to(offsets[None, ...], rel_out.shape),
        atol=1e-5,
    )

  def test_blocks_exceeding_layer_counts_never_produce_noop_accepts(self):
    """When blocks > n_h (e.g. n_e=2, n_h=2, blocks=4), every block moves particles."""
    batch_size = 64
    n_e, n_h = 2, 2
    nelec = n_e + n_h
    blocks = 4

    centers = jnp.array([[10.0, 10.0], [60.0, 60.0]])
    offsets = jnp.array([[1.0, -1.0], [-1.5, 0.5]])
    pos_h = jnp.broadcast_to(centers[None, ...], (batch_size, n_h, 2))
    pos_e = pos_h + offsets[None, ...]
    x0 = jnp.reshape(
        jnp.concatenate([pos_e, pos_h], axis=1), (batch_size, nelec * 2)
    )
    data = _make_dummy_data(x0, nelec)

    # Steeply peaked log_psi around x0 so any actual non-zero move is rejected,
    # whereas an empty block (x2 == x0) would have lp_2 == lp_1 and be falsely accepted.
    def steep_log_psi(params, x, s, a, c):
      del params, s, a, c
      return -1e6 * jnp.sum((x - x0) ** 2, axis=-1)

    lp_1 = 2.0 * steep_log_psi({}, x0, None, None, None)
    key = jax.random.PRNGKey(77)
    for block_idx in range(blocks):
      key, subkey = jax.random.split(key)
      new_data, _, _, num_accepts = mcmc.mh_pair_update(
          params={},
          f=steep_log_psi,
          data=data,
          key=subkey,
          lp_1=lp_1,
          num_accepts=0.0,
          stddev=2.0,
          atoms=None,
          ndim=2,
          blocks=blocks,
          i=block_idx,
          layer_counts=(n_e, n_h),
          pair_radius=10.0,
      )
      self.assertEqual(int(num_accepts), 0)
      np.testing.assert_array_equal(new_data.positions, x0)

    # Also verify that with flat log_psi = 0, every block 0..3 actually moves walkers
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))
    for block_idx in range(blocks):
      key, subkey = jax.random.split(key)
      new_data, _, _, num_accepts = mcmc.mh_pair_update(
          params={},
          f=flat_log_psi,
          data=data,
          key=subkey,
          lp_1=jnp.zeros((batch_size,)),
          num_accepts=0.0,
          stddev=2.0,
          atoms=None,
          ndim=2,
          blocks=blocks,
          i=block_idx,
          layer_counts=(n_e, n_h),
          pair_radius=10.0,
      )
      self.assertEqual(int(num_accepts), batch_size)
      moved_walkers = jnp.any(new_data.positions != x0, axis=-1)
      self.assertTrue(bool(jnp.all(moved_walkers)))

  def test_make_mcmc_step_validation(self):
    log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))
    with self.assertRaisesRegex(ValueError, 'pair_prob'):
      mcmc.make_mcmc_step(log_psi, 4, pair_prob=-0.1)
    with self.assertRaisesRegex(ValueError, 'pair_prob'):
      mcmc.make_mcmc_step(log_psi, 4, pair_prob=1.0)
    with self.assertRaisesRegex(ValueError, 'pair_prob'):
      mcmc.make_mcmc_step(log_psi, 4, pair_prob=1.2)
    with self.assertRaisesRegex(ValueError, 'pair_width_scale'):
      mcmc.make_mcmc_step(log_psi, 4, pair_width_scale=0.0)
    with self.assertRaisesRegex(ValueError, 'pair_min_fraction'):
      mcmc.make_mcmc_step(log_psi, 4, pair_min_fraction=-0.1)
    with self.assertRaisesRegex(ValueError, 'pair_min_fraction'):
      mcmc.make_mcmc_step(log_psi, 4, pair_min_fraction=1.1)
    with self.assertRaisesRegex(ValueError, 'pair_radius'):
      mcmc.make_mcmc_step(
          log_psi, 4, nspins=(2, 2), pair_prob=0.5, pair_radius=0.0
      )

  def test_base_config_and_moire_config_backward_compatibility(self):
    """Default base_config and moire configs expose pair MCMC fields safely."""
    from ferminet import base_config
    from ferminet.configs import WSe2WS2_MoSe2_vh13_ve13_3shell_scaled

    cfg = WSe2WS2_MoSe2_vh13_ve13_3shell_scaled.get_config()
    cfg = base_config.resolve(cfg)
    self.assertEqual(cfg.mcmc.pair_prob, 0.0)
    self.assertEqual(cfg.mcmc.pair_radius, 0.0)
    self.assertEqual(cfg.mcmc.pair_width_scale, 1.0)
    self.assertAlmostEqual(cfg.mcmc.pair_min_fraction, 2.0 / 3.0)

    # Enabling pair_prob and pair_radius via ConfigDict preserves type safety
    cfg.mcmc.pair_prob = 0.25
    cfg.mcmc.pair_radius = 50.0
    cfg.mcmc.pair_width_scale = 1.2
    cfg.mcmc.pair_min_fraction = 0.75
    self.assertAlmostEqual(cfg.mcmc.pair_prob, 0.25)
    self.assertAlmostEqual(cfg.mcmc.pair_radius, 50.0)
    self.assertAlmostEqual(cfg.mcmc.pair_width_scale, 1.2)
    self.assertAlmostEqual(cfg.mcmc.pair_min_fraction, 0.75)

  def test_unpaired_particles_use_unpaired_stddev(self):
    """Matched pairs move with sigma_p while unpaired particles move with sigma_n."""
    batch_size = 512
    n_e, n_h = 2, 2
    nelec = n_e + n_h
    pair_radius = 5.0

    # Pair 0: e0 at (0, 0), h0 at (1, 0) -> paired (distance 1 < 5)
    # Unpaired: e1 at (50, 0), h1 at (-50, 0) -> unpaired (distance 100 > 5)
    pos_e0 = jnp.broadcast_to(
        jnp.array([[0.0, 0.0], [50.0, 0.0]])[None, ...], (batch_size, n_e, 2)
    )
    pos_h0 = jnp.broadcast_to(
        jnp.array([[1.0, 0.0], [-50.0, 0.0]])[None, ...], (batch_size, n_h, 2)
    )
    x0 = jnp.reshape(
        jnp.concatenate([pos_e0, pos_h0], axis=1), (batch_size, nelec * 2)
    )
    data = _make_dummy_data(x0, nelec)
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))
    lp_1 = jnp.zeros((batch_size,))

    sigma_p = 1.5
    sigma_n = 0.1
    (
        new_data,
        _,
        _,
        num_accepts,
        num_map_rej,
        total_pairs,
        active_pair_tries,
        active_pair_accepts,
    ) = mcmc.mh_pair_update(
        params={},
        f=flat_log_psi,
        data=data,
        key=jax.random.PRNGKey(1234),
        lp_1=lp_1,
        num_accepts=0.0,
        stddev=sigma_p,
        unpaired_stddev=sigma_n,
        atoms=None,
        ndim=2,
        layer_counts=(n_e, n_h),
        pair_radius=pair_radius,
        return_pair_diagnostics=True,
    )
    self.assertEqual(int(num_accepts), batch_size)
    self.assertEqual(int(num_map_rej), 0)
    self.assertAlmostEqual(float(total_pairs) / batch_size, 1.0, places=5)
    self.assertEqual(int(active_pair_tries), batch_size)
    self.assertEqual(int(active_pair_accepts), batch_size)

    pos_out = jnp.reshape(new_data.positions, (batch_size, nelec, 2))
    pos_in = jnp.reshape(x0, (batch_size, nelec, 2))
    disp = pos_out - pos_in

    # Matched pair (e0 and h0) must have identical displacements with variance sigma_p^2
    np.testing.assert_allclose(disp[:, 0, :], disp[:, 2, :], atol=1e-6)
    var_paired = float(jnp.var(disp[:, 0, :]))
    var_unpaired_e = float(jnp.var(disp[:, 1, :]))
    var_unpaired_h = float(jnp.var(disp[:, 3, :]))
    self.assertAlmostEqual(var_paired, sigma_p**2, delta=0.25)
    self.assertAlmostEqual(var_unpaired_e, sigma_n**2, delta=0.02)
    self.assertAlmostEqual(var_unpaired_h, sigma_n**2, delta=0.02)

  def test_separate_mcmc_stats_and_diagnostics(self):
    """make_mcmc_step(return_stats=True) tracks normal/pair stats and map rejects."""
    batch_size = 32
    n_e, n_h = 2, 2
    nelec = n_e + n_h
    # e0-h0 tightly paired (dist 0.5), e1-h1 near the cutoff boundary (dist 4.9)
    # so large pair moves frequently change the pairing map or get rejected.
    pos_e0 = jnp.broadcast_to(
        jnp.array([[0.0, 0.0], [20.0, 0.0]])[None, ...], (batch_size, n_e, 2)
    )
    pos_h0 = jnp.broadcast_to(
        jnp.array([[0.5, 0.0], [20.0, 8.0]])[None, ...], (batch_size, n_h, 2)
    )
    x0 = jnp.reshape(
        jnp.concatenate([pos_e0, pos_h0], axis=1), (batch_size, nelec * 2)
    )
    data = _make_dummy_data(x0, nelec)

    def harmonic_log_psi(params, x, s, a, c):
      del params, s, a, c
      return -0.25 * jnp.sum((x - x0) ** 2, axis=-1)

    step_fn = jax.pmap(
        mcmc.make_mcmc_step(
            harmonic_log_psi,
            batch_per_device=batch_size,
            steps=10,
            ndim=2,
            nspins=(n_e, n_h),
            pair_prob=0.5,
            pair_radius=5.0,
            pair_min_fraction=0.5,
            return_stats=True,
        ),
        in_axes=(None, 0, 0, 0),
    )
    sharded_data = jax.tree_util.tree_map(lambda x: x[None, ...], data)
    sharded_key = jax.random.PRNGKey(99)[None, ...]
    # Compiled threshold-controlled sampling requires the 3-element state
    # [width_normal, width_pair, pair_moves_enabled].
    with self.assertRaisesRegex(ValueError, 'pair_moves_enabled'):
      step_fn({}, sharded_data, sharded_key, jnp.array([[0.15, 1.2]]))
    sharded_width = jnp.array([[0.15, 1.2, 1.0]])

    _, stats = step_fn({}, sharded_data, sharded_key, sharded_width)
    n_try_n = float(stats['n_try_normal'][0])
    n_try_p = float(stats['n_try_pair'][0])
    n_acc_n = float(stats['n_acc_normal'][0])
    n_acc_p = float(stats['n_acc_pair'][0])
    n_try_p_act = float(stats['n_try_pair_active'][0])
    n_acc_p_act = float(stats['n_acc_pair_active'][0])
    n_map_rej = float(stats['n_map_rej_pair'][0])

    self.assertEqual(n_try_n + n_try_p, float(10 * batch_size))
    self.assertGreater(n_try_n, 0.0)
    self.assertGreater(n_try_p, 0.0)
    self.assertAlmostEqual(n_try_p_act, n_try_p - n_map_rej, places=5)
    self.assertAlmostEqual(
        float(stats['pmove_normal'][0]), n_acc_n / n_try_n, places=6
    )
    self.assertAlmostEqual(
        float(stats['pmove_pair'][0]), n_acc_p / n_try_p, places=6
    )
    self.assertAlmostEqual(
        float(stats['pmove_pair_active'][0]),
        n_acc_p_act / n_try_p_act,
        places=6,
    )
    self.assertAlmostEqual(
        float(stats['pmove'][0]),
        (n_acc_n + n_acc_p) / (n_try_n + n_try_p),
        places=6,
    )
    self.assertGreaterEqual(float(stats['pair_map_reject_rate'][0]), 0.0)
    self.assertLessEqual(float(stats['pair_map_reject_rate'][0]), 1.0)
    self.assertAlmostEqual(float(stats['mean_num_pairs'][0]), 1.0, delta=0.2)

  def test_update_pair_mcmc_widths_adapts_independently(self):
    """update_pair_mcmc_widths adapts sigma_n and sigma_p independently."""
    adapt_freq = 4
    state = mcmc.init_pair_mcmc_adaptation_state(adapt_freq)
    width = jnp.array([[0.2, 1.0]])

    # Normal acceptance = 80/100 = 0.80 (> 0.55 -> multiply sigma_n by 1.1)
    # Pair acceptance = 20/100 = 0.20 (< 0.50 -> divide sigma_p by 1.1)
    fake_stats = {
        'n_try_normal': jnp.array([100.0]),
        'n_acc_normal': jnp.array([80.0]),
        'n_try_pair': jnp.array([100.0]),
        'n_acc_pair': jnp.array([20.0]),
    }
    for t in range(1, adapt_freq + 1):
      width, state = mcmc.update_pair_mcmc_widths(
          t, width, adapt_freq, fake_stats, state
      )
    self.assertAlmostEqual(float(width[0, 0]), 0.2 * 1.1, places=6)
    self.assertAlmostEqual(float(width[0, 1]), 1.0 / 1.1, places=6)

    # When pair moves have 0 attempts in the window, sigma_p is unchanged.
    zero_pair_stats = {
        'n_try_normal': jnp.array([100.0]),
        'n_acc_normal': jnp.array([30.0]),
        'n_try_pair': jnp.array([0.0]),
        'n_acc_pair': jnp.array([0.0]),
    }
    prev_sigma_p = float(width[0, 1])
    for t in range(adapt_freq + 1, 2 * adapt_freq + 1):
      width, state = mcmc.update_pair_mcmc_widths(
          t, width, adapt_freq, zero_pair_stats, state
      )
    self.assertAlmostEqual(float(width[0, 0]), 0.2, places=6)
    self.assertAlmostEqual(float(width[0, 1]), prev_sigma_p, places=6)

  def test_resolve_layer_counts_four_channel_spin_polarized_and_electron_only(
      self,
  ):
    """4-channel nspins resolves (n_e_up+n_e_down, n_h_up+n_h_down) consistently."""
    # Spin-polarized bilayer: (6, 0, 6, 0) -> (6, 6) -> both layers occupied
    self.assertEqual(mcmc._resolve_layer_counts((6, 0, 6, 0), None), (6, 6))
    # Spin-unpolarized bilayer: (3, 3, 3, 3) -> (6, 6) -> both layers occupied
    self.assertEqual(mcmc._resolve_layer_counts((3, 3, 3, 3), None), (6, 6))
    # Electron-only system: (3, 3, 0, 0) -> (6, 0) -> hole layer empty
    self.assertEqual(mcmc._resolve_layer_counts((3, 3, 0, 0), None), (6, 0))

    # Verify polarized bilayer (1, 0, 1, 0) runs pair moves and returns stats
    batch_size = 8
    pos_e0 = jnp.zeros((batch_size, 1, 2))
    pos_h0 = jnp.ones((batch_size, 1, 2)) * 0.5
    x0 = jnp.reshape(
        jnp.concatenate([pos_e0, pos_h0], axis=1), (batch_size, 4)
    )
    data = _make_dummy_data(x0, 2)
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))
    step_polarized = mcmc.make_mcmc_step(
        flat_log_psi,
        batch_per_device=batch_size,
        steps=10,
        ndim=2,
        nspins=(1, 0, 1, 0),
        pair_prob=0.5,
        pair_radius=5.0,
        return_stats=True,
    )
    _, stats = step_polarized(
        {}, data, jax.random.PRNGKey(7), jnp.array([0.1, 1.0])
    )
    self.assertGreater(float(stats['n_try_pair']), 0.0)
    self.assertEqual(float(stats['n_try_pair_active']), float(stats['n_try_pair']))

  def test_zero_matched_pairs_skips_pair_width_adaptation(self):
    """When 0 pairs are matched, nominal pair moves do not adapt sigma_p."""
    batch_size = 16
    n_e, n_h = 1, 1
    nelec = 2
    # Electron at (0, 0), hole at (100, 0) with pair_radius=5.0 -> 0 pairs
    pos_e0 = jnp.zeros((batch_size, n_e, 2))
    pos_h0 = jnp.broadcast_to(
        jnp.array([[100.0, 0.0]])[None, ...], (batch_size, n_h, 2)
    )
    x0 = jnp.reshape(
        jnp.concatenate([pos_e0, pos_h0], axis=1), (batch_size, nelec * 2)
    )
    data = _make_dummy_data(x0, nelec)
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))

    step_fn = mcmc.make_mcmc_step(
        flat_log_psi,
        batch_per_device=batch_size,
        steps=10,
        ndim=2,
        nspins=(1, 1),
        pair_prob=0.5,
        pair_radius=5.0,
        pair_min_fraction=0.0,
        return_stats=True,
    )
    key = jax.random.PRNGKey(2027)
    data_w1, stats_w1 = step_fn({}, data, key, jnp.array([0.2, 1.0]))
    data_w1000, stats_w1000 = step_fn({}, data, key, jnp.array([0.2, 1000.0]))

    # Because 0 pairs are matched, changing sigma_p from 1 to 1000 yields
    # identical positions with the same RNG key.
    np.testing.assert_allclose(data_w1.positions, data_w1000.positions)

    # Nominal pair acceptance is 1.0, but active pair attempts are 0.
    self.assertGreater(float(stats_w1['n_try_pair']), 0.0)
    self.assertAlmostEqual(float(stats_w1['pmove_pair']), 1.0, places=6)
    self.assertEqual(float(stats_w1['mean_num_pairs']), 0.0)
    self.assertEqual(float(stats_w1['n_try_pair_active']), 0.0)
    self.assertEqual(float(stats_w1['n_acc_pair_active']), 0.0)
    self.assertTrue(bool(jnp.isnan(stats_w1['pmove_pair_active'])))

    # Replaying these stats across 20 adaptation windows must leave sigma_p at 1.0
    adapt_freq = 5
    state = mcmc.init_pair_mcmc_adaptation_state(adapt_freq)
    width = jnp.array([[0.2, 1.0]])
    for t in range(1, 20 * adapt_freq + 1):
      width, state = mcmc.update_pair_mcmc_widths(
          t, width, adapt_freq, stats_w1, state
      )
    self.assertGreater(float(width[0, 0]), 0.2)
    self.assertAlmostEqual(float(width[0, 1]), 1.0, places=6)

  def test_pair_moves_gated_by_pair_min_fraction(self):
    """Pair sweeps wait until mean_num_pairs >= threshold, then latch permanently."""
    batch_size = 16
    n_e, n_h = 3, 3
    nelec = n_e + n_h
    pair_radius = 5.0
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))

    # Default pair_min_fraction = 2/3 -> requires >= 2.0 pairs for n_e=n_h=3.
    step_fn = mcmc.make_mcmc_step(
        flat_log_psi,
        batch_per_device=batch_size,
        steps=10,
        ndim=2,
        nspins=(n_e, n_h),
        pair_prob=0.5,
        pair_radius=pair_radius,
        return_stats=True,
    )

    # Case 1: Only 1 pair matched (1.0 < 2.0) -> 0 pair sweeps executed,
    # 100% normal sweeps, and mean_num_pairs is still reported as 1.0.
    pos_e_1 = jnp.broadcast_to(
        jnp.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])[None, ...],
        (batch_size, n_e, 2),
    )
    pos_h_1 = jnp.broadcast_to(
        jnp.array([[0.5, 0.0], [100.0, 50.0], [200.0, 50.0]])[None, ...],
        (batch_size, n_h, 2),
    )
    x_1 = jnp.reshape(
        jnp.concatenate([pos_e_1, pos_h_1], axis=1), (batch_size, nelec * 2)
    )
    data_1 = _make_dummy_data(x_1, nelec)
    _, stats_1 = step_fn(
        {}, data_1, jax.random.PRNGKey(42), jnp.array([0.1, 1.0])
    )
    self.assertEqual(float(stats_1['n_try_pair']), 0.0)
    self.assertEqual(float(stats_1['n_try_normal']), float(10 * batch_size))
    self.assertAlmostEqual(float(stats_1['mean_num_pairs']), 1.0, places=6)
    self.assertFalse(bool(stats_1['pair_moves_enabled']))
    self.assertTrue(bool(jnp.isnan(stats_1['pmove_pair_active'])))

    # Case 2: 2 pairs matched (2.0 >= 2.0) -> pair gate latches open and pair sweeps run.
    pos_h_2 = jnp.broadcast_to(
        jnp.array([[0.5, 0.0], [100.5, 0.0], [200.0, 50.0]])[None, ...],
        (batch_size, n_h, 2),
    )
    x_2 = jnp.reshape(
        jnp.concatenate([pos_e_1, pos_h_2], axis=1), (batch_size, nelec * 2)
    )
    data_2 = _make_dummy_data(x_2, nelec)
    _, stats_2 = step_fn(
        {}, data_2, jax.random.PRNGKey(42), jnp.array([0.1, 1.0])
    )
    self.assertTrue(bool(stats_2['pair_moves_enabled']))
    self.assertGreater(float(stats_2['n_try_pair']), 0.0)
    self.assertAlmostEqual(float(stats_2['mean_num_pairs']), 2.0, delta=0.2)

    # Case 3: Once latched, even if configurations fall back below threshold
    # (1 pair matched < 2.0), pair moves remain permanently enabled!
    _, stats_3 = step_fn(
        {}, data_1, jax.random.PRNGKey(42), jnp.array([0.1, 1.0])
    )
    self.assertTrue(bool(stats_3['pair_moves_enabled']))
    self.assertGreater(float(stats_3['n_try_pair']), 0.0)

    # Verify the same one-way latch through 3-element mcmc_width and
    # update_pair_mcmc_widths (used in pmapped training and checkpoints).
    w_state = mcmc.init_pair_mcmc_width(
        0.1, pair_width_scale=10.0, pair_min_fraction=2.0 / 3.0
    )[None, :]
    adapt_state = mcmc.init_pair_mcmc_adaptation_state(
        4, pair_moves_enabled=False
    )
    self.assertEqual(float(w_state[0, 2]), 0.0)
    _, stats_pre = step_fn({}, data_1, jax.random.PRNGKey(42), w_state[0])
    self.assertEqual(float(stats_pre['n_try_pair']), 0.0)
    w_state, adapt_state = mcmc.update_pair_mcmc_widths(
        1, w_state, 4, stats_pre, adapt_state
    )
    self.assertEqual(float(w_state[0, 2]), 0.0)
    self.assertFalse(adapt_state['pair_moves_enabled'])

    # Crossing threshold in stats_2 latches w_state[..., 2] = 1.0 permanently.
    w_state, adapt_state = mcmc.update_pair_mcmc_widths(
        2, w_state, 4, stats_2, adapt_state
    )
    self.assertEqual(float(w_state[0, 2]), 1.0)
    self.assertTrue(adapt_state['pair_moves_enabled'])

    # Subsequent below-threshold stats_pre never reset w_state[..., 2].
    w_state, adapt_state = mcmc.update_pair_mcmc_widths(
        3, w_state, 4, stats_pre, adapt_state
    )
    self.assertEqual(float(w_state[0, 2]), 1.0)
    self.assertTrue(adapt_state['pair_moves_enabled'])
    _, stats_post = step_fn({}, data_1, jax.random.PRNGKey(42), w_state[0])
    self.assertGreater(float(stats_post['n_try_pair']), 0.0)

  def test_pair_threshold_normalized_for_multi_state(self):
    """Multi-state pair count is normalized by num_states before threshold check."""
    batch_size = 8
    n_e, n_h = 1, 1
    nelec = n_e + n_h
    num_states = 2
    pair_radius = 5.0
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))

    step_fn = mcmc.make_mcmc_step(
        flat_log_psi,
        batch_per_device=batch_size,
        steps=10,
        ndim=2,
        blocks=num_states,
        nspins=(n_e, n_h),
        pair_prob=0.5,
        pair_radius=pair_radius,
        pair_min_fraction=2.0 / 3.0,
        return_stats=True,
    )

    # State 0 is paired: e at (0, 0), h at (0.2, 0) -> dist 0.2 < 5.0
    # State 1 is unpaired: e at (0, 0), h at (50.0, 0) -> dist 50.0 > 5.0
    # Total pairs across 2 states = 1, so per-state paired fraction = 0.5 < 2/3.
    state0 = jnp.array([0.0, 0.0, 0.2, 0.0])
    state1 = jnp.array([0.0, 0.0, 50.0, 0.0])
    x_half = jnp.broadcast_to(
        jnp.concatenate([state0, state1])[None, :],
        (batch_size, num_states * nelec * 2),
    )
    data_half = _make_dummy_data(x_half, num_states * nelec)
    _, stats_half = step_fn(
        {}, data_half, jax.random.PRNGKey(99), jnp.array([0.1, 1.0])
    )
    self.assertAlmostEqual(float(stats_half['mean_num_pairs']), 0.5, places=6)
    self.assertFalse(bool(stats_half['pair_moves_enabled']))
    self.assertEqual(float(stats_half['n_try_pair']), 0.0)

    # When both states are paired (mean_num_pairs = 1.0 >= 2/3), pair moves activate.
    x_both = jnp.broadcast_to(
        jnp.concatenate([state0, state0])[None, :],
        (batch_size, num_states * nelec * 2),
    )
    data_both = _make_dummy_data(x_both, num_states * nelec)
    _, stats_both = step_fn(
        {}, data_both, jax.random.PRNGKey(99), jnp.array([0.1, 1.0])
    )
    self.assertAlmostEqual(float(stats_both['mean_num_pairs']), 1.0, delta=0.1)
    self.assertTrue(bool(stats_both['pair_moves_enabled']))
    self.assertGreater(float(stats_both['n_try_pair']), 0.0)

  def test_latched_pair_threshold_preserves_stationary_distribution(self):
    """Latched pair threshold preserves the exact Gaussian paired probability."""
    # Target: 1 electron and 1 hole in 2D, each coordinate ~ N(0, 1),
    # so log|psi(x)| = -0.25 * sum(x^2) and 2*log|psi(x)| = -0.5 * sum(x^2).
    # Since r_e - r_h ~ N(0, 2*I_2), |r_e - r_h|^2 / 2 ~ Exp(1), so for
    # pair_radius = 1.0, exact P(|r_e - r_h| < 1) = 1 - exp(-1/4) = 0.221199.
    # With batch_size = 1, a single walker fluctuates between 0 pairs (78% of
    # the time, < 2/3) and 1 pair (22% of the time, >= 2/3). Once the threshold
    # is crossed the first time, pair_moves_enabled latches to True and never
    # switches back off, preserving the exact stationary distribution.
    batch_size = 1
    n_e, n_h = 1, 1
    nelec = 2
    pair_radius = 1.0
    exact_paired_prob = 1.0 - float(np.exp(-0.25))

    def gaussian_log_psi(params, x, s, a, c):
      del params, s, a, c
      return -0.25 * jnp.sum(x**2, axis=-1)

    step_fn = mcmc.make_mcmc_step(
        gaussian_log_psi,
        batch_per_device=batch_size,
        steps=30,
        ndim=2,
        nspins=(n_e, n_h),
        pair_prob=0.5,
        pair_radius=pair_radius,
        pair_min_fraction=2.0 / 3.0,
        return_stats=True,
    )

    # Step 0: Start unpaired at distance 3.0 (> 1.0) -> pair moves initially off.
    data_unpaired = _make_dummy_data(
        jnp.array([[0.0, 0.0, 3.0, 0.0]]), nelec
    )
    key = jax.random.PRNGKey(20261107)
    key, subkey = jax.random.split(key)
    data, stats0 = step_fn({}, data_unpaired, subkey, jnp.array([0.1, 0.1]))
    self.assertFalse(bool(stats0['pair_moves_enabled']))
    self.assertEqual(float(stats0['n_try_pair']), 0.0)

    # Step 1: Walker visits a paired configuration (distance 0.2 < 1.0) ->
    # latches pair_moves_enabled=True permanently.
    data_paired = _make_dummy_data(
        jnp.array([[0.0, 0.0, 0.2, 0.0]]), nelec
    )
    key, subkey = jax.random.split(key)
    data, stats1 = step_fn({}, data_paired, subkey, jnp.array([0.1, 0.1]))
    self.assertTrue(bool(stats1['pair_moves_enabled']))

    # Step 2: Even when walker is unpaired again (distance 3.0 > 1.0),
    # pair_moves_enabled remains True.
    key, subkey = jax.random.split(key)
    _, stats2 = step_fn({}, data_unpaired, subkey, jnp.array([0.1, 0.1]))
    self.assertTrue(bool(stats2['pair_moves_enabled']))

    # Verify stationary paired fraction across an ensemble of walkers after
    # the latch has been triggered.
    ens_batch = 512
    ens_step = jax.jit(
        mcmc.make_mcmc_step(
            gaussian_log_psi,
            batch_per_device=ens_batch,
            steps=30,
            ndim=2,
            nspins=(n_e, n_h),
            pair_prob=0.5,
            pair_radius=pair_radius,
            pair_min_fraction=2.0 / 3.0,
            return_stats=True,
        )
    )
    key, init_key = jax.random.split(key)
    ens_data = _make_dummy_data(
        jax.random.normal(init_key, (ens_batch, 4)), nelec
    )
    w_state = mcmc.init_pair_mcmc_width(
        0.1, pair_width_scale=1.0, pair_min_fraction=2.0 / 3.0,
        pair_moves_enabled=True,
    )
    paired_fractions = []
    for step_idx in range(200):
      key, subkey = jax.random.split(key)
      ens_data, _ = ens_step({}, ens_data, subkey, w_state)
      if step_idx >= 50:
        pos_3d = jnp.reshape(ens_data.positions, (ens_batch, 2, 2))
        dist_sq = jnp.sum((pos_3d[:, 0, :] - pos_3d[:, 1, :]) ** 2, axis=-1)
        paired_fractions.append(float(jnp.mean(dist_sq < pair_radius**2)))

    measured = float(np.mean(paired_fractions))
    self.assertAlmostEqual(measured, exact_paired_prob, delta=0.025)

  def test_map_rejects_excluded_from_active_pair_probability_and_width_adaptation(
      self,
  ):
    """Map-change rejections do not count toward pmove_pair_active or shrink sigma_p."""
    batch_size = 256
    n_e, n_h = 1, 4
    nelec = n_e + n_h
    pair_radius = 5.0
    # Electron e0 at (0, 0) is mutually paired with h0 at (4.8, 0) (dist 4.8).
    # Three unpaired holes h1, h2, h3 sit at (-4.85, 0), (0, 4.85), (0, -4.85)
    # (dist 4.85 > 4.8). When (e0, h0) translates jointly with sigma_p = 2.0
    # while unpaired holes barely move (sigma_n = 0.01), >50% of proposals move
    # e0 closer than 4.8 to one of h1, h2, h3 and violate same_pairing, while
    # flat_log_psi accepts 100% of proposals that preserve same_pairing.
    pos_e0 = jnp.broadcast_to(
        jnp.array([[0.0, 0.0]])[None, ...], (batch_size, n_e, 2)
    )
    pos_h0 = jnp.broadcast_to(
        jnp.array([[4.8, 0.0], [-4.85, 0.0], [0.0, 4.85], [0.0, -4.85]])[
            None, ...
        ],
        (batch_size, n_h, 2),
    )
    x0 = jnp.reshape(
        jnp.concatenate([pos_e0, pos_h0], axis=1), (batch_size, nelec * 2)
    )
    data = _make_dummy_data(x0, nelec)
    flat_log_psi = lambda params, x, s, a, c: jnp.zeros((x.shape[0],))

    (
        _,
        _,
        _,
        num_accepts,
        num_map_rej,
        total_pairs,
        active_pair_tries,
        active_pair_accepts,
    ) = mcmc.mh_pair_update(
        params={},
        f=flat_log_psi,
        data=data,
        key=jax.random.PRNGKey(777),
        lp_1=jnp.zeros((batch_size,)),
        num_accepts=0.0,
        stddev=2.0,
        unpaired_stddev=0.01,
        atoms=None,
        ndim=2,
        layer_counts=(n_e, n_h),
        pair_radius=pair_radius,
        return_pair_diagnostics=True,
    )

    # Over 50% of proposals are rejected due to pairing-map changes, so raw
    # acceptance rate num_accepts / batch_size < 0.50.
    self.assertGreater(float(num_map_rej) / batch_size, 0.50)
    self.assertLess(float(num_accepts) / batch_size, 0.50)
    self.assertAlmostEqual(float(total_pairs) / batch_size, 1.0, places=5)

    # Active pair tries exclude map-change rejections, so active acceptance is 1.0.
    self.assertAlmostEqual(
        float(active_pair_tries), float(batch_size - num_map_rej), places=5
    )
    self.assertGreater(float(active_pair_tries), 0.0)
    self.assertAlmostEqual(
        float(active_pair_accepts / active_pair_tries), 1.0, places=6
    )

    # Feeding these counts into update_pair_mcmc_widths increases sigma_p by 1.1
    # instead of shrinking it due to map rejections.
    adapt_freq = 4
    state = mcmc.init_pair_mcmc_adaptation_state(adapt_freq)
    width = jnp.array([[0.2, 3.0]])
    stats = {
        'n_try_normal': jnp.array([float(batch_size)]),
        'n_acc_normal': jnp.array([float(batch_size) * 0.52]),
        'n_try_pair': jnp.array([float(batch_size)]),
        'n_acc_pair': jnp.array([float(num_accepts)]),
        'n_try_pair_active': jnp.array([float(active_pair_tries)]),
        'n_acc_pair_active': jnp.array([float(active_pair_accepts)]),
    }
    for t in range(1, adapt_freq + 1):
      width, state = mcmc.update_pair_mcmc_widths(
          t, width, adapt_freq, stats, state
      )
    self.assertAlmostEqual(float(width[0, 0]), 0.2, places=6)
    self.assertAlmostEqual(float(width[0, 1]), 3.0 * 1.1, places=6)


if __name__ == '__main__':
  absltest.main()
