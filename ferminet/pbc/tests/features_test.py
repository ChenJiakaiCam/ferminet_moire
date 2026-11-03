# Copyright 2022 DeepMind Technologies Limited.
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
# limitations under the License

"""Tests for ferminet.pbc.feature_layer."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import networks
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class FeatureLayerTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_shape(self, heg):
    """Asserts that output shape of apply matches what is expected by init."""
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=heg
    )

    dims, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features, ee_features = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    assert dims[0] == ae_features.shape[-1]
    assert dims[1] == ee_features.shape[-1]

  def test_periodicity(self):
    nspins = (6, 5)
    atoms = jnp.asarray([[0., 0., 0.2], [1.2, 1., -0.2], [2.5, -0.8, 0.6]])
    natom = atoms.shape[0]
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    xs = jax.random.uniform(subkey, shape=(sum(nspins), 3))

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natom, nspins, 3, lattice=jnp.eye(3), include_r_ae=False
    )

    _, params = feature_layer.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_1, ee_features_1 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    # Select random electron coordinate to displace by a random lattice vector
    key, subkey = jax.random.split(key)
    e_idx = jax.random.randint(subkey, (1,), 0, xs.shape[0])
    key, subkey = jax.random.split(key)
    randvec = jax.random.randint(subkey, (3,), 0, 100).astype(jnp.float32)
    xs = xs.at[e_idx].add(randvec)

    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms)

    ae_features_2, ee_features_2 = feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params)

    atol, rtol = 4.e-3, 4.e-3
    np.testing.assert_allclose(
        ae_features_1, ae_features_2, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        ee_features_1, ee_features_2, atol=atol, rtol=rtol)

  @parameterized.parameters([True, False])
  def test_primitive_lattice_ae_vs_supercell_ee(self, include_r_ae):
    """Tests that ae uses primitive_lattice while ee uses supercell lattice."""
    a_m = 151.18
    prim_lattice = jnp.array(
        [[a_m, -0.5 * a_m], [0.0, 0.5 * jnp.sqrt(3.0) * a_m]],
        dtype=jnp.float32,
    )
    supercell_matrix = jnp.array([[6.0, 0.0], [0.0, 3.0]], dtype=jnp.float32)
    sim_lattice = prim_lattice @ supercell_matrix
    nspins = (2, 2, 2, 2)
    atoms = jnp.zeros((1, 2), dtype=jnp.float32)
    key = jax.random.PRNGKey(7)
    xs = jax.random.uniform(
        key, shape=(sum(nspins), 2), minval=-200.0, maxval=200.0
    )

    fl_prim = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=sim_lattice,
        primitive_lattice=prim_lattice,
        include_r_ae=include_r_ae,
        interlayer_distance=37.79,
        distance_scale=a_m,
    )
    fl_Prim_only = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=prim_lattice,
        include_r_ae=include_r_ae,
        interlayer_distance=37.79,
        distance_scale=a_m,
    )
    fl_sim_only = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=sim_lattice,
        include_r_ae=include_r_ae,
        interlayer_distance=37.79,
        distance_scale=a_m,
    )

    dims, params = fl_prim.init()
    ae, ee, r_ae, r_ee = networks.construct_input_features(xs, atoms, ndim=2)
    ae_feat, ee_feat = fl_prim.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params
    )
    ae_expected, _ = fl_Prim_only.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params
    )
    _, ee_expected = fl_sim_only.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params
    )

    self.assertEqual(ae_feat.shape[-1], dims[0])
    self.assertEqual(ee_feat.shape[-1], dims[1])
    np.testing.assert_allclose(ae_feat, ae_expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(ee_feat, ee_expected, atol=1e-5, rtol=1e-5)

    # Displacing particle 0 by a single primitive moire lattice vector a_1 + 2 a_2
    # must leave ae_features invariant, while changing ee_features.
    prim_shift = prim_lattice @ jnp.array([1.0, 2.0], dtype=jnp.float32)
    xs_prim_shifted = xs.at[0].add(prim_shift)
    ae_p, ee_p, r_ae_p, r_ee_p = networks.construct_input_features(
        xs_prim_shifted, atoms, ndim=2
    )
    ae_feat_p, ee_feat_p = fl_prim.apply(
        ae=ae_p, r_ae=r_ae_p, ee=ee_p, r_ee=r_ee_p, **params
    )
    np.testing.assert_allclose(ae_feat, ae_feat_p, atol=1e-4, rtol=1e-4)
    self.assertGreater(
        float(jnp.max(jnp.abs(ee_feat - ee_feat_p))), 1e-2
    )

    # Displacing particle 0 by a full supercell lattice vector must leave both
    # ae_features and ee_features invariant.
    sim_shift = sim_lattice @ jnp.array([1.0, -1.0], dtype=jnp.float32)
    xs_sim_shifted = xs.at[0].add(sim_shift)
    ae_s, ee_s, r_ae_s, r_ee_s = networks.construct_input_features(
        xs_sim_shifted, atoms, ndim=2
    )
    ae_feat_s, ee_feat_s = fl_prim.apply(
        ae=ae_s, r_ae=r_ae_s, ee=ee_s, r_ee=r_ee_s, **params
    )
    np.testing.assert_allclose(ae_feat, ae_feat_s, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ee_feat, ee_feat_s, atol=1e-4, rtol=1e-4)

  def test_primitive_lattice_shape_validation(self):
    with self.assertRaises(ValueError):
      pbc_feature_layer.make_pbc_feature_layer(
          natoms=1,
          nspins=(1, 1),
          ndim=2,
          lattice=jnp.eye(2),
          primitive_lattice=jnp.eye(3),
      )

  def test_fermi_net_periodicity_and_laplacian_with_primitive_lattice(self):
    """Tests full FermiNet forward pass and Laplacian with primitive_lattice."""
    from ferminet.pbc import envelopes as pbc_envelopes

    a_m = 151.18
    prim_lattice = jnp.array(
        [[a_m, -0.5 * a_m], [0.0, 0.5 * jnp.sqrt(3.0) * a_m]],
        dtype=jnp.float32,
    )
    sim_lattice = prim_lattice @ jnp.array(
        [[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float32
    )
    nspins = (1, 1, 1, 1)
    charges = jnp.zeros((1,), dtype=jnp.float32)
    atoms = jnp.zeros((1, 2), dtype=jnp.float32)

    feature_layer = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=nspins,
        ndim=2,
        lattice=sim_lattice,
        primitive_lattice=prim_lattice,
        include_r_ae=False,
        interlayer_distance=37.79,
        distance_scale=a_m,
    )
    kpoints = pbc_envelopes.make_kpoints(sim_lattice, nspins, ndim=2)
    envelope = pbc_envelopes.make_multiwave_envelope(kpoints)
    net = networks.make_fermi_net(
        nspins=nspins,
        charges=charges,
        ndim=2,
        determinants=2,
        hidden_dims=((16, 8), (16, 8)),
        feature_layer=feature_layer,
        envelope=envelope,
        full_det=True,
    )
    params = net.init(jax.random.PRNGKey(0))
    pos = jnp.array(
        [10.0, -20.0, 45.0, 30.0, -15.0, 60.0, 80.0, -10.0], dtype=jnp.float32
    )
    spins = jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float32)

    sign_0, logpsi_0 = net.apply(params, pos, spins, atoms, charges)
    self.assertTrue(bool(jnp.isfinite(logpsi_0)))

    # Shifting particle 0 by a supercell lattice vector must preserve log|psi|.
    sim_shift = sim_lattice @ jnp.array([1.0, -1.0], dtype=jnp.float32)
    pos_shifted = pos.at[:2].add(sim_shift)
    sign_s, logpsi_s = net.apply(params, pos_shifted, spins, atoms, charges)
    np.testing.assert_allclose(sign_0, sign_s, atol=1e-5)
    np.testing.assert_allclose(logpsi_0, logpsi_s, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
  absltest.main()
