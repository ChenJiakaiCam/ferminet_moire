"""Strained 0-deg P-MoSe2 Q-valley config (nu=2, unpolarized, 3x3, valley_index=2, delta_t=+9 meV)."""

from ferminet.configs import tmose2_0deg_cb_wigner
import ml_collections

finalize = tmose2_0deg_cb_wigner.finalize


def get_config() -> ml_collections.ConfigDict:
  """Returns ConfigDict for strained 0-deg tMoSe2 at nu=2 on a 3x3 supercell."""
  return tmose2_0deg_cb_wigner.build_preset_config(
      nu=2.0,
      supercell_matrix=((3, 0), (0, 3)),
      config_module='.tmose2_0deg_cb_nu2_unpol_3x3_valley2_dt9',
      valley_index=2,
      delta_t_mev=9.0,
      unstrained=False,
  )
