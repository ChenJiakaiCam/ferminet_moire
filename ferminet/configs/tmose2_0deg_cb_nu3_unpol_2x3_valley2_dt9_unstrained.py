"""Unstrained 0-deg P-MoSe2 Q-valley config (nu=3, unpolarized, 2x3, valley_index=2, delta_t=+9 meV)."""

from ferminet.configs import tmose2_0deg_cb_wigner
import ml_collections

finalize = tmose2_0deg_cb_wigner.finalize


def get_config() -> ml_collections.ConfigDict:
  """Returns ConfigDict for unstrained 0-deg tMoSe2 at nu=3 on a 2x3 supercell."""
  return tmose2_0deg_cb_wigner.build_preset_config(
      nu=3.0,
      supercell_matrix=((2, 0), (0, 3)),
      config_module='.tmose2_0deg_cb_nu3_unpol_2x3_valley2_dt9_unstrained',
      valley_index=2,
      delta_t_mev=9.0,
      unstrained=True,
  )
