"""Named simulator behaviour profiles.

Some simulator behaviour changed after a paper's experiments were run. A profile
restores such behaviour as one explicit, opt-in choice: historical experiments
stay reproducible while later work keeps the current defaults.

Select a profile with the ``sim_profile`` key of the ``env.jsbsim`` config, e.g.
``+env.jsbsim.sim_profile=icinco2024`` on a FW-FlightControl command line.
Without the key, the ``current`` profile is used.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional, Tuple


DEFAULT_SIM_PROFILE = "current"


@dataclass(frozen=True, eq=False)
class SimProfile:
    """
        Attr:
            - `name`: profile name used in the `env.jsbsim.sim_profile` config key
            - `description`: what the profile is for
            - `jsbsim_models`: supported aircraft IDs mapped to the JSBSim model to load. The initial conditions are read from
              `initial_conditions/<model>_basic_ic.xml`. `None` loads any requested aircraft unchanged.
            - `reset_integral_errors_every_target_call`: if False, an integral error is only reset when its own target changes
            - `record_surfaces_after_fdm_steps`: record surface positions in `fcs_pos_hist` after advancing the FDM,
              instead of before applying the action
            - `record_throttle_position`: also record the throttle position in `fcs_pos_hist`
            - `gust_start_steps`: episode steps at which a gust starts, when gusts are enabled
    """
    name: str
    description: str
    jsbsim_models: Optional[Mapping[str, str]]
    reset_integral_errors_every_target_call: bool
    record_surfaces_after_fdm_steps: bool
    record_throttle_position: bool
    gust_start_steps: Tuple[int, ...]

    def jsbsim_model(self, aircraft_id: str) -> str:
        """
            Returns the JSBSim model to load for `aircraft_id` under this profile.
        """
        if self.jsbsim_models is None:
            return aircraft_id
        if aircraft_id not in self.jsbsim_models:
            raise ValueError(
                f"Simulator profile '{self.name}' does not support aircraft '{aircraft_id}'. "
                f"Supported aircraft: {sorted(self.jsbsim_models)}"
            )
        return self.jsbsim_models[aircraft_id]


SIM_PROFILES: Mapping[str, SimProfile] = MappingProxyType({
    profile.name: profile for profile in (
        SimProfile(
            name="current",
            description="Current simulator behaviour (default).",
            jsbsim_models=None,
            reset_integral_errors_every_target_call=True,
            record_surfaces_after_fdm_steps=False,
            record_throttle_position=True,
            gust_start_steps=(250,),
        ),
        # Evidence: FW-FlightControl/reproducibility/reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md
        SimProfile(
            name="icinco2024",
            description="Simulator behaviour of the ICINCO 2024 attitude-control experiments.",
            # X8 model before the 2025-06-30 drag update (97e356e) and initial conditions as of 1072782.
            jsbsim_models=MappingProxyType({"x8": "x8_icinco2024"}),
            # Conditional resets as before 95d4b53.
            reset_integral_errors_every_target_call=False,
            # Aileron/elevator positions only, sampled after the FDM step, as before 95d4b53 and 98580b4.
            record_surfaces_after_fdm_steps=True,
            record_throttle_position=False,
            # Gusts at timesteps 500 and 1500 (ICINCO 2024 paper, Section 6.2.2; code at 1072782).
            gust_start_steps=(500, 1500),
        ),
    )
})


def get_sim_profile(name: str) -> SimProfile:
    """
        Returns the simulator profile called `name`.
    """
    try:
        return SIM_PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"Unknown simulator profile {name!r}. Available profiles: {sorted(SIM_PROFILES)}"
        ) from None
