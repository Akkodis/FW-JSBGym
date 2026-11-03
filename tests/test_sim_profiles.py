"""Tests of the simulator behaviour profiles (fw_jsbgym/envs/sim_profiles.py).

The attitude-control tasks import FW-FlightControl, so run from the FW-JSBGym root with:
    PYTHONPATH=.:../FW-FlightControl python -m unittest discover -s tests
"""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

import fw_jsbgym
from fw_jsbgym.envs.sim_profiles import DEFAULT_SIM_PROFILE, SIM_PROFILES, get_sim_profile
from fw_jsbgym.utils import jsbsim_properties as prp


PACKAGE = Path(fw_jsbgym.__file__).resolve().parent
# Exact copies of Git blobs, see fw_jsbgym/fdm_descriptions/aircraft/x8_icinco2024/README.md
HISTORICAL_ASSETS = {
    # git show 97e356e^:fw_jsbgym/fdm_descriptions/aircraft/x8/x8.xml
    "fdm_descriptions/aircraft/x8_icinco2024/x8_icinco2024.xml":
        "e26f40d33742d5ac2fd6930ff2ba582ee401213a9c55ccb9775a0c361d96e01f",
    # git show 1072782:fw_jsbgym/initial_conditions/x8_basic_ic.xml
    "initial_conditions/x8_icinco2024_basic_ic.xml":
        "0ec5bc4a33902dc96af7d0e172109c9b95b9f537e3f36c53a128bc02bcdf0c5a",
}
FT_TO_M = 0.3048


def setUpModule() -> None:
    # The environments write telemetry files relative to the working directory.
    global _workdir, _previous_cwd
    _workdir = tempfile.TemporaryDirectory()
    _previous_cwd = os.getcwd()
    os.chdir(_workdir.name)


def tearDownModule() -> None:
    os.chdir(_previous_cwd)
    _workdir.cleanup()


def make_cfg(sim_profile=None):
    atmosphere = {
        "severity": "off", "variable": False,
        "wind": {"enable": False, "rand_continuous": False},
        "turb": {"enable": False}, "gust": {"enable": False},
    }

    def sim_options(evaluating: bool) -> dict:
        return {
            "seed": 1.0, "atmosphere": deepcopy(atmosphere), "rand_fdm": {"enable": False},
            "was_evaluating": evaluating, "eval": evaluating,
        }

    jsbsim = {
        "fdm_freq": 100.0, "agent_freq": 100.0, "episode_length_s": 20.0, "viz_time_factor": 50.0,
        "debug": False, "render_mode": "none",
        "train_sim_options": sim_options(False), "eval_sim_options": sim_options(True),
    }
    if sim_profile is not None:
        jsbsim["sim_profile"] = sim_profile
    cfg = OmegaConf.create({
        "jsbsim": jsbsim,
        "task": {
            # FW-FlightControl config/env/task/mdp/no_hist_all.yaml and reward/ac_my_bohn.yaml
            "mdp": {
                "obs_is_matrix": False, "obs_hist_size": 1, "action_avg": False,
                "act_hist_size": 5, "act_pos_in_state": True,
            },
            "reward": {
                "name": "bohn_mod",
                "weights": {
                    "roll": {"scaling": 3.3, "clip_max": 0.5},
                    "pitch": {"scaling": 2.25, "clip_max": 0.5},
                    "act_var": {"enabled": False, "clip_max": 0.0},
                },
            },
        },
    })
    OmegaConf.set_struct(cfg, True)  # as Hydra does
    return cfg


def make_env(sim_profile=None):
    cfg = make_cfg(sim_profile)
    with redirect_stdout(io.StringIO()):
        env = gym.make(
            "ACBohnNoVaIErr-v0", cfg_env=cfg, telemetry_file="telemetry/telemetry.csv",
            render_mode="none",
        ).unwrapped
        env.init()
        env.reset(options=cfg.jsbsim.eval_sim_options)
    return env


def step(env, action) -> None:
    with redirect_stdout(io.StringIO()):
        env.step(np.asarray(action, dtype=np.float32))


def surface_positions(env) -> list:
    return [env.sim[prp.aileron_combined_pos_rad], env.sim[prp.elevator_pos_rad], env.sim[prp.throttle_pos]]


class SimProfileRegistryTests(unittest.TestCase):
    def test_available_profiles(self) -> None:
        self.assertEqual(DEFAULT_SIM_PROFILE, "current")
        self.assertEqual(sorted(SIM_PROFILES), ["current", "icinco2024"])

    def test_unknown_profile_lists_available_profiles(self) -> None:
        for name in ("icinco2023", "", None):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "current.*icinco2024"):
                    get_sim_profile(name)

    def test_current_profile_loads_requested_aircraft(self) -> None:
        current = get_sim_profile("current")
        self.assertEqual(current.jsbsim_model("x8"), "x8")
        self.assertEqual(current.jsbsim_model("c172p"), "c172p")

    def test_icinco2024_profile_only_supports_x8(self) -> None:
        icinco = get_sim_profile("icinco2024")
        self.assertEqual(icinco.jsbsim_model("x8"), "x8_icinco2024")
        with self.assertRaisesRegex(ValueError, "does not support aircraft 'c172p'"):
            icinco.jsbsim_model("c172p")

    def test_profiles_are_immutable(self) -> None:
        with self.assertRaises(AttributeError):
            get_sim_profile("current").gust_start_steps = (500,)
        with self.assertRaises(TypeError):
            get_sim_profile("icinco2024").jsbsim_models["x8"] = "x8"

    def test_historical_assets_are_unchanged(self) -> None:
        for name, expected in HISTORICAL_ASSETS.items():
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest(), expected)


class SimProfileEnvironmentTests(unittest.TestCase):
    def test_profile_selection(self) -> None:
        self.assertEqual(make_env().sim_profile.name, "current")
        self.assertEqual(make_env("current").sim_profile.name, "current")
        self.assertEqual(make_env("icinco2024").sim_profile.name, "icinco2024")
        with self.assertRaisesRegex(ValueError, "Unknown simulator profile"):
            make_env("icinco2023")

    def test_model_and_initial_conditions(self) -> None:
        current, icinco = make_env(), make_env("icinco2024")
        self.assertEqual(current.aircraft_id, "x8")
        self.assertEqual(icinco.aircraft_id, "x8")
        self.assertEqual(current.sim.fdm.get_model_name(), "x8")
        self.assertEqual(icinco.sim.fdm.get_model_name(), "x8_icinco2024")
        self.assertAlmostEqual(current.sim["velocities/u-fps"] * FT_TO_M, 16.67, places=6)
        self.assertAlmostEqual(icinco.sim["velocities/u-fps"] * FT_TO_M, 16.97, places=6)
        self.assertNotAlmostEqual(current.sim["ic/lat-geod-deg"], 0.0)
        self.assertEqual(icinco.sim["ic/lat-geod-deg"], 0.0)
        self.assertEqual(icinco.sim["ic/long-gc-deg"], 0.0)

    def integral_errors_after(self, sim_profile, next_target) -> list:
        env = make_env(sim_profile)
        with redirect_stdout(io.StringIO()):
            env.set_target_state(np.array([0.2, -0.1]))
            env.sim[prp.roll_integ_err] = 1.25
            env.sim[prp.pitch_integ_err] = -2.5
            env.set_target_state(np.array(next_target))
        return [env.sim[prp.roll_integ_err], env.sim[prp.pitch_integ_err]]

    def test_integral_errors_reset_on_every_target_call_by_default(self) -> None:
        for next_target in ([0.2, -0.1], [0.3, -0.1], [0.2, 0.0]):
            with self.subTest(next_target=next_target):
                self.assertEqual(self.integral_errors_after(None, next_target), [0.0, 0.0])

    def test_icinco2024_resets_only_integral_errors_of_changed_targets(self) -> None:
        cases = {
            "unchanged": ([0.2, -0.1], [1.25, -2.5]),
            "roll changed": ([0.3, -0.1], [0.0, -2.5]),
            "pitch changed": ([0.2, 0.0], [1.25, 0.0]),
            "both changed": ([0.3, 0.0], [0.0, 0.0]),
        }
        for case, (next_target, expected) in cases.items():
            with self.subTest(case=case):
                self.assertEqual(self.integral_errors_after("icinco2024", next_target), expected)

    def test_reset_clears_integral_errors_in_both_profiles(self) -> None:
        for sim_profile in (None, "icinco2024"):
            with self.subTest(sim_profile=sim_profile):
                env = make_env(sim_profile)
                env.sim[prp.roll_integ_err] = 1.25
                env.sim[prp.pitch_integ_err] = -2.5
                with redirect_stdout(io.StringIO()):
                    env.reset(options=env.sim_options)
                # reset() zeroes the integrals, then accumulates the first (tiny) tracking error
                for integral_error in (env.sim[prp.roll_integ_err], env.sim[prp.pitch_integ_err]):
                    self.assertLess(abs(integral_error), 1e-9)

    def test_surface_history_channels_and_timing(self) -> None:
        # current: [aileron, elevator, throttle] before the action is applied
        # icinco2024: [aileron, elevator] after the FDM step
        for sim_profile, recorded_after_step, channels in ((None, False, 3), ("icinco2024", True, 2)):
            with self.subTest(sim_profile=sim_profile):
                env = make_env(sim_profile)
                for action in ([0.6, -0.4], [-0.5, 0.3]):
                    before = surface_positions(env)
                    step(env, action)
                    after = surface_positions(env)
                    self.assertNotEqual(before[:2], after[:2])
                    expected = after if recorded_after_step else before
                    self.assertEqual(env.fcs_pos_hist[-1], expected[:channels])
                self.assertEqual(np.asarray(env.fcs_pos_hist).shape, (2, channels))

    def test_gust_start_steps(self) -> None:
        for sim_profile, expected in ((None, [250.0]), ("icinco2024", [500.0, 1500.0])):
            with self.subTest(sim_profile=sim_profile):
                env = make_env(sim_profile)
                started_at = []
                env.gust_start = lambda env=env: started_at.append(env.sim[env.current_step])
                env.sim_options.atmosphere.gust.enable = True
                for current_step in (249, 250, 251, 500, 1000, 1500):
                    env.sim[env.current_step] = current_step
                    step(env, [0.0, 0.0])
                self.assertEqual(started_at, expected)

    def test_gusts_disabled(self) -> None:
        for sim_profile in (None, "icinco2024"):
            with self.subTest(sim_profile=sim_profile):
                env = make_env(sim_profile)
                started_at = []
                env.gust_start = lambda env=env: started_at.append(env.sim[env.current_step])
                for current_step in (250, 500, 1500):
                    env.sim[env.current_step] = current_step
                    step(env, [0.0, 0.0])
                self.assertEqual(started_at, [])


if __name__ == "__main__":
    unittest.main()
