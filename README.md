# FW-JSBGym
RL compatible framework of the JSBSim simulator. All control algorithms, training and testing scripts are on the [FWFlightControl](https://github.com/Akkodis/FW-FlightControl/tree/jsbsim) repository.

## Installation
Requires python 3.9

If you wish to use the FlightGear visualization:
- Install the AppImage from [here](https://www.flightgear.org/download/) and place the file into your `$HOME/Apps/` directory (create it if it does not exist).
- Launch the AppImage for it to create all the config and data folders.
- Copy the contents of `fgdata/Aircraft/` of the repository in the `$HOME/Apps/fgdata/Aircraft/`.

```
conda env create --file environment.yml
pip install -e .
```

## Directory organization
- `envs/` contains the `JSBSimEnv` abstract class, describing the basis of a JSBSim env.
    - `tasks/attitude_control` contains the child classes of JSBSimEnv for the attitude control tasks.
    - `sim_profiles.py` defines the simulator behaviour profiles (see below).
- `fdm_descriptions/` JSBSim description of the c172p and x8 aircrafts.
    - `aircraft/x8_icinco2024/` frozen X8 model of the ICINCO 2024 experiments (used by the `icinco2024` profile only).
- `fgdata/` FlightGear visualization files.
- `models/` Saves of NN models.
- `simulation/jsb_simulation.py` contains core functions of JSBSim for setting up, stepping the simulation, accessing properties and converting units.
- `utils/jsbsim_properties.py` references properties to be accessed, inspired from existing work: https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/properties.py
- `visualizers/`
    - `attitude_control_telemetry.py` handles the 'live' plot visualization of a flight.
    - `visualizer.py` handles FlightGear viz or calls the live plot visualizer.
- `../tests/` unit tests (run from the repository root with `PYTHONPATH=.:../FW-FlightControl python -m unittest discover -s tests`).

## Simulator profiles
Some simulator behaviour changed after the ICINCO 2024 experiments. To reproduce them, select the `icinco2024` profile with the `sim_profile` key of the `env.jsbsim` config, e.g. `+env.jsbsim.sim_profile=icinco2024` on a FW-FlightControl command line. Without the key, the `current` behaviour is used.

> [!WARNING]
> Use JSBSim 1.1.13 for ICINCO disturbance reproduction. The profile does not
> change the installed JSBSim version; turbulence results differ with 1.2.3.
> See the FW-FlightControl reproducibility guide for evidence and limitations.

The `icinco2024` profile loads the paper-era X8 model and initial conditions, only resets the integral error of an axis whose target changed, records only the aileron and elevator positions (after the FDM step) for the actuator fluctuation metric, samples wind and gust directions from the historical full cube, and starts gusts at steps 500 and 1500. Its validation is documented in FW-FlightControl `reproducibility/reports/2026-10-30_icinco_ppo_seed1_drift_diagnosis.md`.


## Citation
```
@inproceedings{olivares2024mfvsmb,
  title={Model-Free versus Model-Based Reinforcement Learning for Fixed-Wing UAV Attitude Control Under Varying Wind Conditions}, 
  author={David Olivares and Pierre Fournier and Pavan Vasishta and Julien Marzat},
  booktitle={International Conference on Informatics in Control, Automation and Robotics (ICINCO)},
  year={2024}
}
```