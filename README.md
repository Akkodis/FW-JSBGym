# FW-JSBGym
RL compatible framework of the JSBSim simulator. All control algorithms, training and testing scripts are on the [FWFlightControl](https://github.com/Akkodis/FW-FlightControl/tree/jsbsim) repository.

## Installation
Requires python 3.10

If you wish to use the FlightGear visualization:
- Install the AppImage from [here](https://www.flightgear.org/download/) and place the file into your `$HOME/Apps/` directory (create it if it does not exist).
- Launch the AppImage for it to create all the config and data folders.
- Copy the contents of `fgdata/Aircraft/` of the repository in the `$HOME/Apps/fgdata/Aircraft/`.
```
pip install -r pip_requirements.txt
pip install -e .
```
or
```
conda env create --file environment.yml
pip install -e .
```

## Directory organization
- `envs/` contains the `JSBSimEnv` abstract class, describing the basis of a JSBSim env.
    - `tasks/attitude_control` contains the child classes of JSBSimEnv for the attitude control tasks.
    - `fdm_descriptions/` JSBSim description of the c172p and x8 aircrafts.
    - `fgdata/` FlightGear visualization files.
    - `models/` Saves of NN models.
    - `simulation/jsb_simulation.py` contains core functions of JSBSim for setting up, stepping the simulation, accessing properties and converting units.
    - `utils/jsbsim_properties.py` references properties to be accessed, inspired from existing work: https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/properties.py
    - `visualizers/`
        - `attitude_control_telemetry.py` handles the 'live' plot visualization of a flight.
        - `visualizer.py` handles FlightGear viz or calls the live plot visualizer.