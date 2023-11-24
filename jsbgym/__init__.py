import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='jsbgym.envs.jsbsim_env:JSBSimEnv',
    autoreset=True
)

gymnasium.register(
    id='AttitudeControlTask-v0',
    entry_point='jsbgym.envs.tasks.attitude_control:AttitudeControlTask',
    autoreset=True
)

gymnasium.register(
    id='AttitudeControlNoVaTask-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:AttitudeControlNoVaTask',
    autoreset=True
)