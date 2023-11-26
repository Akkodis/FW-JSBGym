import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='jsbgym.envs.jsbsim_env:JSBSimEnv',
    autoreset=True
)

gymnasium.register(
    id='AttitudeControl-v0',
    entry_point='jsbgym.envs.tasks.attitude_control:AttitudeControlTask',
    autoreset=True
)

gymnasium.register(
    id='AttitudeControlNoVa-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:AttitudeControlNoVaTask',
    autoreset=True
)