import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='jsbgym.envs.jsbsim_env:JSBSimEnv',
    autoreset=True
)

gymnasium.register(
    id='AttitudeControlTaskEnv-v0',
    entry_point='jsbgym.envs.attitude_control:AttitudeControlTaskEnv',
    autoreset=True
)