import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='jsbgym.envs.jsbsim_env:JSBSimEnv',
    # autoreset=True
)

gymnasium.register(
    id='AC-v0',
    entry_point='jsbgym.envs.tasks.attitude_control:AttitudeControlTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVa-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:ACNoVaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaIntegErr-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:ACNoVaIntegErrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaPIDRLAdd-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:ACNoVaPIDRLAddTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaPIDRL-v0',
    entry_point='jsbgym.envs.tasks.attitude_control_no_va:ACNoVaPIDRLTask',
    # autoreset=True
)

gymnasium.register(
    id='SimpleAC-v0',
    entry_point='jsbgym.envs.tasks.simple_ac:SimpleAC'
)
