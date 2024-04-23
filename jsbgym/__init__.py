import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='jsbgym.envs.jsbsim_env:JSBSimEnv',
    # autoreset=True
)

gymnasium.register(
    id='ACVanilla-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaIErr-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaIErrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAct-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaActTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAlpha-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaAlphaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaBeta-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaBetaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAlphaBeta-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaAlphaBetaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaThr-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaThrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohn-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.ac_bohn:ACBohnTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVa-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACNoVaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVaIErr-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACBohnNoVaIErrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaPIDRLAdd-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.ac_bohn_nova_pidrl:ACNoVaPIDRLAddTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaPIDRL-v0',
    entry_point='jsbgym.envs.tasks.attitude_control.ac_bohn_nova_pidrl:ACNoVaPIDRLTask',
    # autoreset=True
)

gymnasium.register(
    id='SimpleAC_OMAC-v0',
    entry_point='jsbgym.envs.tasks.simple_ac:SimpleAC_OMAC'
)
