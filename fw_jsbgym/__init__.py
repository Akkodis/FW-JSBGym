import gymnasium

gymnasium.register(
    id='JSBSimEnv-v0',
    entry_point='fw_jsbgym.envs.jsbsim_env:JSBSimEnv',
    # autoreset=True
)

gymnasium.register(
    id='ACVanilla-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaYaw-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaYawTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaIErr-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaIErrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAct-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaActTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAlpha-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaAlphaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaBeta-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaBetaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaAlphaBeta-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaAlphaBetaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaThr-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaThrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACVanillaActIErr-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaActIErrTask',
)

gymnasium.register(
    id='ACVanillaActIErrAlpha-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaActIErrAlphaTask',
)

gymnasium.register(
    id='ACVanillaActIErrBeta-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.vanilla:ACVanillaActIErrBetaTask',
)

gymnasium.register(
    id='ACBohn-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn:ACBohnTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVa-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACBohnNoVaTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVaIErr-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACBohnNoVaIErrTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVaIErrYaw-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACBohnNoVaIErrYawTask',
    # autoreset=True
)

gymnasium.register(
    id='ACBohnNoVaIErrWindOracle-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova:ACBohnNoVaIErrWindOracleTask'
)

gymnasium.register(
    id='ACNoVaPIDRLAdd-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova_pidrl:ACNoVaPIDRLAddTask',
    # autoreset=True
)

gymnasium.register(
    id='ACNoVaPIDRL-v0',
    entry_point='fw_jsbgym.envs.tasks.attitude_control.ac_bohn_nova_pidrl:ACNoVaPIDRLTask',
    # autoreset=True
)

gymnasium.register(
    id='SimpleAC_OMAC-v0',
    entry_point='fw_jsbgym.envs.tasks.simple_ac:SimpleAC_OMAC'
)

gymnasium.register(
    id='WaypointTracking-v0',
    entry_point='fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking:WaypointTracking'
)

gymnasium.register(
    id='WaypointVaTracking-v0',
    entry_point='fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking:WaypointVaTracking'
)

gymnasium.register(
    id='WaypointTrackingNoVa-v0',
    entry_point='fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking:WaypointTrackingNoVa'
)

gymnasium.register(
    id='AltitudeTracking-v0',
    entry_point='fw_jsbgym.envs.tasks.waypoint_tracking.wp_tracking:AltitudeTracking'
)