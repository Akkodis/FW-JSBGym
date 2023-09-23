import math
import collections


"""
ATTRIBUTION: Based on https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/properties.py by Gordon Rennie, defines parameters
and bounds properties to ensure out-of-bound values are not assigned to JSBSim properties
"""

class AttributeFormatter:
    ILLEGAL_CHARS = '\-/.'
    TRANSLATE_TO = '_' * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)


class BoundedProperty(collections.namedtuple('BoundedProperty', ['name', 'description', 'min', 'max'])):
    def get_legal_name(self):
        return AttributeFormatter.translate(self.name)


class Property(collections.namedtuple('Property', ['name', 'description'])):
    def get_legal_name(self):
        return AttributeFormatter.translate(self.name)


# position and attitude
altitude_sl_ft = BoundedProperty('position/h-sl-ft', 'altitude above mean sea level [ft]', -1400, 85000)
altitude_sl_m = BoundedProperty('position/h-sl-meters', 'altitude above mean sea level [m]', -427, 26000)
pitch_rad = BoundedProperty('attitude/pitch-rad', 'pitch [rad]', -math.pi, math.pi)
roll_rad = BoundedProperty('attitude/roll-rad', 'roll [rad]', -math.pi, math.pi)
heading_rad = BoundedProperty('attitude/psi-rad', 'yaw [rad', -math.pi, math.pi)
heading_deg = BoundedProperty('attitude/psi-deg', 'heading [deg]', 0, 360)
flight_path_rad: float = BoundedProperty('flight-path/psi-gt-rad', 'flight path angle [rad]', -math.pi, math.pi)
sideslip_deg = BoundedProperty('aero/beta-deg', 'sideslip [deg]', -180, +180)
lat_gc_deg = BoundedProperty('position/lat-gc-deg', 'geocentric latitude [deg]', -180, 180)
lng_gc_deg = BoundedProperty('position/long-gc-deg', 'geocentric longitude [deg]', -180, 180)
lat_travel_m = BoundedProperty('position/distance-from-start-lat-mt', 'latitude distance travelled from start [m]',
                               float('-inf'), float('+inf'))
lng_travel_m = BoundedProperty('position/distance-from-start-lon-mt', 'longitude distance travelled from start [m]',
                               float('-inf'), float('+inf'))
dist_travel_m = Property('position/distance-from-start-mag-mt', 'distance travelled from starting position [m]')

# velocities
u_fps = BoundedProperty('velocities/u-fps', 'body frame x-axis velocity [ft/s]', -2200, 2200)
v_fps = BoundedProperty('velocities/v-fps', 'body frame y-axis velocity [ft/s]', -2200, 2200)
w_fps = BoundedProperty('velocities/w-fps', 'body frame z-axis velocity [ft/s]', -2200, 2200)
v_north_fps = BoundedProperty('velocities/v-north-fps', 'velocity true north [ft/s]', float('-inf'), float('+inf'))
v_east_fps = BoundedProperty('velocities/v-east-fps', 'velocity east [ft/s]', float('-inf'), float('+inf'))
v_down_fps = BoundedProperty('velocities/v-down-fps', 'velocity downwards [ft/s]', float('-inf'), float('+inf'))
p_radps = BoundedProperty('velocities/p-rad_sec', 'roll rate [rad/s]', -20, 20)
q_radps = BoundedProperty('velocities/q-rad_sec', 'pitch rate [rad/s]', -20, 20)
r_radps = BoundedProperty('velocities/r-rad_sec', 'yaw rate [rad/s]', -20, 20)
altitude_rate_fps = Property('velocities/h-dot-fps', 'Rate of altitude change [ft/s]')
airspeed_fps = BoundedProperty('velocities/vt-fps', 'True aircraft airspeed [ft/s]', float('-inf'), float('+inf'))
airspeed_kts = BoundedProperty('velocities/vtrue-kts', 'True aircraft airspeed [kts]', float('-inf'), float('+inf'))
airspeed_mps = BoundedProperty('velocities/vt-mps', 'True aircraft airspeed [m/s]', 0, 42) # 42 m/s = 150 km/h = 81 knots
alpha = Property('aero/alpha-rad', 'aircraft angle of attack [rad]')
ci2vel = Property('aero/ci2vel', 'chord/2*airspeed')

# controls state
aileron_left = BoundedProperty('fcs/left-aileron-pos-norm', 'left aileron position, normalised', -1, 1)
aileron_right = BoundedProperty('fcs/right-aileron-pos-norm', 'right aileron position, normalised', -1, 1)
elevator = BoundedProperty('fcs/elevator-pos-norm', 'elevator position, normalised', -1, 1)
rudder = BoundedProperty('fcs/rudder-pos-norm', 'rudder position, normalised', -1, 1)
throttle = BoundedProperty('fcs/throttle-pos-norm', 'throttle position, normalised', 0, 1)
gear = BoundedProperty('gear/gear-pos-norm', 'landing gear position, normalised', 0, 1)

aileron_left_rad = Property('fcs/left-aileron-pos-rad', 'left aileron deflection [rad]')
aileron_right_rad = Property('fcs/right-aileron-pos-rad', 'right aileron deflection [rad]')
aileron_combined_rad = BoundedProperty('fcs/effective-aileron-pos', 'combined effective aileron deflection [rad]', -1.04, 1.04)
elevator_rad = BoundedProperty('fcs/elevator-pos-rad', 'elevator deflection [rad]', -0.52, 0.52)
rudder_rad = Property('fcs/rudder-pos-rad', 'rudder deflection [rad]')

# engines
engine_running = Property('propulsion/engine/set-running', 'engine running (0/1 bool)')
all_engine_running = Property('propulsion/set-running', 'set engine running (-1 for all engines)')
engine_thrust_lbs = Property('propulsion/engine/thrust-lbs', 'engine thrust [lb]')

# controls command
aileron_cmd = BoundedProperty('fcs/aileron-cmd-norm', 'aileron commanded position, normalised', -1., 1.)
elevator_cmd = BoundedProperty('fcs/elevator-cmd-norm', 'elevator commanded position, normalised', -1., 1.)
rudder_cmd = BoundedProperty('fcs/rudder-cmd-norm', 'rudder commanded position, normalised', -1., 1.)
throttle_cmd = BoundedProperty('fcs/throttle-cmd-norm', 'throttle commanded position, normalised', 0., 1.)
mixture_cmd = BoundedProperty('fcs/mixture-cmd-norm', 'engine mixture setting, normalised', 0., 1.)
throttle_1_cmd = BoundedProperty('fcs/throttle-cmd-norm[1]', 'throttle 1 commanded position, normalised', 0., 1.)
mixture_1_cmd = BoundedProperty('fcs/mixture-cmd-norm[1]', 'engine mixture 1 setting, normalised', 0., 1.)
gear_all_cmd = BoundedProperty('gear/gear-cmd-norm', 'all landing gear commanded position, normalised', 0, 1)

# autopilot commands
heading_des = BoundedProperty('ap/heading_setpoint', 'desired heading [deg]', -180, 180)
level_des = BoundedProperty('ap/altitude_setpoint', 'desired altitude [ft]', -1400, 85000)
heading_switch = BoundedProperty('ap/heading_hold', 'engage heading mode [bool]', 0, 1)
level_switch = BoundedProperty('ap/altitude_hold', 'engage alt hold [bool]', 0, 1)
attitude_switch = BoundedProperty('ap/attitude_hold', 'engage att hold [bool]', 0, 1)
wing_level_switch = BoundedProperty('fcs/wing-leveler-ap-on-off', 'engage wing leveler [bool]', -1, 0)

# simulation
sim_dt = Property('simulation/dt', 'JSBSim simulation timestep [s]')
sim_time_s = Property('simulation/sim-time-sec', 'Simulation time [s]')
trim_switch = BoundedProperty('simulation/do_simple_trim', 'engage trimming [bool]', 0, 1)

# initial conditions
initial_altitude_ft = Property('ic/h-sl-ft', 'initial altitude MSL [ft]')
initial_terrain_altitude_ft = Property('ic/terrain-elevation-ft', 'initial terrain alt [ft]')
initial_longitude_geoc_deg = Property('ic/long-gc-deg', 'initial geocentric longitude [deg]')
initial_latitude_geod_deg = Property('ic/lat-geod-deg', 'initial geodesic latitude [deg]')
initial_roll_rad = Property('ic/phi-rad', 'initial roll angle [rad]')
initial_pitch_rad = Property('ic/theta-rad', 'initial pitch angle [rad]')
initial_u_fps = Property('ic/u-fps', 'body frame x-axis velocity; positive forward [ft/s]')
initial_v_fps = Property('ic/v-fps', 'body frame y-axis velocity; positive right [ft/s]')
initial_w_fps = Property('ic/w-fps', 'body frame z-axis velocity; positive down [ft/s]')
initial_p_radps = Property('ic/p-rad_sec', 'roll rate [rad/s]')
initial_q_radps = Property('ic/q-rad_sec', 'pitch rate [rad/s]')
initial_r_radps = Property('ic/r-rad_sec', 'yaw rate [rad/s]')
initial_roc_fpm = Property('ic/roc-fpm', 'initial rate of climb [ft/min]')
initial_heading_deg = Property('ic/psi-true-deg', 'initial (true) heading [deg]')
initial_airspeed_kts = Property('ic/vt-kts', 'initial airspeeed [kts]')

# metrics
qbar_area = Property('aero/qbar-area', 'dynamic pressure * wing-planform area')
Sw = Property('metrics/Sw-sqft', 'wing area [sqft]')
rho = Property('atmosphere/rho-slugs_ft3', 'air density [slug/ft^3]')

# fdm values
Clo = Property('aero/coefficient/CLo', 'zero lift')
Clalpha = Property('aero/coefficient/CLalpha', 'alpha lift')
Clq = Property('aero/coefficient/CLq', 'pitch-rate lift')
ClDe = Property('aero/coefficient/CLDe', 'elevator deflection lift')
Cmo = Property('aero/coefficient/Cmo', 'zero lift pitch')
Cmalpha = Property('aero/coefficient/Cmalpha', 'alpha pitch')
Cmq = Property('aero/coefficient/Cmq', 'pitch rate pitch')
CmDe = Property('aero/coefficient/CmDe', 'pitch due to elevator')

# additional custom properties for error and target values
airspeed_err = BoundedProperty("error/airspeed-err", "airspeed error", -42, 42)
roll_err = BoundedProperty("error/roll-err", "roll error", -2*math.pi, 2*math.pi)
pitch_err = BoundedProperty("error/pitch-err", "pitch error", -2*math.pi, 2*math.pi)
target_airspeed_kts = BoundedProperty("target/airspeed", "desired airspeed", float('-inf'), float('+inf'))
target_airspeed_mps = BoundedProperty("target/airspeed", "desired airspeed", 0, 42)
target_roll_rad = BoundedProperty("target/roll-rad", "desired roll angle [rad]", -math.pi, math.pi)
target_pitch_rad = BoundedProperty("target/pitch-rad", "desired pitch angle [rad]", -math.pi, math.pi)

# action avg over last n steps
aileron_avg = BoundedProperty("fcs/aileron_avg", "aileron action average", aileron_cmd.min, aileron_cmd.max)
elevator_avg = BoundedProperty("fcs/elevator_avg", "elevator action average", elevator_cmd.min, elevator_cmd.max)
throttle_avg = BoundedProperty("fcs/throttle_avg", "throttle action average", throttle_cmd.min, throttle_cmd.max)

# reward properties
reward_total = BoundedProperty("reward/total", "total reward", float('-inf'), 0)
reward_roll = BoundedProperty("reward/roll", "roll reward", float('-inf'), 0)
reward_pitch = BoundedProperty("reward/pitch", "pitch reward", float('-inf'), 0)
reward_airspeed = BoundedProperty("reward/airspeed", "airspeed reward", float('-inf'), 0)
reward_actvar = BoundedProperty("reward/actvar", "action variation reward", float('-inf'), 0)
