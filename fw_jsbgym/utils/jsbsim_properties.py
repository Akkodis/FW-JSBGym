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

# Helper class : setting those properties won't have any effect on the simulation, only for unit conversion and logging
class HelperProperty(collections.namedtuple('HelperProperty', ['name', 'description'])):
    def get_legal_name(self):
        return AttributeFormatter.translate(self.name)

class BoundedHelperProperty(collections.namedtuple('BoundedHelperProperty', ['name', 'description', 'min', 'max'])):
    def get_legal_name(self):
        return AttributeFormatter.translate(self.name)


# position and attitude
altitude_sl_ft = BoundedProperty('position/h-sl-ft', 'altitude above mean sea level [ft]', -1400, 85000)
altitude_sl_m = BoundedProperty('position/h-sl-meters', 'altitude above mean sea level [m]', -427, 26000)
pitch_rad = BoundedProperty('attitude/pitch-rad', 'pitch [rad]', -math.pi, math.pi)
roll_rad = BoundedProperty('attitude/roll-rad', 'roll [rad]', -math.pi, math.pi)
heading_rad = BoundedProperty('attitude/psi-rad', 'yaw [rad]', 0, 2*math.pi)
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

# ECEF position coordinates
ecef_x_ft = BoundedProperty('position/ecef-x-ft', 'ECEF x coordinate [ft]', float('-inf'), float('+inf'))
ecef_y_ft = BoundedProperty('position/ecef-y-ft', 'ECEF y coordinate [ft]', float('-inf'), float('+inf'))
ecef_z_ft = BoundedProperty('position/ecef-z-ft', 'ECEF z coordinate [ft]', float('-inf'), float('+inf'))

# Conversions
ecef_x_m = BoundedHelperProperty('position/ecef-x-m', 'ECEF x coordinate [m]', float('-inf'), float('+inf'))
ecef_y_m = BoundedHelperProperty('position/ecef-y-m', 'ECEF y coordinate [m]', float('-inf'), float('+inf'))
ecef_z_m = BoundedHelperProperty('position/ecef-z-m', 'ECEF z coordinate [m]', float('-inf'), float('+inf'))
ecef_x_km = BoundedHelperProperty('position/ecef-x-km', 'ECEF x coordinate [km]', float('-inf'), float('+inf'))
ecef_y_km = BoundedHelperProperty('position/ecef-y-km', 'ECEF y coordinate [km]', float('-inf'), float('+inf'))
ecef_z_km = BoundedHelperProperty('position/ecef-z-km', 'ECEF z coordinate [km]', float('-inf'), float('+inf'))

# ENU
enu_x_m = BoundedHelperProperty('position/enu-x-m', 'ENU x coordinate [m]', float('-inf'), float('+inf'))
enu_y_m = BoundedHelperProperty('position/enu-y-m', 'ENU y coordinate [m]', float('-inf'), float('+inf'))
enu_z_m = BoundedHelperProperty('position/enu-z-m', 'ENU z coordinate [m]', float('-inf'), float('+inf'))

# zero
zero = BoundedHelperProperty('zero', 'zero value', 0, 0)
zero_ = BoundedHelperProperty('zero_', 'zero value', 0, 0)

# velocities
u_fps = BoundedProperty('velocities/u-fps', 'body frame x-axis velocity [ft/s]', -2200, 2200)
v_fps = BoundedProperty('velocities/v-fps', 'body frame y-axis velocity [ft/s]', -2200, 2200)
w_fps = BoundedProperty('velocities/w-fps', 'body frame z-axis velocity [ft/s]', -2200, 2200)
v_north_fps = BoundedProperty('velocities/v-north-fps', 'velocity true north [ft/s]', float('-inf'), float('+inf'))
v_east_fps = BoundedProperty('velocities/v-east-fps', 'velocity east [ft/s]', float('-inf'), float('+inf'))
v_down_fps = BoundedProperty('velocities/v-down-fps', 'velocity downwards [ft/s]', float('-inf'), float('+inf'))
p_radps = BoundedProperty('velocities/p-rad_sec', 'roll rate [rad/s]', -35, 35)
q_radps = BoundedProperty('velocities/q-rad_sec', 'pitch rate [rad/s]', -35, 35)
r_radps = BoundedProperty('velocities/r-rad_sec', 'yaw rate [rad/s]', -35, 35)
altitude_rate_fps = Property('velocities/h-dot-fps', 'Rate of altitude change [ft/s]')
airspeed_fps = BoundedProperty('velocities/vt-fps', 'True aircraft airspeed [ft/s]', float('-inf'), float('+inf'))
airspeed_kts = BoundedProperty('velocities/vtrue-kts', 'True aircraft airspeed [kts]', float('-inf'), float('+inf'))
airspeed_mps = BoundedHelperProperty('velocities/vt-mps', 'True aircraft airspeed [m/s]', 0, 111)# 111 m/s = 400 km/h = 216 kts 
airspeed_kph = BoundedHelperProperty('velocities/vt-kph', 'True aircraft airspeed [m/s]', 0, 400) # 111 m/s = 400 km/h = 216 kts
alpha_rad = BoundedProperty('aero/alpha-rad', 'aircraft angle of attack [rad]', float('-inf'), float('+inf'))
beta_rad = BoundedProperty('aero/beta-rad', 'aircraft sideslip angle [rad]', float('-inf'), float('+inf'))
ci2vel = Property('aero/ci2vel', 'chord/2*airspeed')

# windspeeds
total_windspeed_north_fps = Property('atmosphere/total-wind-north-fps', 'total wind speed north [ft/s]')
total_windspeed_north_mps = HelperProperty('atmosphere/total-wind-north-mps', 'total wind speed north [m/s]')
total_windspeed_north_kph = BoundedHelperProperty('atmosphere/total-wind-north-kph', 'total wind speed north [km/h]', -400, 400)
total_windspeed_east_fps = Property('atmosphere/total-wind-east-fps', 'total wind speed east [ft/s]')
total_windspeed_east_mps = HelperProperty('atmosphere/total-wind-east-mps', 'total wind speed east [m/s]')
total_windspeed_east_kph = BoundedHelperProperty('atmosphere/total-wind-east-kph', 'total wind speed east [km/h]', -400, 400)
total_windspeed_down_fps = Property('atmosphere/total-wind-down-fps', 'total wind speed down [ft/s]')
total_windspeed_down_mps = HelperProperty('atmosphere/total-wind-down-mps', 'total wind speed down [m/s]')
total_windspeed_down_kph = BoundedHelperProperty('atmosphere/total-wind-down-kph', 'total wind speed down [km/h]', -400, 400)
windspeed_north_fps = Property('atmosphere/wind-north-fps', 'wind speed north [ft/s]')
windspeed_north_mps = HelperProperty('atmosphere/wind-north-mps', 'wind speed north [m/s]')
windspeed_north_kph = HelperProperty('atmosphere/wind-north-kph', 'wind speed north [km/h]')
windspeed_east_fps = Property('atmosphere/wind-east-fps', 'wind speed east [ft/s]')
windspeed_east_mps = HelperProperty('atmosphere/wind-east-mps', 'wind speed east [m/s]')
windspeed_east_kph = HelperProperty('atmosphere/wind-east-kph', 'wind speed east [km/h]')
windspeed_down_fps = Property('atmosphere/wind-down-fps', 'wind speed down [ft/s]')
windspeed_down_mps = HelperProperty('atmosphere/wind-down-mps', 'wind speed down [m/s]')
windspeed_down_kph = HelperProperty('atmosphere/wind-down-kph', 'wind speed down [km/h]')

# turbulences
turb_north_fps = Property('atmosphere/turb-north-fps', 'turbulence wind speed north [ft/s]')
turb_north_mps = HelperProperty('atmosphere/turb-north-mps', 'turbulence wind speed north [m/s]')
turb_north_kph = HelperProperty('atmosphere/turb-north-kph', 'turbulence wind speed north [km/h]')
turb_east_fps = Property('atmosphere/turb-east-fps', 'turbulence wind speed east [ft/s]')
turb_east_mps = HelperProperty('atmosphere/turb-east-mps', 'turbulence wind speed east [m/s]')
turb_east_kph = HelperProperty('atmosphere/turb-east-kph', 'turbulence wind speed east [km/h]')
turb_down_fps = Property('atmosphere/turb-down-fps', 'turbulence wind speed down [ft/s]')
turb_down_mps = HelperProperty('atmosphere/turb-down-mps', 'turbulence wind speed down [m/s]')
turb_down_kph = HelperProperty('atmosphere/turb-down-kph', 'turbulence wind speed down [km/h]')
turb_type = Property('atmosphere/turb-type', 'turbulence type')
turb_severity = Property('atmosphere/turbulence/milspec/severity', 'turbulence severity')
turb_w20_fps = Property('atmosphere/turbulence/milspec/windspeed_at_20ft_AGL-fps', 'turbulence wind speed at 20ft AGL [ft/s]')

# gusts
gust_startup_duration_sec = Property('atmosphere/cosine-gust/startup-duration-sec', 'time it takes for the gust to reach its max value [s]')
gust_steady_duration_sec = Property('atmosphere/cosine-gust/steady-duration-sec', 'duration of the gust at its max value [s]')
gust_end_duration_sec = Property('atmosphere/cosine-gust/end-duration-sec', 'time it takes for the gust go back to 0 [s]')
gust_mag_fps =  Property('atmosphere/cosine-gust/magnitude-ft_sec', 'magnitude of the gust [ft/s]')
gust_frame = Property('atmosphere/cosine-gust/frame', 'frame where the gust is applied, 0: None, 1: Body, 2: Wind, 3: Local NED')
gust_dir_x_fps  = Property('atmosphere/cosine-gust/X-velocity-ft_sec', 'X component of the gust direction vector [ft/s]')
gust_dir_y_fps  = Property('atmosphere/cosine-gust/Y-velocity-ft_sec', 'Y component of the gust direction vector [ft/s]')
gust_dir_z_fps  = Property('atmosphere/cosine-gust/Z-velocity-ft_sec', 'Z component of the gust direction vector [ft/s]')
gust_start = Property('atmosphere/cosine-gust/start', 'set to 1 to start the gust')

# controls state
aileron_left = BoundedProperty('fcs/left-aileron-pos-norm', 'left aileron position, normalised', -1, 1)
aileron_right = BoundedProperty('fcs/right-aileron-pos-norm', 'right aileron position, normalised', -1, 1)
elevator = BoundedProperty('fcs/elevator-pos-norm', 'elevator position, normalised', -1, 1)
rudder = BoundedProperty('fcs/rudder-pos-norm', 'rudder position, normalised', -1, 1)
throttle_pos = BoundedProperty('fcs/throttle-pos-norm', 'throttle position, normalised', 0, 1)
gear = BoundedProperty('gear/gear-pos-norm', 'landing gear position, normalised', 0, 1)

aileron_left_rad = Property('fcs/left-aileron-pos-rad', 'left aileron deflection [rad]')
aileron_right_rad = Property('fcs/right-aileron-pos-rad', 'right aileron deflection [rad]')
aileron_combined_pos_rad = BoundedProperty('fcs/effective-aileron-pos', 'combined effective aileron deflection [rad]', -0.5235, 0.5235)
elevator_pos_rad = BoundedProperty('fcs/elevator-pos-rad', 'elevator deflection [rad]', -0.5235, 0.5235)
aileron_combined_pos_norm = BoundedProperty('fcs/effective-aileron-pos-norm', 'combined effective aileron deflection, normalised', -1, 1)
elevator_pos_norm = BoundedProperty('fcs/elevator-pos-norm', 'elevator deflection, normalised', -1, 1)
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
ic_altitude_ft = Property('ic/h-sl-ft', 'initial altitude MSL [ft]')
ic_terrain_altitude_ft = Property('ic/terrain-elevation-ft', 'initial terrain alt [ft]')
ic_lat_gc_deg = Property('ic/lat-gc-deg', 'initial geocentric latitude [deg]')
ic_long_gc_deg = Property('ic/long-gc-deg', 'initial geocentric longitude [deg]')
ic_lat_gd_deg = Property('ic/lat-geod-deg', 'initial geodetic latitude [deg]')
ic_alt_gd_ft = Property('ic/geod-alt-ft', 'initial geodetic altitude [ft]')
ic_alt_gd_m = HelperProperty('ic/geod-alt-m', 'initial geodetic altitude [m]')
ic_roll_rad = Property('ic/phi-rad', 'initial roll angle [rad]')
ic_pitch_rad = Property('ic/theta-rad', 'initial pitch angle [rad]')
ic_u_fps = Property('ic/u-fps', 'body frame x-axis velocity; positive forward [ft/s]')
ic_v_fps = Property('ic/v-fps', 'body frame y-axis velocity; positive right [ft/s]')
ic_w_fps = Property('ic/w-fps', 'body frame z-axis velocity; positive down [ft/s]')
ic_p_radps = Property('ic/p-rad_sec', 'roll rate [rad/s]')
ic_q_radps = Property('ic/q-rad_sec', 'pitch rate [rad/s]')
ic_r_radps = Property('ic/r-rad_sec', 'yaw rate [rad/s]')
ic_roc_fpm = Property('ic/roc-fpm', 'initial rate of climb [ft/min]')
ic_heading_deg = Property('ic/psi-true-deg', 'initial (true) heading [deg]')
ic_airspeed_kts = Property('ic/vt-kts', 'initial airspeeed [kts]')

# metrics
qbar_area = Property('aero/qbar-area', 'dynamic pressure * wing-planform area')
Sw = Property('metrics/Sw-sqft', 'wing area [sqft]')
rho = Property('atmosphere/rho-slugs_ft3', 'air density [slug/ft^3]')

# mass related dynamic properties
fdm_ixx = Property('fdm/inertia-matrix/ixx', 'inertia matrix element Ixx [KG*M^2]')
fdm_iyy = Property('fdm/inertia-matrix/iyy', 'inertia matrix element Iyy [KG*M^2]')
fdm_izz = Property('fdm/inertia-matrix/izz', 'inertia matrix element Izz [KG*M^2]')
fdm_ixz = Property('fdm/inertia-matrix/ixz', 'inertia matrix element Ixz [KG*M^2]')
fdm_mass_kg = Property('fdm/emptywt', 'empty mass [KG]')

# dynamic aero stability coefficients
aero_CLo = Property('aero/stab-coef/CLo', 'coef: alpha independent lift')
aero_CLalpha = Property('aero/stab-coef/CLalpha', 'coef: lift per alpha rad')
aero_CLq = Property('aero/stab-coef/CLq', 'coef: lift due to pitch rate')
aero_CLDe = Property('aero/stab-coef/CLDe', 'coef: lift due to elevator deflection')

aero_CDo = Property('aero/stab-coef/CDo', 'coef: drag at zero alpha')
aero_CDalpha = Property('aero/stab-coef/CDalpha', 'coef: drag due to alpha')
aero_CDalpha2 = Property('aero/stab-coef/CDalpha2', 'coef: drag due to alpha squared')
aero_CDbeta = Property('aero/stab-coef/CDbeta', 'coef: drag due to sideslip')
aero_CDbeta2 = Property('aero/stab-coef/CDbeta2', 'coef: drag due to sideslip squared')
aero_CDq = Property('aero/stab-coef/CDq', 'coef: drag due to pitch rate')
aero_CDe = Property('aero/stab-coef/CDe', 'coef: drag due to elevator deflection')

aero_CYb = Property('aero/stab-coef/CYb', 'coef: sideforce due to sideslip')
aero_CYp = Property('aero/stab-coef/CYp', 'coef: sideforce due to roll rate')
aero_CYr = Property('aero/stab-coef/CYr', 'coef: sideforce due to yaw rate')
aero_CYda = Property('aero/stab-coef/CYda', 'coef: sideforce due to aileron deflection')

aero_Clb = Property('aero/stab-coef/Clb', 'coef: roll moment due to sideslip')
aero_Clp = Property('aero/stab-coef/Clp', 'coef: roll moment due to roll rate')
aero_Clr = Property('aero/stab-coef/Clr', 'coef: roll moment due to yaw rate')
aero_Clda = Property('aero/stab-coef/Clda', 'coef: roll moment due to aileron deflection')

aero_Cmo = Property('aero/stab-coef/Cmo', 'coef: pitch moment at zero alpha')
aero_Cmalpha = Property('aero/stab-coef/Cmalpha', 'coef: pitch moment due to alpha')
aero_Cmq = Property('aero/stab-coef/Cmq', 'coef: pitch moment due to pitch rate')
aero_CmDe = Property('aero/stab-coef/CmDe', 'coef: pitch moment due to elevator deflection')

aero_Cnb = Property('aero/stab-coef/Cnb', 'coef: yaw moment due to sideslip')
aero_Cnp = Property('aero/stab-coef/Cnp', 'coef: yaw moment due to roll rate')
aero_Cnr = Property('aero/stab-coef/Cnr', 'coef: yaw moment due to yaw rate')
aero_Cnda = Property('aero/stab-coef/Cnda', 'coef: yaw moment due to aileron deflection')

# error values
# additional custom properties for error and target values
airspeed_err = BoundedProperty("error/airspeed-err", "airspeed error", float('-inf'), float('+inf'))
roll_err = BoundedProperty("error/roll-err", "roll error", -2*math.pi, 2*math.pi)
pitch_err = BoundedProperty("error/pitch-err", "pitch error", -2*math.pi, 2*math.pi)
roll_integ_err = BoundedProperty("error/roll-integ-err", "roll integral error", float('-inf'), float('+inf'))
pitch_integ_err = BoundedProperty("error/pitch-integ-err", "pitch integral error", float('-inf'), float('+inf'))

# waypoint tracking - lat, lon, alt
lat_err = BoundedProperty("error/lat-err", "latitude error", -180, 180)
lon_err = BoundedProperty("error/lon-err", "longitude error", -180, 180)
alt_err = BoundedProperty("error/alt-err", "altitude error", float('-inf'), float('+inf'))

# waypoint tracking - x, y, z in ECEF
ecef_x_err_m = BoundedHelperProperty("error/ecef-x-err-m", "ECEF x error [m]", float('-inf'), float('+inf'))
ecef_y_err_m = BoundedHelperProperty("error/ecef-y-err-m", "ECEF y error [m]", float('-inf'), float('+inf'))
ecef_z_err_m = BoundedHelperProperty("error/ecef-z-err-m", "ECEF z error [m]", float('-inf'), float('+inf'))
ecef_x_err_km = BoundedHelperProperty("error/ecef-x-err-km", "ECEF x error [km]", float('-inf'), float('+inf'))
ecef_y_err_km = BoundedHelperProperty("error/ecef-y-err-km", "ECEF y error [km]", float('-inf'), float('+inf'))
ecef_z_err_km = BoundedHelperProperty("error/ecef-z-err-km", "ECEF z error [km]", float('-inf'), float('+inf'))

enu_x_err_m = BoundedHelperProperty("error/enu-x-err-m", "ENU x error [m]", float('-inf'), float('+inf'))
enu_y_err_m = BoundedHelperProperty("error/enu-y-err-m", "ENU y error [m]", float('-inf'), float('+inf'))
enu_z_err_m = BoundedHelperProperty("error/enu-z-err-m", "ENU z error [m]", float('-inf'), float('+inf'))

# target values
# target_airspeed_kts = BoundedProperty("target/airspeed-kts", "desired airspeed [knots]", float('-inf'), float('+inf'))
target_airspeed_mps = BoundedProperty("target/airspeed-mps", "desired airspeed [m/s]", float('-inf'), float('+inf'))
target_airspeed_kph = BoundedProperty("target/airspeed-kph", "desired airspeed [km/h]", float('-inf'), float('+inf'))
target_roll_rad = BoundedProperty("target/roll-rad", "desired roll angle [rad]", -math.pi, math.pi)
target_pitch_rad = BoundedProperty("target/pitch-rad", "desired pitch angle [rad]", -math.pi, math.pi)

# waypoint tracking - lat, lon, alt
target_lat_deg = BoundedProperty("target/lat-deg", "desired latitude [deg]", -90, 90)
target_lon_deg = BoundedProperty("target/lon-deg", "desired longitude [deg]", -90, 90)
target_alt_m = BoundedProperty("target/altitude-m", "desired altitude [m]", 0, float('+inf'))

# waypoint tracking - x, y, z in ECEF
target_ecef_x_m = BoundedProperty("target/ecef-x-m", "desired ECEF x [m]", float('-inf'), float('+inf'))
target_ecef_y_m = BoundedProperty("target/ecef-y-m", "desired ECEF y [m]", float('-inf'), float('+inf'))
target_ecef_z_m = BoundedProperty("target/ecef-z-m", "desired ECEF z [m]", float('-inf'), float('+inf'))
target_ecef_x_km = BoundedProperty("target/ecef-x-km", "desired ECEF x [km]", float('-inf'), float('+inf'))
target_ecef_y_km = BoundedProperty("target/ecef-y-km", "desired ECEF y [km]", float('-inf'), float('+inf'))
target_ecef_z_km = BoundedProperty("target/ecef-z-km", "desired ECEF z [km]", float('-inf'), float('+inf'))

# action avg over last n steps
aileron_avg = BoundedProperty("fcs/aileron_avg", "aileron action average", aileron_cmd.min, aileron_cmd.max)
elevator_avg = BoundedProperty("fcs/elevator_avg", "elevator action average", elevator_cmd.min, elevator_cmd.max)
throttle_avg = BoundedProperty("fcs/throttle_avg", "throttle action average", throttle_cmd.min, throttle_cmd.max)

# reward properties
# attitude control
reward_total = BoundedProperty("reward/total", "total reward", float('-inf'), 0)
reward_roll = BoundedProperty("reward/roll", "roll reward", 0, float('+inf'))
reward_pitch = BoundedProperty("reward/pitch", "pitch reward", 0, float('+inf'))
reward_airspeed = BoundedProperty("reward/airspeed", "airspeed reward", 0, float('+inf'))
reward_actvar = BoundedProperty("reward/act_var", "action variation reward", 0, float('+inf'))
reward_actvar_raw = BoundedProperty("reward/act_var_raw", "raw action variation reward", 0, float('+inf'))
reward_act_bounds = BoundedProperty("reward/act_bounds", "action bound reward", 0, float('+inf'))
reward_int_roll = BoundedProperty("reward/int_roll", "roll integral reward", 0, float('+inf'))
reward_int_pitch = BoundedProperty("reward/int_pitch", "pitch integral reward", 0, float('+inf'))

# waypoint tracking
reward_latitude = BoundedProperty("reward/latitude", "latitude reward", 0, float('+inf'))
reward_ecef_x = BoundedProperty("reward/ecef-x", "ECEF x reward", 0, float('+inf'))
reward_longitude = BoundedProperty("reward/longitude", "longitude reward", 0, float('+inf'))
reward_ecef_y = BoundedProperty("reward/ecef-y", "ECEF y reward", 0, float('+inf'))
reward_altitude = BoundedProperty("reward/altitude", "altitude reward", 0, float('+inf'))
reward_ecef_z = BoundedProperty("reward/ecef-z", "ECEF z reward", 0, float('+inf'))


# PID-RL properties
kp_roll = BoundedProperty("pidrl/roll/kp", "roll kp", float('-inf'), float('+inf'))
ki_roll = BoundedProperty("pidrl/roll/ki", "roll ki", float('-inf'), float('+inf'))
kd_roll = BoundedProperty("pidrl/roll/kd", "roll kd", float('-inf'), float('+inf'))

kp_pitch = BoundedProperty("pidrl/pitch/kp", "pitch kp", float('-inf'), float('+inf'))
ki_pitch = BoundedProperty("pidrl/pitch/ki", "pitch ki", float('-inf'), float('+inf'))
kd_pitch = BoundedProperty("pidrl/pitch/kd", "pitch kd", float('-inf'), float('+inf'))

kp_roll_act = BoundedProperty("pidrl/roll/kp_piact", "additive roll kp RL action", float('-inf'), float('+inf'))
ki_roll_act = BoundedProperty("pidrl/roll/ki_piact", "additive roll ki RL action", float('-inf'), float('+inf'))
kd_roll_act = BoundedProperty("pidrl/roll/kd_piact", "additive roll kd RL action", float('-inf'), float('+inf'))

kp_pitch_act = BoundedProperty("pidrl/pitch/kp_piact", "additive pitch kp RL action", float('-inf'), float('+inf'))
ki_pitch_act = BoundedProperty("pidrl/pitch/ki_piact", "additive pitch ki RL action", float('-inf'), float('+inf'))
kd_pitch_act = BoundedProperty("pidrl/pitch/kd_piact", "additive pitch kd RL action", float('-inf'), float('+inf'))

kp_roll_dt = BoundedProperty("pidrl/roll/kp_dt", "roll kp temporal increment", -0.01, 0.01)
ki_roll_dt = BoundedProperty("pidrl/roll/ki_dt", "roll ki temporal increment", -0.01, 0.01)
kd_roll_dt = BoundedProperty("pidrl/roll/kd_dt", "roll kd temporal increment", -0.01, 0.01)

kp_pitch_dt = BoundedProperty("pidrl/pitch/kp_dt", "pitch kp temporal increment", -0.01, 0.01)
ki_pitch_dt = BoundedProperty("pidrl/pitch/ki_dt", "pitch ki temporal increment", -0.01, 0.01)
kd_pitch_dt = BoundedProperty("pidrl/pitch/kd_dt", "pitch kd temporal increment", -0.01, 0.01)

# PID-RL properties stability ensured parametrized version
roll_tau_1 = BoundedProperty("pidrl/roll/tau_1", "roll tau 1", float(0), float('+inf'))
roll_tau_2 = BoundedProperty("pidrl/roll/tau_2", "roll tau 2", float(0), float('+inf'))
roll_tau_3 = BoundedProperty("pidrl/roll/tau_3", "roll tau 3", float(0), float('+inf'))

pitch_tau_1 = BoundedProperty("pidrl/pitch/tau_1", "pitch tau 1", float(0), float('+inf'))
pitch_tau_2 = BoundedProperty("pidrl/pitch/tau_2", "pitch tau 2", float(0), float('+inf'))
pitch_tau_3 = BoundedProperty("pidrl/pitch/tau_3", "pitch tau 3", float(0), float('+inf'))
