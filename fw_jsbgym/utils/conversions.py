import numpy as np
from typing import Tuple
from enum import Enum
from fw_jsbgym.utils.jsbsim_properties import BoundedProperty
from fw_jsbgym.utils import jsbsim_properties as prp


class ConvFactor(Enum):
    kts2mps = 0.51444
    kts2kph = 1.852
    fps2mps = 0.3048
    fps2kph = 1.09728
    kph2mps = 0.277778
    ft2m = 0.3048

prp_conv_fps2mps: Tuple[BoundedProperty, ...] = (
    prp.windspeed_north_mps, prp.windspeed_east_mps, prp.windspeed_down_mps,
    prp.total_windspeed_north_mps, prp.total_windspeed_east_mps, prp.total_windspeed_down_mps,
    prp.turb_north_mps, prp.turb_east_mps, prp.turb_down_mps
)

prp_conv_fps2kph: Tuple[BoundedProperty, ...] = (
    prp.u_kph, prp.v_kph, prp.w_kph,
    prp.windspeed_north_kph, prp.windspeed_east_kph, prp.windspeed_down_kph,
    prp.total_windspeed_north_kph, prp.total_windspeed_east_kph, prp.total_windspeed_down_kph,
    prp.turb_north_kph, prp.turb_east_kph, prp.turb_down_kph
)

prp_conv_ft2m: Tuple[BoundedProperty, ...] = (
    prp.ecef_x_m, prp.ecef_y_m, prp.ecef_z_m,
    prp.ic_alt_gd_m
)


def props2si(sim) -> None:
    """
        Converts some properties from imperial to international metric system
    """
    for prop in prp_conv_fps2mps:
        sim[prop] = sim[prop.name[:-3]+'fps'] * ConvFactor.fps2mps.value
    for prop in prp_conv_fps2kph:
        sim[prop] = sim[prop.name[:-3]+'fps'] * ConvFactor.fps2kph.value
    for prop in prp_conv_ft2m:
        sim[prop] = sim[prop.name[:-1]+'ft'] * ConvFactor.ft2m.value

    sim[prp.airspeed_mps] = sim[prp.airspeed_kts] * ConvFactor.kts2mps.value
    sim[prp.airspeed_kph] = sim[prp.airspeed_kts] * ConvFactor.kts2kph.value


def geodetic2ecef(lat, lon, alt):
    # WGS-84 ellipsoid parameters
    a = 6378137.0  # Semi-major axis (meters)
    e2 = 6.69437999014e-3  # Square of first eccentricity
    
    lat, lon = np.radians(lat), np.radians(lon)
    
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    
    return np.array([x, y, z])


# convert enu to ecef
def enu2ecef(x, y, z, ref_lat, ref_lon, ref_alt):
    # Convert reference geodetic coordinates to ECEF
    ref_ecef = geodetic2ecef(ref_lat, ref_lon, ref_alt)
    
    # Compute transformation matrix
    lat, lon = np.radians(ref_lat), np.radians(ref_lon)
    
    R = np.array([[-np.sin(lon),  np.cos(lon),  0],
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)]])
    
    # Convert ENU to ECEF
    enu_vec = np.array([x, y, z])
    ecef = ref_ecef + R.T @ enu_vec
    
    return ecef


def ecef2enu(x, y, z, ref_lat, ref_lon, ref_alt):
    # Convert reference geodetic coordinates to ECEF
    ref_ecef = geodetic2ecef(ref_lat, ref_lon, ref_alt)
    
    # Compute transformation matrix
    lat, lon = np.radians(ref_lat), np.radians(ref_lon)
    
    R = np.array([[-np.sin(lon),  np.cos(lon),  0],
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [ np.cos(lat) * np.cos(lon),  np.cos(lat) * np.sin(lon), np.sin(lat)]])
    
    # Convert ECEF to ENU
    ecef_vec = np.array([x, y, z]) - ref_ecef
    enu = R @ ecef_vec
    
    return enu


def ecef2ned(x, y, z, ref_lat, ref_lon, ref_alt):
    # Convert reference geodetic coordinates to ECEF
    ref_ecef = geodetic2ecef(ref_lat, ref_lon, ref_alt)
    
    # Compute transformation matrix
    lat, lon = np.radians(ref_lat), np.radians(ref_lon)

    R = np.array([
                  [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
                  [-np.sin(lon)              ,  np.cos(lon)              ,  0],
                  [-np.cos(lat) * np.cos(lon),  -np.cos(lat) * np.sin(lon), -np.sin(lat)]
                  ])

    # Convert ECEF to NED
    ecef_vec = np.array([x, y, z]) - ref_ecef
    ned = R @ ecef_vec
    
    return ned


def ecef2geodetic(x, y, z, tol=1e-12):
    a = 6378137.0 
    f = 1 / 298.257223563
    e_sq = f * (2 - f)
    lam = np.degrees(np.arctan2(y, x))
    p = np.hypot(x, y)
    phi = np.arctan2(z, p * (1 - e_sq))  # initial guess
    N = 0
    h = 0
    for _ in range(5):
        N = a / np.sqrt(1 - e_sq * np.sin(phi)**2)
        h = p / np.cos(phi) - N
        phi = np.arctan2(z, p * (1 - e_sq * N / (N + h)))
    return np.degrees(phi), lam, h


def enu2geodetic(x, y, z, ref_lat, ref_lon, ref_alt):
    enu = enu2ecef(x, y, z, ref_lat, ref_lon, ref_alt)
    return ecef2geodetic(enu[0], enu[1], enu[2])


def euler2quaternion(roll=None, pitch=None, yaw=None, sim=None):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion representation.
    
    Parameters:
        roll  (float): Rotation around x-axis in radians (-π, π)
        pitch (float): Rotation around y-axis in radians (-π, π)
        yaw   (float): Rotation around z-axis in radians (0, 2π)
        sim   (Simulation): JSBSim simulation object, if not None, the Euler angles will be taken from the simulation.
        Similarly, the quaternion will be stored in the simulation object.

    Returns:
        tuple: (qx, qy, qz, qw) representing the quaternion
    """
    if sim is not None:
        roll = sim[prp.roll_rad]
        pitch = sim[prp.pitch_rad]
        yaw = sim[prp.heading_rad]

    assert roll is not None and pitch is not None and yaw is not None, "Euler angles must be provided"

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    if sim is not None:
        sim[prp.att_qx] = qx
        sim[prp.att_qy] = qy
        sim[prp.att_qz] = qz
        sim[prp.att_qw] = qw

    return np.array([qx, qy, qz, qw])


def wpENU_to_wpCourseAlt(target_enu):
    """Converts waypoint positions (ENU) to course and altitude targets."""
    course_target = np.arctan2(target_enu[0], target_enu[1])
    altitude_target = target_enu[2]
    path_targets = np.array([course_target, altitude_target])
    return path_targets


def angle_rad_wrap_to_pi(angle_rad):
    """
    Wraps an angle in radians to the range [-π, π].
    
    Parameters:
        angle_rad (float): Angle in radians.
        
    Returns:
        float: Wrapped angle in radians.
    """
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def angle_rad_wrap_to_2pi(angle_rad):
    """
    Wraps an angle in radians to the range [0, 2π].
    
    Parameters:
        angle_rad (float): Angle in radians.
        
    Returns:
        float: Wrapped angle in radians.
    """
    return angle_rad % (2 * np.pi)
