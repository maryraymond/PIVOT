import math
import numpy as np
from numpy.typing import NDArray


def ned_from_gps(lat_deg, lon_deg, alt_m, lat0_deg=0, lon0_deg=0, alt0_m=0.0, use_absolute_altitude=True):
    
    R_EARTH=6378137.0 # meters (WGS84)
    # NED -> North east down
    # Convert degrees to radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x_north = R_EARTH * dlat
    
    y_east  = R_EARTH * math.cos(lat0) * dlon

    if not use_absolute_altitude:
        # DJI "world" is NED: z is Down
        # RelativeAltitude is typically "Up from takeoff", so convert to Down:
        z_down = -(alt_m)
    else: 
        # Abslout is from sea level and increases up where Ned z+ve is down so
        # we need to multiply by -ve
        z_down = -(alt_m - alt0_m)

    return x_north, y_east, z_down

def ecef_from_gps(lat_deg, long_deg, alt_m):

    a = 6378137.0
    e2 = 6.69437999014e-3 

    lat = math.radians(lat_deg)
    long = math.radians(long_deg)

    N = a / math.sqrt(1 - (e2 * math.sin(lat)**2))

    X = (N + alt_m) * math.cos(lat) * math.cos(long)
    Y = (N + alt_m) * math.cos(lat) * math.sin(long)
    Z = (N * (1 - e2) + alt_m) * math.sin(lat)

    return X, Y, Z

def ned_from_ecef(point, X0, Y0, Z0, lat0_deg, long0_deg):

    lat0 = math.radians(lat0_deg)
    long0 = math.radians(long0_deg)
    
    R_ecef2ned = np.array([[-math.sin(lat0)*math.cos(long0),  -math.sin(lat0)*math.sin(long0),   math.cos(lat0)],
                           [-math.sin(long0),                  math.cos(long0),                  0                ],
                           [-math.cos(lat0)* math.cos(long0),  -math.cos(lat0)*math.sin(long0),   -math.sin(lat0) ]])
    
    points_ref = np.array(point)
    points_ref -= np.array([X0, Y0, Z0])

    ned_point = points_ref @ R_ecef2ned.T

    return ned_point[0], ned_point[1], ned_point[2]

def convert_ned_cam_to_opengl(drone_pose:NDArray) -> NDArray:

    # assume camera coordinate system is openGL (so camera facing z -ve)
    c2gimble = np.array([[0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]])
    
    pose = drone_pose.copy()
    pose[:3, :3] = pose[:3, :3] @  c2gimble

    return pose
