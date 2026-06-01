import numpy as np
import math


def get_vertical_fov(H, fy):     
    v_fov = 2 * np.atan(H/(2*fy))
    v_fov_degree =np.rad2deg(v_fov)

    return v_fov_degree

def get_horizontal_fov(W, fx):     
    h_fov = 2 * np.atan(W/(2*fx))
    h_fov_degree =np.rad2deg(h_fov)

    return h_fov_degree

def get_diag_fov(W, H, fx, fy=None, degrees=True):
    """
    Diagonal FOV for a rectilinear / pinhole camera.

    width, height: image size in pixels
    fx, fy: focal lengths in pixels
    """
    if fy is None:
        fy = fx

    diag_norm = math.sqrt((W / fx) ** 2 + (H / fy) ** 2)
    fov = 2.0 * math.atan(0.5 * diag_norm)

    return math.degrees(fov) if degrees else fov


def get_horizontal_fov_fe(W: int, fx: float, degrees: bool = True) -> float:
    """
    Horizontal FOV for OpenCV fisheye/equidistant camera model.

    OpenCV fisheye uses approximately:
        r = f * theta

    So at the image edge:
        theta_edge = (W / 2) / fx

    Horizontal FOV:
        fov = 2 * theta_edge
    """
    fov_rad = W / fx

    if degrees:
        return np.degrees(fov_rad)

    return fov_rad



def get_vertical_fov_fe(H: int, fy: float, degrees: bool = True) -> float:
    """
    Horizontal FOV for OpenCV fisheye/equidistant camera model.

    OpenCV fisheye uses approximately:
        r = f * theta

    So at the image edge:
        theta_edge = (H / 2) / fy

    Horizontal FOV:
        fov = 2 * theta_edge
    """
    fov_rad = H / fy

    if degrees:
        return np.degrees(fov_rad)

    return fov_rad


def get_diag_fov_fe(W, H, fx, fy=None, cx=None, cy=None, degrees=True):
    """
    Approximate diagonal FOV for an OpenCV fisheye / equidistant camera.

    width, height: image size in pixels
    fx, fy: focal lengths in pixels
    cx, cy: principal point in pixels

    This assumes the fisheye projection is approximately equidistant:
        r = f * theta
    """
    if fy is None:
        fy = fx

    if cx is None:
        cx = W / 2.0

    if cy is None:
        cy = H / 2.0

    corners = [
        (0, 0),
        (W - 1, 0),
        (0, H - 1),
        (W - 1, H - 1),
    ]

    theta_max = max(
        math.sqrt(((x - cx) / fx) ** 2 + ((y - cy) / fy) ** 2)
        for x, y in corners
    )

    fov = 2.0 * theta_max

    return math.degrees(fov) if degrees else fov