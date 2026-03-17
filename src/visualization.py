import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict


def make_camera_frustu_lines(width:int, height:int, fx:float, fy:float, 
                              cx:float, cy:float, scale:float=0.2, color=[0.1, 0.7, 1.0]):
    """
    Build a camera frustum as an Open3D LineSet in the CAMERA coordinate frame.
    Camera center is at origin. Image plane is at z = -1 (looking down -Z).
    'scale' controls the size in world units.
    """
    # Image corners at z=-1 in camera coords
    # Pixel -> normalized camera coords: x = (u-cx)/fx, y=(v-cy)/fy, z=-1

    corners_px = np.array([[0, 0],
                           [width-1, 0],
                           [width-1, height-1],
                           [0, height-1]], dtype=np.float64)
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    corners_cam[:, 0] = (corners_px[:, 0] - cx) / fx
    corners_cam[:, 1] = (corners_px[:, 1] - cy) / fy
    corners_cam[:, 2] = -1 # forward for NeRF/OpenCV-ish camera is -Z

    corners_cam *= scale

    orgin = np.zeros((1, 3), dtype=np.float64)
    pts = np.vstack([orgin, corners_cam])

    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    
    colors = np.tile(np.array(color), (lines.shape[0], 1))

    ls = o3d.geometry.LineSet()
    ls.points= o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    return ls

def make_textured_image_plane(
    img: np.ndarray,
    width: int,
    height: int,
    fx: float, fy: float, cx: float, cy: float,
    plane_scale: float = 0.25,
    z: float = -1.0,  # NeRF/OpenCV convention: camera looks along -Z
):
    """
    Create a textured quad at z (in camera coords), sized by intrinsics.
    img: HxWx3 uint8 (RGB) or HxWx4.
    Returns: (mesh, material) to add to Open3D rendering scene (or mesh for legacy).
    """

    # 4 image corners in pixel coords
    corners_px = np.array([
        [0,       0      ],
        [width-1, 0      ],
        [width-1, height-1],
        [0,       height-1],
    ], dtype=np.float64)

    # Pixel -> camera normalized plane (at z)
    # x = (u-cx)/fx * |z|, y = (v-cy)/fy * |z|
    # Note: y axis direction depends on convention; this mapping matches typical image coords.
    corners_cam = np.zeros((4, 3), dtype=np.float64)
    corners_cam[:, 0] = (corners_px[:, 0] - cx) / fx * abs(z)
    corners_cam[:, 1] = (corners_px[:, 1] - cy) / fy * abs(z)
    corners_cam[:, 2] = z

    # Scale to make it visually nice in world units
    corners_cam *= plane_scale

    # Create a quad as two triangles
    # Vertex order: 0:TL, 1:TR, 2:BR, 3:BL (in pixel terms)
    vertices = corners_cam

    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    # UVs for each triangle vertex (Open3D expects per-triangle-vertex UVs)
    # Note: v is usually flipped depending on texture coordinates; adjust if image appears upside down.
    uvs = np.array([
        [0.0, 1.0], [1.0, 1.0], [1.0, 0.0],  # tri 0: 0,1,2
        [0.0, 1.0], [1.0, 0.0], [0.0, 0.0],  # tri 1: 0,2,3
    ], dtype=np.float64)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(uvs)

    # Attach texture (legacy visualization uses mesh.textures)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
        
    if img.shape[2] == 3:
        # assume RGB
        o3d_img = o3d.geometry.Image(img)
    else:
        o3d_img = o3d.geometry.Image(img[:, :, :3])

    mesh.textures = [o3d_img]

    return mesh

def get_cameras_visualization(poses:NDArray, images:Optional[List[NDArray]]=None, image_size:Tuple[int]=(800, 600), intrinsics:Optional[List]=None,
                      fov_degrees:float=60.0, frustum_scale:float=0.2, axis_size:float=0.05, stride:int=1, 
                      draw_trajectory:bool=True, c2w:bool=True, show_world_corrdinate:bool=True, show_cam_coord:bool=True,
                      trajectory_color:List[float] = [0.9, 0.2, 0.2], show_images=False, geoms:Optional[List]=None):
    """
    return geometry of a list of camera poses like Nerfstudio: frustum + coordinate frame per pose.

    poses: list/np.ndarray of shape (N,4,4)
    images: list[np.array]
    image_size: (W,H)
    intrinsics: (fx,fy,cx,cy) in pixels. If None, uses fov_degrees + image_size.
    stride: show every k-th camera to reduce clutter.
    c2w: if False, will invert each pose.
    """
    W, H = image_size

    if intrinsics is None:
        #Approximate value from Horizontal FOV
        fov = np.deg2rad(fov_degrees)
        fx = (W / 2.0) / np.tan(fov / 2.0)
        fy = fx
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
    else:
        fx, fy, cx, cy = intrinsics

    
    if geoms is None:
        geoms = []

    if show_world_corrdinate:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5*axis_size))
    
    centers = []

    for i in range(0, len(poses), stride):
        T = poses[i].copy()

        if not c2w:
            T= np.linalg.inv(T)
        
        if show_cam_coord:
            frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)
            frame_coord.transform(T)
            geoms.append(frame_coord)
        
        if show_images and images is not None:
            img_plan = make_textured_image_plane(images[i], W, H, fx, fy, cx, cy, plane_scale=frustum_scale)
            img_plan.transform(T)
            geoms.append(img_plan)

        frustum = make_camera_frustu_lines(W, H, fx, fy, cx, cy, scale=frustum_scale, color=trajectory_color)
        frustum.transform(T)
        geoms.append(frustum)

        # The ceneter is the translation component
        centers.append(T[:3, 3])

    if draw_trajectory and len(centers) >= 2:
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(centers)
        traj.lines = o3d.utility.Vector2iVector(np.column_stack([np.arange(len(centers)-1), np.arange(1, len(centers))]).astype(np.int32))
        traj.colors = o3d.utility.Vector3dVector(np.tile(np.array([trajectory_color]), (len(centers)-1, 1)))

        geoms.append(traj)

    return geoms

def visualize_cameras(poses:List[NDArray], images:Optional[List[NDArray]]=None, image_size:Tuple[int]=(800, 600), intrinsics:Optional[List]=None,
                      fov_degrees:float=60.0, frustum_scale:float=0.2, axis_size:float=0.05, stride:int=1, 
                      draw_trajectory:bool=True, c2w:bool=True, show_world_corrdinate:bool=True, show_cam_coord:bool=True, show_images=False,
                      trajectory_color:List[float] = [0.9, 0.2, 0.2], geoms:Optional[List]= None):
    """
    Visualize a list of camera poses like Nerfstudio: frustum + coordinate frame per pose.

    poses: list/np.ndarray of shape (N,4,4)
    images: list[np.array]
    image_size: (W,H)
    intrinsics: (fx,fy,cx,cy) in pixels. If None, uses fov_degrees + image_size.
    stride: show every k-th camera to reduce clutter.
    c2w: if False, will invert each pose.
    """

    poses = np.array(poses)
    assert poses.ndim==3 and poses.shape[1:]==(4, 4)

    if geoms  is None:
        geoms = []

    geoms += get_cameras_visualization(poses=poses, images=images, image_size=image_size, intrinsics=intrinsics, fov_degrees=fov_degrees,
                                      frustum_scale=frustum_scale, axis_size=axis_size, stride=stride, draw_trajectory=draw_trajectory,
                                      c2w=c2w, show_world_corrdinate=show_world_corrdinate, show_cam_coord=show_cam_coord, show_images=show_images,
                                      trajectory_color=trajectory_color, geoms=geoms)
    
    o3d.visualization.draw_geometries(geoms)

    return geoms
   

def visualize_points(points):
    """
    Visualize a list of 3D points using Open3D.

    Args:
        points (list of tuple): [(x, y, z), ...]
    """
    if len(points) == 0:
        print("No points to visualize.")
        return

    # Convert to numpy array
    points_np = np.array(points)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Optional: color all points (red)
    pcd.paint_uniform_color([1, 0, 0])

    # Visualize
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    pass