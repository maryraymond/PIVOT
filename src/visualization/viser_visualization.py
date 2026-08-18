# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import sys
import time
import numpy as np
from numpy.typing import NDArray
import viser
from plyfile import PlyData
from typing import List, Optional, Dict
import json
import cv2
import math
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage
import numpy as np
import plotly.graph_objects as go

from utils.processing_utils import get_poses_from_data, get_traj_frames_data
from utils.geometry_utils import R_to_quat, homo_pose_to_quat
from utils.camera_utils import get_diag_fov_fe, get_vertical_fov_fe, get_horizontal_fov_fe, get_diag_fov, get_horizontal_fov, get_vertical_fov

def get_traj_camera_centers_pairs(scene_traj_data, traj_name, step=5):
    frames = scene_traj_data[traj_name]["frames"]

    camera_centers_measured = []
    camera_centers_colmap = []
    for i in range(0, len(frames), step):
        frame = frames[i]
        if "colmap_pose_c2w" not in frame:
            continue
        else:
            camera_centers_measured.append(np.array(frame["measured_pose_c2w"])[:3, 3])
            camera_centers_colmap.append(np.array(frame["colmap_pose_c2w"])[:3, 3])
    
    return camera_centers_measured, camera_centers_colmap

def compute_effective_frustum_step(num_frames: int, step: int, max_frame_number: int) -> int:
    """Pick the frame stride to use when rendering camera frustums.

    Never decreases below the configured `step`. Only increases it, and only
    when `step` would still produce more than `max_frame_number` frustums.
    """
    step = max(1, step)
    if num_frames <= 0:
        return step

    frames_at_step = math.ceil(num_frames / step)
    if frames_at_step <= max_frame_number:
        return step

    return math.ceil(num_frames / max_frame_number)

def drone_DS_2_viser_pose(c2w):
    openGL_2_openCV_T = np.array([[1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, 0, 0, 1]])
    c2w = c2w @ openGL_2_openCV_T
    xyzw, position = homo_pose_to_quat(c2w)

    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    position = np.array(position)

    return wxyz, position

def make_trajectory_error_bubble_plot(traj_stats: dict):
    """
    traj_stats example:
    {
        "circular_low": {
            "avg_translation_error": 0.42,
            "avg_rotation_error": 1.8,
            "num_frames": 120,
            "color": (255, 80, 80),   # same color as frustums
        },
        ...
    }
    """

    fig = go.Figure()

    max_frames = max(v["number_frames_in_traj"] for v in traj_stats.values())

    for traj_name, s in traj_stats.items():
        r, g, b = s["color_value"]
        color = f"rgb({r},{g},{b})"

        # Plotly marker sizes are pixel diameters.
        bubble_size = 10 + 40 * (s["number_frames_in_traj"] / max_frames)
        # frames = s["number_frames_in_traj"]
        # bubble_size = 2 + 40 * (
        #         np.log1p(frames) / np.log1p(max_frames)
        # )

        # bubble_size = 15 + 20 * np.log1p(frames)

        fig.add_trace(
            go.Scatter(
                x=[s["average_cam_center_error_distance"]],
                y=[s["average_rot_error"]],
                mode="markers",
                name=traj_name,  # legend entry,
                marker=dict(
                    size=bubble_size,
                    color=color,
                    opacity=0.55,
                    line=dict(width=1, color="black"),
                ),
                hovertemplate=(
                    f"<b>{traj_name}</b><br>"
                    f"Frames: {s['number_frames_in_traj']}<br>"
                    f"Missing frames: {s['missing_colmap_frames']}<br>"
                    "Avg translation error: %{x:.3f} m<br>"
                    "Avg rotation error: %{y:.3f} deg<br>"
                    "<extra></extra>"
                    
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Average translation error",
        yaxis_title="Average rotation error",
        legend_title="Trajectory",
        showlegend=False,
        margin=dict(l=1, r=10, t=20, b=1),
        legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="center",
                x=0.5,
            ),
        width=900,
        height=500,
    )

    return fig

def make_trajectory_error_bar_plot(traj_stats: dict):
    """
    Creates a dual-axis bar chart:
      - x-axis: trajectory name
      - left y-axis: average translation error
      - right y-axis: average rotation error
      - hover: frame count + missing COLMAP frames

    Expected traj_stats format:
    {
        "traj_1": {
            "avg_translation_error": 4.2,
            "avg_rotation_error": 18.5,
            "number_frames_in_traj": 120,
            "missing_colmap_frames": 3,
            "color": (255, 0, 0),
        },
        ...
    }
    """

    traj_names = list(traj_stats.keys())

    translation_errors = [
        traj_stats[name]["average_cam_center_error_distance"] for name in traj_names
    ]

    rotation_errors = [
        traj_stats[name]["average_rot_error"] for name in traj_names
    ]

    frame_counts = [
        traj_stats[name]["number_frames_in_traj"] for name in traj_names
    ]

    missing_frames = [
        traj_stats[name]["missing_colmap_frames"] for name in traj_names
    ]

    colors = [
        f"rgb{tuple(traj_stats[name]['color_value'])}" for name in traj_names
    ]

    customdata = list(zip(frame_counts, missing_frames))

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=traj_names,
            y=translation_errors,
            name="Avg translation error",
            marker=dict(color=colors),
            opacity=0.75,
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg translation error: %{y:.3f} m<br>"
                "Frames: %{customdata[0]}<br>"
                "Missing frames: %{customdata[1]}"
                "<extra></extra>"
            ),
            yaxis="y",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=traj_names,
            y=rotation_errors,
            name="Avg rotation error",
            mode="lines+markers",
            marker=dict(
                size=9,
                color="black",
            ),
            line=dict(width=2),
            customdata=customdata,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg rotation error: %{y:.3f} deg<br>"
                "Frames: %{customdata[0]}<br>"
                "Missing frames: %{customdata[1]}"
                "<extra></extra>"
            ),
            yaxis="y2",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Trajectory",
            tickangle=-45,
        ),
        yaxis=dict(
            title="Average translation error (m)",
            side="left",
        ),
        yaxis2=dict(
            title="Average rotation error (deg)",
            overlaying="y",
            side="right",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=1, r=5, t=20, b=1),
        height=450,
        barmode="group",
    )

    return fig

def make_trajectory_distance_heatmap(
    traj_matrix: dict,
    title: str = "Trajectory Distance Matrix",
    colorscale: str = "Blues",
    show_xy_lables = False
):
    """
    Plot a trajectory-vs-trajectory heatmap.

    Parameters
    ----------
    traj_matrix : dict
        Nested dictionary:
        {
            traj_a: {
                traj_b: distance,
                ...
            },
            ...
        }

    title : str
        Plot title.

    colorscale : str
        Any Plotly colorscale
        e.g. Blues, Viridis, Plasma, Reds.
    """

    traj_names = list(traj_matrix.keys())

    z = np.array([
        [traj_matrix[row][col] for col in traj_names]
        for row in traj_names
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=traj_names,
            y=traj_names,
            colorscale=colorscale,
            # colorbar_title="Distance",
            colorbar=dict(
                title="",
                thickness=8,
                len=0.7,
            ),
            text=np.round(z, 3),
            texttemplate="%{text}",
            hovertemplate=(
                "<b>%{y}</b> → <b>%{x}</b><br>"
                "Distance: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=1, r=1, t=1, b=1),
        showlegend=False,
        width=max(800, len(traj_names) * 60),
        height=max(800, len(traj_names) * 60),
    )
    

    if not show_xy_lables:
        fig.update_xaxes(
        showticklabels=False,
        title=""
        )

        fig.update_yaxes(
            showticklabels=False,
            title=""
        )

    

    return fig

class ViserVisualization():
    def __init__(self, port):
        self.server = viser.ViserServer(port=port)

    def get_server(self):
        return self.server
    
    def add_pointcloud(self, pointcloud_file, scene_object_name, 
                       point_size=0.002, point_shape='circle',
                       color=[0, 0, 0]):
        ply_data = PlyData.read(pointcloud_file)
        vertices = ply_data["vertex"]
        points = np.vstack([vertices["x"],
                            vertices["y"],
                            vertices["z"]]).T.astype(np.float32)
        
        if "red" in vertices and "green" in vertices and "blue" in vertices:
            colors = np.vstack([[vertices["red"]],
                                vertices["green"],
                                vertices["blue"]]).T.astype(np.float32)
            colors /= 255
        else:
            # create a constant color with the given color value
            colors = np.full_like(points, color)
    
        pointcloud_handle = self.server.scene.add_point_cloud(
            name=scene_object_name,
            points=points,
            colors=colors,
            point_size=point_size,
            point_shape=point_shape
        )

        return pointcloud_handle
    
    def add_camera_frustums(self, poses_c2w:List[NDArray],
                            images:Optional[List[NDArray]]=None,
                            image_paths:Optional[List[Optional[str]]]=None,
                            names:Optional[List[str]]=None, H:int=600, W:int=900,
                            v_fov_degree=70, scale=0.15, line_width=1.0, color=[0, 0, 0],
                            variant='wireframe', visible=True, image_downsample=5):
        if images is None and image_paths is None:
            images = [None] * len(poses_c2w)
        if names is None:
            names = [f"cam_{i:05}" for i in range(len(poses_c2w))]

        v_fov = np.deg2rad(v_fov_degree)

        frustums = []

        for i, (c2w, name) in enumerate(zip(poses_c2w, names)):

            wxyz, position = drone_DS_2_viser_pose(c2w.copy())

            # Load images one at a time to avoid holding all frames in RAM simultaneously.
            if image_paths is not None:
                path = image_paths[i]
                if path is not None:
                    raw = cv2.imread(path)
                    if raw is not None:
                        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                        image = raw[::image_downsample, ::image_downsample]
                    else:
                        image = None
                else:
                    image = None
            else:
                image = images[i]
                if image is not None:
                    image = image[::image_downsample, ::image_downsample]

            frustum = self.server.scene.add_camera_frustum(
                            name=name,
                            fov=v_fov,
                            aspect=W/H,
                            scale=scale,
                            wxyz=wxyz,
                            position=position,
                            image=image,
                            line_width=line_width,
                            color=color,
                            variant=variant,
                            visible=visible
                        )

            frustums.append(frustum)

        return frustums

# (scene_data.json key, display label, format spec) for the scene-level stats block.
SCENE_STATS_FIELDS = [
    ("total_frames_number", "Total frames", "{:,}"),
    ("colmap_reg_frames_number", "COLMAP registered frames", "{:,}"),
    ("pointcloud_number", "Point cloud points", "{:,}"),
    ("observations", "COLMAP observations", "{:,}"),
    ("scene_diameter", "Scene diameter", "{:.2f} m"),
    ("aabb_diagonal", "AABB diagonal", "{:.2f} m"),
    ("max_rotation_angle", "Max rotation angle", "{:.2f}°"),
    ("mean_track_length", "Mean track length", "{:.2f}"),
    ("colmap_per_image_observation", "COLMAP obs. per image", "{:.1f}"),
    ("mean_observations_per_image", "Mean obs. per image", "{:.1f}"),
    ("mean_reprojection_error_px", "Mean reprojection error", "{:.3f} px"),
]

class SceneVisualization():
    def __init__(self, dataset_root, scene_name, port, trajectories=None):
        self.dataset_root = dataset_root
        self.scene_name = scene_name
        self.scene_dir = f"{self.dataset_root}/{self.scene_name}"
        self.viser_visualization = ViserVisualization(port)
        # set the correct coordinate system (NED)
        self.viser_visualization.get_server().scene.set_up_direction('-z')

        # set the scene paths
        scene_data_json = f"{self.scene_dir}/scene_data.json"
        self.scene_point_cloud_file = f"{self.scene_dir}/sparse_model.ply"
        # we will load the scene data
        with open(scene_data_json, "r") as f:
            self.scene_data = json.load(f)
        
        # create the visualization dict
        self.scene_vis_dict = {"trajectories":{}}
        if trajectories is None:
            self.trajectories = sorted(self.scene_data["trajectories"].keys())
        else:
            self.trajectories = trajectories
            
        for traj_name in self.trajectories:
             self.scene_vis_dict["trajectories"][traj_name] = {"color":None,
                                                               "frustum_handlers":{}}
             
        # Populated by visualize_scene() with the scene-wide frustum step decision.
        self.frustum_subsampling_report = None

        #Initialization for Frustum selection
        self.selected_frustum = None
        self.selected_frustum_original_color = None
        self.selected_highlight_color = (224/255, 245/255, 39/255)
        self.default_selected_info = """
        ### Selected Camera

        Click a camera frustum to inspect it.
        """
    
    def apply_theme(self):

        buttons = (
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/maryraymond/drone_3d_dataset",
            ),
            TitlebarButton(
                text="Documentation",
                icon="Description",
                href="https://github.com/maryraymond/drone_3d_dataset",
            ),
        )

        image = TitlebarImage(
                image_url_light="http://localhost:8000/dataset_logo.png",
                image_url_dark="http://localhost:8000/dataset_logo.png",
                image_alt="Drone Dataset logo Logo",
                href="https://github.com/maryraymond/drone_3d_dataset",
            )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)

        self.viser_visualization.get_server().gui.configure_theme(
            titlebar_content=titlebar_theme,
            control_layout="floating",
            control_width="large",
        )

    def visualize_point_cloud(self):
        pointcloud_handler = self.viser_visualization.add_pointcloud(self.scene_point_cloud_file, 
                                                                     scene_object_name="/pointcloud")
        
        # populate the scene dict
        self.scene_vis_dict["pointcloud"] = {"handler":pointcloud_handler}

    def add_point_cloud_gui(self):

        server = self.viser_visualization.get_server()

        gui_show_box = server.gui.add_checkbox(
            "Point cloud",
            initial_value=True
        )

        gui_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.1,
            step=0.001,
            initial_value=0.002
        )

        @gui_show_box.on_update
        def _(_event):
            self.scene_vis_dict["pointcloud"]["handler"].visible = gui_show_box.value

        @gui_size_slider.on_update
        def _(_event):
            self.scene_vis_dict["pointcloud"]["handler"].point_size = gui_size_slider.value

        
        # add the GUI handlers
        self.scene_vis_dict["pointcloud"]["gui_box_handler"] = gui_show_box
        self.scene_vis_dict["pointcloud"]["gui_slider_handler"] = gui_size_slider

    def visualize_world_coordinate(self, axes_raduis=0.03, axes_length=0.7):
        world_coord_handle = self.viser_visualization.get_server().scene.add_frame(
            "/world_coordinates",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            axes_radius=axes_raduis,
            axes_length=axes_length
        )

        self.scene_vis_dict["world_coord"] = {"handler":world_coord_handle}
    
    def add_world_coordinate_gui(self):

        gui_show_w_coord = self.viser_visualization.get_server().gui.add_checkbox(
            "World coordinate",
            initial_value=True
        )

        @gui_show_w_coord.on_update
        def _(_event):
            self.scene_vis_dict["world_coord"]["handler"].visible = gui_show_w_coord.value

        # add the GUI handlers
        self.scene_vis_dict["world_coord"]["gui_box_handler"] = gui_show_w_coord
    
    def add_scene_summary_bubble_plot(self):
        fig = make_trajectory_error_bubble_plot(self.scene_data["trajectories"])

        server = self.viser_visualization.get_server()

        scene_summary_folder = server.gui.add_folder("Scene trajectories Summary")
        with scene_summary_folder:
            bubble_plot = server.gui.add_plotly(
                fig,
                aspect=1,
                config={"displayModeBar": True,
                        "scrollZoom": True,},

            )

        self.scene_vis_dict["scene_summary_bubble_plot"] = bubble_plot
        self.scene_vis_dict["scene_summary_folder"] = scene_summary_folder

    def add_scene_stats_summary(self):
        rows = []
        for field, label, fmt in SCENE_STATS_FIELDS:
            value = self.scene_data.get(field)
            value_str = fmt.format(value) if value is not None else "N/A"
            rows.append(f"| {label} | {value_str} |")

        content = "#### Scene statistics\n| Metric | Value |\n|---|---|\n" + "\n".join(rows)

        server = self.viser_visualization.get_server()
        with self.scene_vis_dict["scene_summary_folder"]:
            stats_markdown = server.gui.add_markdown(content)

        self.scene_vis_dict["scene_stats_markdown"] = stats_markdown

    def add_scene_heatmap_plot(self):

        trajectories = self.scene_data["trajectories"]
        traj_names = sorted(trajectories.keys())

        traj_matrix = {
            traj_a: {
                traj_b: trajectories[traj_a]["trajectory_metrics"][traj_b]["directed_norm_chamfer_tr_distance"]
                for traj_b in traj_names
            }
            for traj_a in traj_names
        }

        fig = make_trajectory_distance_heatmap(
            traj_matrix,
            title="",
            colorscale="Viridis",
        )

        server = self.viser_visualization.get_server()

        with server.gui.add_folder("Scene Trajectories Chamfer Distance", expand_by_default=False):
            heatmap_plot = server.gui.add_plotly(
                fig,
                aspect=1,
                config={"displayModeBar": True,
                        "scrollZoom": True},
            )

        self.scene_vis_dict["scene_heatmap_plot"] = heatmap_plot
    

    def add_scene_summary_double_bar_plot(self):
        fig = make_trajectory_error_bar_plot(self.scene_data["trajectories"])

        server = self.viser_visualization.get_server()

        with server.gui.add_folder("Scene trajectories Summary"):
            double_bar_plot = server.gui.add_plotly(
                fig,
                aspect=1,
                config={"displayModeBar": True,
                        "scrollZoom": True,},
                
            )

        self.scene_vis_dict["scene_summary_double_bar_plot"] = double_bar_plot


    def visualize_traj_camera_centers(self, traj_name, line_width=1.0, visible=True, color=[255, 0, 0], step=1):

        measured, optimized = get_traj_camera_centers_pairs(self.scene_data["trajectories"], traj_name, step=step)

        if len(measured) > 0 and len(optimized) > 0:
            optimized_centers = np.asarray(optimized, dtype=np.float32)
            measured_centers = np.asarray(measured, dtype=np.float32)
            
            error_segments = np.stack(
                                        [optimized_centers, measured_centers],
                                        axis=1
                                    )
            
            color = [color_comp/255 for color_comp in color]

            error_handle = self.viser_visualization.get_server().scene.add_line_segments(
                                                                                    name=f"/{traj_name}/error_lines",
                                                                                    points=error_segments,
                                                                                    colors=color,
                                                                                    line_width=line_width,
                                                                                    visible=visible
                                                                                )
            self.scene_vis_dict["trajectories"][traj_name]["error"] = error_handle

        
    def visualize_traj_camera_frustums(self, traj_name, pose_type="colmap_pose_c2w",
                                       variant='filled', intrinsic_type="camera_intrinsic_colmap",
                                       scale=0.15, line_width=2.0, visible=True, image_downsample=5,
                                       step=1):
        # read the trajectory color
        traj_color = self.scene_data["trajectories"][traj_name]["color_value"]
        traj_color = [color_comp /255 for color_comp in traj_color]

        # load the frames data
        frames_data = get_traj_frames_data(self.scene_data["trajectories"],
                                           trajectory_name=traj_name,
                                           cam_intrinsics_type=intrinsic_type,
                                           c2w_pose_type=pose_type)

        num_frames_available = len(frames_data)
        # `step` is the scene-wide value visualize_scene() already picked so that the
        # total frustum count across all shown trajectories stays under the configured budget.
        frames_data = frames_data[::max(1, step)]
        frames_full_data = [f for f in self.scene_data["trajectories"][traj_name]["frames"]
                            if pose_type in f][::max(1, step)]

        # get the camera params
        if len(frames_data) > 0:
            self.scene_vis_dict["trajectories"][traj_name].setdefault("frustum_frame_counts", {})[pose_type] = {
                "num_frames_available": num_frames_available,
                "num_frames_shown": len(frames_data),
            }
            self.scene_vis_dict["trajectories"][traj_name].setdefault("frustum_frames", {})[pose_type] = frames_full_data


            sample_camera = frames_data[0]["intrinsics"]
            H = int(sample_camera["h"])
            W = int(sample_camera["w"])
            fy = float(sample_camera["fl_y"])
            v_fov_degree = get_vertical_fov(H=H, fy=fy)
    
            poses_c2w = get_poses_from_data(frames_data)
            image_paths = [f"{self.scene_dir}/{frame['file_name']}" for frame in frames_data]
    
            if pose_type == "colmap_pose_c2w":
                pose_source = "COLMAP"
            elif pose_type == "measured_pose_c2w":
                pose_source = "Measured"
            else:
                pose_source = "Unknown"
    
            frame_names = [f"/{traj_name}/{pose_source}/{frame['file_name'].split('/')[-1]}" for frame in frames_data]
    
            frustums_handlers = self.viser_visualization.add_camera_frustums(poses_c2w=poses_c2w,
                                                                            image_paths=image_paths,
                                                                            names=frame_names, H=H, W=W,
                                                                            v_fov_degree=v_fov_degree,
                                                                            scale=scale, line_width=line_width,
                                                                            color=traj_color, variant=variant,
                                                                            visible=visible, image_downsample=image_downsample)
            
            # populate the visualization dict
            self.scene_vis_dict["trajectories"][traj_name]["color"] = traj_color
            self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"][pose_type] = frustums_handlers
    
        else:
            print(f"No frames found in traj {traj_name} for pose type {pose_type} "
                  "will continue.")
            
            # raise ValueError(f"No frames found in traj {traj_name}")
        
        
        
    
    def add_trajs_folder(self):
        trajs_folder = self.viser_visualization.get_server().gui.add_folder("Trajectories")
        self.scene_vis_dict["trajs_folder_handler"] = trajs_folder

    def add_trajs_gui(self, visible=True):

        server = self.viser_visualization.get_server()

        with self.scene_vis_dict["trajs_folder_handler"]:

                gui_hide_all_button = server.gui.add_button("Hide all")

                gui_trajs_optim_show = server.gui.add_checkbox(
                    f"All optimized",
                    initial_value=visible
                )
                gui_trajs_meas_show = server.gui.add_checkbox(
                    f"All measured",
                    initial_value=visible
                )

                gui_trajs_error_show = server.gui.add_checkbox(
                    f"All error",
                    initial_value=visible
                )


        @gui_hide_all_button.on_click
        def _(_event):

            trajectories = self.scene_vis_dict["trajectories"].keys()
            for traj_name in trajectories:
                traj_info = self.scene_vis_dict["trajectories"][traj_name]

                for pose_type in ("colmap_pose_c2w", "measured_pose_c2w"):
                    if pose_type in traj_info["frustum_handlers"]:
                        for frustum in traj_info["frustum_handlers"][pose_type]:
                            frustum.visible = False

                if "error" in traj_info:
                    traj_info["error"].visible = False

                # Clear the per-trajectory checkboxes so their state matches reality.
                if "gui_optim_show_checkbox" in traj_info:
                    traj_info["gui_optim_show_checkbox"].value = False
                if "gui_meas_show_checkbox" in traj_info:
                    traj_info["gui_meas_show_checkbox"].value = False
                if "gui_error_show_checkbox" in traj_info:
                    traj_info["gui_error_show_checkbox"].value = False

            # Clear the "All ..." checkboxes too.
            gui_trajs_optim_show.value = False
            gui_trajs_meas_show.value = False
            gui_trajs_error_show.value = False

        @gui_trajs_optim_show.on_update
        def _(_event):

            trajectories = self.scene_vis_dict["trajectories"].keys()
            for traj_name in trajectories:
                if "colmap_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                    frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["colmap_pose_c2w"]
                    for frustum in frustums:
                        frustum.visible = gui_trajs_optim_show.value

        @gui_trajs_meas_show.on_update
        def _(_event):

            trajectories = self.scene_vis_dict["trajectories"].keys()
            for traj_name in trajectories:
                if "measured_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                    frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["measured_pose_c2w"]
                    for frustum in frustums:
                        frustum.visible = gui_trajs_meas_show.value
        
        @gui_trajs_error_show.on_update
        def _(_event):

            trajectories = self.scene_vis_dict["trajectories"].keys()
            for traj_name in trajectories:
                if "error" in self.scene_vis_dict["trajectories"][traj_name]:
                    lines_handler = self.scene_vis_dict["trajectories"][traj_name]["error"]
                    lines_handler.visible = gui_trajs_error_show.value


    def add_frustums_gui(self, traj_name, visible=True):

        server = self.viser_visualization.get_server()

        with self.scene_vis_dict["trajs_folder_handler"]:
            traj_folder = server.gui.add_folder(traj_name, expand_by_default=False)

            with traj_folder:
                gui_frustum_optim_show = server.gui.add_checkbox(
                    f"Optimized",
                    initial_value=visible
                )
                gui_frustum_meas_show = server.gui.add_checkbox(
                    f"Measured",
                    initial_value=visible
                )

                gui_frustum_error_show = server.gui.add_checkbox(
                    f"Error",
                    initial_value=visible
                )

                self.scene_vis_dict["trajectories"][traj_name]["gui_optim_show_checkbox"] = gui_frustum_optim_show
                self.scene_vis_dict["trajectories"][traj_name]["gui_meas_show_checkbox"] = gui_frustum_meas_show
                self.scene_vis_dict["trajectories"][traj_name]["gui_error_show_checkbox"] = gui_frustum_error_show

                color = self.scene_vis_dict["trajectories"][traj_name]["color"]
                color = [color_comp * 255 for color_comp in color]

                gui_optimized_color = server.gui.add_rgb(f"Optimized color", initial_value=color)
                self.scene_vis_dict["trajectories"][traj_name]["colmap_pose_c2w"] = gui_optimized_color


                gui_measured_color = server.gui.add_rgb(f"Measured color", initial_value=color)
                self.scene_vis_dict["trajectories"][traj_name]["measured_color_handler"] = gui_measured_color

                gui_reset_optimized_color = server.gui.add_button(label="Reset optimized color", color='blue')

                gui_reset_measured_color = server.gui.add_button(label="Reset measured color", color='blue')
                

        @gui_frustum_optim_show.on_update
        def _(_event, traj_name=traj_name):
            if "colmap_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["colmap_pose_c2w"]
                for frustum in frustums:
                    frustum.visible = gui_frustum_optim_show.value
            

        @gui_frustum_meas_show.on_update
        def _(_event, traj_name=traj_name):
            if "measured_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["measured_pose_c2w"]
                for frustum in frustums:
                    frustum.visible = gui_frustum_meas_show.value
        
        @gui_frustum_error_show.on_update
        def _(_event, traj_name=traj_name):
            lines_handler = self.scene_vis_dict["trajectories"][traj_name]["error"]
            lines_handler.visible = gui_frustum_error_show.value

        @gui_optimized_color.on_update
        def _(_event, traj_name=traj_name):
            if "colmap_pose_c2w" in  self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["colmap_pose_c2w"]
                for frustum in frustums:
                    frustum.color = gui_optimized_color.value
        

        @gui_measured_color.on_update
        def _(_event, traj_name=traj_name):
            if "measured_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["measured_pose_c2w"]
                for frustum in frustums:
                    frustum.color = gui_measured_color.value


        @gui_reset_optimized_color.on_click
        def _(_event, traj=traj_name):
            color_float = self.scene_vis_dict["trajectories"][traj_name]["color"]
            color = [color_comp * 255 for color_comp in color_float]

            # Reset the color picker
            gui_optimized_color.value = color

            if "colmap_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["colmap_pose_c2w"]
                for frustum in frustums:
                    frustum.color = color

        @gui_reset_measured_color.on_click
        def _(_event, traj=traj_name):
            color_float = self.scene_vis_dict["trajectories"][traj_name]["color"]
            color = [color_comp * 255 for color_comp in color_float]

            # Reset the color picker
            gui_measured_color.value = color

            if "measured_pose_c2w" in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
                frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]["measured_pose_c2w"]
                for frustum in frustums:
                    frustum.color = color

        self.scene_vis_dict["trajectories"][traj_name]["traj_folder"] = traj_folder

    def add_traj_details(self, traj_name):

        server = self.viser_visualization.get_server()
        traj_folder = self.scene_vis_dict["trajectories"][traj_name]["traj_folder"] 

        source_type = self.scene_data["trajectories"][traj_name]["capture_mode"]
        n_frames = self.scene_data["trajectories"][traj_name]["number_frames_in_traj"]
        n_missing_frames = self.scene_data["trajectories"][traj_name]["missing_colmap_frames"]

        average_distance_error = self.scene_data["trajectories"][traj_name]["average_cam_center_error_distance"] 
        average_distance_error = average_distance_error if average_distance_error is not None else "NA"

        average_distance_error_x = self.scene_data["trajectories"][traj_name]["average_cam_center_error_x"] 
        average_distance_error_x = average_distance_error_x if average_distance_error_x is not None else "NA"

        average_distance_error_y = self.scene_data["trajectories"][traj_name]["average_cam_center_error_y"] 
        average_distance_error_y = average_distance_error_y if average_distance_error_y is not None else "NA"

        average_distance_error_z = self.scene_data["trajectories"][traj_name]["average_cam_center_error_z"] 
        average_distance_error_z = average_distance_error_z if average_distance_error_z is not None else "NA"

        average_rotation_error = self.scene_data["trajectories"][traj_name]["average_rot_error"] 
        average_rotation_error = average_rotation_error if average_rotation_error is not None else "NA"

        average_rotation_error_yaw = self.scene_data["trajectories"][traj_name]["average_rot_error_yaw"] 
        average_rotation_error_yaw = average_rotation_error_yaw if average_rotation_error_yaw is not None else "NA"

        average_rotation_error_pitch = self.scene_data["trajectories"][traj_name]["average_rot_error_pitch"] 
        average_rotation_error_pitch = average_rotation_error_pitch if average_rotation_error_pitch is not None else "NA"
        
        average_rotation_error_roll = self.scene_data["trajectories"][traj_name]["average_rot_error_roll"] 
        average_rotation_error_roll = average_rotation_error_roll if average_rotation_error_roll is not None else "NA"
        
        traj_camera = self.scene_data["trajectories"][traj_name]["camera_intrinsic_colmap"]
        if traj_camera is None:
            traj_camera = self.scene_data["trajectories"][traj_name]["camera_intrinsic_calibration"]
            print(f"No COLMAP optimized camera is available for trajectory {traj_name} "
                  f"will use the calibrated camera intrinsic instead for the trajectory details")
        
        H = int(traj_camera["h"])
        W = int(traj_camera["w"])
        fy = float(traj_camera["fl_y"])
        fx = float(traj_camera["fl_x"])
        v_fov_degree = get_vertical_fov(H=H, fy=fy)
        h_fov_degree = get_horizontal_fov(W=W, fx=fx) 

        camera_model = traj_camera["camera_type"]

        def _fmt_m(v): return f"{v:.2f} m" if isinstance(v, float) else v
        def _fmt_deg(v): return f"{v:.2f}°" if isinstance(v, float) else v

        with traj_folder:
            markdown = server.gui.add_markdown(
                content= f"""
            #### Details
            - **Source type**: {source_type}
            - **Number of frames**: {n_frames}
            - **Number of missing frames**: {n_missing_frames}
            - **Frame resolution**: {W} x {H}
            - **Horizontal FOV**: {h_fov_degree:.2f}°
            - **Vertical FOV**: {v_fov_degree:.2f}°
            - **Camera model**: {camera_model}

            #### Statistics
            ##### Translation Error
            - **Avrg. distance**: {_fmt_m(average_distance_error)}
            - **Avrg. X component**: {_fmt_m(average_distance_error_x)}
            - **Avrg. Y component**: {_fmt_m(average_distance_error_y)}
            - **Avrg. Z component**: {_fmt_m(average_distance_error_z)}
            ##### Rotation Error
            - **Avrg. full rotation**: {_fmt_deg(average_rotation_error)}
            - **Avrg. yaw component**: {_fmt_deg(average_rotation_error_yaw)}
            - **Avrg. pitch component**: {_fmt_deg(average_rotation_error_pitch)}
            - **Avrg. roll component**: {_fmt_deg(average_rotation_error_roll)}

            """
            )
    
    def add_camera_info_markdown(self):
        selected_info = self.viser_visualization.get_server().gui.add_markdown(
            self.default_selected_info
        )

        self.scene_vis_dict["selected_info"] = selected_info

    def clear_selected_camera(self):
        if self.selected_frustum is not None:
            self.selected_frustum.color = self.selected_frustum_original_color

        self.selected_frustum = None
        self.selected_frustum_original_color = None
        self.scene_vis_dict["selected_info"].content = self.default_selected_info

    def add_scene_clear_click_callback(self):
        server = self.viser_visualization.get_server()

        @server.scene.on_click(modifier="shift")
        def _(event):
            
            self.clear_selected_camera()

    def make_click_callback(
        self,
        frustum_handle,
        traj_name,
        pose_source,
        frame_id,
        image_name,
        translation_error=None,
        translation_error_x=None,
        translation_error_y=None,
        translation_error_z=None,
        rotation_error=None,
        rotation_error_yaw=None,
        rotation_error_pitch=None,
        rotation_error_roll=None,
    ):
        @frustum_handle.on_click
        def _(_event):
            # Deselect the previously selected frustum.
            if self.selected_frustum is not None:
                self.selected_frustum.color = self.selected_frustum_original_color

            self.selected_frustum = frustum_handle
            self.selected_frustum_original_color = frustum_handle.color
            frustum_handle.color = self.selected_highlight_color

            err_t = "Pose missing" if translation_error is None else f"{translation_error:.3f} m"
            err_t_x = "Pose missing" if translation_error_x is None else f"{translation_error_x:.3f} m"
            err_t_y = "Pose missing" if translation_error_y is None else f"{translation_error_y:.3f} m"
            err_t_z = "Pose missing" if translation_error_z is None else f"{translation_error_z:.3f} m"
            err_r = "Pose missing" if rotation_error is None else f"{rotation_error:.2f}°"
            err_r_yaw = "Pose missing" if rotation_error_yaw is None else f"{rotation_error_yaw:.2f}°"
            err_r_pitch = "Pose missing" if rotation_error_pitch is None else f"{rotation_error_pitch:.2f}°"
            err_r_roll = "Pose missing" if rotation_error_roll is None else f"{rotation_error_roll:.2f}°"

            self.scene_vis_dict["selected_info"].content = f"""
    ### Selected Camera
    Note: Press shift and click in an empty space to clear the following information.

    | Field | Value |
    |---|---|
    | Trajectory | `{traj_name}` |
    | Pose source | `{pose_source}` |
    | Frame | `{frame_id}` |
    | Image | `{image_name}` |
    | Distance error | `{err_t}` |
    | Translation error X | `{err_t_x}` |
    | Translation error Y | `{err_t_y}` |
    | Translation error Z | `{err_t_z}` |
    | Rotation error | `{err_r}` |
    | Rotation error yaw | `{err_r_yaw}` |
    | Rotation error pitch | `{err_r_pitch}` |
    | Rotation error roll | `{err_r_roll}` |
    """
            
    def create_traj_click_callbacks(self, traj_name, pose_type="colmap_pose_c2w"):
        if pose_type in self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"]:
            frustums = self.scene_vis_dict["trajectories"][traj_name]["frustum_handlers"][pose_type]
            # Use the same (possibly subsampled) frame list that the frustums were built
            # from, so click callbacks stay aligned with what's actually on screen.
            frames = self.scene_vis_dict["trajectories"][traj_name]["frustum_frames"][pose_type]

            if pose_type == "colmap_pose_c2w":
                pose_source = "COLMAP"
            elif pose_type == "measured_pose_c2w":
                pose_source = "Measured"
            else:
                pose_source = "Unknown"

            for i, (frustum, frame) in enumerate(zip(frustums, frames)):

                trans_error = frame["camera_center_error_distance"] if "camera_center_error_distance" in frame else None
                trans_error_x = frame["camera_center_error_x"] if "camera_center_error_x" in frame else None
                trans_error_y = frame["camera_center_error_y"] if "camera_center_error_y" in frame else None
                trans_error_z = frame["camera_center_error_z"] if "camera_center_error_z" in frame else None

                rot_error = frame["rot_error"] if "rot_error" in frame else None
                rot_error_yaw = frame["rot_error_yaw"] if "rot_error_yaw" in frame else None
                rot_error_pitch = frame["rot_error_pitch"] if "rot_error_pitch" in frame else None
                rot_error_roll = frame["rot_error_roll"] if "rot_error_roll" in frame else None

                image_name = frame["file_name"].split("/")[-1]

                self.make_click_callback(frustum, traj_name=traj_name,
                                        pose_source=pose_source, frame_id=f"{i:05}",
                                        image_name=image_name, translation_error=trans_error,
                                        translation_error_x=trans_error_x, translation_error_y=trans_error_y,
                                        translation_error_z=trans_error_z, rotation_error=rot_error,
                                        rotation_error_yaw=rot_error_yaw, rotation_error_pitch=rot_error_pitch,
                                        rotation_error_roll=rot_error_roll)
    
    def visualize_scene(self, frustums_visible_default=False, max_frame_number=3000, step=1):

        # Total frame count across every trajectory being shown, computed up front from
        # scene metadata (no per-trajectory frame loading needed yet). Each trajectory is
        # rendered as two frustum layers (optimized/colmap + measured), so the actual
        # frustum node count is ~2x this total; max_frame_number is a budget on that
        # combined total. A single common step is derived from it and reused everywhere,
        # so the scene-wide frustum count (not any one trajectory's or layer's) stays
        # under max_frame_number.
        total_frames_available = sum(
            self.scene_data["trajectories"][traj_name]["number_frames_in_traj"]
            for traj_name in self.trajectories
        )
        pose_layer_count = 2  # optimized (colmap) + measured
        total_frustums_estimate = total_frames_available * pose_layer_count
        effective_step = compute_effective_frustum_step(total_frustums_estimate, step, max_frame_number)
        self.frustum_subsampling_report = {
            "configured_step": step,
            "used_step": effective_step,
            "max_frame_number": max_frame_number,
            "total_frames_available": total_frames_available,
            "total_frustums_estimate": total_frustums_estimate,
        }

        self.apply_theme()
        self.add_scene_summary_bubble_plot()
        self.add_scene_stats_summary()
        self.add_scene_heatmap_plot()
        # scene_vis.add_scene_summary_double_bar_plot()
        self.visualize_world_coordinate()
        self.add_world_coordinate_gui()
        self.visualize_point_cloud()
        self.add_point_cloud_gui()
        divider = self.viser_visualization.get_server().gui.add_divider()
        self.add_camera_info_markdown()
        self.add_scene_clear_click_callback()
        divider = self.viser_visualization.get_server().gui.add_divider()
        self.add_trajs_folder()
        self.add_trajs_gui(visible=False)
        # trajectories = sorted(scene_vis.scene_data["trajectories"].keys())
        for traj in self.trajectories:
            self.visualize_traj_camera_frustums(traj_name=traj,
                                                visible=frustums_visible_default,
                                                image_downsample=15, step=effective_step)
            self.visualize_traj_camera_frustums(traj_name=traj,
                                                intrinsic_type="camera_intrinsic_calibration",
                                                pose_type="measured_pose_c2w",
                                                variant='wireframe',
                                                visible=frustums_visible_default, image_downsample=15, line_width=1,
                                                step=effective_step)
            self.visualize_traj_camera_centers(traj_name=traj,
                                               line_width=0.5,
                                               visible=frustums_visible_default,
                                               step=effective_step)
            
            self.add_frustums_gui(traj, visible=False)
            self.add_traj_details(traj)
            self.create_traj_click_callbacks(traj)
            self.create_traj_click_callbacks(traj, pose_type="measured_pose_c2w")

    def get_frustum_subsampling_report(self):
        """Scene-wide frustum step decision, plus per-trajectory/pose-type frame counts.

        Call after visualize_scene(). Intended for the calling application to report
        to the user what step was actually used and why.
        """
        trajectories_report = {
            traj_name: traj_info["frustum_frame_counts"]
            for traj_name, traj_info in self.scene_vis_dict["trajectories"].items()
            if "frustum_frame_counts" in traj_info
        }

        return {
            "scene": self.frustum_subsampling_report,
            "trajectories": trajectories_report,
        }

