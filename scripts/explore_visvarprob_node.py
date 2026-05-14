#!/usr/bin/env python3
"""
MapEx ROS2 exploration node (visvarprob mode).

Subscribes:
  /project_map        (nav_msgs/OccupancyGrid)  -- accumulated occupancy map
  /state_estimation   (nav_msgs/Odometry)        -- robot pose

Publishes:
  /next_pose          (geometry_msgs/PointStamped)    -- next step along A* path
  /mapex/pred_map     (nav_msgs/OccupancyGrid)         -- LaMa predicted map
  /mapex/path         (nav_msgs/Path)                  -- A* path to frontier
  /mapex/frontiers    (visualization_msgs/MarkerArray) -- scored frontier centers
"""

import os
import sys

# ---- path setup (must come before any local imports) ----
_SCRIPTS_DIR = os.path.dirname(os.path.realpath(__file__))
_LAMA_DIR = os.path.join(os.path.dirname(_SCRIPTS_DIR), "lama")
for _p in (_SCRIPTS_DIR, _LAMA_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# ---------------------------------------------------------

import numpy as np  
import cv2
import torch
import pyastar2d
import albumentations as A
from scipy.ndimage import binary_dilation
import scipy

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker

from lama_pred_utils import (
    load_lama_model,
    visualize_prediction,
    get_lama_transform,
    convert_obsimg_to_model_input,
)
import sim_utils



MODE = "visvarprob"


# ── Helper functions ───────────────────────────────────────────────────────────

def get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device):
    cur_obs_img_3chan = np.stack([cur_obs_img, cur_obs_img, cur_obs_img], axis=2)
    input_lama_batch, lama_mask = convert_obsimg_to_model_input(
        cur_obs_img_3chan, lama_map_transform, device
    )
    lama_pred = lama_model(input_lama_batch)
    lama_pred_viz = visualize_prediction(lama_pred, lama_mask)
    return cur_obs_img_3chan, input_lama_batch, lama_mask, lama_pred, lama_pred_viz


def get_pred_maputils_from_viz(viz_map):
    pred_maputils = np.zeros((viz_map.shape[0], viz_map.shape[1]))
    pred_maputils[viz_map[:, :, 0] > 128] = 1  # occ
    return pred_maputils


def get_lama_padding_transform():
    return A.PadIfNeeded(
        min_height=None,
        min_width=None,
        pad_height_divisor=16,
        pad_width_divisor=16,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )


def get_padded_map(m):
    return get_lama_padding_transform()(image=m)["image"]


def inflate_obs_map(obs_map, dilate_diam, use_dt, unknown_as_occ, dt_floor_val=10):
    """Replicate Mapper.inflate_map without requiring a Mapper instance."""
    dilated = obs_map.copy()
    inv_bin = obs_map.copy() > 0.5
    inv_dilated = binary_dilation(inv_bin, structure=np.ones((dilate_diam, dilate_diam)))
    dilated[inv_dilated] = 1
    occ_grid = np.zeros(dilated.shape, dtype=np.float32)
    if use_dt:
        dt = scipy.ndimage.distance_transform_cdt(dilated == 0) * -1
        bdt = np.clip(dt + dt_floor_val, 1, dt_floor_val)
        if unknown_as_occ:
            occ_grid[dilated == 0] = bdt[dilated == 0]
            occ_grid[dilated > 0] = np.inf
        else:
            occ_grid[dilated >= 0] = bdt[dilated >= 0]
            occ_grid[dilated == 1] = np.inf
    else:
        if unknown_as_occ:
            occ_grid[dilated == 0] = 1
            occ_grid[dilated > 0] = np.inf
        else:
            occ_grid[dilated >= 0] = 1
            occ_grid[dilated == 1] = np.inf
    return occ_grid


def is_locked_frontier_center_valid(
    locked_frontier_center, occ_grid_pyastar, cur_pose, dist_threshold_px
):
    if locked_frontier_center is None:
        return False
    if occ_grid_pyastar[locked_frontier_center[0], locked_frontier_center[1]] == np.inf:
        return False
    if np.linalg.norm(locked_frontier_center - cur_pose) < dist_threshold_px:
        return False
    return True


def reselect_frontier(frontier_region_centers, frontier_cost_list):
    """Remove the cheapest frontier; return (ok, new_center, updated_centers, updated_costs)."""
    idx = np.argmin(frontier_cost_list)
    frontier_region_centers = np.delete(frontier_region_centers, idx, axis=0)
    frontier_cost_list = np.delete(frontier_cost_list, idx, axis=0)
    if len(frontier_region_centers) == 0:
        return False, None, None, None
    locked = frontier_region_centers[np.argmin(frontier_cost_list)]
    return True, locked, frontier_region_centers, frontier_cost_list


def snap_to_free(pose, obs_map, max_search_radius=50):
    """Return the nearest free cell (obs_map == 0.0) to pose, searching BFS outward.
    Falls back to pose itself if no free cell is found within max_search_radius."""
    row, col = int(pose[0]), int(pose[1])
    h, w = obs_map.shape
    if 0 <= row < h and 0 <= col < w and obs_map[row, col] == 0.0:
        return pose  # already free
    from collections import deque
    visited = set()
    queue = deque()
    queue.append((row, col))
    visited.add((row, col))
    while queue:
        r, c = queue.popleft()
        if abs(r - row) > max_search_radius and abs(c - col) > max_search_radius:
            break
        if 0 <= r < h and 0 <= c < w and obs_map[r, c] == 0.0:
            return np.array([r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (r + dr, c + dc)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return pose  # fallback


# ── ROS2 Node ──────────────────────────────────────────────────────────────────

class MapExNode(Node):
    def __init__(self):
        super().__init__("mapex_node")
        self._declare_parameters()
        self._load_parameters()
        self._load_models()
        self._create_interfaces()

        self.obs_map = None          # float32 (H, W): 0=free, 0.5=unknown, 1=occ
        self.map_info = None         # nav_msgs/MapMetaData
        self.cur_pose_world = None   # np.array([x, y]) in metres, world frame
        self.cur_pose_pixel = None   # np.array([row, col]) — kept for reference only
        self.pose_list = None
        self.locked_frontier_center = None
        self.frontier_region_centers = None
        self.frontier_cost_list = None
        self.pred_maputils = None   # cached LaMa prediction (lama_out_size × lama_out_size)
        self.pred_map_info = None    # map_info at prediction time (for correct world alignment)
        self.frontier_planner = sim_utils.FrontierPlanner(score_mode=MODE)
        self.get_logger().info("MapExNode initialised (visvarprob mode)")

    # ── Parameters ────────────────────────────────────────────────────────────

    def _declare_parameters(self):
        self.declare_parameter("planning_cycle", 2.0)
        self.declare_parameter("big_lama_model_path", "")
        self.declare_parameter("ensemble_model_dir", "")
        self.declare_parameter("lama_device", "cuda:0")
        self.declare_parameter("lama_transform_variant", "default_map_eval")
        self.declare_parameter("lama_out_size", 512)
        self.declare_parameter("pixel_per_meter", 10.0)
        self.declare_parameter("cur_pose_dist_threshold_m", 1.0)
        self.declare_parameter("unknown_as_occ", True)
        self.declare_parameter("use_distance_transform_for_planning", True)
        self.declare_parameter("dilate_diam_for_planning", 3)
        self.declare_parameter("laser_range_m", 20.0)
        self.declare_parameter("pred_num_laser", 250)
        self.declare_parameter("ind_to_move_per_step", 3)

    def _load_parameters(self):
        self.planning_cycle = self.get_parameter("planning_cycle").value
        self.big_lama_model_path = self.get_parameter("big_lama_model_path").value
        self.ensemble_model_dir = self.get_parameter("ensemble_model_dir").value
        self.device = self.get_parameter("lama_device").value
        self.lama_transform_variant = self.get_parameter("lama_transform_variant").value
        self.lama_out_size = self.get_parameter("lama_out_size").value
        self.pixel_per_meter = self.get_parameter("pixel_per_meter").value
        self.dist_threshold_px = (
            self.get_parameter("cur_pose_dist_threshold_m").value * self.pixel_per_meter
        )
        self.unknown_as_occ = self.get_parameter("unknown_as_occ").value
        self.use_dt = self.get_parameter("use_distance_transform_for_planning").value
        self.dilate_diam = self.get_parameter("dilate_diam_for_planning").value
        self.pred_vis_configs = {
            "laser_range_m": self.get_parameter("laser_range_m").value,
            "pixel_per_meter": self.pixel_per_meter,
            "num_laser": self.get_parameter("pred_num_laser").value,
        }
        self.ind_to_move_per_step = self.get_parameter("ind_to_move_per_step").value

    def _load_models(self):
        self.get_logger().info("Loading LAMA model ...")
        self.lama_model = load_lama_model(self.big_lama_model_path, device=self.device)
        self.lama_map_transform = get_lama_transform(
            self.lama_transform_variant, self.lama_out_size
        )
        self.model_list = []
        if self.ensemble_model_dir:
            for name in sorted(os.listdir(self.ensemble_model_dir)):
                m = load_lama_model(
                    os.path.join(self.ensemble_model_dir, name), device=self.device
                )
                self.model_list.append(m)
                self.get_logger().info(f"Loaded ensemble model: {name}")
        self.get_logger().info(f"Models ready. Ensemble size: {len(self.model_list)}")

    # ── Interfaces ────────────────────────────────────────────────────────────

    def _create_interfaces(self):
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/projected_map", self._map_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/state_estimation", self._odom_callback, 10
        )
        self.waypoint_pub = self.create_publisher(PointStamped, "/next_pose", 10)
        self.pred_map_pub = self.create_publisher(OccupancyGrid, "/mapex/pred_map", 10)
        self.path_pub = self.create_publisher(Path, "/solution_path", 10)
        self.frontier_pub = self.create_publisher(MarkerArray, "/mapex/frontiers", 10)
        self.timer = self.create_timer(self.planning_cycle, self._planning_callback)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _map_callback(self, msg: OccupancyGrid):
        grid = np.asarray(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        obs_map = np.full(grid.shape, 0.5, dtype=np.float32)  # default: unknown
        obs_map[grid == 0] = 0.0    # free
        obs_map[grid == 100] = 1.0  # occupied
        self.obs_map = obs_map
        self.map_info = msg.info

    def _odom_callback(self, msg: Odometry):
        # Store world-frame position only; pixel conversion is deferred to
        # planning callback so it always uses the *current* map_info/origin.
        self.cur_pose_world = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        ])

    # ── Planning ──────────────────────────────────────────────────────────────

    def _planning_callback(self):
        if self.obs_map is None or self.cur_pose_world is None or self.map_info is None:
            return

        # Snapshot map state atomically so obs_map and map_info are consistent.
        obs_map = self.obs_map.copy()
        map_info = self.map_info

        # Convert world pose → pixel using the *current* map origin/resolution.
        wx, wy = self.cur_pose_world
        col = int((wx - map_info.origin.position.x) / map_info.resolution)
        row = int((wy - map_info.origin.position.y) / map_info.resolution)
        cur_pose = np.array([row, col])

        # Snap robot to nearest free cell if it landed in unknown/obstacle
        cur_pose_free = snap_to_free(cur_pose, obs_map)
        if not np.array_equal(cur_pose_free, cur_pose):
            self.get_logger().warn(
                f"Robot pose {cur_pose} is not in a free cell; "
                f"snapped to {cur_pose_free} for planning"
            )
        cur_pose = cur_pose_free

        if self.pose_list is None:
            self.pose_list = np.atleast_2d(cur_pose)
        else:
            self.pose_list = np.concatenate(
                [self.pose_list, np.atleast_2d(cur_pose)], axis=0
            )

        # Frontier detection
        frontier_region_centers_unscored, _, _ = (
            self.frontier_planner.get_frontier_centers_given_obs_map(obs_map)
        )
        if len(frontier_region_centers_unscored) == 0:
            self.get_logger().warn("No frontiers found, skipping planning cycle")
            return

        occ_grid_pyastar = inflate_obs_map(
            obs_map, self.dilate_diam, self.use_dt, self.unknown_as_occ
        )

        # Unlock frontier if reached
        if self.locked_frontier_center is not None:
            if np.linalg.norm(self.locked_frontier_center - cur_pose) < self.dist_threshold_px:
                self.locked_frontier_center = None

        need_new_frontier = not is_locked_frontier_center_valid(
            self.locked_frontier_center, occ_grid_pyastar, cur_pose, self.dist_threshold_px
        )

        if need_new_frontier:
            # Map prediction
            # Resize obs_map to training resolution (lama_out_size × lama_out_size)
            # before LaMa inference so the model always sees maps at the correct scale.
            lama_sz = self.lama_out_size
            obs_h, obs_w = obs_map.shape
            lama_input_map = cv2.resize(
                obs_map.astype(np.float32),
                (lama_sz, lama_sz),
                interpolation=cv2.INTER_NEAREST,
            )

            (
                _,
                input_lama_batch,
                _,
                _,
                lama_pred_alltrain_viz,
            ) = get_lama_pred_from_obs(
                lama_input_map, self.lama_model, self.lama_map_transform, self.device
            )
            pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)
            self.pred_maputils = pred_maputils  # keep lama_sz×lama_sz for publishing
            self.pred_map_info = map_info        # anchor to world coords at prediction time

            # Ensemble predictions for variance
            lama_pred_list = []
            for model_i, model in enumerate(self.model_list):
                self.get_logger().debug(f"Ensemble prediction {model_i}")
                lama_pred = model(input_lama_batch)
                lama_pred_list.append(lama_pred["inpainted"][0][0])

            lama_pred_stack = torch.stack(lama_pred_list)
            var_map = torch.var(lama_pred_stack, dim=0)
            mean_map = np.mean(lama_pred_stack.cpu().numpy(), axis=0)

            # padded_obs_map is at the original obs_map resolution
            padded_obs_map = get_padded_map(obs_map)
            ph, pw = padded_obs_map.shape

            # Resize predictions (lama_sz×lama_sz) back to obs_map resolution
            # so that frontier pixel coordinates are valid.
            def _resize_to(arr, h, w):
                if arr.shape == (h, w):
                    return arr
                return cv2.resize(arr.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)

            pred_maputils_sc = _resize_to(pred_maputils, ph, pw)
            mean_map_sc = _resize_to(mean_map, ph, pw)
            var_map_sc = _resize_to(var_map.cpu().numpy(), ph, pw)
            var_map_sc = torch.from_numpy(var_map_sc)  # keep as tensor (sim_utils uses torch.sum)

            # Frontier scoring
            frontier_region_centers_unscored, _, _ = (
                self.frontier_planner.get_frontier_centers_given_obs_map(obs_map)
            )
            (
                frontier_region_centers,
                frontier_cost_list,
                _,
                _,
                _,
                _,
            ) = self.frontier_planner.score_frontiers(
                frontier_region_centers_unscored,
                cur_pose,
                self.pose_list,
                pred_maputils_sc,
                self.pred_vis_configs,
                obs_map=padded_obs_map,
                mean_map=mean_map_sc,
                var_map=var_map_sc,
            )

            self.frontier_region_centers = frontier_region_centers
            self.frontier_cost_list = frontier_cost_list
            self.locked_frontier_center = frontier_region_centers[
                np.argmin(frontier_cost_list)
            ]

            # Validate and reselect if needed
            while not is_locked_frontier_center_valid(
                self.locked_frontier_center,
                occ_grid_pyastar,
                cur_pose,
                self.dist_threshold_px,
            ):
                (
                    selected,
                    self.locked_frontier_center,
                    self.frontier_region_centers,
                    self.frontier_cost_list,
                ) = reselect_frontier(self.frontier_region_centers, self.frontier_cost_list)
                if not selected:
                    self.get_logger().warn("No valid frontier found after reselection")
                    return

        # A* path planning
        path = pyastar2d.astar_path(
            occ_grid_pyastar,
            tuple(cur_pose),
            tuple(self.locked_frontier_center),
            allow_diagonal=False,
        )
        while path is None:
            (
                selected,
                self.locked_frontier_center,
                self.frontier_region_centers,
                self.frontier_cost_list,
            ) = reselect_frontier(self.frontier_region_centers, self.frontier_cost_list)
            if not selected:
                self.get_logger().warn("A* failed for all frontiers, skipping")
                return
            path = pyastar2d.astar_path(
                occ_grid_pyastar,
                tuple(cur_pose),
                tuple(self.locked_frontier_center),
                allow_diagonal=False,
            )

        now = self.get_clock().now().to_msg()
        res = map_info.resolution
        origin_x = map_info.origin.position.x
        origin_y = map_info.origin.position.y

        # next_pose: next step along A* path (psuedo-trajectory controller)
        ind = min(self.ind_to_move_per_step, len(path) - 1)
        next_row, next_col = int(path[ind, 0]), int(path[ind, 1])
        pt = PointStamped()
        pt.header.frame_id = "map"
        pt.header.stamp = now
        pt.point.x = (float(next_col) + 0.5) * res + origin_x
        pt.point.y = (float(next_row) + 0.5) * res + origin_y
        pt.point.z = 0.0
        self.waypoint_pub.publish(pt)
        self.get_logger().info(
            f"Published waypoint: x={pt.point.x:.2f} y={pt.point.y:.2f}"
        )

        # Publish predicted map using the map_info from prediction time so that
        # the pixel→world transform stays anchored even as the map grows.
        if self.pred_maputils is not None and self.pred_map_info is not None:
            pub_info = self.pred_map_info
            pub_h = pub_info.height
            pub_w = pub_info.width
            pm = self.pred_maputils
            if pm.shape != (pub_h, pub_w):
                # Use INTER_LINEAR to avoid the 1-pixel floor-truncation shift
                # that INTER_NEAREST produces for non-integer scale factors.
                # Threshold at 0.5 afterwards to restore binary occupancy values.
                pm = cv2.resize(
                    pm.astype(np.float32),
                    (pub_w, pub_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                pm = (pm > 0.5).astype(np.float32)
            pred_og = OccupancyGrid()
            pred_og.header.frame_id = "map"
            pred_og.header.stamp = now
            pred_og.info = pub_info
            pred_og.data = (pm * 100).astype(np.int8).flatten().tolist()
            self.pred_map_pub.publish(pred_og)

        # Publish A* path
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = now
        for pixel_row, pixel_col in path:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = (float(pixel_col) + 0.5) * res + origin_x
            ps.pose.position.y = (float(pixel_row) + 0.5) * res + origin_y
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

        # Publish frontier markers
        if self.frontier_region_centers is not None:
            markers = MarkerArray()
            del_m = Marker()
            del_m.header.frame_id = "map"
            del_m.header.stamp = now
            del_m.ns = "frontiers"
            del_m.action = Marker.DELETEALL
            markers.markers.append(del_m)
            for i, fc in enumerate(self.frontier_region_centers):
                m = Marker()
                m.header.frame_id = "map"
                m.header.stamp = now
                m.ns = "frontiers"
                m.id = i + 1
                m.type = Marker.SPHERE
                m.action = Marker.ADD
                m.pose.position.x = (float(fc[1]) + 0.5) * res + origin_x
                m.pose.position.y = (float(fc[0]) + 0.5) * res + origin_y
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.5
                m.color.r = 1.0
                m.color.g = 0.5
                m.color.b = 0.0
                m.color.a = 1.0
                markers.markers.append(m)
            self.frontier_pub.publish(markers)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = MapExNode()
    node.get_logger().info("MapExNode started, spinning ...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
