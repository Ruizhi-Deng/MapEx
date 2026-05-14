"""
Simplified exploration script for visvarprob mode only.
"""

import numpy as np
import os
import cv2
import time
from omegaconf import OmegaConf
import hydra
import torch
import pyastar2d
import json
import albumentations as A
import traceback
import argparse
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from lama_pred_utils import (
    load_lama_model,
    visualize_prediction,
    get_lama_transform,
    convert_obsimg_to_model_input,
)
import sys

sys.path.append("../")
import simple_mask_utils as smu
import sim_utils
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from datetime import datetime

MODE = "visvarprob"


def get_options_dict_from_yml(config_name):
    cwd = os.getcwd()
    hydra_config_dir_path = os.path.join(cwd, "../configs")
    print(hydra_config_dir_path)
    with hydra.initialize_config_dir(config_dir=hydra_config_dir_path):
        cfg = hydra.compose(config_name=config_name)
    options_dict = OmegaConf.to_container(cfg)
    options = OmegaConf.create(options_dict)
    return options


def update_mission_status(
    start_time, cur_step, mission_complete, fail_reason, mission_status_save_path
):
    mission_status = {}
    mission_status["start_time"] = start_time
    mission_status["cur_step"] = cur_step
    mission_status["mission_complete"] = mission_complete
    mission_status["fail_reason"] = fail_reason
    mission_status["last_exp_time_s"] = time.time() - mission_status["start_time"]
    with open(mission_status_save_path, "w") as f:
        json.dump(mission_status, f)


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
    lama_padding_transform = A.PadIfNeeded(
        min_height=None,
        min_width=None,
        pad_height_divisor=16,
        pad_width_divisor=16,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    )
    return lama_padding_transform


def get_padded_map(m):
    return get_lama_padding_transform()(image=m)["image"]


def is_locked_frontier_center_valid(
    locked_frontier_center, occ_grid_pyastar, cur_pose, collect_opts, pixel_per_meter
):
    if locked_frontier_center is None:
        return False
    if occ_grid_pyastar[locked_frontier_center[0], locked_frontier_center[1]] == np.inf:
        return False
    if (
        np.linalg.norm(locked_frontier_center - cur_pose)
        < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter
    ):
        return False
    return True


def reselect_frontier(
    frontier_region_centers,
    frontier_cost_list,
    t,
    start_exp_time,
    mission_status_save_path,
):
    frontier_region_centers = np.delete(
        frontier_region_centers, np.argmin(frontier_cost_list), axis=0
    )
    frontier_cost_list = np.delete(frontier_cost_list, np.argmin(frontier_cost_list), axis=0)
    if len(frontier_region_centers) == 0:
        update_mission_status(
            start_time=start_exp_time,
            cur_step=t,
            mission_complete=False,
            fail_reason="frontier_region_centers",
            mission_status_save_path=mission_status_save_path,
        )
        return False, None, None, None
    locked_frontier_center = frontier_region_centers[np.argmin(frontier_cost_list)]
    return True, locked_frontier_center, frontier_region_centers, frontier_cost_list


def run_exploration_comparison_for_map(args):
    map_folder_path = args["map_folder_path"]
    model_list = args["model_list"]
    lama_model = args["lama_model"]
    lama_map_transform = args["lama_map_transform"]
    pred_vis_configs = args["pred_vis_configs"]
    lidar_sim_configs = args["lidar_sim_configs"]
    start_pose = args["start_pose"]
    unknown_as_occ = args["unknown_as_occ"]
    use_distance_transform_for_planning = args["use_distance_transform_for_planning"]

    print("Running exploration for map:", map_folder_path)

    map_occ_npy_path = os.path.join(map_folder_path, "occ_map.npy")
    map_valid_space_npy_path = os.path.join(map_folder_path, "valid_space.npy")
    assert os.path.exists(map_occ_npy_path), "Missing: {}".format(map_occ_npy_path)
    assert os.path.exists(map_valid_space_npy_path), "Missing: {}".format(map_valid_space_npy_path)

    occ_map, validspace_map = sim_utils.get_kth_occ_validspace_map(
        map_occ_npy_path, map_valid_space_npy_path
    )
    map_name = os.path.dirname(map_occ_npy_path)
    folder_name = os.path.basename(map_name)

    if start_pose is None:
        start_pose = smu.sample_free_position_given_buffer(occ_map, validspace_map, buffer=2)
        assert start_pose is not None, "Could not sample start pose"

    exp_title = (
        time.strftime("%Y%m%d_%H%M%S")
        + "_"
        + folder_name
        + "_"
        + str(start_pose[0])
        + "_"
        + str(start_pose[1])
        + "_"
        + MODE
    )
    run_exploration_for_map(
        occ_map=occ_map,
        exp_title=exp_title,
        lama_model=lama_model,
        model_list=model_list,
        lama_map_transform=lama_map_transform,
        pred_vis_configs=pred_vis_configs,
        lidar_sim_configs=lidar_sim_configs,
        start_pose=start_pose,
        unknown_as_occ=unknown_as_occ,
        use_distance_transform_for_planning=use_distance_transform_for_planning,
    )


def run_exploration_for_map(
    occ_map,
    exp_title,
    lama_model,
    model_list,
    lama_map_transform,
    pred_vis_configs,
    lidar_sim_configs,
    start_pose,
    unknown_as_occ,
    use_distance_transform_for_planning,
):
    try:
        print("exp_title:", exp_title)
        start_exp_time = time.time()
        pixel_per_meter = lidar_sim_configs["pixel_per_meter"]

        # Planner setup
        mapper = sim_utils.Mapper(
            occ_map,
            lidar_sim_configs,
            use_distance_transform_for_planning=use_distance_transform_for_planning,
        )
        frontier_planner = sim_utils.FrontierPlanner(score_mode=MODE)

        # Output directories
        exp_dir = os.path.join(output_root_dir, exp_title)
        global_obs_dir = os.path.join(exp_dir, "global_obs")
        run_viz_dir = os.path.join(exp_dir, "run_viz")
        for d in [exp_dir, global_obs_dir, run_viz_dir]:
            os.makedirs(d, exist_ok=True)

        gt_map_save_path = os.path.join(exp_dir, "gt_map.png")
        odom_npy_save_path = os.path.join(exp_dir, "odom.npy")
        mission_status_save_path = os.path.join(exp_dir, "mission_status.json")

        # Visualization setup
        fig, ax = plt.subplots(2, 3, figsize=(20, 10))
        ax_flatten = ax.flatten()
        ax_gt = ax_flatten[0]
        ax_obs = ax_flatten[1]
        ax_pred = ax_flatten[2]
        ax_pred_var = ax_flatten[3]
        ax_mean_map = ax_flatten[5]

        # Pad size for lidar boundary padding
        pd_size = 500
        gt_h, gt_w = mapper.gt_map.shape
        pad_h = gt_h % 16
        pad_w = gt_w % 16
        pad_h1 = 0 if pad_h == 0 else int((16 - pad_h) / 2)
        pad_h2 = 0 if pad_h == 0 else 16 - pad_h - pad_h1
        pad_w1 = 0 if pad_w == 0 else int((16 - pad_w) / 2)
        pad_w2 = 0 if pad_w == 0 else 16 - pad_w - pad_w1

        # Initial state
        mission_failed = False
        update_mission_status(
            start_time=start_exp_time,
            cur_step=0,
            mission_complete=False,
            fail_reason="",
            mission_status_save_path=mission_status_save_path,
        )
        cur_pose = start_pose
        mapper.observe_and_accumulate_given_pose(cur_pose)
        ind_to_move_per_step = 3
        pose_list = np.atleast_2d(cur_pose)
        locked_frontier_center = None

        cv2.imwrite(
            gt_map_save_path,
            smu.convert_01_single_channel_to_0_255_3_channel(mapper.gt_map),
        )
        np.save(odom_npy_save_path, pose_list)

        # ── Main loop ──────────────────────────────────────────────────────────
        for t in range(collect_opts.mission_time):
            start_step_time = time.time()
            show_plt = (t % collect_opts.show_plt_freq == 0) or (t == collect_opts.mission_time - 1)

            # Frontier detection (initialise on step 0)
            if t == 0:
                frontier_region_centers_unscored, filtered_map, num_large_regions = (
                    frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                )

            if len(frontier_region_centers_unscored) == 0:
                mission_failed = True
                update_mission_status(
                    start_time=start_exp_time,
                    cur_step=t,
                    mission_complete=False,
                    fail_reason="frontier_region_no_large_region",
                    mission_status_save_path=mission_status_save_path,
                )
                break

            # Inflated planning map
            occ_grid_pyastar = mapper.get_inflated_planning_maps(unknown_as_occ=unknown_as_occ)

            # Unlock frontier if reached
            if locked_frontier_center is not None:
                if (
                    np.linalg.norm(locked_frontier_center - cur_pose)
                    < collect_opts.cur_pose_dist_threshold_m * pixel_per_meter
                ):
                    locked_frontier_center = None

            need_new_locked_frontier = not is_locked_frontier_center_valid(
                locked_frontier_center,
                occ_grid_pyastar,
                cur_pose,
                collect_opts,
                pixel_per_meter,
            )

            if need_new_locked_frontier:
                show_plt = True

                # ── Map prediction ─────────────────────────────────────────────
                cur_obs_img = mapper.obs_map.copy()
                (
                    _,
                    input_lama_batch,
                    lama_mask,
                    lama_pred_alltrain,
                    lama_pred_alltrain_viz,
                ) = get_lama_pred_from_obs(cur_obs_img, lama_model, lama_map_transform, device)
                pred_maputils = get_pred_maputils_from_viz(lama_pred_alltrain_viz)

                # Ensemble predictions for variance
                lama_pred_list = []
                for model_i, model in enumerate(model_list):
                    print("Predicting with ensemble model:", model_i)
                    pred_start = time.time()
                    lama_pred = model(input_lama_batch)
                    print("Prediction took {:.2f}s".format(time.time() - pred_start))
                    lama_pred_onechan = lama_pred["inpainted"][0][0]
                    lama_pred_list.append(lama_pred_onechan)

                lama_pred_stack = torch.stack(lama_pred_list)
                var_map = torch.var(lama_pred_stack, dim=0)
                mean_map = np.mean(lama_pred_stack.cpu().numpy(), axis=0)

                padded_obs_map = get_padded_map(mapper.obs_map)
                padded_gt_map = get_padded_map(mapper.gt_map)

                # ── Frontier scoring ───────────────────────────────────────────
                frontier_region_centers_unscored, filtered_map, num_large_regions = (
                    frontier_planner.get_frontier_centers_given_obs_map(mapper.obs_map)
                )
                (
                    frontier_region_centers,
                    frontier_cost_list,
                    viz_most_flooded_grid,
                    viz_medium_flooded_grid,
                    best_ind,
                    medium_ind,
                ) = frontier_planner.score_frontiers(
                    frontier_region_centers_unscored,
                    cur_pose,
                    pose_list,
                    pred_maputils,
                    pred_vis_configs,
                    obs_map=padded_obs_map,
                    mean_map=mean_map,
                    var_map=var_map,
                )
                locked_frontier_center = frontier_region_centers[np.argmin(frontier_cost_list)]

                while not is_locked_frontier_center_valid(
                    locked_frontier_center,
                    occ_grid_pyastar,
                    cur_pose,
                    collect_opts,
                    pixel_per_meter,
                ):
                    (
                        selected,
                        locked_frontier_center,
                        frontier_region_centers,
                        frontier_cost_list,
                    ) = reselect_frontier(
                        frontier_region_centers,
                        frontier_cost_list,
                        t,
                        start_exp_time,
                        mission_status_save_path,
                    )
                    if not selected:
                        mission_failed = True
                        break
                if mission_failed:
                    break

            # ── A* local planning ──────────────────────────────────────────────
            path = pyastar2d.astar_path(
                occ_grid_pyastar, cur_pose, locked_frontier_center, allow_diagonal=False
            )
            while path is None:
                (
                    selected,
                    locked_frontier_center,
                    frontier_region_centers,
                    frontier_cost_list,
                ) = reselect_frontier(
                    frontier_region_centers,
                    frontier_cost_list,
                    t,
                    start_exp_time,
                    mission_status_save_path,
                )
                if not selected:
                    mission_failed = True
                    break
                path = pyastar2d.astar_path(
                    occ_grid_pyastar,
                    cur_pose,
                    locked_frontier_center,
                    allow_diagonal=False,
                )
            if mission_failed:
                break

            plan_x = path[:, 0]
            plan_y = path[:, 1]
            next_pose = sim_utils.psuedo_traj_controller(
                plan_x, plan_y, plan_ind_to_use=ind_to_move_per_step
            )

            # ── Visualization ──────────────────────────────────────────────────
            if show_plt:
                for a in ax.flatten():
                    a.clear()

                map_kwargs = {"cmap": "gray", "vmin": 0, "vmax": 1}
                white = "#FFFFFF"

                # GT map
                ax_gt.imshow(1 - mapper.gt_map[pd_size:-pd_size, pd_size:-pd_size], **map_kwargs)
                ax_gt.plot(
                    pose_list[:, 1] - pd_size,
                    pose_list[:, 0] - pd_size,
                    c="r",
                    alpha=0.5,
                )
                ax_gt.scatter(
                    pose_list[-1, 1] - pd_size,
                    pose_list[-1, 0] - pd_size,
                    c="g",
                    s=10,
                    marker="*",
                )
                frontier_colors = (
                    -frontier_cost_list if not np.all(frontier_cost_list == 0) else "r"
                )
                ax_gt.scatter(
                    np.array(frontier_region_centers)[:, 1] - pd_size,
                    np.array(frontier_region_centers)[:, 0] - pd_size,
                    c=frontier_colors,
                    s=10,
                    marker="x",
                    cmap="plasma",
                )
                ax_gt.set_title("GT Map")

                # Observed map
                colors_ = ["#FFFFFF", "#D9D9D9", "#000000"]
                cmap_obs = ListedColormap(colors_)
                ax_obs.imshow(mapper.obs_map[pd_size:-pd_size, pd_size:-pd_size], cmap=cmap_obs)
                ax_obs.plot(
                    pose_list[:, 1] - pd_size,
                    pose_list[:, 0] - pd_size,
                    c="#417CF2",
                    alpha=1.0,
                )
                ax_obs.scatter(
                    locked_frontier_center[1] - pd_size,
                    locked_frontier_center[0] - pd_size,
                    c="#D13EF5",
                    s=10,
                )
                ax_obs.plot(plan_y - pd_size, plan_x - pd_size, c="#417CF2", linestyle=":")
                ax_obs.set_title("Observed Map")

                # Predicted map
                blue = "#0000FF"
                cmap_pred = LinearSegmentedColormap.from_list(
                    "customwhiteblue", [white, blue], N=10
                )
                ax_pred.imshow(
                    pred_maputils[
                        pd_size + pad_h1 : -(pd_size + pad_h2),
                        pd_size + pad_w1 : -(pd_size + pad_w2),
                    ],
                    cmap=cmap_pred,
                )

                obs_occ_mask = np.zeros_like(
                    pred_maputils[
                        pd_size + pad_h1 : -(pd_size + pad_h2),
                        pd_size + pad_w1 : -(pd_size + pad_w2),
                    ]
                )
                occ_inds = np.where(mapper.obs_map[pd_size:-pd_size, pd_size:-pd_size] == 1.0)
                obs_occ_mask[occ_inds] = 1
                occ_alpha = np.where(obs_occ_mask == 1, 1.0, 0.0)
                ax_pred.imshow(
                    obs_occ_mask,
                    cmap=LinearSegmentedColormap.from_list(
                        "mask_black", ["#000000", "#000000"], N=2
                    ),
                    alpha=occ_alpha,
                )

                obs_unk_mask = np.zeros_like(obs_occ_mask)
                unk_inds = np.where(mapper.obs_map[pd_size:-pd_size, pd_size:-pd_size] == 0.5)
                obs_unk_mask[unk_inds] = 1
                unk_alpha = np.where(obs_unk_mask == 1, 0.3, 0.0)
                ax_pred.imshow(
                    obs_unk_mask,
                    cmap=LinearSegmentedColormap.from_list(
                        "mask_grey", ["#909090", "#909090"], N=2
                    ),
                    alpha=unk_alpha,
                )

                ax_pred.plot(
                    pose_list[:, 1] - pd_size,
                    pose_list[:, 0] - pd_size,
                    c="#eb4205",
                    alpha=1.0,
                )
                ax_pred.set_title("Predicted Map")

                # Variance map
                orange = "#FF9F1C"
                var_cmap = LinearSegmentedColormap.from_list("custom_var", [white, orange], N=10)
                ax_pred_var.imshow(
                    var_map[
                        pd_size + pad_w1 : -(pd_size + pad_w2),
                        pd_size + pad_h1 : -(pd_size + pad_h2),
                    ]
                    .cpu()
                    .numpy(),
                    vmin=0,
                    vmax=0.3,
                    cmap=var_cmap,
                )
                ax_pred_var.set_title("Predicted Map Variance")

                # Mean map
                cmap_mean = LinearSegmentedColormap.from_list(
                    "customgreenred", [white, blue], N=100
                )
                ax_mean_map.imshow(mean_map[pd_size:-pd_size, pd_size:-pd_size], cmap=cmap_mean)

                mean_occ_mask = np.zeros_like(mean_map[pd_size:-pd_size, pd_size:-pd_size])
                mean_occ_inds = np.where(
                    mapper.obs_map[
                        pd_size - pad_h1 : -(pd_size - pad_h2),
                        pd_size - pad_w1 : -(pd_size - pad_w2),
                    ]
                    == 1.0
                )
                mean_occ_mask[mean_occ_inds] = 1
                mean_occ_alpha = np.where(mean_occ_mask == 1, 1.0, 0.0)
                ax_mean_map.imshow(
                    mean_occ_mask,
                    cmap=LinearSegmentedColormap.from_list(
                        "mask_black", ["#000000", "#000000"], N=2
                    ),
                    alpha=mean_occ_alpha,
                )

                mean_unk_mask = np.zeros_like(mean_occ_mask)
                mean_unk_inds = np.where(
                    mapper.obs_map[
                        pd_size - pad_h1 : -(pd_size - pad_h2),
                        pd_size - pad_w1 : -(pd_size - pad_w2),
                    ]
                    == 0.5
                )
                mean_unk_mask[mean_unk_inds] = 1
                mean_unk_alpha = np.where(mean_unk_mask == 1, 0.3, 0.0)
                ax_mean_map.imshow(
                    mean_unk_mask,
                    cmap=LinearSegmentedColormap.from_list(
                        "mask_grey", ["#909090", "#909090"], N=2
                    ),
                    alpha=mean_unk_alpha,
                )

                path_color = "#eb4205"
                ax_mean_map.plot(
                    pose_list[:, 1] - (pd_size - pad_w1),
                    pose_list[:, 0] - (pd_size - pad_h1),
                    c=path_color,
                    alpha=1.0,
                )
                ax_mean_map.scatter(
                    locked_frontier_center[1] - (pd_size - pad_w1),
                    locked_frontier_center[0] - (pd_size - pad_h1),
                    c="#f57b3e",
                    s=10,
                )
                ax_mean_map.plot(
                    plan_y - (pd_size - pad_w1),
                    plan_x - (pd_size - pad_h1),
                    c=path_color,
                    linestyle=":",
                )

                if viz_most_flooded_grid is not None:
                    flooded_rgba = np.zeros((*mean_map.shape, 4))
                    flooded_inds = np.where(viz_most_flooded_grid)
                    flooded_rgba[flooded_inds[0], flooded_inds[1], :] = (
                        1.0,
                        159 / 255,
                        28 / 255,
                        0.3,
                    )
                    ax_mean_map.imshow(flooded_rgba[pd_size:-pd_size, pd_size:-pd_size])

                ax_mean_map.set_title("Mean Map of Prediction Ensembles")

                plt.tight_layout()
                print("Saving fig:", t)
                plt.savefig(
                    os.path.join(run_viz_dir, "{}_{}.png".format(exp_title, str(t).zfill(8))),
                    dpi=100,
                )
                show_plt = False

            # Save observations and pose
            cv2.imwrite(
                os.path.join(global_obs_dir, "{}.png".format(str(t).zfill(8))),
                smu.convert_01_single_channel_to_0_255_3_channel(mapper.obs_map),
            )
            np.save(odom_npy_save_path, pose_list)

            # Step: move to next pose
            cur_pose = next_pose
            if mapper.gt_map[cur_pose[0], cur_pose[1]] == 1:
                print("Hit wall!")
                mission_failed = True
                update_mission_status(
                    start_time=start_exp_time,
                    cur_step=t,
                    mission_complete=False,
                    fail_reason="hit_wall",
                    mission_status_save_path=mission_status_save_path,
                )
                break

            pose_list = np.concatenate([pose_list, np.atleast_2d(cur_pose)], axis=0)
            mapper.observe_and_accumulate_given_pose(cur_pose)
            update_mission_status(
                start_time=start_exp_time,
                cur_step=t,
                mission_complete=False,
                fail_reason="",
                mission_status_save_path=mission_status_save_path,
            )
            print("Step {} took {:.2f}s".format(t, time.time() - start_step_time))

        # Final status
        if mission_failed:
            print("\033[91mMission failed for {}!\033[0m".format(exp_title))
        else:
            update_mission_status(
                start_time=start_exp_time,
                cur_step=t,
                mission_complete=True,
                fail_reason="",
                mission_status_save_path=mission_status_save_path,
            )
            print("\033[94mMission complete for {}!\033[0m".format(exp_title))

    except Exception as e:
        print("\033[93mMission failed with exception for {}!\033[0m".format(exp_title))
        print(e)
        print(traceback.format_exc())
        update_mission_status(
            start_time=start_exp_time,
            cur_step=t,
            mission_complete=False,
            fail_reason=str(e),
            mission_status_save_path=mission_status_save_path,
        )


if __name__ == "__main__":
    data_collect_config_name = "base.yaml"
    today = datetime.today()
    output_subdirectory_name = (
        str(today.year) + "{:02d}".format(today.month) + "{:02d}".format(today.day) + "_test"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect_world_list", nargs="+", help="List of worlds to collect data from"
    )
    parser.add_argument("--start_pose", nargs="+", help="Start pose as two integers")
    args = parser.parse_args()

    collect_opts = get_options_dict_from_yml(data_collect_config_name)
    if args.collect_world_list is not None:
        collect_opts.collect_world_list = args.collect_world_list
    if args.start_pose is not None:
        collect_opts.start_pose = [int(p) for p in args.start_pose]

    kth_map_folder_path = os.path.join(collect_opts.root_path, "kth_test_maps")
    kth_map_paths = os.listdir(kth_map_folder_path)

    if collect_opts.collect_world_list is not None:
        kth_map_paths = [p for p in kth_map_paths if p in collect_opts.collect_world_list]

    kth_map_folder_paths = [
        os.path.join(kth_map_folder_path, p) for p in kth_map_paths
    ] * collect_opts.num_data_per_world

    output_root_dir = os.path.join(
        collect_opts.root_path,
        collect_opts.output_folder_name,
        output_subdirectory_name,
    )
    os.makedirs(output_root_dir, exist_ok=True)

    device = collect_opts.lama_device

    # Load ensemble models (G_i)
    model_list = []
    if collect_opts.ensemble_folder_name is not None:
        ensemble_dir = os.path.join(
            collect_opts.root_path,
            "pretrained_models",
            collect_opts.ensemble_folder_name,
        )
        for ensemble_model_dir in sorted(os.listdir(ensemble_dir)):
            model = load_lama_model(os.path.join(ensemble_dir, ensemble_model_dir), device=device)
            print("Loaded ensemble model:", ensemble_model_dir)
            model_list.append(model)

    # Load big lama model (G)
    lama_model = load_lama_model(
        os.path.join(
            collect_opts.root_path,
            "pretrained_models",
            collect_opts.big_lama_model_folder_name,
        ),
        device=device,
    )
    lama_map_transform = get_lama_transform(
        collect_opts.lama_transform_variant, collect_opts.lama_out_size
    )

    for kth_map_folder_path in kth_map_folder_paths:
        run_exploration_comparison_for_map(
            {
                "map_folder_path": kth_map_folder_path,
                "lama_model": lama_model,
                "model_list": model_list,
                "lama_map_transform": lama_map_transform,
                "pred_vis_configs": collect_opts.pred_vis_configs,
                "lidar_sim_configs": collect_opts.lidar_sim_configs,
                "start_pose": collect_opts.start_pose,
                "unknown_as_occ": collect_opts.unknown_as_occ,
                "use_distance_transform_for_planning": collect_opts.use_distance_transform_for_planning,
            }
        )
