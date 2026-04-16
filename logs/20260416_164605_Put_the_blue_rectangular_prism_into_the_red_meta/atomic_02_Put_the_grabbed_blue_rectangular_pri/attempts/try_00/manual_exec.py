#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import sys


def _find_project_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'configs' / 'demo.yaml').exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate project root from {{start!s}}; expected a parent directory containing pyproject.toml and configs/demo.yaml."
    )


def _resolve_project_path(project_root: str | Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return Path(project_root).resolve() / path


PROJECT_ROOT = _find_project_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hpaf.geometry.transforms import load_extrinsic
from hpaf.perception.llm_perception import LLMPerceptionService
from hpaf.perception.classic_cv_perception import ClassicalTabletopPerceptionService
from hpaf.perception.foundation_vision_perception import FoundationVisionPerceptionService
from hpaf.api.runtime_api import RuntimeAPI
from hpaf.robot.piper_backend import PiperArm
from hpaf.llm.factory import make_vision_client
from hpaf.core.io import load_config
from hpaf.core.models import Pose
from hpaf.camera.shared_dir_camera import SharedDirCamera, DualSharedDirCamera

cfg = load_config(str(PROJECT_ROOT / 'configs' / 'demo.yaml'))
vision_client = make_vision_client(cfg.llm)
PRIMARY_DIR = str(_resolve_project_path(PROJECT_ROOT, './shared_scene/primary'))
SECONDARY_DIR = str(_resolve_project_path(PROJECT_ROOT, './shared_scene/secondary'))
EXTRINSIC_PATH = str(_resolve_project_path(PROJECT_ROOT, './assets/eyeinhand.json'))
ARTIFACT_DIR = str(_resolve_project_path(PROJECT_ROOT, 'logs/20260416_164605_Put_the_blue_rectangular_prism_into_the_red_meta/atomic_02_Put_the_grabbed_blue_rectangular_pri/attempts/try_00/artifacts'))
ATOMIC_TASK = 'Put the grabbed blue rectangular prism into the red metal box'
PLAN_BRIEF = 'Detect the red metal box first, estimate the safe placement pose inside the red box, move the robotic arm to the placement pose, open the gripper to release the object, retreat upward, return to the global observation pose, and finally verify whether the task is completed.'

primary_camera = SharedDirCamera(
    shared_dir=PRIMARY_DIR,
    rgb_filename='latest_color.png',
    depth_filename='latest_depth.npy',
    camera_info_filename='latest_camera_info.json',
    freshness_max_age_s=float(getattr(cfg.cameras, 'freshness_max_age_s', 3.0) if cfg.cameras is not None else getattr(cfg.camera, 'freshness_max_age_s', 3.0)),
)
secondary_camera = None
if SECONDARY_DIR and os.path.abspath(SECONDARY_DIR) != os.path.abspath(PRIMARY_DIR):
    secondary_camera = SharedDirCamera(
        shared_dir=SECONDARY_DIR,
        rgb_filename='latest_color.png',
        depth_filename='latest_depth.npy',
        camera_info_filename='latest_camera_info.json',
        freshness_max_age_s=float(getattr(cfg.cameras, 'freshness_max_age_s', 3.0) if cfg.cameras is not None else getattr(cfg.camera, 'freshness_max_age_s', 3.0)),
    )
camera = DualSharedDirCamera(primary_camera, secondary_camera)
snap = camera.snapshot()
print('[ManualExec] using rgb snapshot:', snap['rgb_path'])
print('[ManualExec] using rgb source  :', snap.get('rgb_source_path'))
print('[ManualExec] shared_dir        :', snap.get('shared_dir'))
if snap.get('status'):
    print('[ManualExec] status camera_ns :', snap['status'].get('camera_ns'))
    print('[ManualExec] status rgb_stamp_ns:', snap['status'].get('rgb_stamp_ns'))
extrinsic = load_extrinsic(EXTRINSIC_PATH)
llm_perception = LLMPerceptionService(
    vision_client=vision_client,
    rgb_path=snap['rgb_path'],
    depth=snap['depth'],
    camera_info=snap['camera_info'],
    extrinsic=extrinsic,
    tool_rpy_deg=cfg.api_runtime.default_tool_orientation_deg,
    place_drop_mm=cfg.api_runtime.default_place_drop_mm,
    depth_window_radius=cfg.perception.depth_window_radius,
    debug=cfg.perception.debug,
    grasp_xyz_offset_mm=cfg.api_runtime.grasp_xyz_offset_mm,
    place_xyz_offset_mm=cfg.api_runtime.place_xyz_offset_mm,
    depth_uv_mapping_mode=cfg.perception.depth_uv_mapping_mode,
)
if cfg.perception.backend == 'classic_cv':
    perception = ClassicalTabletopPerceptionService(
        rgb_path=snap['rgb_path'],
        depth=snap['depth'],
        camera_info=snap['camera_info'],
        extrinsic=extrinsic,
        tool_rpy_deg=cfg.api_runtime.default_tool_orientation_deg,
        place_drop_mm=cfg.api_runtime.default_place_drop_mm,
        depth_window_radius=cfg.perception.depth_window_radius,
        debug=cfg.perception.debug,
        llm_fallback=llm_perception if cfg.perception.llm_fallback else None,
        grasp_xyz_offset_mm=cfg.api_runtime.grasp_xyz_offset_mm,
        place_xyz_offset_mm=cfg.api_runtime.place_xyz_offset_mm,
        depth_uv_mapping_mode=cfg.perception.depth_uv_mapping_mode,
        eye_in_hand=True,
    )
elif cfg.perception.backend == 'foundation_vision':
    perception = FoundationVisionPerceptionService(
        vision_client=vision_client,
        rgb_path=snap['rgb_path'],
        depth=snap['depth'],
        camera_info=snap['camera_info'],
        extrinsic=extrinsic,
        tool_rpy_deg=cfg.api_runtime.default_tool_orientation_deg,
        place_drop_mm=cfg.api_runtime.default_place_drop_mm,
        depth_window_radius=cfg.perception.depth_window_radius,
        debug=cfg.perception.debug,
        grasp_xyz_offset_mm=cfg.api_runtime.grasp_xyz_offset_mm,
        place_xyz_offset_mm=cfg.api_runtime.place_xyz_offset_mm,
        depth_uv_mapping_mode=cfg.perception.depth_uv_mapping_mode,
        eye_in_hand=True,
        model_provider=cfg.perception.foundation_model_provider,
        florence_model_id=cfg.perception.florence_model_id,
        grounding_dino_model_id=cfg.perception.grounding_dino_model_id,
        grounding_dino_repo_dir=cfg.perception.grounding_dino_repo_dir,
        grounding_dino_ckpt_path=cfg.perception.grounding_dino_ckpt_path,
        use_sam_refine=cfg.perception.use_sam_refine,
        device=cfg.perception.device,
    )
else:
    perception = llm_perception

arm = PiperArm(
    can_name=cfg.robot.can_name,
    speed_percent=cfg.robot.speed_percent,
    move_mode=cfg.robot.move_mode,
    pose_wait_s=cfg.robot.pose_wait_s,
    grip_wait_s=cfg.robot.grip_wait_s,
    pose_command_hz=cfg.robot.pose_command_hz,
    pose_command_duration_s=cfg.robot.pose_command_duration_s,
    gripper_command_hz=cfg.robot.gripper_command_hz,
    gripper_command_duration_s=cfg.robot.gripper_command_duration_s,
    gripper_max_width_mm=cfg.robot.gripper_max_width_mm,
    gripper_mm_to_ctrl=cfg.robot.gripper_mm_to_ctrl,
    gripper_open_force=cfg.robot.gripper_open_force,
    gripper_close_force=cfg.robot.gripper_close_force,
)
arm.connect()
ox, oy, oz, orx, ory, orz = cfg.api_runtime.observe_pose_xyzrpy_mmdeg
observe_pose = Pose(ox, oy, oz, orx, ory, orz)
runtime_api = RuntimeAPI(
    arm=arm,
    perception=perception,
    default_pregrasp_lift_mm=cfg.api_runtime.default_pregrasp_lift_mm,
    default_retreat_mm=cfg.api_runtime.default_retreat_mm,
    default_close_force=cfg.api_runtime.default_close_force,
    min_safe_z_mm=cfg.api_runtime.min_safe_z_mm,
    observe_pose=observe_pose,
    default_open_width_mm=cfg.api_runtime.default_open_width_mm,
    vision_client=vision_client,
    camera=camera,
)
if hasattr(runtime_api, 'set_artifact_dir'):
    runtime_api.set_artifact_dir(ARTIFACT_DIR)
runtime_api.set_verify_context(atomic_task=ATOMIC_TASK, plan_brief=PLAN_BRIEF)

debug = runtime_api.debug
detect_object_by_text = runtime_api.detect_object_by_text
estimate_top_grasp_pose = runtime_api.estimate_top_grasp_pose
build_pregrasp_pose = runtime_api.build_pregrasp_pose
estimate_place_pose = runtime_api.estimate_place_pose
open_gripper = runtime_api.open_gripper
close_gripper = runtime_api.close_gripper
move_to_pose = runtime_api.move_to_pose
retreat = runtime_api.retreat
return_to_observe_pose = runtime_api.return_to_observe_pose
stabilize_grasp = runtime_api.stabilize_grasp
verify_object_grasped = runtime_api.verify_object_grasped
verify_object_in_region = runtime_api.verify_object_in_region
ai_verify_atomic_task = runtime_api.ai_verify_atomic_task

# ===== GENERATED PROGRAM BODY START =====
debug("Start perception step: detect target red metal box")
red_box = detect_object_by_text("red metal box")
debug("Perception step finished, start aligning to placement pose")
place_pose = estimate_place_pose(red_box)
move_to_pose(place_pose)
debug("Aligned to placement pose, start interaction step")
open_gripper()
debug("Object released, retreat upward")
retreat()
debug("Return to global observation pose")
return_to_observe_pose()
result = ai_verify_atomic_task()
# ===== GENERATED PROGRAM BODY END =====

print('result =', result)
print('secondary_view_dir =', SECONDARY_DIR)
print('artifact_dir =', ARTIFACT_DIR)
