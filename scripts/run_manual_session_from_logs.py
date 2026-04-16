#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def _find_project_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / 'pyproject.toml').exists() and (candidate / 'configs' / 'demo.yaml').exists():
            return candidate
    raise FileNotFoundError('Could not locate project root')


PROJECT_ROOT = _find_project_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hpaf.core.io import load_config
from hpaf.llm.factory import make_vision_client
from hpaf.geometry.transforms import load_extrinsic
from hpaf.perception.llm_perception import LLMPerceptionService
from hpaf.perception.classic_cv_perception import ClassicalTabletopPerceptionService
from hpaf.perception.foundation_vision_perception import FoundationVisionPerceptionService
from hpaf.api.runtime_api import RuntimeAPI
from hpaf.robot.piper_backend import PiperArm
from hpaf.core.models import Pose
from hpaf.camera.shared_dir_camera import SharedDirCamera


def _sanitize_program_for_manual(program: str, atomic_task: str = '') -> str:
    lines = []
    for line in program.splitlines():
        striped = line.strip()
        if not striped:
            lines.append(line)
            continue
        if 'verify_object_grasped' in striped or 'verify_object_in_region' in striped:
            lines.append(f"debug('skip verify in manual session: {striped.replace(chr(39), '')}')")
            continue
        m = re.search(r"close_gripper\(\s*force\s*=\s*(\d+)\s*\)", striped)
        if m and int(m.group(1)) <= 100:
            cleaned = re.sub(r"force\s*=\s*\d+", "", line)
            cleaned = cleaned.replace('(,', '(').replace(',)', ')').replace('  )', ')')
            lines.append(cleaned)
            lines.append(f"debug('manual session sanitize: removed tiny close_gripper force={m.group(1)} and used backend default')")
            continue
        lines.append(line)
    joined = "\n".join(lines)
    task_hint = (atomic_task or '').lower()
    placement_like = any(k in task_hint for k in ['place', 'put', 'grasped', '放入', '放到', '已抓'])
    if placement_like and 'stabilize_grasp(' not in joined:
        joined = "debug('manual session sanitize: stabilize grasp before placement')\nstabilize_grasp()\n" + joined
    if 'return_to_observe_pose()' not in joined:
        joined += "\ndebug('Return to the global observation pose')\nreturn_to_observe_pose()"
    joined += "\nresult = True"
    return joined


def _build_runtime(cfg):
    vision_client = make_vision_client(cfg.llm)
    if cfg.cameras is not None:
        shared_dir = cfg.cameras.primary_shared_dir
        rgb_name = cfg.cameras.rgb_filename
        depth_name = cfg.cameras.depth_filename
        camera_info_name = cfg.cameras.camera_info_filename
        extrinsic_path = cfg.cameras.primary_extrinsic_json
        freshness_max_age_s = cfg.cameras.freshness_max_age_s
    else:
        shared_dir = cfg.camera.shared_dir
        rgb_name = cfg.camera.rgb_filename
        depth_name = cfg.camera.depth_filename
        camera_info_name = cfg.camera.camera_info_filename
        extrinsic_path = cfg.camera.extrinsic_json
        freshness_max_age_s = cfg.camera.freshness_max_age_s

    camera = SharedDirCamera(
        shared_dir=shared_dir,
        rgb_filename=rgb_name,
        depth_filename=depth_name,
        camera_info_filename=camera_info_name,
        freshness_max_age_s=freshness_max_age_s,
    )
    snap = camera.snapshot()
    extrinsic = load_extrinsic(extrinsic_path)

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
    )
    return camera, runtime_api


def _bindings(runtime_api):
    return {
        'debug': runtime_api.debug,
        'detect_object_by_text': runtime_api.detect_object_by_text,
        'estimate_top_grasp_pose': runtime_api.estimate_top_grasp_pose,
        'build_pregrasp_pose': runtime_api.build_pregrasp_pose,
        'estimate_place_pose': runtime_api.estimate_place_pose,
        'open_gripper': runtime_api.open_gripper,
        'close_gripper': runtime_api.close_gripper,
        'move_to_pose': runtime_api.move_to_pose,
        'retreat': runtime_api.retreat,
        'return_to_observe_pose': runtime_api.return_to_observe_pose,
        'stabilize_grasp': runtime_api.stabilize_grasp,
    }


def _collect_steps(run_dir: Path):
    steps = []
    for atomic_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('atomic_')):
        atomic_json = atomic_dir / 'atomic_task.json'
        attempt_dir = atomic_dir / 'attempts' / 'try_00'
        program_json = attempt_dir / 'program_agent_output.json'
        if not atomic_json.exists() or not program_json.exists():
            continue
        atomic_info = json.loads(atomic_json.read_text(encoding='utf-8'))
        program_info = json.loads(program_json.read_text(encoding='utf-8'))
        steps.append({
            'index': atomic_info.get('index', len(steps) + 1),
            'atomic_task': atomic_info.get('atomic_task', atomic_dir.name),
            'program': _sanitize_program_for_manual(program_info.get('program', ''), atomic_info.get('atomic_task', '')),
            'artifact_dir': attempt_dir / 'artifacts',
        })
    return steps


def main():
    parser = argparse.ArgumentParser(description='Run all atomic tasks from one logs run in a single Python process.')
    parser.add_argument('run_dir', help='Path to one logs/<timestamp_task> directory')
    parser.add_argument('--start', type=int, default=1, help='1-based atomic task index to start from')
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = load_config(str(PROJECT_ROOT / 'configs' / 'demo.yaml'))
    camera, runtime_api = _build_runtime(cfg)
    api = _bindings(runtime_api)
    steps = _collect_steps(run_dir)
    if not steps:
        raise RuntimeError(f'No atomic steps found under {run_dir}')

    print('[ManualSession] Loaded', len(steps), 'atomic tasks')
    print('[ManualSession] One arm connection will be reused across all remaining steps.')

    for step in steps:
        if step['index'] < args.start:
            continue
        artifact_dir = step['artifact_dir']
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(runtime_api, 'set_artifact_dir'):
            runtime_api.set_artifact_dir(str(artifact_dir))
        snap = camera.snapshot()
        runtime_api.set_frame(snap['rgb_path'], snap['depth'], snap['camera_info'])
        print(f"\n===== Atomic Task {step['index']}/{len(steps)} =====")
        print(step['atomic_task'])
        exec(step['program'], {'__builtins__': {}}, api)
        print('result =', api.get('result'))
        ans = input('Enter yes if this atomic task is complete, otherwise enter no: ').strip().lower()
        if ans != 'yes':
            raise RuntimeError(f'Atomic task {step["index"]} failed human verification')

    print('\n[ManualSession] All requested atomic tasks finished.')


if __name__ == '__main__':
    main()
