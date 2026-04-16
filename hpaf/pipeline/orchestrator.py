import os
import re
import shutil
import time
from hpaf.core.io import save_json, ensure_dir
from hpaf.core.logging_utils import ts
from hpaf.execution.executor import execute_program

SCRIPT_TEMPLATE = """#!/usr/bin/env python3
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
PRIMARY_DIR = str(_resolve_project_path(PROJECT_ROOT, {primary_dir!r}))
SECONDARY_DIR = str(_resolve_project_path(PROJECT_ROOT, {secondary_dir!r}))
EXTRINSIC_PATH = str(_resolve_project_path(PROJECT_ROOT, {extrinsic_json!r}))
ARTIFACT_DIR = str(_resolve_project_path(PROJECT_ROOT, {artifact_dir!r}))
ATOMIC_TASK = {atomic_task!r}
PLAN_BRIEF = {plan_brief!r}

primary_camera = SharedDirCamera(
    shared_dir=PRIMARY_DIR,
    rgb_filename={rgb_filename!r},
    depth_filename={depth_filename!r},
    camera_info_filename={camera_info_filename!r},
    freshness_max_age_s=float(getattr(cfg.cameras, 'freshness_max_age_s', 3.0) if cfg.cameras is not None else getattr(cfg.camera, 'freshness_max_age_s', 3.0)),
)
secondary_camera = None
if SECONDARY_DIR and os.path.abspath(SECONDARY_DIR) != os.path.abspath(PRIMARY_DIR):
    secondary_camera = SharedDirCamera(
        shared_dir=SECONDARY_DIR,
        rgb_filename={rgb_filename!r},
        depth_filename={depth_filename!r},
        camera_info_filename={camera_info_filename!r},
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
{program}
# ===== GENERATED PROGRAM BODY END =====

print('result =', result)
print('secondary_view_dir =', SECONDARY_DIR)
print('artifact_dir =', ARTIFACT_DIR)
"""


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]+', '_', (text or '').strip())
    s = re.sub(r'_+', '_', s).strip('_')
    return (s or 'task')[:limit]


class PipelineOrchestrator:
    def __init__(self, cfg, task_agent, program_agent, verify_agent, camera, runtime_api):
        self.cfg = cfg
        self.task_agent = task_agent
        self.program_agent = program_agent
        self.verify_agent = verify_agent
        self.camera = camera
        self.runtime_api = runtime_api
        ensure_dir(cfg.pipeline.logs_dir)
        self.run_dir = None
        self.task_dir = None
        self.current_atomic_dir = None
        self.current_attempt_dir = None

    def _init_run_dirs(self, task_text: str):
        run_name = f"{ts()}_{_slug(task_text)}"
        self.run_dir = os.path.join(self.cfg.pipeline.logs_dir, run_name)
        self.task_dir = os.path.join(self.run_dir, 'task')
        ensure_dir(self.task_dir)
        save_json(os.path.join(self.task_dir, 'task_request.json'), {'task_text': task_text, 'run_dir': self.run_dir})
        return self.run_dir

    def _atomic_dir(self, idx: int, atomic_task: str):
        path = os.path.join(self.run_dir, f"atomic_{idx:02d}_{_slug(atomic_task, 36)}")
        ensure_dir(path)
        ensure_dir(os.path.join(path, 'attempts'))
        atomic_artifacts = os.path.join(path, 'artifacts')
        if os.path.isdir(atomic_artifacts):
            try:
                shutil.rmtree(atomic_artifacts)
            except Exception:
                pass
        return path

    def _attempt_dir(self, atomic_dir: str, retry: int):
        path = os.path.join(atomic_dir, 'attempts', f'try_{retry:02d}')
        ensure_dir(path)
        ensure_dir(os.path.join(path, 'artifacts'))
        return path

    def _save_json(self, path, data):
        save_json(path, data)
        return path

    def _save_task_output(self, data):
        return self._save_json(os.path.join(self.task_dir, 'task_agent_output.json'), data)

    def _save_program_output(self, idx, retries, data):
        return self._save_json(os.path.join(self.current_attempt_dir, 'program_agent_output.json'), data)

    def _save_verify_output(self, data):
        return self._save_json(os.path.join(self.current_attempt_dir, 'verify_output.json'), data)

    def _save_exec_output(self, data):
        return self._save_json(os.path.join(self.current_attempt_dir, 'executor_output.json'), data)

    def _strip_nonterminal_verify_calls(self, program: str) -> str:
        stripped_lines = []
        removed = []
        for raw_line in (program or '').splitlines():
            line = raw_line.rstrip()
            compact = line.strip()
            if not compact:
                continue
            if compact.startswith('verify_object_grasped(') or compact.startswith('verify_object_in_region('):
                removed.append(compact)
                continue
            stripped_lines.append(line)
        joined = '\n'.join(stripped_lines)
        if removed:
            if joined and not joined.endswith('\n'):
                joined += '\n'
            joined += "debug('program sanitize: removed intermediate object-specific verify call(s); final decision uses ai_verify_atomic_task only')"
        return joined

    def _ensure_program_has_ai_verify(self, program: str) -> str:
        program = self._strip_nonterminal_verify_calls(program)
        lines = []
        for raw_line in (program or '').splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith('result ='):
                continue
            lines.append(raw_line.rstrip())
        joined = '\n'.join(lines)
        if joined and not joined.endswith('\n'):
            joined += '\n'
        joined += 'result = ai_verify_atomic_task()'
        return joined

    def _sanitize_program_for_manual(self, program: str, atomic_task: str = '') -> str:
        program = self._strip_nonterminal_verify_calls(program)
        lines = []
        for raw_line in (program or '').splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('result ='):
                continue
            if stripped.startswith('close_gripper('):
                m = re.search(r"force\s*=\s*(\d+)", line)
                if m and int(m.group(1)) <= 100:
                    cleaned = re.sub(r"force\s*=\s*\d+", "", line)
                    cleaned = cleaned.replace('(,', '(').replace(',)', ')').replace('  )', ')')
                    lines.append(cleaned)
                    lines.append(f"debug('manual sanitize: removed tiny close_gripper force={m.group(1)} and used backend default')")
                    continue
            lines.append(line)
        joined = '\n'.join(lines)
        if 'return_to_observe_pose()' not in joined:
            joined += "\ndebug('Return to the global observation pose')\nreturn_to_observe_pose()"
        if joined and not joined.endswith('\n'):
            joined += '\n'
        joined += 'result = ai_verify_atomic_task()'
        return joined

    def _camera_paths_for_script(self):
        if self.cfg.cameras is not None:
            return {
                'primary_dir': self.cfg.cameras.primary_shared_dir,
                'secondary_dir': self.cfg.cameras.secondary_shared_dir,
                'rgb_filename': self.cfg.cameras.rgb_filename,
                'depth_filename': self.cfg.cameras.depth_filename,
                'camera_info_filename': self.cfg.cameras.camera_info_filename,
                'extrinsic_json': self.cfg.cameras.primary_extrinsic_json,
            }
        return {
            'primary_dir': self.cfg.camera.shared_dir,
            'secondary_dir': self.cfg.camera.shared_dir,
            'rgb_filename': self.cfg.camera.rgb_filename,
            'depth_filename': self.cfg.camera.depth_filename,
            'camera_info_filename': self.cfg.camera.camera_info_filename,
            'extrinsic_json': self.cfg.camera.extrinsic_json,
        }

    def _save_manual_script(self, idx, program, atomic_task, plan_brief):
        path = os.path.join(self.current_attempt_dir, 'manual_exec.py')
        camera_args = self._camera_paths_for_script()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        artifact_rel = os.path.relpath(os.path.join(self.current_attempt_dir, 'artifacts'), start=project_root)
        replacements = {
            'program': self._sanitize_program_for_manual(program, atomic_task=atomic_task),
            'artifact_dir': repr(artifact_rel),
            'primary_dir': repr(camera_args['primary_dir']),
            'secondary_dir': repr(camera_args['secondary_dir']),
            'rgb_filename': repr(camera_args['rgb_filename']),
            'depth_filename': repr(camera_args['depth_filename']),
            'camera_info_filename': repr(camera_args['camera_info_filename']),
            'extrinsic_json': repr(camera_args['extrinsic_json']),
            'atomic_task': repr(atomic_task),
            'plan_brief': repr(plan_brief),
        }
        text = SCRIPT_TEMPLATE
        for key, value in replacements.items():
            text = text.replace('{' + key + '!r}', value)
            text = text.replace('{' + key + '}', value)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        return path

    def _human_verify(self, atomic_task, secondary=None):
        print(f"\n[Human Verify] Current atomic task: {atomic_task}")
        if secondary is not None:
            print(f"[Human Verify] Reference global-view image: {secondary.get('rgb_path')}")
        ans = input('Enter yes if this atomic task is complete, otherwise enter no: ').strip().lower()
        return {
            'done': ans == 'yes',
            'failure_stage': 'human' if ans != 'yes' else 'none',
            'reason': 'human verification',
            'regeneration_hint': '' if ans == 'yes' else 'Regenerate or rerun the program according to the scene state.',
        }

    def _prepare_attempt_context(self, attempt_dir: str, snap: dict):
        artifacts_dir = os.path.join(attempt_dir, 'artifacts')
        ensure_dir(artifacts_dir)
        if hasattr(self.runtime_api, 'set_artifact_dir'):
            self.runtime_api.set_artifact_dir(artifacts_dir)
        try:
            shutil.copy2(snap['rgb_path'], os.path.join(artifacts_dir, f"primary_view{os.path.splitext(snap['rgb_path'])[1] or '.png'}"))
        except Exception:
            pass
        if snap.get('secondary', {}).get('rgb_path'):
            try:
                shutil.copy2(snap['secondary']['rgb_path'], os.path.join(artifacts_dir, f"secondary_view{os.path.splitext(snap['secondary']['rgb_path'])[1] or '.png'}"))
            except Exception:
                pass
        self._save_json(os.path.join(artifacts_dir, 'snapshot_meta.json'), {
            'primary': snap.get('rgb_path'),
            'secondary': snap.get('secondary', {}).get('rgb_path') if snap.get('secondary') else None,
        })
        atomic_artifacts_dir = os.path.join(os.path.dirname(attempt_dir), 'artifacts')
        if os.path.isdir(atomic_artifacts_dir):
            try:
                shutil.rmtree(atomic_artifacts_dir)
            except Exception:
                pass

    def run(self, task_text: str, mode: str = 'manual'):
        pipeline_t0 = time.perf_counter()
        self._init_run_dirs(task_text)
        snap = self.camera.snapshot()
        task_image_path = snap['secondary']['rgb_path'] if snap.get('secondary') else snap['rgb_path']
        self.runtime_api.set_frame(snap['rgb_path'], snap['depth'], snap['camera_info'])

        t0 = time.perf_counter()
        task_out = self.task_agent.run(task_image_path, task_text)
        task_time = time.perf_counter() - t0
        self._save_task_output(task_out)
        print('\n[TaskAgent]')
        print(task_out)
        print(f"[Timing] TaskAgent: {task_time:.3f}s")
        print(f"[Logs] run_dir: {self.run_dir}")

        atomic_tasks = task_out.get('atomic_tasks', [])
        if not atomic_tasks:
            raise RuntimeError('TaskAgent produced no atomic tasks')

        for idx, atomic_task in enumerate(atomic_tasks, start=1):
            self.current_atomic_dir = self._atomic_dir(idx, atomic_task)
            self._save_json(os.path.join(self.current_atomic_dir, 'atomic_task.json'), {
                'index': idx,
                'atomic_task': atomic_task,
            })
            print(f"\n===== Atomic Task {idx}/{len(atomic_tasks)} =====")
            print(atomic_task)
            retries = 0
            while retries <= self.cfg.pipeline.max_verify_retries:
                self.current_attempt_dir = self._attempt_dir(self.current_atomic_dir, retries)
                snap = self.camera.snapshot()
                image_path = snap['rgb_path']
                self.runtime_api.set_frame(snap['rgb_path'], snap['depth'], snap['camera_info'])
                self._prepare_attempt_context(self.current_attempt_dir, snap)
                self.runtime_api.set_verify_context(atomic_task=atomic_task, plan_brief='')

                t0 = time.perf_counter()
                program_out = self.program_agent.run(image_path, atomic_task, execution_mode=mode)
                prog_time = time.perf_counter() - t0
                program = self._ensure_program_has_ai_verify(program_out['program'])
                program_out['program'] = program
                self.runtime_api.set_verify_context(atomic_task=atomic_task, plan_brief=program_out.get('plan_brief', ''))
                self._save_program_output(idx, retries, program_out)

                print('\n[ProgramAgent plan]')
                print(program_out.get('plan_brief', ''))
                print('\n[ProgramAgent program]')
                print(program)
                print(f"[Timing] ProgramAgent: {prog_time:.3f}s")

                if mode == 'manual':
                    script_path = self._save_manual_script(idx, program, atomic_task, program_out.get('plan_brief', ''))
                    print(f"\n[Manual Script Saved] {script_path}")
                    print('Run this script in another terminal. It will end with result = ai_verify_atomic_task() and save ai_verify_output.json under the attempt artifacts directory. Then return here for human verification.')
                    verify_out = self._human_verify(atomic_task, snap.get('secondary'))
                    self._save_verify_output(verify_out)
                else:
                    if mode == 'review':
                        ans = input('\nEnter yes to execute this program, otherwise abort: ').strip().lower()
                        if ans != 'yes':
                            raise RuntimeError('User aborted execution')
                    t0 = time.perf_counter()
                    exec_result = execute_program(program, self.runtime_api)
                    exec_time = time.perf_counter() - t0
                    print('\n[Executor result]', exec_result)
                    print(f"[Timing] Executor: {exec_time:.3f}s")
                    self._save_exec_output(exec_result)
                    verify_out = exec_result.get('ai_verify_output')
                    if not verify_out:
                        if self.cfg.verify.mode == 'human':
                            verify_out = self._human_verify(atomic_task, snap.get('secondary'))
                        else:
                            t0 = time.perf_counter()
                            verify_image = snap['secondary']['rgb_path'] if snap.get('secondary') else image_path
                            verify_out = self.verify_agent.run(verify_image, atomic_task, program_out.get('plan_brief', ''))
                            verify_time = time.perf_counter() - t0
                            print('\n[VerifyAgent]')
                            print(verify_out)
                            print(f"[Timing] VerifyAgent: {verify_time:.3f}s")
                    self._save_verify_output(verify_out)

                print('\n[Verify]')
                print(verify_out)
                if verify_out.get('done', False):
                    print(f'Atomic task {idx} done.')
                    break
                retries += 1
                if retries > self.cfg.pipeline.max_verify_retries:
                    if self.cfg.pipeline.stop_on_failure:
                        raise RuntimeError(f'Atomic task failed after retries: {atomic_task}')
                    break
                print(f"\n[Retry] {atomic_task} -> try {retries}/{self.cfg.pipeline.max_verify_retries}")

        pipeline_total = time.perf_counter() - pipeline_t0
        self._save_json(os.path.join(self.run_dir, 'run_summary.json'), {
            'task_text': task_text,
            'total_atomic_tasks': len(atomic_tasks),
            'pipeline_total_s': pipeline_total,
        })
        print(f"\n[Timing] Pipeline Total: {pipeline_total:.3f}s")
        print(f"[Logs] Organized run dir: {self.run_dir}")
