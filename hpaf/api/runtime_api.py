import json
import os
import time
from hpaf.core.models import Pose


class RuntimeAPI:
    def __init__(
        self,
        arm,
        perception,
        default_pregrasp_lift_mm=60,
        default_retreat_mm=90,
        default_close_force=1000,
        min_safe_z_mm=120.0,
        observe_pose: Pose = None,
        default_open_width_mm=45.0,
        vision_client=None,
        camera=None,
    ):
        self.arm = arm
        self.perception = perception
        self.default_pregrasp_lift_mm = default_pregrasp_lift_mm
        self.default_retreat_mm = default_retreat_mm
        self.default_close_force = default_close_force
        self.default_open_width_mm = float(default_open_width_mm)
        self.min_safe_z_mm = float(min_safe_z_mm)
        self.observe_pose = observe_pose
        self.vision_client = vision_client
        self.camera = camera
        self._last_pose = None
        self._last_detected_obj = None
        self._artifact_dir = None
        self._verify_context = {
            'atomic_task': '',
            'plan_brief': '',
        }
        self._last_ai_verify_output = None

    def set_frame(self, rgb_path, depth, camera_info):
        if hasattr(self.perception, 'set_frame'):
            self.perception.set_frame(rgb_path, depth, camera_info)

    def debug(self, message):
        print(f'[PROGRAM DEBUG] {message}')

    def set_artifact_dir(self, path: str):
        self._artifact_dir = path
        if hasattr(self.perception, 'set_artifact_dir'):
            self.perception.set_artifact_dir(path)

    def set_verify_context(self, atomic_task: str = '', plan_brief: str = ''):
        self._verify_context = {
            'atomic_task': atomic_task or '',
            'plan_brief': plan_brief or '',
        }

    def get_last_ai_verify_output(self):
        return self._last_ai_verify_output

    def detect_object_by_text(self, text_query: str):
        obj = self.perception.detect_object_by_text(text_query)
        self._last_detected_obj = obj
        return obj

    def estimate_top_grasp_pose(self, obj):
        current_ee_pose = self.arm.get_current_end_pose()
        return self.perception.estimate_top_grasp_pose(obj, current_ee_pose=current_ee_pose)

    def build_pregrasp_pose(self, grasp_pose: Pose, lift_mm: int = None):
        if lift_mm is None:
            lift_mm = self.default_pregrasp_lift_mm
        return Pose(
            x_mm=grasp_pose.x_mm,
            y_mm=grasp_pose.y_mm,
            z_mm=max(grasp_pose.z_mm + lift_mm, self.min_safe_z_mm),
            rx_deg=grasp_pose.rx_deg,
            ry_deg=grasp_pose.ry_deg,
            rz_deg=grasp_pose.rz_deg,
        )

    def estimate_place_pose(self, target_region_obj):
        current_ee_pose = self.arm.get_current_end_pose()
        return self.perception.estimate_place_pose(target_region_obj, current_ee_pose=current_ee_pose)

    def move_to_pose(self, pose):
        self.debug(f'move_to_pose -> {pose}')
        self.arm.move_to_pose(pose)
        self._last_pose = pose

    def open_gripper(self, width_mm: float = None):
        arm_max = float(getattr(self.arm, 'gripper_max_width_mm', self.default_open_width_mm))
        target_width = arm_max if width_mm is None else float(width_mm)
        obj = self._last_detected_obj
        if obj is not None and hasattr(self.perception, 'estimate_gripper_width_mm'):
            try:
                est = float(self.perception.estimate_gripper_width_mm(obj))
                target_width = max(target_width, min(arm_max, est + 8.0))
            except Exception as e:
                self.debug(f'estimate_gripper_width_mm failed, fallback default: {e}')
        target_width = max(self.default_open_width_mm, min(arm_max, target_width))
        self.debug(f'open_gripper width_mm={target_width}')
        if hasattr(self.arm, 'open_gripper'):
            self.arm.open_gripper(width_mm=target_width)
        else:
            self.arm.set_gripper_width_mm(target_width)

    def _normalize_user_force(self, force: int = None):
        if force is None:
            return self.default_close_force
        try:
            raw = int(force)
        except Exception:
            return self.default_close_force
        if raw <= 0:
            return self.default_close_force
        if raw <= 10:
            scaled = min(1000, raw * 200)
            self.debug(f'close_gripper force normalized from {raw} to {scaled}')
            return scaled
        if raw <= 100:
            scaled = min(1000, raw * 10)
            self.debug(f'close_gripper force normalized from {raw} to {scaled}')
            return scaled
        return min(1000, raw)

    def close_gripper(self, force: int = None):
        force = self._normalize_user_force(force)
        self.debug(f'close_gripper force={force}')
        self.arm.close_gripper(force=force)

    def stabilize_grasp(self, force: int = None):
        force = self._normalize_user_force(force)
        self.debug(f'stabilize_grasp force={force}')
        if hasattr(self.arm, 'hold_current_gripper'):
            self.arm.hold_current_gripper(force=force)
        else:
            self.arm.close_gripper(force=force)

    def retreat(self, z_offset_mm: int = None):
        if z_offset_mm is None:
            z_offset_mm = self.default_retreat_mm
        if self._last_pose is None:
            raise RuntimeError('retreat() called before move_to_pose()')
        pose = Pose(
            x_mm=self._last_pose.x_mm,
            y_mm=self._last_pose.y_mm,
            z_mm=max(self._last_pose.z_mm + z_offset_mm, self.min_safe_z_mm),
            rx_deg=self._last_pose.rx_deg,
            ry_deg=self._last_pose.ry_deg,
            rz_deg=self._last_pose.rz_deg,
        )
        self.move_to_pose(pose)

    def return_to_observe_pose(self):
        if self.observe_pose is None:
            raise RuntimeError('observe pose is not configured')
        self.debug(f'return_to_observe_pose -> {self.observe_pose}')
        self.move_to_pose(self.observe_pose)

    def verify_object_grasped(self, label: str, return_bool: bool = True):
        return self.perception.verify_object_grasped(label)

    def verify_object_in_region(self, object_label: str, region_label: str, return_bool: bool = True):
        return self.perception.verify_object_in_region(object_label, region_label)

    def _snapshot_verify_view(self):
        image_path = getattr(self.perception, 'rgb_path', None)
        source = 'primary_perception_frame'
        primary_path = None
        secondary_path = None
        if self.camera is not None and hasattr(self.camera, 'snapshot'):
            try:
                snap = self.camera.snapshot()
                primary_path = snap.get('rgb_path')
                secondary = snap.get('secondary') or {}
                secondary_path = secondary.get('rgb_path')
                if secondary_path:
                    image_path = secondary_path
                    source = 'secondary_global_view'
                elif primary_path:
                    image_path = primary_path
                    source = 'primary_fresh_view'
            except Exception as e:
                self.debug(f'ai_verify_atomic_task snapshot fallback due to: {e}')
        return image_path, source, primary_path, secondary_path

    def ai_verify_atomic_task(self):
        if self.vision_client is None:
            raise RuntimeError('ai_verify_atomic_task() requires RuntimeAPI to be constructed with vision_client')
        image_path, source, primary_path, secondary_path = self._snapshot_verify_view()
        if not image_path or not os.path.exists(image_path):
            raise RuntimeError('ai_verify_atomic_task() could not resolve a readable verification image')

        atomic_task = self._verify_context.get('atomic_task', '').strip()
        plan_brief = self._verify_context.get('plan_brief', '').strip()
        system_prompt = (
            'You are a robot atomic-task verifier. '
            'Judge only whether the current atomic task has been completed in the current scene image. '
            'Return strict JSON only in the format '
            '{"done": true, "reason": "...", "confidence": 0.0, "failure_stage": "none"}. '
            'If the task is not complete, set done to false and failure_stage to one of perception, align, interact, verify.'
        )
        user_text = (
            f'Atomic task: {atomic_task}\n'
            f'Program summary: {plan_brief or "N/A"}\n'
            f'Verification image source: {source}\n'
            'Determine whether the atomic task has been completed from this image.'
        )
        t0 = time.time()
        data = self.vision_client.ask_json_with_image(image_path, system_prompt, user_text)
        elapsed = time.time() - t0
        if 'done' not in data and 'success' in data:
            data['done'] = bool(data.get('success'))
        data['done'] = bool(data.get('done', False))
        data.setdefault('reason', '')
        data.setdefault('confidence', None)
        data.setdefault('failure_stage', 'none' if data['done'] else 'verify')
        data['atomic_task'] = atomic_task
        data['plan_brief'] = plan_brief
        data['verify_image_path'] = image_path
        data['verify_image_source'] = source
        data['primary_verify_image_path'] = primary_path
        data['secondary_verify_image_path'] = secondary_path
        data['elapsed_s'] = round(elapsed, 4)
        self._last_ai_verify_output = data
        self.debug(f"ai_verify_atomic_task -> done={data['done']} source={source} reason={data.get('reason', '')}")
        if self._artifact_dir:
            try:
                os.makedirs(self._artifact_dir, exist_ok=True)
                out_path = os.path.join(self._artifact_dir, 'ai_verify_output.json')
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.debug(f'failed to write ai_verify_output.json: {e}')
        return data['done']
