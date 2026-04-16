import time
from hpaf.core.models import Pose


def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _get_attr_recursive(obj, target_names):
    seen = set()
    queue = [obj]
    lowered = {n.lower() for n in target_names}
    while queue:
        cur = queue.pop(0)
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        for name in dir(cur):
            if name.startswith('_'):
                continue
            try:
                val = getattr(cur, name)
            except Exception:
                continue
            if callable(val):
                continue
            if name.lower() in lowered:
                return val
            if hasattr(val, '__dict__') or 'Msg' in type(val).__name__:
                queue.append(val)
        if hasattr(cur, '__dict__'):
            for val in cur.__dict__.values():
                if hasattr(val, '__dict__') or 'Msg' in type(val).__name__:
                    queue.append(val)
    return None

class PiperArm:
    def __init__(self, can_name='can0', speed_percent=100, move_mode='0x00', pose_wait_s=2.0, grip_wait_s=1.0, pose_command_hz=50, pose_command_duration_s=1.0, gripper_command_hz=100, gripper_command_duration_s=0.5, gripper_max_width_mm=50.0, gripper_mm_to_ctrl=1000.0, gripper_open_force=1000, gripper_close_force=1000, reach_min_x_mm=120.0, reach_max_x_mm=430.0, reach_max_abs_y_mm=260.0, reach_max_planar_radius_mm=460.0):
        self.can_name = can_name
        self.speed_percent = speed_percent
        self.move_mode = int(move_mode, 16) if isinstance(move_mode, str) else int(move_mode)
        self.pose_wait_s = pose_wait_s
        self.grip_wait_s = grip_wait_s
        self.pose_command_hz = int(pose_command_hz)
        self.pose_command_duration_s = float(pose_command_duration_s)
        self.gripper_command_hz = int(gripper_command_hz)
        self.gripper_command_duration_s = float(gripper_command_duration_s)
        self.gripper_max_width_mm = float(gripper_max_width_mm)
        self.gripper_mm_to_ctrl = float(gripper_mm_to_ctrl)
        self.gripper_open_force = int(gripper_open_force)
        self.gripper_close_force = int(gripper_close_force)
        self.reach_min_x_mm = float(reach_min_x_mm)
        self.reach_max_x_mm = float(reach_max_x_mm)
        self.reach_max_abs_y_mm = float(reach_max_abs_y_mm)
        self.reach_max_planar_radius_mm = float(reach_max_planar_radius_mm)
        self.piper = None

    def _normalize_force(self, force, *, default_force: int, action: str = 'gripper') -> int:
        try:
            raw = int(force) if force is not None else int(default_force)
        except Exception:
            raw = int(default_force)
        if raw <= 0:
            return int(default_force)
        normalized = raw
        note = None
        # Many LLM-generated programs use tiny human-scale values such as 5 or 10.
        # Piper expects approximately 0-1000, so rescale these values conservatively.
        if raw <= 10:
            normalized = raw * 200
            note = 'scaled from 0-10 heuristic range'
        elif raw <= 100:
            normalized = raw * 10
            note = 'scaled from 0-100 heuristic range'
        normalized = max(50, min(1000, int(normalized)))
        if note is not None:
            print(f'[PiperArm] normalize_force {action}: raw={raw} -> sdk={normalized} ({note})')
        return normalized

    def _read_current_gripper_ctrl(self):
        try:
            msg = self.piper.GetArmGripperMsgs()
            angle = _get_attr_recursive(msg, ['grippers_angle', 'gripper_angle', 'angle'])
            val = _safe_float(angle)
            if val is None:
                return None
            return int(round(val * 1000.0)) if abs(val) < 1000 else int(round(val))
        except Exception:
            return None

    def hold_current_gripper(self, force: int = None, repeats: int = 6, interval_s: float = 0.02):
        force = self._normalize_force(force, default_force=self.gripper_close_force, action='hold')
        ctrl = self._read_current_gripper_ctrl()
        if ctrl is None:
            # On reconnect, failing closed is much safer than failing open.
            ctrl = 0
        try:
            for _ in range(max(1, int(repeats))):
                # Use only the regular control stream here. In practice this is less likely to produce a
                # brief jaw relaxation on a fresh reconnect than toggling an extra 0x02 wake command first.
                self.piper.GripperCtrl(abs(int(ctrl)), force, 0x01, 0)
                time.sleep(max(0.0, float(interval_s)))
            print(f'[PiperArm] hold_current_gripper ctrl={ctrl}, force={force}, repeats={repeats}')
        except Exception as e:
            print('[PiperArm] gripper hold warning:', e)

    def connect(self):
        from piper_sdk import C_PiperInterface_V2
        self.piper = C_PiperInterface_V2(self.can_name)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        time.sleep(0.05)
        # Preserve the current jaw state on reconnects. This is especially important in manual multi-step
        # execution where the next atomic script starts in a fresh Python process while the object is already grasped.
        self.hold_current_gripper(force=self.gripper_close_force, repeats=10, interval_s=0.02)
        print('[PiperArm] connected')

    def _ensure_cartesian(self):
        # Keep consistent with validated vendor demo.
        self.piper.MotionCtrl_2(0x01, self.move_mode, self.speed_percent, 0x00)

    def _send_pose_command(self, pose: Pose, repeats=None):
        self._ensure_cartesian()
        X = int(round(pose.x_mm * 1000.0))
        Y = int(round(pose.y_mm * 1000.0))
        Z = int(round(pose.z_mm * 1000.0))
        RX = int(round(pose.rx_deg * 1000.0))
        RY = int(round(pose.ry_deg * 1000.0))
        RZ = int(round(pose.rz_deg * 1000.0))
        n = max(1, int(self.pose_command_hz * self.pose_command_duration_s)) if repeats is None else max(1, int(repeats))
        print(f'[PiperArm] move_to_pose target=({X},{Y},{Z},{RX},{RY},{RZ}), repeats={n}')
        for _ in range(n):
            self.piper.MotionCtrl_2(0x01, self.move_mode, self.speed_percent, 0x00)
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            time.sleep(1.0 / max(1, self.pose_command_hz))


    def _project_pose_into_reachable_envelope(self, pose: Pose) -> Pose:
        x = float(pose.x_mm)
        y = float(pose.y_mm)
        z = float(pose.z_mm)
        x0, y0 = x, y
        x = min(max(x, self.reach_min_x_mm), self.reach_max_x_mm)
        y = min(max(y, -self.reach_max_abs_y_mm), self.reach_max_abs_y_mm)
        planar = (x * x + y * y) ** 0.5
        if planar > self.reach_max_planar_radius_mm and planar > 1e-6:
            scale = self.reach_max_planar_radius_mm / planar
            x *= scale
            y *= scale
            x = min(max(x, self.reach_min_x_mm), self.reach_max_x_mm)
            y = min(max(y, -self.reach_max_abs_y_mm), self.reach_max_abs_y_mm)
        if abs(x - x0) > 1e-3 or abs(y - y0) > 1e-3:
            print(f'[PiperArm] projected unreachable xy from ({x0:.1f},{y0:.1f}) to ({x:.1f},{y:.1f}) before IK fallback')
        return Pose(x, y, z, pose.rx_deg, pose.ry_deg, pose.rz_deg)

    def _position_error_mm(self, feedback_pose: Pose, target_pose: Pose) -> float:
        dx = float(feedback_pose.x_mm) - float(target_pose.x_mm)
        dy = float(feedback_pose.y_mm) - float(target_pose.y_mm)
        dz = float(feedback_pose.z_mm) - float(target_pose.z_mm)
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def _orientation_fallback_poses(self, target_pose: Pose, current_pose: Pose):
        candidates = []
        def add(rx, ry, rz):
            key = (round(float(rx), 3), round(float(ry), 3), round(float(rz), 3))
            if key in seen:
                return
            seen.add(key)
            candidates.append(Pose(target_pose.x_mm, target_pose.y_mm, target_pose.z_mm, float(rx), float(ry), float(rz)))
        seen = set()
        # exact target first
        add(target_pose.rx_deg, target_pose.ry_deg, target_pose.rz_deg)
        # then strongly prefer preserving the current reachable wrist orientation while keeping xyz
        add(current_pose.rx_deg, current_pose.ry_deg, current_pose.rz_deg)
        # sweep a few common ry values while keeping target rx/rz near the calibrated top-down convention
        for ry in [target_pose.ry_deg, current_pose.ry_deg, 55.0, 45.0, 40.0, 30.0, 20.0, 70.0]:
            add(target_pose.rx_deg, ry, target_pose.rz_deg)
            add(current_pose.rx_deg, ry, current_pose.rz_deg)
        # also allow rz to stay at the currently reachable wrist yaw
        for rz in [target_pose.rz_deg, current_pose.rz_deg]:
            add(target_pose.rx_deg, target_pose.ry_deg, rz)
            add(current_pose.rx_deg, current_pose.ry_deg, rz)
        return candidates

    def move_to_pose(self, pose: Pose):
        target_pose = self._project_pose_into_reachable_envelope(pose)
        best_pose = target_pose
        best_feedback = None
        best_err = float('inf')
        try:
            current_pose = self.get_current_end_pose()
        except Exception:
            current_pose = target_pose
        candidate_poses = self._orientation_fallback_poses(target_pose, current_pose)
        position_ok_mm = 18.0
        for idx, cand in enumerate(candidate_poses):
            if idx > 0:
                print(f'[PiperArm] position-priority fallback try #{idx}: xyz=({cand.x_mm:.1f},{cand.y_mm:.1f},{cand.z_mm:.1f}) rpy=({cand.rx_deg:.1f},{cand.ry_deg:.1f},{cand.rz_deg:.1f})')
            self._send_pose_command(cand)
            try:
                fb_pose = self.get_current_end_pose()
                err = self._position_error_mm(fb_pose, target_pose)
            except Exception:
                fb_pose = None
                err = float('inf')
            if err < best_err:
                best_err = err
                best_pose = cand
                best_feedback = fb_pose
            if err <= position_ok_mm:
                break
        if best_err > position_ok_mm and best_feedback is not None:
            print(f'[PiperArm] position-priority fallback accepted best-effort pose; xyz_err_mm={best_err:.1f}, used_rpy=({best_pose.rx_deg:.1f},{best_pose.ry_deg:.1f},{best_pose.rz_deg:.1f})')
            # Re-send the best Cartesian candidate a bit longer so the controller settles there.
            self._send_pose_command(best_pose, repeats=max(1, int(self.pose_command_hz * max(self.pose_command_duration_s, 1.2))))
            try:
                print('[PiperArm] feedback', self.piper.GetArmEndPoseMsgs())
            except Exception:
                pass
        else:
            try:
                print('[PiperArm] feedback', self.piper.GetArmEndPoseMsgs())
            except Exception:
                pass
        time.sleep(self.pose_wait_s)

    def width_mm_to_ctrl(self, width_mm: float) -> int:
        width_mm = max(0.0, min(self.gripper_max_width_mm, float(width_mm)))
        return int(round(width_mm * self.gripper_mm_to_ctrl))

    def set_gripper_width_mm(self, width_mm: float, force: int = None):
        force = self._normalize_force(force, default_force=self.gripper_open_force, action='open')
        ctrl = self.width_mm_to_ctrl(width_mm)
        n = max(1, int(self.gripper_command_hz * self.gripper_command_duration_s))
        print(f'[PiperArm] set_gripper_width_mm width_mm={width_mm:.1f}, ctrl={ctrl}, repeats={n}')
        for _ in range(n):
            self.piper.GripperCtrl(abs(ctrl), int(force), 0x01, 0)
            time.sleep(1.0 / max(1, self.gripper_command_hz))
        try:
            print('[PiperArm] gripper feedback', self.piper.GetArmGripperMsgs())
        except Exception:
            pass
        time.sleep(self.grip_wait_s)

    def _wake_gripper(self):
        try:
            self.piper.GripperCtrl(0, self.gripper_open_force, 0x02, 0)
            time.sleep(0.02)
        except Exception:
            pass

    def open_gripper(self, width_mm=None):
        if width_mm is None:
            width_mm = self.gripper_max_width_mm
        width_mm = max(0.0, min(float(width_mm), self.gripper_max_width_mm))
        # Re-arm the module before open commands. In the full pipeline this is more robust than a single 0x01 stream.
        self._wake_gripper()
        # Open fully first, then settle to target width so small estimation errors do not leave the jaw nearly closed.
        self.set_gripper_width_mm(self.gripper_max_width_mm, force=self.gripper_open_force)
        if width_mm < self.gripper_max_width_mm - 0.5:
            self.set_gripper_width_mm(width_mm, force=self.gripper_open_force)

    def close_gripper(self, force=800, close_value=0):
        n = max(1, int(self.gripper_command_hz * self.gripper_command_duration_s))
        actual_force = self._normalize_force(force, default_force=self.gripper_close_force, action='close')
        print(f'[PiperArm] close_gripper close_value={close_value}, force={actual_force}, repeats={n}')
        for _ in range(n):
            self.piper.GripperCtrl(int(close_value), actual_force, 0x01, 0)
            time.sleep(1.0 / max(1, self.gripper_command_hz))
        try:
            print('[PiperArm] gripper feedback', self.piper.GetArmGripperMsgs())
        except Exception:
            pass
        time.sleep(self.grip_wait_s)


    def get_current_end_pose(self):
        msg = self.piper.GetArmEndPoseMsgs()
        x = _get_attr_recursive(msg, ['X_axis', 'x_axis'])
        y = _get_attr_recursive(msg, ['Y_axis', 'y_axis'])
        z = _get_attr_recursive(msg, ['Z_axis', 'z_axis'])
        rx = _get_attr_recursive(msg, ['RX_axis', 'rx_axis'])
        ry = _get_attr_recursive(msg, ['RY_axis', 'ry_axis'])
        rz = _get_attr_recursive(msg, ['RZ_axis', 'rz_axis'])
        vals = [_safe_float(v) for v in [x, y, z, rx, ry, rz]]
        if any(v is None for v in vals):
            raise RuntimeError(f'Failed to parse current end pose from feedback: {msg}')
        x, y, z, rx, ry, rz = vals
        pose = Pose(x / 1000.0, y / 1000.0, z / 1000.0, rx / 1000.0, ry / 1000.0, rz / 1000.0)
        print(f'[PiperArm] current_end_pose={pose}')
        return pose

    def get_fk(self):
        return self.piper.GetFK('feedback')
