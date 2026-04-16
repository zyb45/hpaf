import json
import math
from typing import Iterable, Optional, List, Dict

import numpy as np
from hpaf.core.models import Pose


def load_extrinsic(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pos = data['position']
    rpy = data.get('rpy', [[0.0, 0.0, 0.0]])[0]
    return {
        'position': pos,
        'rpy': rpy,
        'transform_direction': data.get('transform_direction'),
        'raw': data,
    }


def euler_to_matrix(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera_to_base(xyz_cam_mm, extrinsic):
    """Legacy helper for eye-to-hand cameras with a fixed camera->base transform."""
    rpy = extrinsic['rpy']
    R = euler_to_matrix(rpy[0], rpy[1], rpy[2])
    t = np.array(extrinsic['position']) * 1000.0
    p = R @ np.array(xyz_cam_mm, dtype=float) + t
    return p.tolist()


def pose_to_matrix(pose: Pose):
    R = euler_to_matrix(math.radians(pose.rx_deg), math.radians(pose.ry_deg), math.radians(pose.rz_deg))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [pose.x_mm, pose.y_mm, pose.z_mm]
    return T


def transform_points(T, xyz):
    p = np.ones(4)
    p[:3] = np.array(xyz, dtype=float)
    q = T @ p
    return q[:3].tolist()


def _extrinsic_matrix_from_rpy_position(extrinsic):
    rpy = extrinsic['rpy']
    R = euler_to_matrix(rpy[0], rpy[1], rpy[2])
    t = np.array(extrinsic['position'], dtype=float) * 1000.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def camera_to_base_eye_in_hand_candidates(xyz_cam_mm, ee_pose: Pose, extrinsic) -> List[Dict[str, object]]:
    """Return only the calibrated wrist-camera interpretation: camera_in_ee."""
    T_base_ee = pose_to_matrix(ee_pose)
    T_ee_cam = _extrinsic_matrix_from_rpy_position(extrinsic)
    T_base_cam = T_base_ee @ T_ee_cam
    return [{
        'mode': 'camera_in_ee',
        'xyz_base': transform_points(T_base_cam, xyz_cam_mm),
        'T_base_cam': T_base_cam,
    }]


def camera_to_base_eye_in_hand(xyz_cam_mm, ee_pose: Pose, extrinsic):
    return camera_to_base_eye_in_hand_candidates(xyz_cam_mm, ee_pose, extrinsic)[0]['xyz_base']


def _flatten_numeric_array(v: Optional[Iterable]):
    if v is None:
        return None
    if isinstance(v, (int, float, str)):
        try:
            return [float(v)]
        except Exception:
            return None
    out = []
    stack = [v]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, (list, tuple)):
            stack = list(cur) + stack
        else:
            try:
                out.append(float(cur))
            except Exception:
                pass
    return out or None


def _extract_intrinsics_from_array(arr, kind: str):
    arr = _flatten_numeric_array(arr)
    if not arr:
        return None
    if kind == 'k' and len(arr) >= 9:
        fx, fy, cx, cy = float(arr[0]), float(arr[4]), float(arr[2]), float(arr[5])
    elif kind == 'p' and len(arr) >= 12:
        fx, fy, cx, cy = float(arr[0]), float(arr[5]), float(arr[2]), float(arr[6])
    else:
        return None
    if abs(fx) > 1e-9 and abs(fy) > 1e-9:
        return fx, fy, cx, cy
    return None


def _camera_intrinsics(camera_info, stream: str = 'auto'):
    if camera_info is None:
        raise ValueError('camera_info is None')
    stream = (stream or 'auto').lower()
    key_orders = {
        'color': [('color_k', 'k'), ('color_p', 'p'), ('k', 'k'), ('p', 'p'), ('depth_k', 'k'), ('depth_p', 'p')],
        'depth': [('depth_k', 'k'), ('depth_p', 'p'), ('k', 'k'), ('p', 'p'), ('color_k', 'k'), ('color_p', 'p')],
        'auto': [('k', 'k'), ('p', 'p'), ('depth_k', 'k'), ('depth_p', 'p'), ('color_k', 'k'), ('color_p', 'p')],
    }
    tried = []
    for key, kind in key_orders.get(stream, key_orders['auto']):
        if key in camera_info:
            tried.append(key)
            intr = _extract_intrinsics_from_array(camera_info.get(key), kind)
            if intr is not None:
                return intr
    for key, kind in [('K', 'k'), ('P', 'p'), ('depth_K', 'k'), ('depth_P', 'p'), ('color_K', 'k'), ('color_P', 'p')]:
        if key in camera_info:
            tried.append(key)
            intr = _extract_intrinsics_from_array(camera_info.get(key), kind)
            if intr is not None:
                return intr
    summary = {k: (len(_flatten_numeric_array(v) or []), (_flatten_numeric_array(v) or [])[:6]) for k, v in camera_info.items() if isinstance(v, (list, tuple))}
    raise ValueError(f'camera_info missing valid intrinsics for stream={stream}: tried={tried}, summary={summary}, keys={list(camera_info.keys())}')


def pixel_to_camera_mm(u, v, z_mm, camera_info, stream: str = 'auto'):
    fx, fy, cx, cy = _camera_intrinsics(camera_info, stream=stream)
    x = (float(u) - cx) * float(z_mm) / fx
    y = (float(v) - cy) * float(z_mm) / fy
    return [x, y, float(z_mm)]


def rgb_uv_to_depth_uv(u_rgb, v_rgb, rgb_shape, depth_shape, mode: str = 'scale_uv'):
    hr, wr = rgb_shape[:2]
    hd, wd = depth_shape[:2]
    if mode == 'same_uv':
        u_d = int(round(u_rgb))
        v_d = int(round(v_rgb))
    else:
        u_d = int(round(u_rgb * (wd - 1) / max(1, wr - 1)))
        v_d = int(round(v_rgb * (hd - 1) / max(1, hr - 1)))
    u_d = max(0, min(u_d, wd - 1))
    v_d = max(0, min(v_d, hd - 1))
    return u_d, v_d


def make_pose_from_xyz(x_mm, y_mm, z_mm, tool_rpy_deg):
    rx, ry, rz = tool_rpy_deg
    return Pose(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, rx_deg=rx, ry_deg=ry, rz_deg=rz)


def offset_pose_xyz(pose: Pose, dx_mm: float = 0.0, dy_mm: float = 0.0, dz_mm: float = 0.0) -> Pose:
    return Pose(
        x_mm=pose.x_mm + dx_mm,
        y_mm=pose.y_mm + dy_mm,
        z_mm=pose.z_mm + dz_mm,
        rx_deg=pose.rx_deg,
        ry_deg=pose.ry_deg,
        rz_deg=pose.rz_deg,
    )
