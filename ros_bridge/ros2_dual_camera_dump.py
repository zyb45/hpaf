#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class DualCameraDump(Node):
    def __init__(self):
        super().__init__('hpaf_dual_camera_dump')
        self.bridge = CvBridge()
        default_root = Path(__file__).resolve().parents[1] / 'shared_scene'
        self.shared_root = os.path.abspath(os.path.expanduser(os.environ.get('HPAF_SHARED_ROOT', str(default_root))))
        self.primary_ns = os.environ.get('HPAF_PRIMARY_NS', '/camera_gemini')
        self.secondary_ns = os.environ.get('HPAF_SECONDARY_NS', '/camera_astra')
        self.primary_dir = os.path.join(self.shared_root, 'primary')
        self.secondary_dir = os.path.join(self.shared_root, 'secondary')
        self.rgb_ext = os.environ.get('HPAF_RGB_EXT', 'png').lower().strip('.')
        self.dump_max_fps = float(os.environ.get('HPAF_DUMP_MAX_FPS', '5.0'))
        self._last_write = {}
        os.makedirs(self.primary_dir, exist_ok=True)
        os.makedirs(self.secondary_dir, exist_ok=True)
        self._setup_camera(self.primary_ns, self.primary_dir, 'primary')
        self._setup_camera(self.secondary_ns, self.secondary_dir, 'secondary')
        self.get_logger().info(f'shared_root={self.shared_root}, rgb_ext={self.rgb_ext}, dump_max_fps={self.dump_max_fps}')

    def _should_write(self, key: str) -> bool:
        now = time.monotonic()
        last = self._last_write.get(key, 0.0)
        min_interval = 0.0 if self.dump_max_fps <= 0 else (1.0 / self.dump_max_fps)
        if now - last < min_interval:
            return False
        self._last_write[key] = now
        return True

    def _atomic_write_bytes(self, final_path: str, data: bytes):
        tmp_path = f'{final_path}.tmp'
        with open(tmp_path, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)

    def _atomic_write_json(self, final_path: str, data):
        tmp_path = f'{final_path}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)

    def _atomic_write_npy(self, final_path: str, arr):
        tmp_path = f'{final_path}.tmp.npy'
        np.save(tmp_path, arr)
        os.replace(tmp_path, final_path)

    def _atomic_write_image(self, final_path: str, img):
        ext = Path(final_path).suffix.lower()
        if ext == '.png':
            ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f'Failed to encode image for {final_path}')
        self._atomic_write_bytes(final_path, buf.tobytes())

    def _load_json(self, path: str):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _update_status(self, out_dir: str, **kwargs):
        path = os.path.join(out_dir, 'latest_status.json')
        data = self._load_json(path)
        data.update(kwargs)
        data['shared_dir'] = out_dir
        data['writer_pid'] = os.getpid()
        data['writer_wall_time_ns'] = int(time.time_ns())
        self._atomic_write_json(path, data)

    def _msg_stamp_ns(self, msg) -> int:
        try:
            return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        except Exception:
            return 0

    def _setup_camera(self, ns, out_dir, tag):
        qos = qos_profile_sensor_data
        self.create_subscription(Image, f'{ns}/color/image_raw', lambda msg, d=out_dir, t=tag, n=ns: self.on_rgb(msg, d, t, n), qos)
        self.create_subscription(Image, f'{ns}/depth/image_raw', lambda msg, d=out_dir, t=tag, n=ns: self.on_depth(msg, d, t, n), qos)
        self.create_subscription(CameraInfo, f'{ns}/color/camera_info', lambda msg, d=out_dir, t=tag, n=ns: self.on_color_info(msg, d, t, n), qos)
        self.create_subscription(CameraInfo, f'{ns}/depth/camera_info', lambda msg, d=out_dir, t=tag, n=ns: self.on_depth_info(msg, d, t, n), qos)
        self.get_logger().info(f'subscribed {tag}: {ns}')

    def _save_info(self, out_dir, field_name, msg, ns):
        path = os.path.join(out_dir, 'latest_camera_info.json')
        data = self._load_json(path)
        data.update({
            'width': data.get('width', msg.width),
            'height': data.get('height', msg.height),
            f'{field_name}_width': int(msg.width),
            f'{field_name}_height': int(msg.height),
            f'{field_name}_k': [float(x) for x in msg.k],
            f'{field_name}_d': [float(x) for x in msg.d],
            f'{field_name}_p': [float(x) for x in msg.p],
            f'{field_name}_distortion_model': msg.distortion_model,
            f'{field_name}_frame_id': msg.header.frame_id,
            f'{field_name}_stamp_ns': self._msg_stamp_ns(msg),
            'timestamp_ns': int(time.time_ns()),
            'camera_ns': ns,
        })
        if field_name == 'depth':
            data['k'] = [float(x) for x in msg.k]
            data['d'] = [float(x) for x in msg.d]
            data['p'] = [float(x) for x in msg.p]
            data['distortion_model'] = msg.distortion_model
            data['width'] = msg.width
            data['height'] = msg.height
        elif 'k' not in data:
            data['k'] = [float(x) for x in msg.k]
            data['d'] = [float(x) for x in msg.d]
            data['p'] = [float(x) for x in msg.p]
            data['distortion_model'] = msg.distortion_model
            data['width'] = msg.width
            data['height'] = msg.height
        self._atomic_write_json(path, data)
        self._update_status(out_dir, camera_info_path=path, camera_info_mtime_ns=int(os.stat(path).st_mtime_ns), camera_ns=ns)

    def on_rgb(self, msg, out_dir, tag, ns):
        if not self._should_write(f'{tag}:rgb'):
            return
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        path = os.path.join(out_dir, f'latest_color.{self.rgb_ext}')
        self._atomic_write_image(path, img)
        self._update_status(
            out_dir,
            rgb_path=path,
            rgb_mtime_ns=int(os.stat(path).st_mtime_ns),
            rgb_shape=list(img.shape),
            rgb_encoding=msg.encoding,
            rgb_frame_id=msg.header.frame_id,
            rgb_stamp_ns=self._msg_stamp_ns(msg),
            camera_ns=ns,
        )

    def on_depth(self, msg, out_dir, tag, ns):
        if not self._should_write(f'{tag}:depth'):
            return
        depth = self.bridge.imgmsg_to_cv2(msg)
        path = os.path.join(out_dir, 'latest_depth.npy')
        self._atomic_write_npy(path, depth)
        self._update_status(
            out_dir,
            depth_path=path,
            depth_mtime_ns=int(os.stat(path).st_mtime_ns),
            depth_shape=list(depth.shape),
            depth_dtype=str(depth.dtype),
            depth_encoding=msg.encoding,
            depth_frame_id=msg.header.frame_id,
            depth_stamp_ns=self._msg_stamp_ns(msg),
            camera_ns=ns,
        )

    def on_color_info(self, msg, out_dir, tag, ns):
        self._save_info(out_dir, 'color', msg, ns)

    def on_depth_info(self, msg, out_dir, tag, ns):
        self._save_info(out_dir, 'depth', msg, ns)


def main():
    rclpy.init()
    node = DualCameraDump()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
