import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


class SharedDirCamera:
    def __init__(self, shared_dir: str, rgb_filename: str, depth_filename: str, camera_info_filename: str, freshness_max_age_s: float = 3.0):
        self.shared_dir = os.path.abspath(os.path.expanduser(shared_dir))
        self.rgb_filename = rgb_filename
        self.depth_filename = depth_filename
        self.camera_info_filename = camera_info_filename
        self.freshness_max_age_s = float(freshness_max_age_s)
        self.rgb_path = os.path.join(self.shared_dir, rgb_filename)
        self.depth_path = os.path.join(self.shared_dir, depth_filename)
        self.camera_info_path = os.path.join(self.shared_dir, camera_info_filename)
        self.status_path = os.path.join(self.shared_dir, 'latest_status.json')
        self.snapshots_dir = os.path.join(self.shared_dir, '.snapshots')
        os.makedirs(self.snapshots_dir, exist_ok=True)

    def _candidate_rgb_paths(self):
        candidates = [self.rgb_path]
        stem, ext = os.path.splitext(self.rgb_path)
        if ext.lower() == '.jpg':
            candidates.append(stem + '.png')
        elif ext.lower() == '.png':
            candidates.append(stem + '.jpg')
        return candidates

    def _find_existing_rgb_path(self) -> str:
        for path in self._candidate_rgb_paths():
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'RGB image not found in {self.shared_dir}: tried {self._candidate_rgb_paths()}')

    def _wait_for_readable_file(self, path: str, timeout_s: float = 3.0, interval_s: float = 0.05) -> None:
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            if not os.path.exists(path):
                time.sleep(interval_s)
                continue
            try:
                if path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img = cv2.imread(path)
                    if img is not None and img.size > 0:
                        return
                    last_err = RuntimeError('cv2.imread returned None')
                elif path.lower().endswith('.npy'):
                    arr = np.load(path, allow_pickle=False)
                    if arr is not None:
                        return
                    last_err = RuntimeError('np.load returned None')
                elif path.lower().endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    return
                else:
                    with open(path, 'rb'):
                        pass
                    return
            except Exception as e:
                last_err = e
            time.sleep(interval_s)
        raise FileNotFoundError(f'File not readable after retries: {path}. last_error={last_err}')

    def _copy_snapshot_file(self, src: str) -> str:
        ext = ''.join(Path(src).suffixes) or Path(src).suffix or ''
        dst = os.path.join(self.snapshots_dir, f'{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}{ext}')
        shutil.copy2(src, dst)
        return dst

    def _cleanup_old_snapshots(self, keep: int = 30):
        try:
            files = sorted(Path(self.snapshots_dir).glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files[keep:]:
                try:
                    p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

    def _age_seconds(self, path: str) -> float:
        return time.time() - os.path.getmtime(path)

    def _require_fresh(self, path: str, what: str):
        if self.freshness_max_age_s <= 0:
            return
        age_s = self._age_seconds(path)
        if age_s > self.freshness_max_age_s:
            raise RuntimeError(
                f'{what} is stale: {path} age={age_s:.2f}s > freshness_max_age_s={self.freshness_max_age_s:.2f}s. '
                f'This usually means the ROS dump process is not writing into {self.shared_dir} or the camera topic is not updating.'
            )

    def _load_optional_json(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def snapshot(self) -> Dict[str, Any]:
        rgb_src = self._find_existing_rgb_path()
        self._wait_for_readable_file(rgb_src)
        self._require_fresh(rgb_src, 'RGB image')
        rgb_snapshot = self._copy_snapshot_file(rgb_src)

        depth = None
        depth_snapshot = None
        if os.path.exists(self.depth_path):
            self._wait_for_readable_file(self.depth_path)
            self._require_fresh(self.depth_path, 'Depth image')
            depth_snapshot = self._copy_snapshot_file(self.depth_path)
            depth = np.load(depth_snapshot, allow_pickle=False)

        camera_info: Dict[str, Any] = {}
        camera_info_snapshot = None
        if os.path.exists(self.camera_info_path):
            self._wait_for_readable_file(self.camera_info_path)
            self._require_fresh(self.camera_info_path, 'Camera info')
            camera_info_snapshot = self._copy_snapshot_file(self.camera_info_path)
            with open(camera_info_snapshot, 'r', encoding='utf-8') as f:
                camera_info = json.load(f)

        status = self._load_optional_json(self.status_path)
        self._cleanup_old_snapshots()
        return {
            'rgb_path': rgb_snapshot,
            'rgb_source_path': rgb_src,
            'rgb_source_mtime_ns': int(os.stat(rgb_src).st_mtime_ns),
            'rgb_age_s': self._age_seconds(rgb_src),
            'depth': depth,
            'depth_path': depth_snapshot,
            'depth_source_path': self.depth_path if os.path.exists(self.depth_path) else None,
            'depth_source_mtime_ns': int(os.stat(self.depth_path).st_mtime_ns) if os.path.exists(self.depth_path) else None,
            'camera_info': camera_info,
            'camera_info_path': camera_info_snapshot,
            'camera_info_source_path': self.camera_info_path if os.path.exists(self.camera_info_path) else None,
            'camera_info_source_mtime_ns': int(os.stat(self.camera_info_path).st_mtime_ns) if os.path.exists(self.camera_info_path) else None,
            'status': status,
            'shared_dir': self.shared_dir,
        }


class DualSharedDirCamera:
    def __init__(self, primary: SharedDirCamera, secondary: Optional[SharedDirCamera] = None):
        self.primary = primary
        self.secondary = secondary

    def snapshot(self) -> Dict[str, Any]:
        data = self.primary.snapshot()
        if self.secondary is not None:
            try:
                data['secondary'] = self.secondary.snapshot()
            except Exception:
                data['secondary'] = None
        else:
            data['secondary'] = None
        return data
