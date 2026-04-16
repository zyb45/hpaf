from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from hpaf.core.models import Detection
from hpaf.geometry.transforms import (
    pixel_to_camera_mm,
    camera_to_base,
    camera_to_base_eye_in_hand,
    make_pose_from_xyz,
    offset_pose_xyz,
    rgb_uv_to_depth_uv,
)


class ClassicalTabletopPerceptionService:
    def __init__(
        self,
        rgb_path: str,
        depth,
        camera_info: Dict[str, Any],
        extrinsic,
        tool_rpy_deg=(0.0, 85.0, 0.0),
        place_drop_mm=20,
        depth_window_radius: int = 3,
        debug: bool = True,
        llm_fallback=None,
        grasp_xyz_offset_mm=(0.0, 0.0, 0.0),
        place_xyz_offset_mm=(0.0, 0.0, 0.0),
        depth_uv_mapping_mode: str = 'scale_uv',
        eye_in_hand: bool = True,
    ):
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self.extrinsic = extrinsic
        self.tool_rpy_deg = tuple(tool_rpy_deg)
        self.place_drop_mm = place_drop_mm
        self.depth_window_radius = int(depth_window_radius)
        self.debug = debug
        self.llm_fallback = llm_fallback
        self.grasp_xyz_offset_mm = tuple(grasp_xyz_offset_mm)
        self.place_xyz_offset_mm = tuple(place_xyz_offset_mm)
        self.depth_uv_mapping_mode = depth_uv_mapping_mode
        self.eye_in_hand = bool(eye_in_hand)
        self._load_rgb()

    def _load_rgb(self):
        self.rgb = cv2.imread(self.rgb_path)
        if self.rgb is None:
            raise FileNotFoundError(f'Failed to read RGB image: {self.rgb_path}')
        self.image_height, self.image_width = self.rgb.shape[:2]

    def set_frame(self, rgb_path: str, depth, camera_info: Dict[str, Any]):
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self._load_rgb()

    def _debug(self, *msg):
        if self.debug:
            print(*msg)

    def _parse_query(self, text_query: str):
        q = text_query.lower()
        color = None
        for c in ['red', 'blue', 'green', 'brown', '红', '蓝', '绿', '棕']:
            if c in q:
                color = c
                break
        shape = None
        if any(k in q for k in ['box', '盒', '收纳盒', '纸盒']):
            shape = 'box'
        elif any(k in q for k in ['圆柱', 'cylinder']):
            shape = 'cylinder'
        elif any(k in q for k in ['长方', 'cuboid', '长方体']):
            shape = 'cuboid'
        elif any(k in q for k in ['方块', '立方', 'cube', 'square']):
            shape = 'cube'
        return color, shape

    def _color_masks(self):
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)
        masks = {}
        m1 = cv2.inRange(hsv, (0, 80, 50), (12, 255, 255))
        m2 = cv2.inRange(hsv, (165, 80, 50), (180, 255, 255))
        masks['red'] = cv2.bitwise_or(m1, m2)
        masks['红'] = masks['red']
        masks['blue'] = cv2.inRange(hsv, (90, 70, 40), (135, 255, 255))
        masks['蓝'] = masks['blue']
        masks['green'] = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
        masks['绿'] = masks['green']
        masks['brown'] = cv2.inRange(hsv, (5, 40, 20), (30, 255, 200))
        masks['棕'] = masks['brown']
        for k, m in list(masks.items()):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            masks[k] = m
        return masks

    def _classify_contour(self, cnt) -> str:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        rect_area = max(1, w * h)
        fill_ratio = area / rect_area
        peri = max(cv2.arcLength(cnt, True), 1e-6)
        circularity = 4 * np.pi * area / (peri * peri)
        aspect = w / max(h, 1)
        if area > 10000:
            return 'box'
        if circularity > 0.72:
            return 'cylinder'
        if 0.75 <= aspect <= 1.33 and fill_ratio > 0.55:
            return 'cube'
        if aspect > 1.33 or aspect < 0.75:
            return 'cuboid'
        return 'cube'

    def _find_candidates(self, color_key: Optional[str]) -> List[Detection]:
        masks = self._color_masks()
        candidate_masks = []
        if color_key and color_key in masks:
            candidate_masks.append((color_key, masks[color_key]))
        else:
            for ck, m in masks.items():
                if ck in ['red', 'blue', 'green', 'brown']:
                    candidate_masks.append((ck, m))
        dets = []
        for ck, mask in candidate_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 120:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                shape = self._classify_contour(cnt)
                dets.append(Detection(label=f'{ck}:{shape}', bbox=[x, y, x + w, y + h], score=float(area), metadata={'color': ck, 'shape': shape, 'area': area}))
        return dets

    def detect_object_by_text(self, text_query: str) -> Detection:
        color, shape = self._parse_query(text_query)
        dets = self._find_candidates(color)
        scored = []
        color_map = {'红': 'red', '蓝': 'blue', '绿': 'green', '棕': 'brown'}
        for det in dets:
            score = det.score
            meta = det.metadata or {}
            x1, y1, x2, y2 = det.bbox
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            area = w * h
            aspect = w / max(h, 1)
            touches_edge = (x1 <= 2 or y1 <= 2 or x2 >= self.image_width - 2 or y2 >= self.image_height - 2)
            if color and meta.get('color') in [color, color_map.get(color, color)]:
                score += 1e6
            if shape and meta.get('shape') == shape:
                score += 5e5
            if shape == 'cube':
                score -= area * 2.0
                score -= abs(aspect - 1.0) * 3000.0
                if touches_edge:
                    score -= 2e5
                if area > 15000:
                    score -= 5e5
            elif shape == 'box':
                score += area * 0.5
                if touches_edge:
                    score += 1e4
            elif shape == 'cuboid':
                score += abs(aspect - 1.8) * 500.0
            scored.append((score, det))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            det = scored[0][1]
            self._debug('[ClassicPerception det]', text_query, det)
            return det
        if self.llm_fallback is not None:
            self._debug('[ClassicPerception fallback to LLM]', text_query)
            return self.llm_fallback.detect_object_by_text(text_query)
        raise RuntimeError(f'Classic perception failed to detect target: {text_query}')

    def _bbox_center_depth_mm(self, det: Detection) -> Dict[str, Any]:
        x1, y1, x2, y2 = det.bbox
        u_rgb = int(round((x1 + x2) / 2.0))
        v_rgb = int(round((y1 + y2) / 2.0))
        if self.depth is None:
            raise RuntimeError('Depth image is required for pose estimation')
        h, w = self.depth.shape[:2]
        u, v = rgb_uv_to_depth_uv(
            u_rgb,
            v_rgb,
            self.rgb.shape,
            self.depth.shape,
            mode=self.depth_uv_mapping_mode,
        )
        rad = max(1, self.depth_window_radius)
        valid = np.array([], dtype=self.depth.dtype)
        used_rad = rad
        for cur_rad in [rad, rad * 2, max(rad * 4, 12)]:
            patch = self.depth[max(0, v-cur_rad):min(h, v+cur_rad+1), max(0, u-cur_rad):min(w, u+cur_rad+1)]
            valid = patch[patch > 0]
            if valid.size > 0:
                used_rad = cur_rad
                break
        if valid.size == 0:
            raise RuntimeError('No valid depth around bbox center')
        z = float(np.median(valid))
        z_mm = z * 1000.0 if z < 20 else z
        self._debug(f'[Depth lookup] rgb_uv=({u_rgb},{v_rgb}) -> depth_uv=({u},{v}), depth_shape={self.depth.shape}, win={used_rad}, z_mm={z_mm:.1f}')
        return {'u': u, 'v': v, 'u_cam': u_rgb, 'v_cam': v_rgb, 'z_mm': z_mm, 'rgb_shape': self.rgb.shape[:2], 'depth_shape': self.depth.shape[:2]}

    def _base_xyz_from_detection(self, det: Detection, current_ee_pose=None):
        info = self._bbox_center_depth_mm(det)
        xyz_cam = pixel_to_camera_mm(info['u_cam'], info['v_cam'], info['z_mm'], self.camera_info, stream='color')
        if self.eye_in_hand:
            if current_ee_pose is None:
                raise RuntimeError('Eye-in-hand mode requires current_ee_pose')
            xyz_base = camera_to_base_eye_in_hand(xyz_cam, current_ee_pose, self.extrinsic)
        else:
            xyz_base = camera_to_base(xyz_cam, self.extrinsic)
        return xyz_base

    def estimate_top_grasp_pose(self, det: Detection, current_ee_pose=None):
        xyz_base = self._base_xyz_from_detection(det, current_ee_pose=current_ee_pose)
        tool_rpy = self.tool_rpy_deg
        if current_ee_pose is not None:
            tool_rpy = (current_ee_pose.rx_deg, current_ee_pose.ry_deg, current_ee_pose.rz_deg)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2], tool_rpy)
        pose = offset_pose_xyz(pose, *self.grasp_xyz_offset_mm)
        self._debug('[estimate_top_grasp_pose]', det.label, pose)
        return pose

    def estimate_place_pose(self, target_region: Detection, current_ee_pose=None):
        xyz_base = self._base_xyz_from_detection(target_region, current_ee_pose=current_ee_pose)
        tool_rpy = self.tool_rpy_deg
        if current_ee_pose is not None:
            tool_rpy = (current_ee_pose.rx_deg, current_ee_pose.ry_deg, current_ee_pose.rz_deg)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2] + self.place_drop_mm, tool_rpy)
        pose = offset_pose_xyz(pose, *self.place_xyz_offset_mm)
        self._debug('[estimate_place_pose]', target_region.label, pose)
        return pose

    def verify_object_grasped(self, label: str) -> bool:
        return False

    def verify_object_in_region(self, object_label: str, region_label: str) -> bool:
        return False
