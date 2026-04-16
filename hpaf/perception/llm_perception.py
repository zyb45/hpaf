from typing import Any, Dict
import cv2
import numpy as np
from hpaf.core.models import Detection
from hpaf.geometry.transforms import (
    pixel_to_camera_mm,
    camera_to_base,
    make_pose_from_xyz,
    offset_pose_xyz,
    rgb_uv_to_depth_uv,
)


class LLMPerceptionService:
    def __init__(
        self,
        vision_client,
        rgb_path: str,
        depth,
        camera_info: Dict[str, Any],
        extrinsic,
        tool_rpy_deg=(0.0, 85.0, 0.0),
        place_drop_mm=20,
        depth_window_radius=3,
        debug=True,
        grasp_xyz_offset_mm=(0.0, 0.0, 0.0),
        place_xyz_offset_mm=(0.0, 0.0, 0.0),
        depth_uv_mapping_mode: str = 'scale_uv',
    ):
        self.vision_client = vision_client
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self.extrinsic = extrinsic
        self.tool_rpy_deg = tool_rpy_deg
        self.place_drop_mm = place_drop_mm
        self.depth_window_radius = int(depth_window_radius)
        self.debug = debug
        self.grasp_xyz_offset_mm = tuple(grasp_xyz_offset_mm)
        self.place_xyz_offset_mm = tuple(place_xyz_offset_mm)
        self.depth_uv_mapping_mode = depth_uv_mapping_mode
        self._load_rgb()

    def _load_rgb(self):
        self.rgb = cv2.imread(self.rgb_path)
        if self.rgb is None:
            raise FileNotFoundError(f'Failed to read RGB image: {self.rgb_path}')
        self.rgb_h, self.rgb_w = self.rgb.shape[:2]
        self.image_width = self.rgb_w
        self.image_height = self.rgb_h

    def set_frame(self, rgb_path: str, depth, camera_info: Dict[str, Any]):
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self._load_rgb()

    def _clamp_bbox(self, bbox):
        if bbox is None:
            raise RuntimeError('Detector returned null bbox')
        x1, y1, x2, y2 = bbox
        w = max(1, self.image_width)
        h = max(1, self.image_height)
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w - 1))
        y2 = max(0, min(int(round(y2)), h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def detect_object_by_text(self, text_query: str) -> Detection:
        prompt = (
            'You are a visual object localizer. Return the bounding box that best matches the text description. Output JSON only.'
            f'Image size width={self.image_width}, height={self.image_height}.'
            'Output format: {"label":"...","bbox":[x1,y1,x2,y2],"score":0.0}.'
            'bbox must satisfy 0 <= x1 < x2 < width and 0 <= y1 < y2 < height.'
            'Do not output any extra text.'
            f'Text description: {text_query}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a visual object localizer.', prompt)
        bbox = self._clamp_bbox(data['bbox'])
        if self.debug:
            print(f'[Perception bbox raw] {data}')
            print(f'[Perception bbox clamped] {bbox}')
        return Detection(label=data.get('label', text_query), bbox=bbox, score=float(data.get('score', 1.0)), metadata=data)

    def _bbox_center_depth_mm(self, det: Detection) -> Dict[str, Any]:
        x1, y1, x2, y2 = det.bbox
        u_rgb = int(round((x1 + x2) / 2.0))
        v_rgb = int(round((y1 + y2) / 2.0))
        if self.depth is None:
            raise RuntimeError('Depth image is required for real execution pose estimation')
        h, w = self.depth.shape[:2]
        u, v = rgb_uv_to_depth_uv(
            u_rgb,
            v_rgb,
            (self.rgb_h, self.rgb_w),
            (h, w),
            mode=self.depth_uv_mapping_mode,
        )
        rad = max(1, self.depth_window_radius)
        patch = self.depth[max(0, v-rad):min(h, v+rad+1), max(0, u-rad):min(w, u+rad+1)]
        valid = patch[patch > 0]
        if valid.size == 0:
            raise RuntimeError('No valid depth around mapped bbox center')
        z = float(np.median(valid))
        if self.debug:
            print(f'[Depth lookup] rgb_uv=({u_rgb},{v_rgb}) -> depth_uv=({u},{v}), depth_shape={self.depth.shape}, z_raw={z}')
        z_mm = z * 1000.0 if z < 20 else z
        return {'u': u, 'v': v, 'u_cam': u_rgb, 'v_cam': v_rgb, 'z_mm': z_mm}

    def estimate_top_grasp_pose(self, det: Detection):
        info = self._bbox_center_depth_mm(det)
        xyz_cam = pixel_to_camera_mm(info.get('u_cam', info['u']), info.get('v_cam', info['v']), info['z_mm'], self.camera_info, stream='color')
        xyz_base = camera_to_base(xyz_cam, self.extrinsic)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2], self.tool_rpy_deg)
        return offset_pose_xyz(pose, *self.grasp_xyz_offset_mm)

    def estimate_place_pose(self, target_region: Detection):
        info = self._bbox_center_depth_mm(target_region)
        xyz_cam = pixel_to_camera_mm(info.get('u_cam', info['u']), info.get('v_cam', info['v']), info['z_mm'], self.camera_info, stream='color')
        xyz_base = camera_to_base(xyz_cam, self.extrinsic)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2] + self.place_drop_mm, self.tool_rpy_deg)
        return offset_pose_xyz(pose, *self.place_xyz_offset_mm)

    def verify_object_grasped(self, label: str) -> bool:
        prompt = (
            'Judge whether the target object has already been lifted by the robot gripper and is no longer on its original tabletop position. Return JSON only: '
            '{"grasped": true, "reason": "..."}。'
            f'Target object: {label}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a grasp-result verifier.', prompt)
        return bool(data.get('grasped', False))

    def verify_object_in_region(self, object_label: str, region_label: str) -> bool:
        prompt = (
            'Judge whether the target object is already inside the target region. Return JSON only: '
            '{"done": true, "reason": "..."}。'
            f'Target object: {object_label}; target region: {region_label}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a placement-result verifier.', prompt)
        return bool(data.get('done', False))
