from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile

from hpaf.core.models import Detection
from hpaf.geometry.transforms import (
    camera_to_base,
    camera_to_base_eye_in_hand,
    make_pose_from_xyz,
    offset_pose_xyz,
    pixel_to_camera_mm,
    rgb_uv_to_depth_uv,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore', message='Importing from timm.models.layers is deprecated.*')
warnings.filterwarnings('ignore', message='torch.meshgrid: in an upcoming release.*')
warnings.filterwarnings('ignore', message='You are using `torch.load` with `weights_only=False`.*')
warnings.filterwarnings('ignore', message='The `device` argument is deprecated.*')
warnings.filterwarnings('ignore', message='torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly.*')
warnings.filterwarnings('ignore', message='None of the inputs have requires_grad=True.*')
warnings.filterwarnings('ignore', message=r'`torch.cuda.amp.autocast\(args\.\.\.\)` is deprecated.*')

@dataclass
class Candidate:
    bbox: List[int]
    score: float
    label: str
    source: str
    metadata: Dict[str, Any]


class FoundationVisionPerceptionService:
    """
    Stronger perception backend backed by pretrained open-vocabulary models.

    Main strategy:
      1) Use Florence-2 phrase grounding and GroundingDINO together.
      2) Heuristically filter/rerank candidates with size/depth/edge priors.
      3) For multi-object scenes, ask the VLM to choose among indexed candidates.
      4) If the scene is still ambiguous, fail safe instead of moving dangerously.
    """

    def __init__(
        self,
        vision_client,
        rgb_path: str,
        depth,
        camera_info: Dict[str, Any],
        extrinsic,
        tool_rpy_deg=(0.0, 85.0, 0.0),
        place_drop_mm=20,
        depth_window_radius: int = 5,
        debug: bool = True,
        grasp_xyz_offset_mm=(0.0, 0.0, 0.0),
        place_xyz_offset_mm=(0.0, 0.0, 0.0),
        depth_uv_mapping_mode: str = 'scale_uv',
        eye_in_hand: bool = True,
        model_provider: str = 'florence2',
        florence_model_id: str = 'microsoft/Florence-2-base',
        grounding_dino_model_id: str = 'IDEA-Research/grounding-dino-base',
        grounding_dino_repo_dir: str = '~/GroundingDINO',
        grounding_dino_ckpt_path: str = '~/GroundingDINO/weights/groundingdino_swint_ogc.pth',
        use_sam_refine: bool = False,
        device: str = 'auto',
    ):
        self.vision_client = vision_client
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self.extrinsic = extrinsic
        self.tool_rpy_deg = tuple(tool_rpy_deg)
        self.place_drop_mm = float(place_drop_mm)
        self.depth_window_radius = int(depth_window_radius)
        self.debug = bool(debug)
        self.grasp_xyz_offset_mm = tuple(grasp_xyz_offset_mm)
        self.place_xyz_offset_mm = tuple(place_xyz_offset_mm)
        self.depth_uv_mapping_mode = depth_uv_mapping_mode
        self.eye_in_hand = bool(eye_in_hand)
        self.model_provider = model_provider
        self.florence_model_id = florence_model_id
        self.grounding_dino_model_id = grounding_dino_model_id
        self.grounding_dino_repo_dir = os.path.expanduser(grounding_dino_repo_dir)
        self.grounding_dino_ckpt_path = os.path.expanduser(grounding_dino_ckpt_path)
        self.use_sam_refine = bool(use_sam_refine)
        self.device = device
        self._florence = None
        self._grounding_dino = None
        self._last_detection: Optional[Detection] = None
        self._artifact_dir: Optional[str] = None
        self._detect_counter: int = 0
        self._pose_counter: int = 0
        self._load_rgb()

    def _debug(self, *args):
        if self.debug:
            print(*args)

    def _safe_open_pil(self, path: str):
        with Image.open(path) as img:
            return img.convert('RGB').copy()

    def _load_rgb(self):
        self.rgb = cv2.imread(self.rgb_path)
        if self.rgb is None:
            pil_img = self._safe_open_pil(self.rgb_path)
            self.rgb = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if self.rgb is None:
            raise FileNotFoundError(f'Failed to read RGB image: {self.rgb_path}')
        self.image_height, self.image_width = self.rgb.shape[:2]

    def set_frame(self, rgb_path: str, depth, camera_info: Dict[str, Any]):
        self.rgb_path = rgb_path
        self.depth = depth
        self.camera_info = camera_info or {}
        self._load_rgb()
        self._last_detection = None

    def get_last_detection(self) -> Optional[Detection]:
        return self._last_detection

    def set_artifact_dir(self, path: str):
        self._artifact_dir = path
        if path:
            os.makedirs(path, exist_ok=True)

    def _artifact_path(self, name: str) -> Optional[str]:
        if not self._artifact_dir:
            return None
        os.makedirs(self._artifact_dir, exist_ok=True)
        return os.path.join(self._artifact_dir, name)

    def _save_json_artifact(self, name: str, data: Dict[str, Any]):
        path = self._artifact_path(name)
        if path is None:
            return None
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path


    def _source_display_name(self, source: str) -> str:
        mapping = {
            'florence2': 'Florence-2',
            'grounding_dino_local': 'GroundingDINO',
            'grounding_dino': 'GroundingDINO',
            'color_box_cv': 'ColorBoxCV',
            'vlm': 'VLM Fallback',
        }
        return mapping.get(source, source.replace('_', ' ').title())

    def _draw_label_box(self, image, text: str, x: int, y: int, color, font_scale: float = 0.58, thickness: int = 2):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = int(max(8, min(x, image.shape[1] - tw - 16)))
        y = int(max(th + 12, min(y, image.shape[0] - 8)))
        cv2.rectangle(image, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 4), (255, 255, 255), -1)
        cv2.rectangle(image, (x - 6, y - th - 8), (x + tw + 6, y + baseline + 4), color, 2)
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _bbox_iou(self, a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = [int(v) for v in a]
        bx1, by1, bx2, by2 = [int(v) for v in b]
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        area_a = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        area_b = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / max(area_a + area_b - inter, 1.0)

    def _save_candidate_grid(self, file_name: str, title: str, query: str, candidates: List[Candidate], selected_bbox=None, candidate_details: Optional[List[Dict[str, Any]]] = None, footer_lines: Optional[List[str]] = None):
        if not self._artifact_dir or not candidates:
            return None
        canvas = self.rgb.copy()
        color_map = [(47, 158, 255), (90, 200, 120), (255, 180, 0), (255, 110, 110), (180, 120, 255)]
        for idx, cand in enumerate(candidates[:5], start=1):
            x1, y1, x2, y2 = cand.bbox
            is_selected = selected_bbox is not None and self._bbox_iou(cand.bbox, selected_bbox) > 0.92
            color = (0, 220, 120) if is_selected else color_map[(idx - 1) % len(color_map)]
            thickness = 4 if is_selected else 3
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            detail = candidate_details[idx - 1] if candidate_details and idx - 1 < len(candidate_details) else {}
            src = detail.get('source_display') or self._source_display_name(cand.source)
            raw = detail.get('raw_score', cand.score)
            final_score = detail.get('final_score')
            depth_mm = detail.get('z_mm')
            line1 = f'#{idx} {src}'
            suffix = ' SELECTED' if is_selected else ''
            line2 = f'raw={raw:.3f}'
            if final_score is not None:
                line2 += f' final={final_score:.1f}'
            if depth_mm is not None:
                line2 += f' z={depth_mm:.0f}mm'
            line2 += suffix
            self._draw_label_box(canvas, line1, x1 + 6, max(26, y1 - 28), color, font_scale=0.56, thickness=2)
            self._draw_label_box(canvas, line2, x1 + 6, max(48, y1 - 6), color, font_scale=0.50, thickness=2)
        self._draw_label_box(canvas, f'{title} | Query: {query}', 14, 28, (35, 35, 35), font_scale=0.64, thickness=2)
        if footer_lines:
            y = self.image_height - 14 - 22 * (len(footer_lines) - 1)
            for line in footer_lines:
                self._draw_label_box(canvas, line, 14, y, (35, 35, 35), font_scale=0.52, thickness=2)
                y += 22
        out = self._artifact_path(file_name)
        if out:
            cv2.imwrite(out, canvas)
        return out

    def _save_final_selection_artifacts(self, text_query: str, all_candidates: List[Candidate], det: Detection):
        if not self._artifact_dir:
            return
        ranked_details = list(det.metadata.get('ranked_candidates', []) or [])
        florence = [c for c in all_candidates if c.source == 'florence2']
        gdino = [c for c in all_candidates if 'grounding_dino' in c.source]
        florence_details = [d for d in ranked_details if d.get('source') == 'florence2']
        gdino_details = [d for d in ranked_details if 'grounding_dino' in str(d.get('source', ''))]
        florence_name = f'florence_review_{self._detect_counter:02d}.png'
        gdino_name = f'groundingdino_review_{self._detect_counter:02d}.png'
        final_name = f'final_selection_{self._detect_counter:02d}.png'
        crop_name = f'final_object_crop_{self._detect_counter:02d}.png'
        footer_lines = []
        if det.metadata.get('selection_strategy'):
            footer_lines.append(f"selection={det.metadata.get('selection_strategy')}")
        if det.metadata.get('selection_reason'):
            footer_lines.append(f"reason={det.metadata.get('selection_reason')}")
        self._save_candidate_grid(florence_name, 'Florence-2 candidates', text_query, florence, selected_bbox=det.bbox, candidate_details=florence_details)
        self._save_candidate_grid(gdino_name, 'GroundingDINO candidates', text_query, gdino, selected_bbox=det.bbox, candidate_details=gdino_details)
        final_candidates = []
        final_details = []
        detail_map = {}
        for d in ranked_details:
            key = tuple(int(v) for v in d.get('bbox', []))
            detail_map[key] = d
        for cand in all_candidates:
            key = tuple(int(v) for v in cand.bbox)
            if key in detail_map and all(tuple(int(v) for v in c.bbox) != key for c in final_candidates):
                final_candidates.append(cand)
                final_details.append(detail_map[key])
        if not final_candidates:
            final_candidates = [Candidate(det.bbox, det.score, det.label, det.metadata.get('source', 'final'), det.metadata)]
            final_details = [
                {
                    'bbox': det.bbox,
                    'source': det.metadata.get('source', 'final'),
                    'source_display': self._source_display_name(det.metadata.get('source', 'final')),
                    'raw_score': float(det.metadata.get('raw_score', det.score)),
                    'final_score': float(det.score),
                    'z_mm': float(det.metadata.get('z_mm', 0.0)),
                }
            ]
        self._save_candidate_grid(final_name, 'Final selection', text_query, final_candidates[:5], selected_bbox=det.bbox, candidate_details=final_details[:5], footer_lines=footer_lines)
        x1, y1, x2, y2 = det.bbox
        pad = 16
        crop = self.rgb[max(0, y1-pad):min(self.image_height, y2+pad), max(0, x1-pad):min(self.image_width, x2+pad)].copy()
        crop_path = self._artifact_path(crop_name)
        if crop_path and crop.size > 0:
            cv2.imwrite(crop_path, crop)
        self._save_json_artifact(f'final_selection_{self._detect_counter:02d}.json', {
            'query': text_query,
            'selected_label': det.label,
            'selected_source': det.metadata.get('source'),
            'selected_depth_mm': det.metadata.get('z_mm'),
            'selected_bbox': det.bbox,
            'selection_strategy': det.metadata.get('selection_strategy'),
            'selection_reason': det.metadata.get('selection_reason'),
            'review_choice': det.metadata.get('review_choice'),
            'ranked_candidates': ranked_details,
            'images': {
                'florence': florence_name,
                'groundingdino': gdino_name,
                'final': final_name,
                'crop': crop_name,
            },
        })

    def _save_pose_debug_artifact(self, det: Detection, pose_name: str, info: Dict[str, Any], xyz_cam, xyz_base, current_ee_pose=None):
        if not self._artifact_dir:
            return
        self._pose_counter += 1
        canvas = self.rgb.copy()
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 120), 3)
        u = int(info['u_cam'])
        v = int(info['v_cam'])
        cv2.circle(canvas, (u, v), 6, (255, 255, 255), -1)
        cv2.circle(canvas, (u, v), 10, (35, 35, 35), 2)
        dist_base = float(np.linalg.norm(np.array(xyz_base, dtype=float)))
        lines = [
            f'{pose_name}: {det.label}',
            f'Camera depth: {float(info["z_mm"]):.1f} mm',
            f'Base XYZ: ({float(xyz_base[0]):.1f}, {float(xyz_base[1]):.1f}, {float(xyz_base[2]):.1f}) mm',
            f'Base distance: {dist_base:.1f} mm',
        ]
        if current_ee_pose is not None:
            ee_xyz = np.array([current_ee_pose.x_mm, current_ee_pose.y_mm, current_ee_pose.z_mm], dtype=float)
            dist_ee = float(np.linalg.norm(np.array(xyz_base, dtype=float) - ee_xyz))
            lines.append(f'EE distance: {dist_ee:.1f} mm')
        lines.append(f'Transform: {info.get("transform_mode", "unknown")}')
        for idx, line in enumerate(lines):
            self._draw_label_box(canvas, line, 14, 28 + idx * 30, (35, 35, 35), font_scale=0.6, thickness=2)
        img_name = f'{pose_name.lower()}_{self._pose_counter:02d}.png'
        img_path = self._artifact_path(img_name)
        if img_path:
            cv2.imwrite(img_path, canvas)
        self._save_json_artifact(f'{pose_name.lower()}_{self._pose_counter:02d}.json', {
            'pose_name': pose_name,
            'label': det.label,
            'bbox': det.bbox,
            'depth_mm': info['z_mm'],
            'pixel_rgb': [info['u_cam'], info['v_cam']],
            'pixel_depth': [info['u'], info['v']],
            'xyz_camera_mm': [float(v) for v in xyz_cam],
            'xyz_base_mm': [float(v) for v in xyz_base],
            'transform_mode': info.get('transform_mode'),
            'image': img_name,
        })


    def _normalize_bbox(self, bbox: List[float]) -> List[int]:
        x1, y1, x2, y2 = bbox
        x1 = int(round(max(0, min(self.image_width - 1, x1))))
        y1 = int(round(max(0, min(self.image_height - 1, y1))))
        x2 = int(round(max(0, min(self.image_width - 1, x2))))
        y2 = int(round(max(0, min(self.image_height - 1, y2))))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]

    def _bbox_iou(self, a: List[int], b: List[int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / max(1, area_a + area_b - inter)

    def _dedupe_candidates(self, candidates: List[Candidate], iou_threshold: float = 0.72) -> List[Candidate]:
        out: List[Candidate] = []
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            if any(self._bbox_iou(cand.bbox, keep.bbox) >= iou_threshold for keep in out):
                continue
            out.append(cand)
        return out

    def _load_florence(self):
        if self._florence is not None:
            return self._florence
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as e:
            self._debug('[FoundationVision] Florence import unavailable:', e)
            self._florence = False
            return None
        try:
            device = 'cuda' if self.device == 'auto' and torch.cuda.is_available() else 'cpu'
            dtype = torch.float16 if device == 'cuda' else torch.float32
            processor = AutoProcessor.from_pretrained(
                self.florence_model_id,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.florence_model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
                local_files_only=True,
            ).to(device)
            self._florence = {'torch': torch, 'processor': processor, 'model': model, 'device': device, 'dtype': dtype}
            self._debug('[FoundationVision] loaded Florence-2', self.florence_model_id, 'on', device)
            return self._florence
        except Exception as e:
            self._debug('[FoundationVision] Florence load failed:', e)
            self._florence = False
            return None

    def _load_grounding_dino(self):
        if self._grounding_dino is not None:
            return self._grounding_dino
        try:
            import torch
            device = 'cuda' if self.device == 'auto' and torch.cuda.is_available() else 'cpu'
        except Exception as e:
            self._debug('[FoundationVision] GroundingDINO torch unavailable:', e)
            self._grounding_dino = False
            return None

        try:
            from groundingdino.util.inference import load_model
            cfg = os.path.join(self.grounding_dino_repo_dir, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
            ckpt = self.grounding_dino_ckpt_path
            if os.path.exists(cfg) and os.path.exists(ckpt):
                model = load_model(cfg, ckpt)
                if hasattr(model, 'to'):
                    model = model.to(device)
                self._grounding_dino = {'mode': 'local_repo', 'torch': torch, 'model': model, 'device': device}
                self._debug('[FoundationVision] loaded local GroundingDINO repo model')
                return self._grounding_dino
        except Exception as e:
            self._debug('[FoundationVision] local GroundingDINO load failed:', e)

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id, local_files_only=True)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_dino_model_id, local_files_only=True).to(device)
            self._grounding_dino = {'mode': 'hf', 'torch': torch, 'processor': processor, 'model': model, 'device': device}
            self._debug('[FoundationVision] loaded HF GroundingDINO', self.grounding_dino_model_id, 'on', device)
            return self._grounding_dino
        except Exception as e:
            self._debug('[FoundationVision] GroundingDINO load failed:', e)
            self._grounding_dino = False
            return None

    def _run_florence_phrase_grounding(self, text_query: str) -> List[Candidate]:
        mod = self._load_florence()
        if not mod:
            return []
        try:
            image = self._safe_open_pil(self.rgb_path)
            task = '<CAPTION_TO_PHRASE_GROUNDING>'
            prompt = f'{task}{text_query}'
            inputs = mod['processor'](text=prompt, images=image, return_tensors='pt')
            for k, v in list(inputs.items()):
                if hasattr(v, 'to'):
                    if hasattr(v, 'is_floating_point') and v.is_floating_point():
                        inputs[k] = v.to(mod['device'], dtype=mod['dtype'])
                    else:
                        inputs[k] = v.to(mod['device'])
            with mod['torch'].no_grad():
                generated_ids = mod['model'].generate(**inputs, max_new_tokens=256, num_beams=3)
            generated_text = mod['processor'].batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = mod['processor'].post_process_generation(generated_text, task=task, image_size=(self.image_width, self.image_height))
            raw_bboxes = parsed.get(task, {}).get('bboxes', []) if isinstance(parsed, dict) else []
            raw_labels = parsed.get(task, {}).get('labels', []) if isinstance(parsed, dict) else []
            out = []
            for idx, bbox in enumerate(raw_bboxes):
                out.append(Candidate(
                    bbox=self._normalize_bbox(bbox),
                    score=max(0.05, 1.0 - idx * 0.05),
                    label=str(raw_labels[idx]) if idx < len(raw_labels) else text_query,
                    source='florence2',
                    metadata={'query': text_query},
                ))
            self._debug('[FoundationVision] Florence candidates:', out)
            return out
        except Exception as e:
            self._debug('[FoundationVision] Florence inference failed:', e)
            return []

    def _run_grounding_dino(self, text_query: str) -> List[Candidate]:
        mod = self._load_grounding_dino()
        if not mod:
            return []
        try:
            if mod.get('mode') == 'local_repo':
                from groundingdino.util.inference import load_image, predict
                _image_source, image = load_image(self.rgb_path)
                caption = text_query if text_query.strip().endswith('.') else f'{text_query.strip()} .'
                boxes, logits, phrases = predict(
                    model=mod['model'],
                    image=image,
                    caption=caption,
                    box_threshold=0.22,
                    text_threshold=0.18,
                    device=mod['device'],
                )
                out = []
                for box, score, label in zip(boxes, logits, phrases):
                    box = box.detach().cpu().numpy().tolist() if hasattr(box, 'detach') else list(box)
                    cx, cy, w, h = box
                    x1 = (cx - w / 2.0) * self.image_width
                    y1 = (cy - h / 2.0) * self.image_height
                    x2 = (cx + w / 2.0) * self.image_width
                    y2 = (cy + h / 2.0) * self.image_height
                    out.append(Candidate(
                        bbox=self._normalize_bbox([x1, y1, x2, y2]),
                        score=float(score),
                        label=str(label),
                        source='grounding_dino_local',
                        metadata={'query': text_query},
                    ))
                self._debug('[FoundationVision] GroundingDINO local candidates:', out)
                return out

            image = self._safe_open_pil(self.rgb_path)
            text = text_query if text_query.strip().endswith('.') else f'{text_query.strip()}.'
            inputs = mod['processor'](images=image, text=text, return_tensors='pt').to(mod['device'])
            with mod['torch'].no_grad():
                outputs = mod['model'](**inputs)
            results = mod['processor'].post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.20,
                text_threshold=0.18,
                target_sizes=[image.size[::-1]],
            )[0]
            out = []
            for box, score, label in zip(results.get('boxes', []), results.get('scores', []), results.get('labels', [])):
                bbox = box.detach().cpu().tolist() if hasattr(box, 'detach') else list(box)
                out.append(Candidate(
                    bbox=self._normalize_bbox(bbox),
                    score=float(score),
                    label=str(label),
                    source='grounding_dino_hf',
                    metadata={'query': text_query},
                ))
            self._debug('[FoundationVision] GroundingDINO HF candidates:', out)
            return out
        except Exception as e:
            self._debug('[FoundationVision] GroundingDINO inference failed:', e)
            return []

    def _run_vlm_fallback(self, text_query: str) -> List[Candidate]:
        prompt = (
            'You are a robot visual localizer. Find the object that best matches the text description. Use category, shape, size, and relative position instead of color alone. '
            'Return JSON only: {"label":"...","bbox":[x1,y1,x2,y2],"score":0.0,"reason":"..."}. '
            f'Image size: width={self.image_width}, height={self.image_height}. Text description: {text_query}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a robot visual localizer.', prompt)
        return [Candidate(
            bbox=self._normalize_bbox(data['bbox']),
            score=float(data.get('score', 0.5)),
            label=data.get('label', text_query),
            source='vlm_bbox',
            metadata=data,
        )]

    def _depth_stats_for_bbox(self, bbox: List[int]) -> Dict[str, float]:
        x1, y1, x2, y2 = bbox
        if self.depth is None:
            return {'z_mm': 1000.0, 'median_z_mm': 1000.0}
        u_rgb = int(round((x1 + x2) / 2.0))
        v_rgb = int(round((y1 + y2) / 2.0))
        h, w = self.depth.shape[:2]
        u, v = rgb_uv_to_depth_uv(u_rgb, v_rgb, self.rgb.shape, self.depth.shape, mode=self.depth_uv_mapping_mode)
        rad = max(1, self.depth_window_radius)
        patch = self.depth[max(0, v-rad):min(h, v+rad+1), max(0, u-rad):min(w, u+rad+1)]
        valid = patch[patch > 0]
        if valid.size == 0:
            z_mm = 1000.0
        else:
            z = float(np.median(valid))
            z_mm = z * 1000.0 if z < 20 else z
        return {'z_mm': z_mm, 'median_z_mm': z_mm}

    def _query_shape_bias(self, text_query: str) -> Dict[str, Any]:
        t = text_query.lower()
        return {
            'want_small': any(k in text_query for k in ['小', '小块', '小方块', '小立方体']) or 'small' in t,
            'want_large': any(k in text_query for k in ['大', '盒', '箱']) or 'large' in t,
            'want_cube': any(k in text_query for k in ['立方', '方块', 'cube', 'square']) or 'cube' in t or 'square' in t,
            'want_rect_prism': any(k in text_query for k in ['长方体', '长方块', '矩形', 'rectangular prism', 'rectangular block', 'cuboid', 'prism']) or 'rectangular' in t or 'cuboid' in t or 'prism' in t,
            'want_box': any(k in text_query for k in ['盒', '箱', '容器']) or 'box' in t or 'container' in t,
        }

    def _query_color_bias(self, text_query: str) -> List[str]:
        t = str(text_query).lower()
        colors: List[str] = []
        mapping = {
            'red': ['red', '红'],
            'green': ['green', '绿'],
            'blue': ['blue', '蓝'],
            'brown': ['brown', '棕', '褐', '牛皮'],
            'yellow': ['yellow', '黄'],
            'white': ['white', '白'],
            'black': ['black', '黑'],
        }
        for name, keys in mapping.items():
            if any(k in t for k in keys):
                colors.append(name)
        return colors


    def _query_material_bias(self, text_query: str) -> List[str]:
        t = str(text_query).lower()
        mats = []
        mapping = {
            'metal': ['metal', 'iron', 'steel', 'tin', '金属', '铁'],
            'paper': ['paper', 'cardboard', 'carton', '纸', '纸盒', '纸箱'],
            'plastic': ['plastic', '塑料'],
        }
        for name, keys in mapping.items():
            if any(k in t for k in keys):
                mats.append(name)
        return mats

    def _run_color_box_detector(self, text_query: str) -> List[Candidate]:
        t = str(text_query).lower()
        if not any(k in t for k in ['box', 'container', 'bin', 'tray', 'basket', 'drawer', '盒', '箱', '容器']):
            return []
        query_colors = self._query_color_bias(text_query)
        if not query_colors:
            return []
        hsv = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        masks = []
        for c in query_colors:
            if c == 'red':
                mask = (((h <= 12) | (h >= 170)) & (s > 55) & (v > 35)).astype(np.uint8) * 255
            elif c == 'brown':
                mask = (((h >= 5) & (h <= 25)) & (s > 45) & (v >= 25) & (v <= 190)).astype(np.uint8) * 255
            elif c == 'blue':
                mask = (((h >= 95) & (h <= 135)) & (s > 50) & (v > 35)).astype(np.uint8) * 255
            elif c == 'green':
                mask = (((h >= 35) & (h <= 90)) & (s > 50) & (v > 35)).astype(np.uint8) * 255
            elif c == 'yellow':
                mask = (((h >= 15) & (h <= 35)) & (s > 50) & (v > 60)).astype(np.uint8) * 255
            else:
                continue
            masks.append(mask)
        if not masks:
            return []
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        img_area = float(self.image_width * self.image_height)
        for cnt in contours:
            x, y, w, h2 = cv2.boundingRect(cnt)
            area = float(w * h2)
            if area < 0.01 * img_area:
                continue
            bbox = [int(x), int(y), int(x + w), int(y + h2)]
            crop = self._crop_from_bbox(bbox)
            if crop is None:
                continue
            profile = self._candidate_color_profile(Candidate(bbox, 1.0, text_query, 'color_box_cv', {}))
            match = max(float(profile.get(c, 0.0)) for c in query_colors)
            if match < 0.08:
                continue
            out.append(Candidate(
                bbox=bbox,
                score=min(0.99, 0.55 + match),
                label=text_query,
                source='color_box_cv',
                metadata={'query': text_query, 'color_match': match, 'profile': profile},
            ))
        out.sort(key=lambda c: c.score, reverse=True)
        if out:
            self._debug(f'[FoundationVision] color-box candidates: {out}')
        return out[:3]

    def _crop_from_bbox(self, bbox: List[int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(self.image_width - 1, x1))
        x2 = max(x1 + 1, min(self.image_width, x2))
        y1 = max(0, min(self.image_height - 1, y1))
        y2 = max(y1 + 1, min(self.image_height, y2))
        crop = self.rgb[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop

    def _candidate_color_profile(self, cand: Candidate) -> Dict[str, float]:
        crop = self._crop_from_bbox(cand.bbox)
        if crop is None:
            return {}
        h0, w0 = crop.shape[:2]
        if h0 >= 12 and w0 >= 12:
            y1 = int(round(h0 * 0.15))
            y2 = int(round(h0 * 0.85))
            x1 = int(round(w0 * 0.15))
            x2 = int(round(w0 * 0.85))
            crop = crop[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h = hsv[:, :, 0].astype(np.uint8)
        s = hsv[:, :, 1].astype(np.uint8)
        v = hsv[:, :, 2].astype(np.uint8)
        colorful = s > 40
        dark = v < 70
        bright = v > 180
        total = float(h.size)
        def ratio(mask):
            return float(mask.mean()) if mask.size else 0.0
        red = ratio((((h <= 10) | (h >= 170)) & (s > 60) & (v > 40)).astype(np.uint8))
        green = ratio(((h >= 35) & (h <= 90) & (s > 50) & (v > 35)).astype(np.uint8))
        blue = ratio(((h >= 95) & (h <= 135) & (s > 50) & (v > 35)).astype(np.uint8))
        yellow = ratio(((h >= 15) & (h <= 35) & (s > 50) & (v > 60)).astype(np.uint8))
        brown = ratio(((h >= 5) & (h <= 25) & (s > 45) & (v >= 40) & (v <= 180)).astype(np.uint8))
        white = ratio(((s < 35) & bright).astype(np.uint8))
        black = ratio(((s < 60) & dark).astype(np.uint8))
        colorful_ratio = ratio(colorful.astype(np.uint8))
        return {
            'red': red,
            'green': green,
            'blue': blue,
            'yellow': yellow,
            'brown': brown,
            'white': white,
            'black': black,
            'colorful': colorful_ratio,
        }

    def _candidate_features(self, cand: Candidate) -> Dict[str, float]:
        x1, y1, x2, y2 = cand.bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = float(w * h)
        area_ratio = area / max(1.0, float(self.image_width * self.image_height))
        aspect = float(w) / max(float(h), 1.0)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = (cx - self.image_width * 0.5) / max(self.image_width * 0.5, 1.0)
        dy = (cy - self.image_height * 0.5) / max(self.image_height * 0.5, 1.0)
        center_dist = float(np.sqrt(dx * dx + dy * dy))
        edge_touch = 1.0 if (x1 <= 3 or y1 <= 3 or x2 >= self.image_width - 4 or y2 >= self.image_height - 4) else 0.0
        z_mm = self._depth_stats_for_bbox(cand.bbox)['z_mm']
        color_profile = self._candidate_color_profile(cand)
        return {
            'w': w,
            'h': h,
            'area': area,
            'area_ratio': area_ratio,
            'aspect': aspect,
            'center_dist': center_dist,
            'edge_touch': edge_touch,
            'z_mm': z_mm,
            'color_profile': color_profile,
        }

    def _heuristic_score(self, text_query: str, cand: Candidate) -> Tuple[float, Dict[str, float]]:
        feat = self._candidate_features(cand)
        shape_bias = self._query_shape_bias(text_query)
        score = float(cand.score) * 1000.0
        score += min(feat['area'], 50000.0) * 0.018
        score += max(0.0, 1200.0 - feat['z_mm']) * 0.06
        score -= feat['center_dist'] * 70.0
        score -= feat['edge_touch'] * 120.0
        # strong priors for multi-object tabletop scenes
        if feat['area_ratio'] > 0.55:
            score -= 400.0
        if feat['z_mm'] > 1500.0:
            score -= 180.0
        if feat['z_mm'] < 40.0:
            score -= 180.0
        if shape_bias['want_small']:
            score -= feat['area'] * 0.045
            if feat['area_ratio'] > 0.18:
                score -= 250.0
        if shape_bias['want_large']:
            score += feat['area'] * 0.03
        if shape_bias['want_cube']:
            score -= abs(feat['aspect'] - 1.0) * 180.0
            if feat['aspect'] > 1.45:
                score -= 240.0
        if shape_bias['want_rect_prism']:
            score += min(abs(feat['aspect'] - 1.9), 1.2) * 220.0
            if feat['aspect'] < 1.25:
                score -= 320.0
            if feat['area_ratio'] < 0.008:
                score -= 160.0
        if shape_bias['want_box']:
            score += abs(feat['aspect'] - 1.35) * 12.0 + feat['area'] * 0.008
            if feat['area_ratio'] < 0.01:
                score -= 120.0
        query_colors = self._query_color_bias(text_query)
        if query_colors:
            color_profile = feat.get('color_profile', {}) or {}
            match_strength = max(float(color_profile.get(c, 0.0)) for c in query_colors)
            mismatch_strength = max(
                [float(v) for k, v in color_profile.items() if k not in query_colors and k in {'red', 'green', 'blue', 'yellow', 'brown'}] + [0.0]
            )
            score += match_strength * 1350.0
            score -= mismatch_strength * 420.0
            if match_strength < 0.06:
                score -= 260.0
            if mismatch_strength > max(0.12, match_strength * 1.2):
                score -= 260.0
        query_mats = self._query_material_bias(text_query)
        if query_mats:
            cp = feat.get('color_profile', {}) or {}
            if 'paper' in query_mats:
                score += float(cp.get('brown', 0.0)) * 260.0
            if 'metal' in query_mats:
                score += float(cp.get('red', 0.0)) * 160.0 + float(cp.get('white', 0.0)) * 60.0
                if float(cp.get('brown', 0.0)) > 0.25 and float(cp.get('red', 0.0)) < 0.25:
                    score -= 180.0
        label_l = str(cand.label or '').lower()
        if shape_bias['want_cube'] and any(k in label_l for k in ['rect', 'prism', 'cuboid']):
            score -= 260.0
        if shape_bias['want_rect_prism'] and any(k in label_l for k in ['cube', 'square']):
            score -= 220.0
        if shape_bias['want_rect_prism'] and any(k in label_l for k in ['rect', 'prism', 'cuboid']):
            score += 180.0
        if shape_bias['want_cube'] and any(k in label_l for k in ['cube', 'square']):
            score += 160.0
        if cand.source == 'color_box_cv':
            score += 280.0
        return score, feat

    def _candidate_review_image(self, text_query: str, ranked: List[Tuple[float, Candidate, Dict[str, float]]], max_items: int = 5) -> Optional[str]:
        if not ranked:
            return None
        canvas = self.rgb.copy()
        for idx, (final_score, cand, feat) in enumerate(ranked[:max_items], start=1):
            x1, y1, x2, y2 = cand.bbox
            color = (0, 255, 0) if idx == 1 else (0, 200, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            line1 = f'#{idx} {self._source_display_name(cand.source)}'
            line2 = f'raw={cand.score:.3f} final={final_score:.1f} z={int(feat["z_mm"])}'
            self._draw_label_box(canvas, line1, x1 + 4, max(28, y1 - 26), color, font_scale=0.52, thickness=2)
            self._draw_label_box(canvas, line2, x1 + 4, max(48, y1 - 4), color, font_scale=0.48, thickness=2)
        header = f'Query: {text_query}'
        self._draw_label_box(canvas, header, 10, 24, (255, 80, 80), font_scale=0.62, thickness=2)
        path = self._artifact_path(f'candidate_review_{self._detect_counter:02d}.png')
        if path is None:
            fd, path = tempfile.mkstemp(prefix='hpaf_review_', suffix='.png')
            os.close(fd)
        cv2.imwrite(path, canvas)
        return path

    def _vlm_choose_candidate(self, text_query: str, ranked: List[Tuple[float, Candidate, Dict[str, float]]]) -> Optional[int]:
        if self.vision_client is None or len(ranked) <= 1:
            return 0 if ranked else None
        review_path = self._candidate_review_image(text_query, ranked)
        if review_path is None:
            return 0 if ranked else None
        lines = []
        for idx, (_, cand, feat) in enumerate(ranked[:5], start=1):
            lines.append(
                f'{idx}. label={cand.label}, bbox={cand.bbox}, z_mm={feat["z_mm"]:.1f}, area_ratio={feat["area_ratio"]:.3f}, source={cand.source}'
            )
        prompt = (
            'You are a conservative robot grasp-safety reviewer. Candidate objects are marked with numbered boxes.\n'
            'Choose the candidate that best matches the target description using object category, shape, size, and relative position.\n'
            'If the scene is ambiguous or you are not confident, return ambiguous=true.\n'
            'When the target description explicitly specifies a color, treat color as an important cue together with category and shape.\n'
            'Return strict JSON only: {"choice": 1, "ambiguous": false, "confidence": 0.0, "reason": "..."}\n'
            f'Target description: {text_query}\nCandidate list:\n' + '\n'.join(lines)
        )
        try:
            data = self.vision_client.ask_json_with_image(review_path, 'You are a conservative robot visual reviewer.', prompt)
            self._debug('[FoundationVision] VLM candidate review:', data)
            if bool(data.get('ambiguous', False)):
                return None
            choice = int(data.get('choice', 0))
            confidence = float(data.get('confidence', 0.0))
            if choice < 1 or choice > min(5, len(ranked)):
                return None
            if confidence < 0.45:
                return None
            return choice - 1
        except Exception as e:
            self._debug('[FoundationVision] VLM candidate review failed:', e)
            return 0 if ranked else None
        finally:
            try:
                if not (self._artifact_dir and os.path.abspath(review_path).startswith(os.path.abspath(self._artifact_dir))):
                    os.remove(review_path)
            except Exception:
                pass


    def _save_detection_visualization(self, text_query: str, candidates: List[Candidate], det: Detection):
        self._save_final_selection_artifacts(text_query, candidates, det)
        if self.debug and self._artifact_dir:
            self._debug(
                f'[FoundationVision] review images saved: florence_review_{self._detect_counter:02d}.png, ' 
                f'groundingdino_review_{self._detect_counter:02d}.png, final_selection_{self._detect_counter:02d}.png, ' 
                f'final_object_crop_{self._detect_counter:02d}.png'
            )

    def _rerank(self, text_query: str, candidates: List[Candidate]) -> Detection:
        if not candidates:
            raise RuntimeError(f'No candidates for query: {text_query}')
        candidates = self._dedupe_candidates(candidates)
        ranked: List[Tuple[float, Candidate, Dict[str, float]]] = []
        for cand in candidates:
            score, feat = self._heuristic_score(text_query, cand)
            ranked.append((score, cand, feat))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked_debug = [
            {
                'idx': i + 1,
                'source': c.source,
                'source_display': self._source_display_name(c.source),
                'label': c.label or text_query,
                'bbox': list(c.bbox),
                'raw_score': round(float(c.score), 4),
                'final_score': round(float(s), 2),
                'depth_mm': round(float(f['z_mm']), 1),
                'z_mm': round(float(f['z_mm']), 1),
                'area_ratio': round(float(f['area_ratio']), 4),
                'aspect': round(float(f['aspect']), 3),
            }
            for i, (s, c, f) in enumerate(ranked[:5])
        ]
        self._debug('[FoundationVision] candidate summary:', ranked_debug[:4])

        selection_strategy = 'heuristic'
        selection_reason = 'top heuristic score'
        review_choice = None
        review_override = None
        if len(ranked) > 1:
            top_gap = ranked[0][0] - ranked[1][0]
            explicit_color = bool(self._query_color_bias(text_query))
            need_review = top_gap < 140.0 or explicit_color
            if need_review:
                choice = self._vlm_choose_candidate(text_query, ranked[:5])
                review_choice = None if choice is None else int(choice + 1)
                if choice is None:
                    best_score, best, best_feat = ranked[0]
                    selection_strategy = 'heuristic_after_review_abstain'
                    selection_reason = 'reviewer abstained or confidence too low; kept top heuristic candidate'
                else:
                    best_score, best, best_feat = ranked[choice]
                    selection_strategy = 'vlm_review'
                    selection_reason = f'reviewer chose candidate #{choice + 1}'
                    if best_feat['z_mm'] >= 900.0:
                        reviewed_bbox = list(best.bbox)
                        reviewed_source = str(best.source)
                        best_iou_alt = None
                        for alt_score, alt_cand, alt_feat in ranked:
                            same_region = self._bbox_iou(alt_cand.bbox, reviewed_bbox) >= 0.85
                            same_source_family = (('grounding_dino' in reviewed_source and 'grounding_dino' in alt_cand.source) or (reviewed_source == alt_cand.source))
                            if alt_feat['z_mm'] < 900.0 and same_region and same_source_family:
                                best_iou_alt = (alt_score, alt_cand, alt_feat)
                                break
                        if best_iou_alt is None:
                            for alt_score, alt_cand, alt_feat in ranked:
                                same_region = self._bbox_iou(alt_cand.bbox, reviewed_bbox) >= 0.85
                                if alt_feat['z_mm'] < 900.0 and same_region:
                                    best_iou_alt = (alt_score, alt_cand, alt_feat)
                                    break
                        if best_iou_alt is not None:
                            best_score, best, best_feat = best_iou_alt
                            review_override = 'same-region depth-stable alternative'
                            selection_strategy = 'vlm_review_with_same_region_depth_override'
                            selection_reason = f'reviewer chose candidate #{choice + 1}, but depth was unstable; switched only within same bbox region'
                    best_score += 180.0
            else:
                best_score, best, best_feat = ranked[0]
        else:
            best_score, best, best_feat = ranked[0]

        if best_feat['area_ratio'] > 0.72:
            raise RuntimeError(f'Detected region too large for safe grasp: {best.bbox}')
        if best_feat['z_mm'] > 1800.0:
            raise RuntimeError(f'Detected target too far / unstable depth for safe grasp: z={best_feat["z_mm"]:.1f}mm')

        det = Detection(label=best.label or text_query, bbox=best.bbox, score=float(best_score), metadata={
            'source': best.source,
            'query': text_query,
            'raw_score': best.score,
            'z_mm': best_feat['z_mm'],
            'area_ratio': best_feat['area_ratio'],
            'selection_strategy': selection_strategy,
            'selection_reason': selection_reason,
            'review_choice': review_choice,
            'review_override': review_override,
            'ranked_candidates': ranked_debug,
            **(best.metadata or {}),
        })
        self._debug(f"[FoundationVision] selected object: label={det.label}, source={self._source_display_name(det.metadata.get('source', 'unknown'))}, depth_mm={float(det.metadata.get('z_mm', 0.0)):.1f}")
        return det

    def detect_object_by_text(self, text_query: str) -> Detection:
        self._detect_counter += 1
        candidates: List[Candidate] = []
        # Combine two strong detectors in all normal modes. Prefer over single-backend reliance.
        if self.model_provider in ('florence2', 'auto', 'grounding_dino'):
            candidates.extend(self._run_color_box_detector(text_query))
            candidates.extend(self._run_florence_phrase_grounding(text_query))
            candidates.extend(self._run_grounding_dino(text_query))
        if not candidates:
            candidates.extend(self._run_vlm_fallback(text_query))
        det = self._rerank(text_query, candidates)
        self._last_detection = det
        self._save_detection_visualization(text_query, candidates, det)
        return det

    def _depth_info_from_rgb_uv(self, u_rgb: int, v_rgb: int, tag: str = 'sample') -> Dict[str, Any]:
        if self.depth is None:
            raise RuntimeError('Depth image is required for pose estimation')
        h, w = self.depth.shape[:2]
        u, v = rgb_uv_to_depth_uv(u_rgb, v_rgb, self.rgb.shape, self.depth.shape, mode=self.depth_uv_mapping_mode)
        rad = max(1, self.depth_window_radius)
        valid = np.array([], dtype=self.depth.dtype)
        for cur_rad in [rad, rad * 2, max(rad * 4, 12)]:
            patch = self.depth[max(0, v-cur_rad):min(h, v+cur_rad+1), max(0, u-cur_rad):min(w, u+cur_rad+1)]
            valid = patch[patch > 0]
            if valid.size > 0:
                rad = cur_rad
                break
        if valid.size == 0:
            raise RuntimeError(f'No valid depth around rgb_uv=({u_rgb},{v_rgb})')
        z = float(np.median(valid))
        z_mm = z * 1000.0 if z < 20 else z
        self._debug(f'[FoundationVision] depth sample ({tag}): depth_mm={z_mm:.1f}, depth_window={rad}, rgb_uv=({u_rgb},{v_rgb})')
        return {'u': u, 'v': v, 'u_cam': u_rgb, 'v_cam': v_rgb, 'z_mm': z_mm, 'tag': tag}

    def _workspace_violation(self, xyz_base: List[float]) -> float:
        """Workspace checks are intentionally disabled.

        The user wants to trust the calibrated camera_in_ee transform result and
        execute the resulting Cartesian xyz directly, even when the platform is
        lower than before. Reachability should therefore be handled by the arm
        controller / IK fallback layer rather than by perception-side clipping.
        """
        return 0.0

    def _base_xyz_from_depth_info(self, info: Dict[str, Any], current_ee_pose=None, validate: bool = True):
        xyz_cam = pixel_to_camera_mm(info['u_cam'], info['v_cam'], info['z_mm'], self.camera_info, stream='color')
        chosen_mode = 'eye_to_hand'
        if self.eye_in_hand:
            if current_ee_pose is None:
                raise RuntimeError('Eye-in-hand mode requires current_ee_pose')
            # Project policy-wide convention: only use the calibrated camera_in_ee
            # interpretation for the wrist camera. Never silently switch to
            # ee_in_camera based on heuristics.
            self.extrinsic['transform_direction'] = 'camera_in_ee'
            xyz_base = camera_to_base_eye_in_hand(xyz_cam, current_ee_pose, self.extrinsic)
            chosen_mode = 'camera_in_ee'
            self._debug(
                f'[FoundationVision] transform mode: {chosen_mode}; '
                f'xyz_base={[float(v) for v in xyz_base]}'
            )
        else:
            xyz_base = camera_to_base(xyz_cam, self.extrinsic)
        violation = self._workspace_violation(xyz_base)
        info['transform_mode'] = chosen_mode
        info['workspace_violation_mm'] = violation
        if validate:
            self._validate_workspace_xyz(xyz_base)
        return xyz_base, xyz_cam, info

    def _bbox_center_depth_mm(self, det: Detection) -> Dict[str, Any]:
        x1, y1, x2, y2 = det.bbox
        u_rgb = int(round((x1 + x2) / 2.0))
        v_rgb = int(round((y1 + y2) / 2.0))
        return self._depth_info_from_rgb_uv(u_rgb, v_rgb, tag='center')

    def _bbox_depth_from_anchor_specs(self, det: Detection, specs: List[Tuple[str, float, float]], choose: str = 'min_z') -> Dict[str, Any]:
        x1, y1, x2, y2 = det.bbox
        infos = []
        for name, rx, ry in specs:
            u_rgb = int(round(x1 + rx * max(1, x2 - x1)))
            v_rgb = int(round(y1 + ry * max(1, y2 - y1)))
            u_rgb = int(min(max(u_rgb, x1 + 1), x2 - 1))
            v_rgb = int(min(max(v_rgb, y1 + 1), y2 - 1))
            try:
                infos.append(self._depth_info_from_rgb_uv(u_rgb, v_rgb, tag=name))
            except Exception:
                continue
        if not infos:
            return self._bbox_center_depth_mm(det)
        if choose == 'median':
            infos.sort(key=lambda d: d['z_mm'])
            return infos[len(infos) // 2]
        if choose == 'max_z':
            return max(infos, key=lambda d: d['z_mm'])
        return min(infos, key=lambda d: d['z_mm'])

    def _grasp_depth_info(self, det: Detection) -> Dict[str, Any]:
        # Prefer upper / top-surface samples. For tabletop objects, the smallest valid depth in the upper
        # half of the bbox usually corresponds to the visible top surface and is much more stable than bbox center.
        specs = [
            ('top_center', 0.50, 0.28),
            ('top_left', 0.34, 0.30),
            ('top_right', 0.66, 0.30),
            ('upper_center', 0.50, 0.40),
            ('center', 0.50, 0.50),
        ]
        return self._bbox_depth_from_anchor_specs(det, specs, choose='min_z')

    def _container_opening_depth_info(self, det: Detection, anchor_uv: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        x1, y1, x2, y2 = det.bbox
        specs = [
            ('opening_upper_center', 0.50, 0.18),
            ('opening_upper_left', 0.30, 0.20),
            ('opening_upper_right', 0.70, 0.20),
            ('opening_mid_center', 0.50, 0.30),
            ('opening_mid_left', 0.32, 0.32),
            ('opening_mid_right', 0.68, 0.32),
        ]
        infos = []
        if anchor_uv is not None:
            try:
                infos.append(self._depth_info_from_rgb_uv(int(anchor_uv[0]), int(anchor_uv[1]), tag='container_anchor'))
            except Exception:
                pass
        for name, rx, ry in specs:
            u_rgb = int(round(x1 + rx * max(1, x2 - x1)))
            v_rgb = int(round(y1 + ry * max(1, y2 - y1)))
            u_rgb = int(min(max(u_rgb, x1 + 1), x2 - 1))
            v_rgb = int(min(max(v_rgb, y1 + 1), y2 - 1))
            try:
                infos.append(self._depth_info_from_rgb_uv(u_rgb, v_rgb, tag=name))
            except Exception:
                continue
        if not infos:
            if anchor_uv is not None:
                return self._depth_info_from_rgb_uv(int(anchor_uv[0]), int(anchor_uv[1]), tag='container_anchor_fallback')
            return self._bbox_center_depth_mm(det)
        # Prefer the nearest valid depth around the opening/rim rather than the box bottom.
        return min(infos, key=lambda d: d['z_mm'])

    def _validate_workspace_xyz(self, xyz_base: List[float]):
        # Intentionally no hard workspace rejection here. Cartesian xyz should be
        # trusted and passed to execution, where position-priority IK fallback can
        # attempt to realize the pose even when some orientations are infeasible.
        return None

    def _base_xyz_from_detection(self, det: Detection, current_ee_pose=None, validate: bool = True):
        info = self._grasp_depth_info(det)
        return self._base_xyz_from_depth_info(info, current_ee_pose=current_ee_pose, validate=validate)

    def estimate_gripper_width_mm(self, det: Detection) -> float:
        x1, y1, x2, y2 = det.bbox
        px_w = max(1.0, float(x2 - x1))
        info = self._grasp_depth_info(det)
        fx = float(self.camera_info.get('fx') or self.camera_info.get('color_fx') or self.camera_info.get('K', [0, 0, 0, 0])[0] or 600.0)
        est_mm = (px_w * info['z_mm']) / max(fx, 1e-6)
        # open slightly larger to improve tolerance; clamp to realistic PiPER range.
        est_mm = float(np.clip(est_mm * 0.78 + 8.0, 25.0, 50.0))
        self._debug(f'[FoundationVision] estimated gripper width {est_mm:.1f} mm for bbox={det.bbox}')
        return est_mm

    def estimate_top_grasp_pose(self, det: Detection, current_ee_pose=None):
        xyz_base, xyz_cam, info = self._base_xyz_from_detection(det, current_ee_pose=current_ee_pose)
        tool_rpy = self.tool_rpy_deg if current_ee_pose is None else (current_ee_pose.rx_deg, current_ee_pose.ry_deg, current_ee_pose.rz_deg)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2], tool_rpy)
        pose = offset_pose_xyz(pose, *self.grasp_xyz_offset_mm)
        self._save_pose_debug_artifact(det, 'grasp_pose', info, xyz_cam, xyz_base, current_ee_pose=current_ee_pose)
        self._debug(f'[FoundationVision] grasp pose ready: base_xyz_mm=({float(xyz_base[0]):.1f}, {float(xyz_base[1]):.1f}, {float(xyz_base[2]):.1f}), transform={info.get("transform_mode")}')
        return pose

    def _looks_like_open_container(self, label: str) -> bool:
        t = str(label).lower()
        return any(k in t for k in ['box', 'container', 'bin', 'tray', 'basket', 'drawer', '盒', '箱', '容器', '框'])

    def _estimate_container_inner_anchor(self, det: Detection) -> Tuple[int, int, Dict[str, Any]]:
        x1, y1, x2, y2 = det.bbox
        crop = self._crop_from_bbox(det.bbox)
        if crop is None:
            return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)), {'method': 'bbox_center'}
        h, w = crop.shape[:2]
        if h < 12 or w < 12:
            return int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)), {'method': 'bbox_center_small'}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 35, 110)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        blocked = edges.copy()
        border = max(3, int(round(min(h, w) * 0.06)))
        blocked[:border, :] = 255
        blocked[-border:, :] = 255
        blocked[:, :border] = 255
        blocked[:, -border:] = 255
        free = cv2.bitwise_not(blocked)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        max_val = float(dist.max()) if dist.size else 0.0
        if max_val < 4.0:
            u = int(round(x1 + 0.5 * (x2 - x1)))
            v = int(round(y1 + 0.56 * (y2 - y1)))
            return u, v, {'method': 'fallback_center', 'radius_px': max_val}
        _, _, _, max_loc = cv2.minMaxLoc(dist)
        u = int(x1 + max_loc[0])
        v = int(y1 + max_loc[1])
        meta = {'method': 'distance_transform', 'radius_px': max_val}
        debug_path = self._artifact_path(f'container_inner_{self._pose_counter + 1:02d}.png')
        if debug_path:
            vis = crop.copy()
            cv2.circle(vis, (max_loc[0], max_loc[1]), 6, (0, 255, 0), -1)
            cv2.imwrite(debug_path, vis)
            meta['image'] = os.path.basename(debug_path)
        return u, v, meta

    def estimate_place_pose(self, target_region: Detection, current_ee_pose=None):
        x1, y1, x2, y2 = target_region.bbox
        is_container = self._looks_like_open_container(target_region.label)
        anchor_specs = []
        if is_container:
            inner_u, inner_v, inner_meta = self._estimate_container_inner_anchor(target_region)
            anchor_specs.append(('container_inner_center', inner_u, inner_v, 0.0))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            anchor_specs.extend([
                ('container_inner_up', inner_u, int(round(inner_v - 0.08 * h)), 1.0),
                ('container_inner_down', inner_u, int(round(inner_v + 0.08 * h)), 1.2),
                ('container_inner_left', int(round(inner_u - 0.08 * w)), inner_v, 1.2),
                ('container_inner_right', int(round(inner_u + 0.08 * w)), inner_v, 1.2),
                ('container_bbox_center', int(round((x1 + x2) / 2.0)), int(round(y1 + 0.56 * (y2 - y1))), 1.6),
            ])
        else:
            anchor_specs.extend([
                ('region_center', int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)), 0.0),
                ('region_upper_center', int(round((x1 + x2) / 2.0)), int(round(y1 + 0.42 * (y2 - y1))), 0.8),
                ('region_lower_center', int(round((x1 + x2) / 2.0)), int(round(y1 + 0.62 * (y2 - y1))), 1.0),
            ])
        candidates = []
        for name, u_rgb, v_rgb, center_cost in anchor_specs:
            u_rgb = int(min(max(u_rgb, x1 + 1), x2 - 1))
            v_rgb = int(min(max(v_rgb, y1 + 1), y2 - 1))
            try:
                if is_container:
                    info = self._container_opening_depth_info(target_region, anchor_uv=(u_rgb, v_rgb))
                    info['u_cam'] = u_rgb
                    info['v_cam'] = v_rgb
                else:
                    info = self._depth_info_from_rgb_uv(u_rgb, v_rgb, tag=name)
                xyz_base, xyz_cam, info = self._base_xyz_from_depth_info(info, current_ee_pose=current_ee_pose, validate=False)
                violation = self._workspace_violation(xyz_base)
                inside = violation <= 1e-6
                z_pref = abs(float(xyz_base[2]) - 25.0)
                candidates.append({
                    'name': name,
                    'info': info,
                    'xyz_base': xyz_base,
                    'xyz_cam': xyz_cam,
                    'inside': inside,
                    'violation': violation,
                    'center_cost': float(center_cost),
                    'z_pref': float(z_pref),
                })
            except Exception as e:
                self._debug(f'[FoundationVision] place anchor {name} failed: {e}')
        if not candidates:
            raise RuntimeError('Failed to estimate any placement anchor inside the target region')
        valid = [c for c in candidates if c['inside']]
        if valid:
            chosen = sorted(valid, key=lambda c: (c['center_cost'], c['z_pref']))[0]
        else:
            chosen = sorted(candidates, key=lambda c: (c['violation'], c['center_cost'], c['z_pref']))[0]
            chosen['inside'] = True
        chosen['info']['placement_anchor'] = chosen['name']
        if is_container:
            chosen['info']['container_inner_method'] = inner_meta.get('method')
            if 'image' in inner_meta:
                chosen['info']['container_inner_image'] = inner_meta['image']
        self._debug('[FoundationVision] placement anchors: ' + str([
            {
                'name': c['name'],
                'xyz_base': [round(float(v), 1) for v in c['xyz_base']],
                'inside': c['inside'],
                'violation_mm': round(float(c['violation']), 1),
                'center_cost': round(float(c['center_cost']), 2),
            }
            for c in candidates
        ]))
        xyz_base, xyz_cam, info = chosen['xyz_base'], chosen['xyz_cam'], chosen['info']
        if is_container:
            # For open containers, depth often lands on the bottom interior and yields unsafe negative/biased Z.
            # Keep XY from the opening anchor but snap Z to a safe near-table placement height.
            xyz_base = [float(xyz_base[0]), float(xyz_base[1]), max(10.0, min(35.0, float(xyz_base[2])))]
            info['container_safe_place_z_mm'] = float(xyz_base[2])
        tool_rpy = self.tool_rpy_deg if current_ee_pose is None else (current_ee_pose.rx_deg, current_ee_pose.ry_deg, current_ee_pose.rz_deg)
        pose = make_pose_from_xyz(xyz_base[0], xyz_base[1], xyz_base[2] + self.place_drop_mm, tool_rpy)
        pose = offset_pose_xyz(pose, *self.place_xyz_offset_mm)
        self._save_pose_debug_artifact(target_region, 'place_pose', info, xyz_cam, xyz_base, current_ee_pose=current_ee_pose)
        self._debug(f'[FoundationVision] place pose ready: anchor={chosen["name"]}, base_xyz_mm=({float(xyz_base[0]):.1f}, {float(xyz_base[1]):.1f}, {float(xyz_base[2]):.1f}), transform={info.get("transform_mode")}')
        return pose

    def verify_object_grasped(self, label: str) -> bool:
        prompt = (
            'Judge whether the target object has already been lifted by the robot gripper and is no longer resting at its original tabletop position. '
            'Return JSON only: {"grasped": true, "reason": "..."}. '
            f'Target object: {label}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a grasp-result verifier.', prompt)
        return bool(data.get('grasped', False))

    def verify_object_in_region(self, object_label: str, region_label: str) -> bool:
        prompt = (
            'Judge whether the target object is already inside the target region. '
            'Return JSON only: {"done": true, "reason": "..."}. '
            f'Target object: {object_label}; target region: {region_label}'
        )
        data = self.vision_client.ask_json_with_image(self.rgb_path, 'You are a placement-result verifier.', prompt)
        return bool(data.get('done', False))
