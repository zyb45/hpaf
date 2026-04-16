from typing import Optional, Tuple
from pydantic import BaseModel


class LLMConfig(BaseModel):
    provider: str = "doubao"
    base_url: str
    api_key_env: str = "ARK_API_KEY"
    model: str
    max_output_tokens: int = 4096
    temperature: float = 0.2


class CameraConfig(BaseModel):
    source: str = "shared_dir"
    shared_dir: str = "./shared_scene"
    rgb_filename: str = "latest_color.png"
    depth_filename: str = "latest_depth.npy"
    camera_info_filename: str = "latest_camera_info.json"
    extrinsic_json: str = "./assets/cur.json"
    extrinsic_direction: Optional[str] = None
    freshness_max_age_s: float = 3.0


class MultiCameraConfig(BaseModel):
    primary_shared_dir: str = "./shared_scene/primary"
    secondary_shared_dir: str = "./shared_scene/secondary"
    rgb_filename: str = "latest_color.png"
    depth_filename: str = "latest_depth.npy"
    camera_info_filename: str = "latest_camera_info.json"
    primary_extrinsic_json: str = "./assets/eyeinhand.json"
    primary_extrinsic_direction: Optional[str] = None
    secondary_extrinsic_json: str = "./assets/cur.json"
    secondary_extrinsic_direction: Optional[str] = None
    freshness_max_age_s: float = 3.0


class RobotConfig(BaseModel):
    backend: str = "piper"
    can_name: str = "can0"
    speed_percent: int = 100
    move_mode: str = "0x00"
    pose_wait_s: float = 0.5
    grip_wait_s: float = 0.5
    review_require_yes: bool = True
    connect_on_start: bool = True
    pose_command_hz: int = 50
    pose_command_duration_s: float = 1.0
    gripper_command_hz: int = 100
    gripper_command_duration_s: float = 0.5
    gripper_max_width_mm: float = 50.0
    gripper_mm_to_ctrl: float = 1000.0
    gripper_open_force: int = 1000
    gripper_close_force: int = 1000


class PipelineConfig(BaseModel):
    max_verify_retries: int = 2
    logs_dir: str = "./logs"
    stop_on_failure: bool = True
    save_generated_scripts: bool = True


class VerifyConfig(BaseModel):
    mode: str = "human"  # ai / human
    require_yes_on_human_verify: bool = True


class APIRuntimeConfig(BaseModel):
    default_pregrasp_lift_mm: int = 60
    default_retreat_mm: int = 90
    default_close_force: int = 1000
    default_open_width_mm: float = 50.0
    default_place_drop_mm: int = 20
    default_tool_orientation_deg: Tuple[float, float, float] = (0.0, 85.0, 0.0)
    grasp_xyz_offset_mm: Tuple[float, float, float] = (0.0, -8.0, -18.0)
    place_xyz_offset_mm: Tuple[float, float, float] = (0.0, -8.0, -6.0)
    min_safe_z_mm: float = 120.0
    observe_pose_xyzrpy_mmdeg: Tuple[float, float, float, float, float, float] = (
        178.305,
        -8.032,
        207.138,
        -178.186,
        55.230,
        179.628,
    )


class PerceptionConfig(BaseModel):
    backend: str = "foundation_vision"  # foundation_vision / classic_cv / llm
    llm_fallback: bool = True
    debug: bool = True
    depth_window_radius: int = 5
    depth_uv_mapping_mode: str = "scale_uv"  # same_uv / scale_uv
    foundation_model_provider: str = "florence2"  # florence2 / grounding_dino / auto
    florence_model_id: str = "microsoft/Florence-2-large"
    grounding_dino_model_id: str = "IDEA-Research/grounding-dino-base"
    grounding_dino_repo_dir: str = "~/GroundingDINO"
    grounding_dino_ckpt_path: str = "~/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    use_sam_refine: bool = False
    device: str = "auto"


class AppConfig(BaseModel):
    llm: LLMConfig
    camera: Optional[CameraConfig] = None
    cameras: Optional[MultiCameraConfig] = None
    robot: RobotConfig
    pipeline: PipelineConfig
    verify: VerifyConfig = VerifyConfig()
    api_runtime: APIRuntimeConfig
    perception: PerceptionConfig = PerceptionConfig()
