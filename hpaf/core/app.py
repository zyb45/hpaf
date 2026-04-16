from hpaf.core.io import load_config
from hpaf.llm.factory import make_vision_client
from hpaf.agents.task_agent import TaskAgent
from hpaf.agents.program_agent import ProgramAgent
from hpaf.agents.verify_agent import VerifyAgent
from hpaf.camera.shared_dir_camera import SharedDirCamera, DualSharedDirCamera
from hpaf.geometry.transforms import load_extrinsic
from hpaf.perception.llm_perception import LLMPerceptionService
from hpaf.perception.classic_cv_perception import ClassicalTabletopPerceptionService
from hpaf.perception.foundation_vision_perception import FoundationVisionPerceptionService
from hpaf.robot.piper_backend import PiperArm
from hpaf.robot.dummy_backend import DummyArm
from hpaf.api.runtime_api import RuntimeAPI
from hpaf.pipeline.orchestrator import PipelineOrchestrator
from hpaf.core.models import Pose


class HPAFSystem:
    def __init__(self, cfg, orchestrator):
        self.cfg = cfg
        self.orchestrator = orchestrator

    @classmethod
    def build(cls, config_path: str, connect_robot: bool = True):
        cfg = load_config(config_path)
        vision_client = make_vision_client(cfg.llm)

        if cfg.cameras is not None:
            primary_cam = SharedDirCamera(
                shared_dir=cfg.cameras.primary_shared_dir,
                rgb_filename=cfg.cameras.rgb_filename,
                depth_filename=cfg.cameras.depth_filename,
                camera_info_filename=cfg.cameras.camera_info_filename,
                freshness_max_age_s=cfg.cameras.freshness_max_age_s,
            )
            secondary_cam = SharedDirCamera(
                shared_dir=cfg.cameras.secondary_shared_dir,
                rgb_filename=cfg.cameras.rgb_filename,
                depth_filename=cfg.cameras.depth_filename,
                camera_info_filename=cfg.cameras.camera_info_filename,
                freshness_max_age_s=cfg.cameras.freshness_max_age_s,
            )
            camera = DualSharedDirCamera(primary_cam, secondary_cam)
            extrinsic_primary = load_extrinsic(cfg.cameras.primary_extrinsic_json)
            extrinsic_primary['transform_direction'] = 'camera_in_ee'
        elif cfg.camera is not None:
            primary_cam = SharedDirCamera(
                shared_dir=cfg.camera.shared_dir,
                rgb_filename=cfg.camera.rgb_filename,
                depth_filename=cfg.camera.depth_filename,
                camera_info_filename=cfg.camera.camera_info_filename,
                freshness_max_age_s=cfg.camera.freshness_max_age_s,
            )
            camera = DualSharedDirCamera(primary_cam, None)
            extrinsic_primary = load_extrinsic(cfg.camera.extrinsic_json)
            extrinsic_primary['transform_direction'] = 'camera_in_ee'
        else:
            raise RuntimeError('Either camera or cameras must be configured')

        snap = camera.snapshot()

        llm_perception = LLMPerceptionService(
            vision_client=vision_client,
            rgb_path=snap['rgb_path'],
            depth=snap['depth'],
            camera_info=snap['camera_info'],
            extrinsic=extrinsic_primary,
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
                extrinsic=extrinsic_primary,
                tool_rpy_deg=cfg.api_runtime.default_tool_orientation_deg,
                place_drop_mm=cfg.api_runtime.default_place_drop_mm,
                depth_window_radius=cfg.perception.depth_window_radius,
                debug=cfg.perception.debug,
                llm_fallback=llm_perception if cfg.perception.llm_fallback else None,
                grasp_xyz_offset_mm=cfg.api_runtime.grasp_xyz_offset_mm,
                place_xyz_offset_mm=cfg.api_runtime.place_xyz_offset_mm,
                depth_uv_mapping_mode=cfg.perception.depth_uv_mapping_mode,
            )
        elif cfg.perception.backend == 'foundation_vision':
            perception = FoundationVisionPerceptionService(
                vision_client=vision_client,
                rgb_path=snap['rgb_path'],
                depth=snap['depth'],
                camera_info=snap['camera_info'],
                extrinsic=extrinsic_primary,
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

        if cfg.robot.backend == 'piper':
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
        else:
            arm = DummyArm()
        if connect_robot and cfg.robot.connect_on_start:
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

        task_agent = TaskAgent(vision_client, 'configs/prompts.yaml')
        program_agent = ProgramAgent(vision_client, 'configs/prompts.yaml', 'configs/api_registry.yaml')
        verify_agent = VerifyAgent(vision_client, 'configs/prompts.yaml')

        orchestrator = PipelineOrchestrator(cfg, task_agent, program_agent, verify_agent, camera, runtime_api)
        return cls(cfg, orchestrator)
