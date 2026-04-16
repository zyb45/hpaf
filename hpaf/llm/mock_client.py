from typing import Any, Dict

class MockVisionClient:
    def ask_json_with_image(self, image_path: str, system_prompt: str, user_text: str) -> Dict[str, Any]:
        if 'atomic_tasks' in system_prompt:
            return {
                'scene_summary': 'mock scene',
                'atomic_tasks': ['grasp the red object', 'put the red object into the left box']
            }
        if 'program' in system_prompt:
            return {
                'atomic_task': 'grasp the red object',
                'plan_brief': 'mock',
                'program': (
                    'target = detect_object_by_text("red object")\n'
                    'pose = estimate_top_grasp_pose(target)\n'
                    'pre = build_pregrasp_pose(pose, lift_mm=80)\n'
                    'open_gripper()\n'
                    'move_to_pose(pre)\n'
                    'move_to_pose(pose)\n'
                    'close_gripper(force=800)\n'
                    'retreat(z_offset_mm=80)\n'
                    'result = verify_object_grasped("red object")'
                )
            }
        return {
            'done': True,
            'failure_stage': 'none',
            'reason': 'mock',
            'regeneration_hint': ''
        }
