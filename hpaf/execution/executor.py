from hpaf.execution.program_validator import validate_program


def execute_program(program: str, runtime_api):
    validate_program(program)
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        'debug': runtime_api.debug,
        'detect_object_by_text': runtime_api.detect_object_by_text,
        'estimate_top_grasp_pose': runtime_api.estimate_top_grasp_pose,
        'build_pregrasp_pose': runtime_api.build_pregrasp_pose,
        'estimate_place_pose': runtime_api.estimate_place_pose,
        'open_gripper': runtime_api.open_gripper,
        'close_gripper': runtime_api.close_gripper,
        'move_to_pose': runtime_api.move_to_pose,
        'retreat': runtime_api.retreat,
        'return_to_observe_pose': runtime_api.return_to_observe_pose,
        'stabilize_grasp': runtime_api.stabilize_grasp,
        'verify_object_grasped': runtime_api.verify_object_grasped,
        'verify_object_in_region': runtime_api.verify_object_in_region,
        'ai_verify_atomic_task': runtime_api.ai_verify_atomic_task,
    }
    exec(program, safe_globals, safe_locals)
    return {
        'result': safe_locals.get('result', None),
        'ai_verify_output': runtime_api.get_last_ai_verify_output(),
    }
