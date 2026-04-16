import ast


class ProgramValidationError(Exception):
    pass


ALLOWED_NODES = (
    ast.Module, ast.Assign, ast.Expr, ast.Call, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.keyword, ast.Attribute, ast.JoinedStr, ast.FormattedValue,
)

ALLOWED_FUNC_NAMES = {
    'debug', 'detect_object_by_text', 'estimate_top_grasp_pose', 'build_pregrasp_pose',
    'estimate_place_pose', 'open_gripper', 'close_gripper', 'move_to_pose', 'retreat',
    'return_to_observe_pose', 'stabilize_grasp', 'ai_verify_atomic_task'
}


def validate_program(program: str):
    try:
        tree = ast.parse(program)
    except SyntaxError as e:
        raise ProgramValidationError(str(e))
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ProgramValidationError(f'Disallowed AST node: {type(node).__name__}')
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_FUNC_NAMES:
                raise ProgramValidationError(f'Disallowed function: {node.func.id}')
    return tree
