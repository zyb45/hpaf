import yaml


class ProgramAgent:
    def __init__(self, vision_client, prompts_path: str, api_registry_path: str):
        self.vision_client = vision_client
        self.prompts = yaml.safe_load(open(prompts_path, 'r', encoding='utf-8'))
        self.api_registry = yaml.safe_load(open(api_registry_path, 'r', encoding='utf-8'))

    def _task_hint(self, atomic_task: str) -> str:
        t = (atomic_task or '').lower()
        if any(k in t for k in ['grasp', 'pick', '抓取', '拾取', '夹取']):
            return 'This is a grasping atomic task: generate only grasp-related actions and do not generate placement actions.'
        if any(k in t for k in ['put', 'place', 'drop', '放入', '放到', '放进', '放置']):
            return 'This is a placement atomic task: assume the object is already grasped; only localize the target region, move to a placement pose, open the gripper, retreat, and return to observe pose. Do not re-grasp the object.'
        return 'This is a general atomic task: keep the program focused on the current step only.'

    def run(self, image_path: str, atomic_task: str, execution_mode: str = 'manual'):
        system = self.prompts['program_agent_system']
        user_text = (
            f'Execution mode: {execution_mode}\n\n'
            f'Current atomic task: {atomic_task}\n\n'
            f'{self._task_hint(atomic_task)}\n\n'
            f'Available API documentation:\n{self.api_registry["docs"]}'
        )
        return self.vision_client.ask_json_with_image(image_path, system, user_text)
