import json
from hpaf.core.io import read_text

class TaskAgent:
    def __init__(self, vision_client, prompts_path: str):
        self.vision_client = vision_client
        import yaml
        self.prompts = yaml.safe_load(open(prompts_path, 'r', encoding='utf-8'))

    def run(self, image_path: str, task_text: str):
        system = self.prompts['task_agent_system']
        user_text = f'Complex task: {task_text}'
        return self.vision_client.ask_json_with_image(image_path, system, user_text)
