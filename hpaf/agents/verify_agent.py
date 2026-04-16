import yaml

class VerifyAgent:
    def __init__(self, vision_client, prompts_path: str):
        self.vision_client = vision_client
        self.prompts = yaml.safe_load(open(prompts_path, 'r', encoding='utf-8'))

    def run(self, image_path: str, atomic_task: str, previous_plan_brief: str):
        system = self.prompts['verify_agent_system']
        user_text = (
            f'Current atomic task: {atomic_task}\n'
            f'Previous program summary: {previous_plan_brief}'
        )
        return self.vision_client.ask_json_with_image(image_path, system, user_text)
