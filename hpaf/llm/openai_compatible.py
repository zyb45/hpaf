import base64
import json
from typing import Any, Dict

from openai import OpenAI

from .base import VisionLanguageClient


class OpenAICompatibleVisionClient(VisionLanguageClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 4096,
        temperature: float = 0.2,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    @staticmethod
    def _encode_image_to_data_url(image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")
        lower = image_path.lower()
        if lower.endswith('.png'):
            mime = 'image/png'
        elif lower.endswith('.webp'):
            mime = 'image/webp'
        else:
            mime = 'image/jpeg'
        return f"data:{mime};base64,{base64_image}"

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        return stripped

    def ask_image(self, image_path: str, text_prompt: str) -> str:
        image_data_url = self._encode_image_to_data_url(image_path)

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": image_data_url,
                        },
                        {
                            "type": "input_text",
                            "text": text_prompt,
                        },
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
        )

        if hasattr(response, "output") and response.output:
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            return content.text

        return str(response)

    def ask_image_json(self, image_path: str, text_prompt: str) -> Dict[str, Any]:
        text = self.ask_image(image_path=image_path, text_prompt=text_prompt)
        text = self._strip_code_fence(text)
        return json.loads(text)

    # -------------------------
    # Abstract interface methods
    # -------------------------

    def task_decompose(self, image_path: str, task_text: str) -> Dict[str, Any]:
        prompt = f"""
You are a tabletop robot task decomposition agent (TaskAgent).

Given a current tabletop image and a complex task, please:
1. Briefly summarize the scene.
2. Decompose the complex task into an ordered list of atomic tasks.

Requirements:
- Return strict JSON only
- Do not output explanations or markdown code fences
- The JSON format must be:
{{
  "scene_summary": "...",
  "atomic_tasks": [
    "...",
    "..."
  ]
}}

Complex task:
{task_text}
""".strip()

        return self.ask_image_json(image_path=image_path, text_prompt=prompt)

    def generate_program(self, image_path: str, atomic_task: str, api_docs: str) -> Dict[str, Any]:
        prompt = f"""
You are a tabletop robot program generation agent (ProgramAgent).

Given the current image, an atomic task, and the available API docs, generate an executable Python script body.

Goals:
- Organize the program as perception -> align -> interact -> verify
- Call only functions allowed by the API documentation
- Do not output def / class / import
- Do not output markdown code fences
- The final result must be saved to the variable result

Available API documentation:
{api_docs}

Atomic task:
{atomic_task}

Return strict JSON only, in the format:
{{
  "atomic_task": "{atomic_task}",
  "plan_brief": "...",
  "program": "..."
}}
""".strip()

        return self.ask_image_json(image_path=image_path, text_prompt=prompt)

    def verify_task(self, image_path: str, atomic_task: str) -> Dict[str, Any]:
        prompt = f"""
You are a tabletop robot task verification agent (VerifyAgent).

Given the post-execution image and an atomic task, determine whether the task is complete.

Return strict JSON only. Do not output explanations or markdown code fences.
The format must be:
{{
  "atomic_task": "{atomic_task}",
  "success": true,
  "failure_stage": "",
  "reason": ""
}}

Where:
- success: true/false
- failure_stage must be one of perception / align / interact / verify / ""
- If success is true, failure_stage must be an empty string
- If success is false, indicate the most likely failure stage
- reason should briefly explain the judgement

Atomic task:
{atomic_task}
""".strip()

        return self.ask_image_json(image_path=image_path, text_prompt=prompt)
    
    def ask_json_with_image(self, image_path: str, system_prompt: str, user_text: str):
        prompt = f"{system_prompt.strip()}\n\n{user_text.strip()}"
        return self.ask_image_json(image_path=image_path, text_prompt=prompt)

    def ask_text_with_image(self, image_path: str, system_prompt: str, user_text: str):
        prompt = f"{system_prompt.strip()}\n\n{user_text.strip()}"
        return self.ask_image(image_path=image_path, text_prompt=prompt)

    def ask_json(self, system_prompt: str, user_text: str):
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{system_prompt.strip()}\n\n{user_text.strip()}",
                        }
                    ],
                }
            ],
            max_output_tokens=self.max_output_tokens,
        )

        text = None
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for content in getattr(item, "content", []):
                        if getattr(content, "type", None) == "output_text":
                            text = content.text
                            break

        if text is None:
            text = str(response)

        text = self._strip_code_fence(text)
        return json.loads(text)    
