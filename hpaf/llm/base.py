from abc import ABC, abstractmethod
from typing import Any, Dict


class VisionLanguageClient(ABC):
    @abstractmethod
    def task_decompose(self, image_path: str, task: str, prompt_template: str) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate_program(
        self,
        image_path: str,
        atomic_task: str,
        api_docs: str,
        prompt_template: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def verify_task(
        self,
        image_path: str,
        atomic_task: str,
        prompt_template: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError
