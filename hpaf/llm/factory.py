import os
from .openai_compatible import OpenAICompatibleVisionClient
from .mock_client import MockVisionClient

def make_vision_client(llm_cfg):
    provider = str(llm_cfg.provider).lower().strip()

    if provider in ("doubao", "openai", "openai_compatible"):
        api_key = os.getenv(llm_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {llm_cfg.api_key_env} is empty. "
                f"Refusing to fall back to mock client."
            )

        print(
            f"[LLM] provider={provider}, "
            f"model={llm_cfg.model}, "
            f"base_url={llm_cfg.base_url}"
        )

        return OpenAICompatibleVisionClient(
            base_url=llm_cfg.base_url,
            api_key=api_key,
            model=llm_cfg.model,
            max_output_tokens=llm_cfg.max_output_tokens,
            temperature=llm_cfg.temperature,
        )

    elif provider == "mock":
        print("[LLM] using MockVisionClient")
        return MockVisionClient()

    raise ValueError(f"Unsupported llm.provider: {llm_cfg.provider}")
