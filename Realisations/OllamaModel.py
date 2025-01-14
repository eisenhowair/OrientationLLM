from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.schema.runnable import Runnable

from langchain_ollama.llms import OllamaLLM


class OllamaModel:
    """Implémentation pour les modèles Ollama."""

    def __init__(
        self, model_name: str, base_url: str = "http://localhost:11434", **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.params = {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3, **kwargs}

    def get_model(self) -> OllamaLLM:
        return OllamaLLM(base_url=self.base_url, model=self.model_name, **self.params)

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            **self.params,
        }

    def prepare_for_ensemble(self, few_shot) -> Runnable:
        from prepare_prompt import prepare_prompt_zero_shot, prepare_prompt_few_shot

        model = self.get_model()

        return (
            prepare_prompt_few_shot(model=model)
            if few_shot
            else prepare_prompt_zero_shot(model=model)
        )
