# model_factory.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.schema.runnable import Runnable
from langchain_community.llms import Ollama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class BaseLanguageModel(ABC):
    """Classe abstraite définissant l'interface commune pour tous les modèles."""

    @abstractmethod
    def get_model(self) -> Any:
        """Retourne l'instance du modèle."""
        pass

    @abstractmethod
    def get_model_params(self) -> Dict[str, Any]:
        """Retourne les paramètres du modèle."""
        pass

    @abstractmethod
    def prepare_for_ensemble(self) -> Runnable:
        """Prépare le modèle pour une utilisation dans l'ensemble."""
        pass


class OllamaModel(BaseLanguageModel):
    """Implémentation pour les modèles Ollama."""

    def __init__(
        self, model_name: str, base_url: str = "http://localhost:11434", **kwargs
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.params = {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3, **kwargs}

    def get_model(self) -> Ollama:
        return Ollama(base_url=self.base_url, model=self.model_name, **self.params)

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": "ollama",
            "model_name": self.model_name,
            "base_url": self.base_url,
            **self.params,
        }

    def prepare_for_ensemble(self) -> Runnable:
        from prepare_prompt import prepare_prompt_zero_shot

        model = self.get_model()
        return prepare_prompt_zero_shot(model=model)


class HuggingFaceModel(BaseLanguageModel):
    """Implémentation pour les modèles Hugging Face."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs

    def get_model(self) -> HuggingFacePipeline:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", **self.params
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
        )

        return HuggingFacePipeline(pipeline=pipe)

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": "huggingface",
            "model_name": self.model_name,
            **self.params,
        }

    def prepare_for_ensemble(self) -> Runnable:
        from prepare_prompt import prepare_prompt_zero_shot

        model = self.get_model()
        return prepare_prompt_zero_shot(model=model)


class ModelFactory:
    """Factory pour créer les instances de modèles appropriées."""

    @staticmethod
    def create_model(model_config: Dict[str, Any]) -> BaseLanguageModel:
        model_type = model_config.get("model_type", "ollama")

        if model_type == "ollama":
            return OllamaModel(
                model_name=model_config["model_name"],
                base_url=model_config.get("base_url", "http://localhost:11434"),
                **model_config.get("params", {}),
            )
        elif model_type == "huggingface":
            return HuggingFaceModel(
                model_name=model_config["model_name"], **model_config.get("params", {})
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
