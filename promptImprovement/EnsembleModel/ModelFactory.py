# model_factory.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.schema.runnable import Runnable

# from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM

# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from langchain_huggingface import HuggingFacePipeline
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import os
import gc
import platform
from transformers import TextStreamer
from queue import Empty  # Importer Empty depuis le module queue


class CustomTextStreamer(TextStreamer):
    def __init__(self, tokenizer, timeout: Optional[float] = None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.timeout = timeout  # None signifie pas de timeout

    def __next__(self):
        try:
            # Utiliser None comme timeout désactive le timeout
            value = self.text_queue.get(timeout=self.timeout)
            if value is StopIteration:
                raise StopIteration()
            return value
        except Empty:
            # Vous pouvez personnaliser la gestion des timeouts ici
            return ""  # ou gérer autrement


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
    def prepare_for_ensemble(self, few_shot) -> Runnable:
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


class HuggingFaceModel(BaseLanguageModel):
    """Implémentation pour les modèles Hugging Face."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs
        self.is_windows = platform.system() == "Windows"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self) -> HuggingFacePipeline:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False,  # Utiliser le tokenizer lent qui est plus stable
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        try:
            # Sur Windows, on évite la quantification qui peut causer des problèmes
            if self.is_windows:
                model = self._load_model_with_quantization()  # _load_model_safe()
            else:
                model = self._load_model_with_quantization()
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            print("Tentative de chargement en mode sécurisé...")
            model = self._load_model_safe()

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            streamer=CustomTextStreamer(tokenizer, timeout=5000),
            max_new_tokens=1024,
            trust_remote_code=True,
            device_map="auto",
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,  # Ajout explicite du pad_token_id
        )

        return HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"device_map": "auto"},  # Ajoutez device_map ici aussi
            pipeline_kwargs={
                "max_new_tokens": 1024,  # une seconde fois ici pour être sûr
                "do_sample": True,
            },
        )

    def _load_model_safe(self):
        """Chargement sécurisé du modèle sans quantification"""
        print("Chargement du modèle en mode sécurisé (sans quantification)...")

        # Calculer la mémoire GPU disponible
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            max_memory = {
                0: f"{int(gpu_mem * 0.9 / 1024**2)}MiB"
            }  # Utilise 90% de la VRAM
        else:
            max_memory = None

        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            **{
                k: v
                for k, v in self.params.items()
                if k
                not in [
                    "load_in_8bit",
                    "load_in_4bit",
                ]  # Ignore les paramètres de quantification
            },
        )

    def _load_model_with_quantization(self):
        """Tente de charger le modèle avec quantification (pour non-Windows)"""
        print("Tentative de chargement avec quantification...")
        config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )

        config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,  # Activez le mode 4-bit au lieu de 8-bit
            bnb_4bit_compute_dtype=torch.float16,  # Type de données pour les calculs
            bnb_4bit_use_double_quant=True,  # Double quantification pour une meilleure compression
            bnb_4bit_quant_type="nf4",  # Type de quantification (nf4 ou fp4)
        )

        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=config_4bit,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **self.params,
        )

    def get_model_params(self) -> Dict[str, Any]:
        return {
            "model_type": "huggingface",
            "model_name": self.model_name,
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
