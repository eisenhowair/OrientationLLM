from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain.schema.runnable import Runnable

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from langchain_huggingface import HuggingFacePipeline
import torch
import gc
import platform
from transformers import TextStreamer
from queue import Empty  # Importer Empty depuis le module queue


class CustomTextStreamer(TextStreamer):
    """
    Cette classe est uniquement là pour pouvoir changer le délai d'attente permis pour qu'un modèle réponde.
    Sinon au bout de quelques secondes chainlit envoie une erreur asyncio.
    """

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
            return ""  # ou gérer autrement


class HuggingFaceModel:
    """Implémentation pour les modèles Hugging Face."""

    def __init__(self, model_name: str, response_length: int = 1, **kwargs):
        self.model_name = model_name
        self.response_length = response_length  # to 1 for single character response
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
            use_fast=False,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        try:
            if self.is_windows:
                model = self._load_model_safe()
            else:
                model = self._load_model_with_quantization()
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            print(f"Tentative de chargement de {self.model_name} en mode sécurisé...")
            model = self._load_model_safe()

        # Disable streaming for single character response
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.response_length,
            streamer=(
                CustomTextStreamer(tokenizer, timeout=9000)
                if self.response_length > 2
                else None
            ),
            trust_remote_code=True,
            device_map="auto",
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
        )

        return HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"device_map": "auto"},
            pipeline_kwargs={
                "max_new_tokens": self.response_length,
                "do_sample": True,
            },
        )

    def _load_model_safe(self):
        """Chargement sécurisé du modèle sans quantification"""
        print("Chargement du modèle en mode sécurisé (sans quantification)...")

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            max_memory = {0: f"{int(gpu_mem * 0.9 / 1024**2)}MiB"}
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
                if k not in ["load_in_8bit", "load_in_4bit"]
            },
        )

    def _load_model_with_quantization(self):
        """Tente de charger le modèle avec quantification (pour non-Windows)"""
        print(f"Tentative de chargement de {self.model_name} avec quantification...")
        config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=config_4bit,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **self.params,
        )
