from HuggingFaceModel import HuggingFaceModel
from prepare_prompt import *
from prompt_warehouse import *


class RAGDecider:
    """
    Classe pour mettre en place et utiliser un modèle
    servant à déterminer si le contexte par RAG est nécessaire
    ou non à une requête utilisateur
    """

    def __init__(
        self,
        # model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # model_name: str = "utter-project/EuroLLM-1.7B",
        # model_name="meta-llama/Llama-3.2-1B-Instruct",
        model_name="mistralai/Ministral-8B-Instruct-2410",
        response_length: int = 1,
    ):
        """
        Initialise l'agent.

        Args:
            model_name: Nom du modèle qui sera utilisé
        """
        self.model_name = model_name
        self.prompt = prompt_rag_decider_simple
        self.response_length = response_length

    def prepare_runnable(self):
        entity = HuggingFaceModel(
            model_name=self.model_name, response_length=self.response_length
        )
        self.model = entity.get_model()

        self.runnable = prepare_prompt_few_shot_rag_decider(
            model=self.model, version="advanced"
        )

    def invoke_agent(self, user_input: str):
        formatted_prompt = {"input": user_input}

        # Exécution du runnable
        response = self.runnable.invoke(formatted_prompt)
        print(f"Responseeee:{response}\n")
        return response
