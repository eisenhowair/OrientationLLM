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
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        """
        Initialise l'agent.

        Args:
            model_name: Nom du modèle qui sera utilisé
        """
        self.model_name = model_name
        self.prompt = prompt_rag_decider

    def prepare_runnable(self):
        entity = HuggingFaceModel(model_name=self.model_name, response_length=1)
        self.model = entity.get_model()

        self.runnable = prepare_prompt_few_shot_rag_decider(
            model=self.model, corps_prompt=self.prompt, version="simple"
        )

    def invoke_agent(self, user_input: str):
        formatted_prompt = {"input": user_input}

        # Exécution du runnable
        response = self.runnable.invoke(formatted_prompt)
        print(f"Besoin d'utiliser le RAG: {response}===\n")
        return response
