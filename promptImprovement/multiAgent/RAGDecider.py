import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.getcwd()))))
from HuggingFaceModel import HuggingFaceModel
from OllamaModel import OllamaModel
from prepare_prompt import prepare_prompt_few_shot_rag_decider


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
        provider: str = "HF",
    ):
        """
        Initialise l'agent.

        Args:
            model_name: Nom du modèle qui sera utilisé
        """
        self.model_name = model_name
        self.response_length = response_length
        self.provider = provider

    def prepare_runnable(self):
        if self.provider == "HF":
            entity = HuggingFaceModel(
                model_name=self.model_name, response_length=self.response_length
            )
        elif self.provider == "OL":
            entity = OllamaModel(model_name=self.model_name)

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

    def prepare_career_intent_classifier(model):  # à essayer
        """
        Prépare un classificateur simple pour détecter les intentions liées aux métiers.

        Args:
            model: Le modèle de language à utiliser

        Returns:
            Callable: Une fonction qui prend un message utilisateur et retourne "1" ou "2"
        """
        prompt_template = """Tu es un assistant qui analyse les messages des utilisateurs pour déterminer s'ils expriment un intérêt pour des informations sur les métiers.

    Ta tâche est de classifier chaque message :
    - Réponds "1" si l'utilisateur demande des informations sur les métiers ou les débouchés professionnels
    - Réponds "2" dans tous les autres cas

    IMPORTANT:
    - Réponds uniquement par "1" ou "2", sans aucun autre texte
    - Ne fais pas de supposition, base-toi uniquement sur ce qui est explicitement exprimé
    - Si le message contient plusieurs intentions, réponds "1" dès qu'une intention liée aux métiers est présente

    Voici quelques exemples:

    Message: À quelle heure se terminent les cours à l'université ?
    Réponse: 2

    Message: Franchement je ne sais pas quels métiers sont possibles depuis ce domaine
    Réponse: 1

    Message: Je ne suis pas intéressé par les métiers, uniquement par les cours
    Réponse: 2

    Message: {input}
    Réponse:"""

        def classify(user_message: str) -> str:
            """
            Classifie un message utilisateur.

            Args:
                user_message (str): Le message à classifier

            Returns:
                str: "1" si l'intention est liée aux métiers, "2" sinon
            """
            prompt = prompt_template.format(input=user_message)
            return model.predict(prompt).strip()

        return classify
