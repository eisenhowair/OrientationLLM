# ensemble_model_gestion.py
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema.runnable import Runnable, RunnableParallel
import chainlit as cl
from ModelFactory import ModelFactory, BaseLanguageModel
from prompt_warehouse import *
from langchain.schema.runnable.config import RunnableConfig
from vector_store_manager import *

""" trop lourd, trop lent (30 minutes pour une question minimum)
# LLaMA 3.1 Minitron 4B (NVIDIA)
            "llama3.1-minitron-4b-nvidia": {
                "weight": 1.0,
                "config": {
                    "model_type": "huggingface",
                    "model_name": "nvidia/Llama-3.1-Minitron-4B-Width-Base",
                    "params": {"trust_remote_code": True},
                },
            },
            # Modèles Hugging Face vraiment pas bons, galèrent en français
            "TinyLlama-1.1B-Chat-v1.0": {
                "weight": 1.0,
                "config": {
                    "model_type": "huggingface",
                    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "params": {
                        "trust_remote_code": True,
                    },
                },
            },
            "SmolLM2-1.7B-Instruct": {
                "weight": 1.0,
                "config": {
                    "model_type": "huggingface",
                    "model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    "params": {
                        "trust_remote_code": True,
                    },
                },
            },
"""


class EnsembleModelManager:
    def __init__(self):
        # modèles disponibles
        self.available_models = {
            # Modèles Ollama
            "llama3.1:8b-instruct-q4_1": {
                "weight": 1.0,
                "config": {
                    "model_type": "ollama",
                    "model_name": "llama3.1:8b-instruct-q4_1",
                },
            },
            "qwen-2.5:3b-instruct": {
                "weight": 1.0,
                "config": {
                    "model_type": "ollama",
                    "model_name": "qwen2.5:3b-instruct",
                },
            },
            "Llama-3.2-3B-Instruct": {
                "weight": 1.0,
                "config": {
                    "model_type": "ollama",
                    "model_name": "llama3.2:3b-instruct-q4_0",
                },
            },
        }
        self.active_models: List[str] = []
        self.model_instances: Dict[str, BaseLanguageModel] = {}

    def set_model_weights(self, weights: Dict[str, float]) -> None:
        for model_name, weight in weights.items():
            if model_name in self.available_models:
                self.available_models[model_name]["weight"] = weight

    def _combine_responses(self, responses: Dict[str, str]) -> str:
        """
        Combine les réponses des modèles en utilisant un système de vote basé sur la similarité.
        Affiche également chaque réponse avec son modèle correspondant.

        Args:
            responses: Dictionnaire des réponses de chaque modèle

        Returns:
            La réponse qui représente le meilleur consensus
        """
        if len(responses) == 1:
            return list(responses.values())[0]

        # Convertir le dictionnaire en listes et afficher les réponses
        model_names = list(responses.keys())
        response_texts = list(responses.values())

        print("\n=== Réponses par modèle ===")
        print("------------------------")
        for model_name, response in zip(model_names, response_texts):
            print(f"\nModèle: {model_name}")
            print("------------------------")
            print(response)
            print("------------------------")
        print("\n")

        # Créer une matrice de similarité entre toutes les réponses
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(response_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Calculer le score de consensus pour chaque réponse
        consensus_scores = []
        for i, model_name in enumerate(model_names):
            # Similarité moyenne avec les autres réponses
            avg_similarity = np.mean(
                [similarity_matrix[i][j] for j in range(len(model_names)) if i != j]
            )

            # Pondérer par le poids du modèle
            model_weight = self.available_models[model_name]["weight"]
            consensus_score = avg_similarity * model_weight

            consensus_scores.append((response_texts[i], consensus_score))

            # Afficher les scores de similarité pour ce modèle
            print(f"\nScores de similarité pour {model_name}:")
            print(f"Similarité moyenne: {avg_similarity:.2f}")
            print(f"Poids du modèle: {model_weight}")
            print(f"Score final: {consensus_score:.2f}")

        # Choisir la réponse avec le meilleur score de consensus
        best_response = max(consensus_scores, key=lambda x: x[1])[0]
        best_model = model_names[response_texts.index(best_response)]

        print(f"\n=== Réponse choisie ===")
        print(f"Modèle sélectionné: {best_model}")
        print("------------------------")

        return best_response

    def get_response_statistics(self, responses: Dict[str, str]) -> Dict[str, float]:
        """
        Calcule des statistiques sur les réponses pour aider à comprendre le consensus.

        Args:
            responses: Dictionnaire des réponses des modèles

        Returns:
            Dictionnaire contenant des statistiques sur les réponses
        """

        # Si responses est une seule chaîne
        if isinstance(responses, str):
            # Convertir en liste avec un seul élément
            responses = [responses]
        # Si c'est déjà un dictionnaire
        elif isinstance(responses, dict):
            responses = list(responses.values())
        else:
            responses = list(responses)

        if len(responses) <= 1:
            return {"consensus_level": 1.0}

        # Calculer la similarité entre les réponses
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(responses.values())
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Calculer le niveau moyen de consensus
        avg_similarity = np.mean(
            [
                similarity_matrix[i][j]
                for i in range(len(responses))
                for j in range(i + 1, len(responses))
            ]
        )

        # Identifier les outliers (réponses très différentes)
        similarity_scores = np.mean(similarity_matrix, axis=1)
        outliers = [
            model_name
            for model_name, score in zip(responses.keys(), similarity_scores)
            if score < avg_similarity - np.std(similarity_scores)
        ]

        result = {
            "consensus_level": float(avg_similarity),
            "num_outliers": len(outliers),
            "outlier_models": outliers,
            "min_similarity": float(
                np.min(similarity_matrix[similarity_matrix != 1.0])
            ),
        }

        if not outliers:
            del result["num_outliers"]

        return result

    def _generate_prompt(self, domaine: Optional[str], formation: Optional[str]) -> str:
        """
        Génère le prompt en fonction du domaine et de la formation.
        """
        if domaine and formation:
            return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}. L'utilisateur sort de la formation {formation}."
        elif domaine:
            return f"Ton rôle est de conseiller l'utilisateur sur les métiers du domaine {domaine}."
        elif formation:
            return f"L'utilisateur sort de la formation {formation}."
        else:
            return prompt_no_domain_no_formation_v3

    def activate_models(self, model_names: List[str]) -> None:
        self.active_models = [
            name for name in model_names if name in self.available_models
        ]

        # Instantiate models using factory
        self.model_instances = {}
        for model_name in self.active_models:
            config = self.available_models[model_name]["config"]
            self.model_instances[model_name] = ModelFactory.create_model(config)

    def create_ensemble_runnable(
        self,
        domaine: Optional[str] = None,
        formation: Optional[str] = None,
        use_few_shot: bool = False,
    ) -> Runnable:
        if not self.active_models:
            raise ValueError("Aucun modèle n'est activé dans l'ensemble")

        model_runnables = {}
        for model_name in self.active_models:
            model_instance = self.model_instances[model_name]
            model_runnable = model_instance.prepare_for_ensemble(few_shot=use_few_shot)
            model_runnables[model_name] = model_runnable
            print(f"==> {model_name} runnable fetched")

        parallel_runnable = RunnableParallel(model_runnables)
        final_runnable = parallel_runnable | self._combine_responses
        print(f"====> final_runnable obtained")
        return final_runnable

    async def stream_ensemble_response(
        self, message: cl.Message, runnable: Runnable
    ) -> cl.Message:
        vectorstore = cl.user_session.get("vectorstore")  # type: VectorStoreFAISS
        msg = cl.Message(content="")
        context = vectorstore.similarity_search(query=message.content)
        for result in context:
            print(f"Contexte récupéré:{result}\n======\n")

        async for chunk in runnable.astream(
            {
                "input": message.content,
                "context": context,  # rajouté ça pour donner accès au contexte, et rajouté {context dans le prompt lui-même}
            },
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        # Afficher les statistiques si disponibles
        if len(self.active_models) > 1:
            stats = self.get_response_statistics(msg.content)
            stats_msg = (
                f"\n\nStatistiques de l'ensemble:\n"
                f"- Niveau de consensus: {stats['consensus_level']:.2f}\n"
                # f"- Nombre de modèles divergents: {stats['num_outliers']}"
            )
            await cl.Message(content=stats_msg).send()

        await msg.send()
        return msg

    # Les autres méthodes (_combine_responses, get_response_statistics, etc.)
    # restent identiques car elles travaillent sur le texte déjà généré
