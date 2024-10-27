from langchain_community.llms import Ollama
from langchain.schema.runnable import Runnable, RunnableParallel
from langchain.schema import StrOutputParser
from typing import List, Dict, Optional
import chainlit as cl
from prepare_prompt import prepare_prompt_zero_shot, prepare_prompt_few_shot
from prompt_warehouse import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.schema.runnable.config import RunnableConfig


class EnsembleModelManager:
    def __init__(self):
        self.available_models = {
            "llama3.1:8b-instruct-q4_1": {
                "weight": 1.0,
                "params": {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3},
            },
            "llama3:instruct": {
                "weight": 1.0,
                "params": {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3},
            },
            "llama3.2:3b-instruct-q8_0": {
                "weight": 1.0,
                "params": {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3},
            },
            "nemotron-mini": {
                "weight": 1.0,
                "params": {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3},
            },
            "nemotron-mini:4b-instruct-q5_0": {
                "weight": 1.0,
                "params": {"cache": True, "num_ctx": 32768, "repeat_penalty": 1.3},
            },
        }
        self.active_models: List[str] = []
        self.base_url = "http://localhost:11434"

    def set_model_weights(self, weights: Dict[str, float]) -> None:
        """
        Met à jour les poids des modèles disponibles.

        Args:
            weights: Dictionnaire avec les noms des modèles comme clés et leurs poids comme valeurs
        """
        for model_name, weight in weights.items():
            if model_name in self.available_models:
                self.available_models[model_name]["weight"] = weight

    def activate_models(self, model_names: List[str]) -> None:
        """
        Active les modèles spécifiés pour l'ensemble.

        Args:
            model_names: Liste des noms des modèles à activer
        """
        self.active_models = [
            name for name in model_names if name in self.available_models
        ]

    def create_ensemble_runnable(
        self,
        domaine: Optional[str] = None,
        formation: Optional[str] = None,
        use_few_shot: bool = False,
    ) -> Runnable:
        """
        Crée un Runnable qui combine les résultats de tous les modèles actifs.

        Args:
            domaine: Le domaine d'études spécifié
            formation: Le niveau de formation spécifié
            use_few_shot: Utiliser ou non le few-shot prompting

        Returns:
            Un Runnable combinant les résultats des modèles
        """
        if not self.active_models:
            raise ValueError("Aucun modèle n'est activé dans l'ensemble")

        # Créer un dictionnaire de Runnables pour chaque modèle actif
        model_runnables = {}

        for model_name in self.active_models:
            # Initialiser le modèle Ollama avec ses paramètres
            model = Ollama(
                base_url=self.base_url,
                model=model_name,
                **self.available_models[model_name]["params"],
            )

            # Créer le runnable spécifique au modèle
            prepare_fn = (
                prepare_prompt_few_shot if use_few_shot else prepare_prompt_zero_shot
            )
            model_runnable = prepare_fn(
                corps_prompt=self._generate_prompt(domaine, formation), model=model
            )

            model_runnables[model_name] = model_runnable

        # Créer un RunnableParallel pour exécuter tous les modèles en parallèle
        parallel_runnable = RunnableParallel(model_runnables)

        # Créer un runnable final qui combine les résultats
        final_runnable = parallel_runnable | self._combine_responses

        return final_runnable

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
            # Importer le prompt par défaut depuis prompt_warehouse si nécessaire
            return "Ton rôle est de conseiller l'utilisateur sur les opportunités de carrière."

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

        return {
            "consensus_level": float(avg_similarity),
            "num_outliers": len(outliers),
            "outlier_models": outliers,
            "min_similarity": float(
                np.min(similarity_matrix[similarity_matrix != 1.0])
            ),
        }

    async def stream_ensemble_response(
        self, message: cl.Message, runnable: Runnable
    ) -> None:
        """
        Diffuse la réponse de l'ensemble via Chainlit.

        Args:
            message: Le message de l'utilisateur
            runnable: Le runnable d'ensemble à utiliser
        """
        msg = cl.Message(content="")

        async for chunk in runnable.astream(
            {"input": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()
