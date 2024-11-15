from typing import Dict, List, Any
from pathlib import Path
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from vector_store_manager import VectorStoreFAISS


class EmbeddingComparator:
    """
    Classe pour comparer différents modèles d'embedding sur un même ensemble de documents.
    """

    def __init__(
        self,
        data_dir: str,
        models_config: List[Dict[str, Any]],
        index_base_path: str = "embedding_indexes",
    ):
        """
        Initialise le comparateur.

        Args:
            data_dir: Chemin vers le dossier contenant les documents
            models_config: Liste des configurations des modèles à comparer
            index_base_path: Dossier où sauvegarder les index FAISS
        """
        self.data_dir = Path(data_dir)
        self.models_config = models_config
        self.index_base_path = Path(index_base_path)
        self.vectorstores: Dict[str, VectorStoreFAISS] = {}
        self.console = Console()

    def initialize_vectorstores(self) -> None:
        """Crée et initialise un vectorstore pour chaque modèle d'embedding."""
        self.index_base_path.mkdir(parents=True, exist_ok=True)

        for config in self.models_config:
            model_name = config["name"]
            self.console.print(
                f"\n[bold blue]Initializing vectorstore for {model_name}...[/]"
            )

            # Initialiser le vectorstore
            vectorstore = VectorStoreFAISS(
                embedding_model_name=config["model"],
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
                index_path=str(self.index_base_path),
            )

            # Dans tous les cas, vectoriser les documents si le vectorstore est vide
            if not vectorstore.documents:
                self.console.print(
                    f"[yellow]Vectorizing documents for {model_name}...[/]"
                )
                vectorstore.vectorize_from_local_directory(str(self.data_dir))
            else:
                self.console.print(
                    f"[green]Loaded existing index for {model_name} with {len(vectorstore.documents)} documents[/]"
                )

            self.vectorstores[model_name] = vectorstore

    def compare_responses(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Compare les réponses des différents modèles pour une même requête.

        Args:
            query: Question ou requête à poser
            k: Nombre de résultats à retourner par modèle

        Returns:
            DataFrame contenant les comparaisons
        """
        results = []

        for model_name, vectorstore in self.vectorstores.items():
            similar_docs = vectorstore.similarity_search(query, k=k)

            for rank, doc in enumerate(similar_docs, 1):
                results.append(
                    {
                        "model": model_name,
                        "rank": rank,
                        "score": doc["score"],
                        "content": doc["content"][:300]
                        + "...",  # Tronquer pour la lisibilité
                        "metadata": str(doc.get("metadata", {})),
                    }
                )

        return pd.DataFrame(results)

    def display_comparison_table(self, df: pd.DataFrame) -> None:
        """Affiche un tableau formaté des résultats."""
        table = Table(title="Embedding Models Comparison")

        table.add_column("Model", style="cyan")
        table.add_column("Rank", style="magenta")
        table.add_column("Score", style="green")
        table.add_column("Content Preview", style="blue")
        table.add_column("Metadata", style="yellow")

        for _, row in df.iterrows():
            table.add_row(
                row["model"],
                str(row["rank"]),
                f"{row['score']:.4f}",
                row["content"],
                row["metadata"],
            )

        self.console.print(table)


def main():
    # Configuration des modèles à comparer
    models_config = [
        {
            "name": "MPNet",
            "model": "all-mpnet-base-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "name": "MiniLM",
            "model": "all-MiniLM-L6-v2",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "name": "nomic-embed-text",
            "model": "ollama!nomic-embed-text",
            # on met ollama! au début pour signaler qu'il faut le chercher sur OllamaEmbedding
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "name": "Instructor L",
            "model": "hkunlp/instructor-large",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
    ]

    # Initialiser le comparateur
    comparator = EmbeddingComparator(
        data_dir="C:/Users/elias/yugioh_cardlist_scraper/data/en",
        models_config=models_config,
        index_base_path="embedding_indexes",
    )

    # Initialiser tous les vectorstores
    comparator.initialize_vectorstores()

    while True:
        # Interface utilisateur simple
        query = input("\nEntrez votre question (ou 'q' pour quitter): ")
        if query.lower() == "q":
            break

        # Comparer les réponses
        results_df = comparator.compare_responses(query)

        # Afficher les résultats
        comparator.display_comparison_table(results_df)

        # Sauvegarder les résultats dans un CSV (optionnel)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"comparison_results/{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
