from typing import List, Dict, Union, Optional, Any
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
    OllamaEmbeddings,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.schema.vectorstore import VectorStoreRetriever
import faiss
import numpy as np
import os
import pickle
from pathlib import Path


class VectorStoreFAISS:
    """
    Classe améliorée pour gérer un vectorstore avec FAISS, avec support de sauvegarde/chargement
    et intégration facilitée avec LangChain.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-mpnet-base-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_path: Optional[str] = None,
    ):
        """
        Initialise le VectorStoreFAISS.

        Args:
            embedding_model_name (str): Nom du modèle d'embedding
            chunk_size (int): Taille des chunks pour la vectorisation
            chunk_overlap (int): Overlap entre les chunks
            index_path (str, optional): Chemin où sauvegarder/charger l'index
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.index_path = Path(index_path) if index_path else None

        # Initialisation du modèle d'embedding
        if embedding_model_name == "openai":
            self.embeddings = OpenAIEmbeddings()
        elif embedding_model_name == "ollama":
            self.embeddings = OllamaEmbeddings(
                base_url="http://localhost:11434",
                model=embedding_model_name,
                show_progress="true",
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        # Stockage des documents et de l'index FAISS de LangChain
        self.documents: List[Document] = []
        self.langchain_faiss: Optional[FAISS] = None

        # Charger l'index existant si spécifié et disponible
        if self.index_path and self.index_path.exists():
            self.load_index()

    def load_index(self) -> None:
        """Charge l'index FAISS et les documents associés depuis le disque."""
        if not self.index_path:
            raise ValueError("No index path specified")

        print(f"Loading index from {self.index_path}...")
        self.langchain_faiss = FAISS.load_local(
            str(self.index_path), self.embeddings, allow_dangerous_deserialization=True
        )

        # Charger les documents si disponibles
        docs_path = self.index_path.parent / f"{self.index_path.stem}_documents.pkl"
        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

    def save_index(self) -> None:
        """Sauvegarde l'index FAISS et les documents sur le disque."""
        if not self.index_path or not self.langchain_faiss:
            raise ValueError("No index path specified or no index to save")

        # Créer le dossier parent si nécessaire
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder l'index
        self.langchain_faiss.save_local(str(self.index_path))

        # Sauvegarder les documents
        docs_path = self.index_path.parent / f"{self.index_path.stem}_documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def add_texts(self, texts: List[str]) -> None:
        """
        Ajoute des textes bruts à l'index.

        Args:
            texts (List[str]): Liste des textes à ajouter
        """
        docs = self.text_splitter.create_documents(texts)
        self.add_documents(docs)

    def add_documents(self, docs: List[Document]) -> None:
        """
        Ajoute des documents à l'index.

        Args:
            docs (List[Document]): Liste des documents à ajouter
        """
        self.documents.extend(docs)

        if self.langchain_faiss is None:
            self.langchain_faiss = FAISS.from_documents(docs, self.embeddings)
        else:
            self.langchain_faiss.add_documents(docs)

        if self.index_path:
            self.save_index()

    def vectorize_from_urls(self, urls: List[str]) -> None:
        """Vectorise le contenu de plusieurs URLs."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        self.add_documents(docs)

    def vectorize_from_local_directory(self, directory_path: str) -> None:
        """Vectorise tous les fichiers supportés d'un répertoire local."""
        loaders = [
            DirectoryLoader(directory_path, glob=pattern)
            for pattern in ["*.csv", "*.json", "*.pdf", "*.txt"]
        ]

        docs = []
        for loader in loaders:
            try:
                docs.extend(loader.load())
            except Exception as e:
                print(f"Error loading files with {loader}: {e}")

        if docs:
            self.add_documents(docs)

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Recherche les documents les plus similaires à une requête.

        Args:
            query (str): Requête de recherche
            k (int): Nombre de résultats à retourner

        Returns:
            List[Dict[str, Union[str, float]]]: Documents similaires avec scores
        """
        if not self.langchain_faiss:
            raise ValueError("No index available for search")

        docs_and_scores = self.langchain_faiss.similarity_search_with_score(query, k=k)

        return [
            {"content": doc.page_content, "metadata": doc.metadata, "score": score}
            for doc, score in docs_and_scores
        ]

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """
        Retourne un retrieveur LangChain compatible pour l'utilisation dans des chaînes.

        Returns:
            VectorStoreRetriever: Le retrieveur LangChain
        """
        if not self.langchain_faiss:
            raise ValueError("No index available")

        return self.langchain_faiss.as_retriever(**kwargs)
