from typing import List, Dict, Union, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.schema.vectorstore import VectorStoreRetriever
import faiss
import numpy as np
import os
import pickle
from pathlib import Path
import pandas as pd
import glob


class VectorStoreFAISS:
    def __init__(
        self,
        embedding_model_name: str = "all-mpnet-base-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        index_path: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name

        # Extraire le nom du modèle sans le préfixe du fournisseur
        model_name = embedding_model_name.split("/")[-1]

        # Construire le chemin complet pour l'index
        if index_path:
            base_path = Path(index_path)
            # Créer le dossier embedding_indexes s'il n'existe pas
            base_path.mkdir(parents=True, exist_ok=True)
            # Créer le sous-dossier spécifique au modèle
            model_path = base_path / model_name
            model_path.mkdir(exist_ok=True)
            # Le fichier index sera dans ce sous-dossier
            self.index_path = model_path
        else:
            self.index_path = None

        # Initialize embeddings
        if "ollama!" in embedding_model_name:
            self.embeddings = OllamaEmbeddings(
                model=embedding_model_name.replace("ollama!", ""),
            )
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        self.documents: List[Document] = []
        self.index_faiss: Optional[FAISS] = None

        # Initialize the index
        if self.index_path and self.index_path.parent.exists():
            self.load_index()

    def load_index(self) -> None:
        """Charge l'index FAISS et les documents associés depuis le disque."""
        if not self.index_path:
            raise ValueError("No index path specified")

        # Construire les chemins corrects
        index_file = self.index_path  # / self.embedding_model_name.split("/")[-1]
        docs_path = (
            self.index_path
            / f"{self.embedding_model_name.split('/')[-1]}_documents.pkl"
        )

        print(f"Looking for index at {index_file}...")

        if not index_file.exists():
            print(
                "No existing index found. A new one will be created when documents are added."
            )
            return

        try:
            self.index_faiss = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
                index_name=self.embedding_model_name.split("/")[-1],
            )

            # Load documents if available
            if docs_path.exists():
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                print(f"Successfully loaded index with {len(self.documents)} documents")
            else:
                print("Warning: Documents file not found")
        except Exception as e:
            print(f"Error loading index: {e}")

    def save_index(self) -> None:
        """Sauvegarde l'index FAISS et les documents sur le disque."""
        if not self.index_path or not self.index_faiss:
            raise ValueError("No index path specified or no index to save")

        # Create parent directory if necessary
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save index
        self.index_faiss.save_local(
            str(self.index_path),  # Utiliser le chemin complet jusqu'au dossier 'index'
            index_name=self.embedding_model_name.split("/")[-1],
        )

        # Save documents
        docs_path = (
            self.index_path
            / f"{self.embedding_model_name.split('/')[-1]}_documents.pkl"
        )
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Successfully saved index with {len(self.documents)} documents")

    def vectorize_from_local_directory(self, directory_path: str) -> None:
        """Vectorise tous les fichiers supportés d'un répertoire local."""
        print(f"Loading documents from {directory_path}")
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

        docs = []

        for csv_file in csv_files:
            try:
                # Lire le CSV avec pandas
                df = pd.read_csv(csv_file, sep="$")

                # Convertir chaque ligne en document
                for idx, row in df.iterrows():
                    # Concaténer toutes les colonnes en un seul texte
                    text = " ".join(
                        str(value) for value in row.values if pd.notna(value)
                    )
                    metadata = {"source": os.path.basename(csv_file), "row": idx}

                    # Créer un Document Langchain
                    doc = Document(page_content=text, metadata=metadata)
                    docs.append(doc)

                print(f"Processed {len(df)} rows from {os.path.basename(csv_file)}")

            except Exception as e:
                print(f"Error processing CSV file {csv_file}: {e}")

        # Traitement des autres types de fichiers avec DirectoryLoader
        other_patterns = ["*.json", "*.pdf", "*.txt"]
        try:
            # Load .md files using TextLoader
            md_files = glob.glob(os.path.join(directory_path, "*.md"))
            md_documents = []
            for file_path in md_files:
                loader = TextLoader(file_path, encoding="utf-16le")
                md_documents.extend(loader.load())  # Load documents from .md files

            # Load other files using DirectoryLoader
            other_documents = []
            for pattern in other_patterns:
                loader = DirectoryLoader(directory_path, glob=pattern)
                other_documents.extend(
                    loader.load()
                )  # Load documents from other file types

            # Combine all documents
            documents = md_documents + other_documents
            docs.extend(documents)

            # Now `documents` contains all the loaded documents
            print(f"Total documents loaded: {len(docs)}")

        except Exception as e:
            print(f"Erreur : {str(e)}")
        """
        for loader in loaders:
            try:
                new_docs = loader.load()
                print(f"Loaded {len(new_docs)} documents from pattern {loader.glob}")
                docs.extend(new_docs)
            except Exception as e:
                print(f"Error loading files with {loader}: {e}")
        """

        if docs:
            print(f"Total documents to process: {len(docs)}")
            processed_docs = self.text_splitter.split_documents(docs)
            print(f"Split into {len(processed_docs)} chunks")
            self.add_documents(processed_docs)
        else:
            print(f"No documents were found in {directory_path}")

    def add_documents(self, docs: List[Document]) -> None:
        """Ajoute des documents à l'index."""
        if not docs:
            print("No documents to add")
            return

        self.documents.extend(docs)
        print(f"Adding {len(docs)} documents to index")

        if self.index_faiss is None:
            print("Initializing new FAISS index")
            self.index_faiss = FAISS.from_documents(docs, self.embeddings)
        else:
            print("Adding documents to existing index")
            self.index_faiss.add_documents(docs)

        if self.index_path:
            print("Saving updated index")
            self.save_index()

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Union[str, float]]]:
        """Recherche les documents les plus similaires à une requête."""
        if not self.index_faiss:
            print("Warning: No documents have been indexed yet")
            return []

        try:
            docs_and_scores = self.index_faiss.similarity_search_with_relevance_scores(
                query, k=k
            )
            results = []
            for doc, score in docs_and_scores:
                content = str(doc.page_content) if doc.page_content else ""

                # Créer un dictionnaire avec les résultats
                result = {
                    "content": content,
                    "metadata": dict(doc.metadata) if doc.metadata else {},
                    "score": float(score),
                }
                results.append(result)

            return results
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
