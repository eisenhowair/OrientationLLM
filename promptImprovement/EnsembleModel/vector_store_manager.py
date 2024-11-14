from typing import List, Dict, Union, Optional, Any
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
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
        self.index_path = Path(index_path) if index_path else None

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
        self.langchain_faiss: Optional[FAISS] = None

        # Initialize the index
        if self.index_path and self.index_path.exists():
            self.load_index()

    def load_index(self) -> None:
        """Charge l'index FAISS et les documents associés depuis le disque."""
        if not self.index_path:
            raise ValueError("No index path specified")

        print(f"Loading index from {self.index_path}...")
        try:
            self.langchain_faiss = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            # Load documents if available
            docs_path = self.index_path.parent / f"{self.index_path.stem}_documents.pkl"
            if docs_path.exists():
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
            print(f"Successfully loaded index with {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Don't initialize empty index here, let it be initialized when documents are added

    def save_index(self) -> None:
        """Sauvegarde l'index FAISS et les documents sur le disque."""
        if not self.index_path or not self.langchain_faiss:
            raise ValueError("No index path specified or no index to save")

        # Create parent directory if necessary
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save index
        self.langchain_faiss.save_local(str(self.index_path))

        # Save documents
        docs_path = self.index_path.parent / f"{self.index_path.stem}_documents.pkl"
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"Successfully saved index with {len(self.documents)} documents")

    def vectorize_from_local_directorys_csv_specifique(
        self, directory_path: str
    ) -> None:
        """Vectorise tous les fichiers CSV d'un répertoire local."""
        print(f"Loading documents from {directory_path}")

        # Rechercher tous les fichiers CSV dans le répertoire
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
        print(f"Found {len(csv_files)} CSV files")

        docs = []
        for csv_file in csv_files:
            try:
                # Lire le CSV avec pandas
                df = pd.read_csv(csv_file)

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
                print(f"Error processing {csv_file}: {e}")

        if docs:
            print(f"Total documents to process: {len(docs)}")
            # Découper les documents en chunks si nécessaire
            processed_docs = self.text_splitter.split_documents(docs)
            print(f"Split into {len(processed_docs)} chunks")
            self.add_documents(processed_docs)
        else:
            print(f"No documents were successfully processed from {directory_path}")

    def vectorize_from_local_directory(self, directory_path: str) -> None:
        """Vectorise tous les fichiers supportés d'un répertoire local."""
        print(f"Loading documents from {directory_path}")
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

        docs = []

        # Traiter les fichiers CSV comme dans la première fonction
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
        loaders = [
            DirectoryLoader(directory_path, glob=pattern) for pattern in other_patterns
        ]

        for loader in loaders:
            try:
                new_docs = loader.load()
                print(f"Loaded {len(new_docs)} documents from pattern {loader.glob}")
                docs.extend(new_docs)
            except Exception as e:
                print(f"Error loading files with {loader}: {e}")

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

        if self.langchain_faiss is None:
            print("Initializing new FAISS index")
            self.langchain_faiss = FAISS.from_documents(docs, self.embeddings)
        else:
            print("Adding documents to existing index")
            self.langchain_faiss.add_documents(docs)

        if self.index_path:
            print("Saving updated index")
            self.save_index()

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Dict[str, Union[str, float]]]:
        """Recherche les documents les plus similaires à une requête."""
        if not self.langchain_faiss:
            print("Warning: No documents have been indexed yet")
            return []

        try:
            docs_and_scores = self.langchain_faiss.similarity_search_with_score(
                query, k=k
            )
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
                for doc, score in docs_and_scores
            ]
        except Exception as e:
            print(f"Error during similarity search: {e}")
            return []
