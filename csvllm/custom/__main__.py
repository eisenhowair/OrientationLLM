import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import requests
from typing import List, Dict
import numpy as np

class CustomHuggingFaceEmbeddings(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

class CSVRAGSystem:
    def __init__(self, ollama_base_url: str = "http://localhost:11434", 
                 model_name: str = "qwen2.5:7b"):
        # Initialize ChromaDB
        self.client = chromadb.Client()
        
        # Initialize HuggingFace embedding function
        self.embedding_function = CustomHuggingFaceEmbeddings()
        
        # Ollama configuration
        self.ollama_url = ollama_base_url
        self.model_name = model_name

    def prepare_documents(self, csv_path: str, text_columns: List[str]) -> pd.DataFrame:
        """
        Prépare les documents à partir du CSV en combinant les colonnes spécifiées
        """
        df = pd.read_csv(csv_path)
        print(df)
        # Combine specified columns into a single text field
        df['combined_text'] = df[text_columns].apply(
            lambda row: ' | '.join(str(cell) for cell in row),
            axis=1
        )
        return df

    def create_collection(self, collection_name: str, df: pd.DataFrame, metadata_columns: List[str] = None):
        """
        Crée une collection Chroma et y ajoute les documents
        """
        # Créer (ou obtenir si elle existe) la collection
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        documents = df['combined_text'].tolist()
        ids = [str(i) for i in range(len(documents))]
        
        # Préparer les métadonnées si spécifiées
        metadatas = None
        if metadata_columns:
            metadatas = df[metadata_columns].to_dict('records')
        
        # Ajouter les documents à la collection
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        return collection

    def query_collection(self, 
                        collection_name: str, 
                        query: str, 
                        n_results: int = 5,
                        metadata_filter: Dict = None):
        """
        Interroge la collection avec une question
        """
        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        results = collection.query(
            query_texts=query,
            n_results=n_results,
            where=metadata_filter
        )
        
        return results

    def get_llm_response(self, query: str, context: str) -> str:
        """
        Obtient une réponse du modèle Ollama en utilisant le contexte
        """
        prompt = f"""Context: {context}

Question: {query}

Please provide a concise answer based on the context above."""

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Error from Ollama API: {response.text}")

    def query_with_rag(self, collection_name: str, query: str, n_results: int = 3):
        """
        Effectue une requête RAG complète : recherche + réponse LLM
        """
        # Récupérer les documents pertinents
        results = self.query_collection(collection_name, query, n_results)
        context = "\n".join(results["documents"][0])
        
        # Obtenir la réponse du LLM
        answer = self.get_llm_response(query, context)
        
        return {
            "answer": answer,
            "sources": self.format_results(results)
        }

    def format_results(self, results: Dict) -> List[Dict]:
        """
        Formate les résultats de manière lisible
        """
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            result = {
                'document': results['documents'][0][i],
                'distance': results['distances'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else None
            }
            formatted_results.append(result)
            
        return formatted_results

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    CSV_PATH = "../../formations/formations.csv"

    TEXT_COLUMNS = ["Diplôme","Nom de la formation", "Discipline(s)", "Secteur(s) d'activité", "Domaine"]  # Colonnes à combiner pour le texte
    METADATA_COLUMNS = ["Régime(s) d'études", "Niveau d'entrée"]  # Colonnes à utiliser comme métadonnées
    
    # Initialisation du système
    rag_system = CSVRAGSystem()  # Par défaut utilise mistral sur localhost:11434
    
    # Préparation des données
    df = rag_system.prepare_documents(CSV_PATH, TEXT_COLUMNS)
    # for d in df:
    #     print(d)
    #     print(df[d])
    #     print("---------")

    
    # Création de la collection
    collection = rag_system.create_collection(
        "ma_collection",
        df,
        METADATA_COLUMNS
    )
    
    # Exemple de requête RAG
    query = "Combien de diplôme B.U.T existe t-il ?"
    response = rag_system.query_with_rag("ma_collection", query)
    
    # Affichage de la réponse
    print("\nRéponse du LLM:")
    print(response["answer"])
    print("\nSources utilisées:")
    for i, source in enumerate(response["sources"], 1):
        print(f"\nSource {i}:")
        print(f"Document: {source['document']}")
        print(f"Distance: {source['distance']}")
        if source['metadata']:
            print(f"Métadonnées: {source['metadata']}")
    