from typing import Optional, List
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from pydantic import BaseModel
import os
import pickle

class OrientationCriteria(BaseModel):
    diplome: str
    domaine_interet: str
    user_query: Optional[str] = None

class OrientationChatbot:
    def __init__(self, model_name: str = "myqwen:latest", base_url: str = "http://localhost:11434", vector_store_dir: str = "vectorstores"):
        self.llm = OllamaLLM(model=model_name, base_url=base_url)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.df = None
        self.vector_store = None
        self.rag_chain = None
        self.vector_store_dir = vector_store_dir
        self.index_file = os.path.join(vector_store_dir, "faiss_index.bin")
        self.docstore_file = os.path.join(vector_store_dir, "docstore.pkl")

        os.makedirs(vector_store_dir, exist_ok=True)

    def load_data(self, file_path: str = "../../formations/formations.csv", force_rebuild: bool = False):
        """Charge les données depuis le CSV avec option de charger depuis le stockage local"""
        self.df = pd.read_csv(file_path)
        
        # Check if we can load from existing vector store
        if not force_rebuild and self._load_vector_store():
            print("Loaded existing vector store from disk")
            self._setup_rag_chain()
            return
            
        print("Building new vector store")
        docs = self._dataframe_to_docs(self.df)
        self._setup_vector_store(docs)
        self._save_vector_store()
        self._setup_rag_chain()

    def _load_vector_store(self) -> bool:
        """Charge le vector store depuis le stockage local"""
        try:
            if not (os.path.exists(self.index_file) and os.path.exists(self.docstore_file)):
                return False
                
            #Load the FAISS index
            index = faiss.read_index(self.index_file)
            
            #Load the document store
            with open(self.docstore_file, 'rb') as f:
                docstore_data = pickle.load(f)
                docstore = InMemoryDocstore(docstore_data['docstore'])
                index_to_docstore_id = docstore_data['index_to_id']
            
            #Recreate the FAISS vector store
            self.vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )
            return True
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def _save_vector_store(self):
        """Sauvegarde le vector store localement"""
        try:
            #Save the FAISS index
            faiss.write_index(self.vector_store.index, self.index_file)
            
            #Save the document store and the index mapping
            docstore_data = {
                'docstore': self.vector_store.docstore._dict,
                'index_to_id': self.vector_store.index_to_docstore_id
            }
            with open(self.docstore_file, 'wb') as f:
                pickle.dump(docstore_data, f)
                
            print(f"Vector store saved to {self.vector_store_dir}")
            
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def _dataframe_to_docs(self, df: pd.DataFrame) -> List[Document]:
        """Convertit le DataFrame en documents avec métadonnées"""
        docs = []
        for idx, row in df.iterrows():
            # Ajout des métadonnées pertinentes pour le filtrage
            metadata = {
                "formation": str(row.get("Nom de la formation", "")).lower(),
                "domaine": str(row.get("Domaine", "")).lower(),
                "diplome": str(row.get("Diplôme", "")).lower(),
                "type": str(row.get("Régime(s) d études", "")).lower(),
                "row": idx
            }
            # Création du contenu du document
            content = f"Formation: {row.get('Nom de la formation', '')}. "
            content += f"Domaine: {row.get('Domaine', '')}. "
            content += f"Diplôme: {row.get('Diplôme', '')}. "
            content += f"Régime d'étude: {row.get('Régime d étude', '')}. "
            content += f"Disciplines: {row.get('Discipline(s)', '')}. "
            content += f"Secteur d'activité: {row.get('Secteur(s) d activité', '')}. "
            content += f"Niveau d'entrée: {row.get('Niveau d entrée', '')}. "

            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
        return docs
    

    def _setup_vector_store(self, docs: List[Document]):
        """Configure le vector store avec les documents"""
        dimension = 384
        index = faiss.IndexFlatL2(dimension)
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        self.vector_store.add_documents(documents=docs)

    def _setup_rag_chain(self):
        """Configure la chaîne RAG avec un prompt personnalisé"""
        system_prompt = """
        Tu es un assistant spécialisé dans l'orientation étudiante de l'Université de Haute-Alsace (UHA).
        Utilise le contexte suivant pour répondre aux questions sur les formations.
        Présente les formations de manière claire et structurée.
        Mentionne uniquement les formations qui correspondent exactement aux critères.
        
        Context: {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}  # Récupère les 10 documents les plus pertinents
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    def process_orientation(self, criteria: OrientationCriteria) -> str:
        """Traite une demande d'orientation avec filtrage"""
        #Construction de la requête sous forme de mots clés
        keys_criteria_query = f"{criteria.diplome} {criteria.domaine_interet}".strip()

        #documents pertinents
        retriever = self.vector_store.as_retriever()
        docs = retriever.invoke(keys_criteria_query)

        print("Docs similaires")
        for doc in docs:
            print(repr(doc))
            print()
        
        context = "\n".join(doc.page_content for doc in docs)

        str_criteria_action = f"Liste et décris les formations trouvées qui correspondent aux critères suivants : "
        str_criteria=  (f"diplome {criteria.diplome}, "
            f"domaine {criteria.domaine_interet}"
            + f"\nContexte :\n"
            f"{context}")
        
        if criteria.user_query:
            print("USER QUERY")

            str_questions_action = f"Réponds à la question en prenant en compte les critères."
            str_questions = (f"Question :\n"
                            f"{criteria.user_query}")
            
            response = self.rag_chain.invoke({
                "input": f"{str_questions_action}{str_criteria}{str_questions}"
            })
        else:
            print("NO USER QUERY")
            response = self.rag_chain.invoke({
                "input": f"{str_criteria_action}{str_criteria}"
            })

        return response["answer"]