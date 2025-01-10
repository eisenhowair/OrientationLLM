from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document

# Configuration initiale
llm = Ollama(
    model="qwen2.5:7b",  
    base_url="http://localhost:11434"
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chargement des données dans un DataFrame
file_path = '../../../formations/formations.csv'
df = pd.read_csv(file_path)

# Conversion du DataFrame en documents
def dataframe_to_docs(df):
    docs = []
    for idx, row in df.iterrows():
        # Concaténation des colonnes en une seule chaîne
        content = " ".join(str(value) for value in row.values)
        metadata = {"source": file_path, "row": idx}
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)
    return docs

# Alternative: si vous voulez des colonnes spécifiques
def dataframe_to_docs_specific_columns(df, content_columns, metadata_columns=None):
    docs = []
    for idx, row in df.iterrows():
        # Création du contenu à partir des colonnes spécifiées
        content = " ".join(str(row[col]) for col in content_columns)
        
        # Création des métadonnées
        metadata = {"source": file_path, "row": idx}
        if metadata_columns:
            metadata.update({col: row[col] for col in metadata_columns})
            
        doc = Document(page_content=content, metadata=metadata)
        docs.append(doc)
    return docs

# Création des documents
# Option 1: Utiliser toutes les colonnes
docs = dataframe_to_docs(df)

# Option 2: Spécifier les colonnes (décommenter pour utiliser)
# content_columns = ['nom_formation', 'description']  # Ajustez selon vos colonnes
# metadata_columns = ['niveau', 'discipline']        # Ajustez selon vos colonnes
# docs = dataframe_to_docs_specific_columns(df, content_columns, metadata_columns)

# Configuration du vector store
dimension = 384
index = faiss.IndexFlatL2(dimension)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Ajout des documents au vector store
vector_store.add_documents(documents=docs)

# Configuration du RAG
retriever = vector_store.as_retriever()
system_prompt = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer 
the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the 
answer concise.

{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Création de la chaîne question-réponse
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Test
answer = rag_chain.invoke({"input": "Liste moi 10 formations"})
print(answer['answer'])